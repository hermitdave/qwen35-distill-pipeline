#!/usr/bin/env python3
"""Per-task benchmark with full logging: base vs distilled 2B.

Runs every task, records pass/fail, and for failures shows the generated code
and the error so we can tell a logic failure from a thinking-block artifact.
"""
import json, re, subprocess, tempfile, os, time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODELS = ["Qwen3.5-2B-MLX-8bit", "Qwen3.5-2B-Distilled"]

def call(model, messages, max_tokens=400):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["choices"][0]["message"].get("content", "")
        except Exception as e:
            if attempt == 2:
                return f"ERROR: {e}"
            time.sleep(2)
    return ""

def extract_code(text):
    m = re.search(r"```(?:python|javascript|js)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    text = re.sub(r".*?", "", text, flags=re.DOTALL)
    return text.strip()

def run_py(code, test):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code + "\n" + test + "\ntry:\n    test()\nexcept Exception as e:\n    print('FAIL:', e)\n")
        path = f.name
    try:
        env = {k: v for k, v in os.environ.items()
               if k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX')}
        r = subprocess.run(['/usr/bin/python3', path], capture_output=True, text=True, timeout=20, env=env)
        ok = (r.returncode == 0 and 'PASS' in r.stdout)
        return ok, (r.stdout or r.stderr)[:300]
    finally:
        os.unlink(path)

def parses(code):
    try:
        compile(code, '<gen>', 'exec')
        return True
    except SyntaxError as e:
        return False

# import tasks from benchmark.py
import sys
sys.path.insert(0, 'work')
from benchmark import SYNTAX_TASKS, LOGIC_TASKS, SUBTLE_TASKS, RED_HERRING_TASKS, MULTI_TURN_TASKS

def run_category(name, tasks, kind):
    print(f"\n{'='*80}\n{name}\n{'='*80}")
    for model in MODELS:
        print(f"\n--- {model} ---")
        for task in tasks:
            tname, prompt = task[0], task[1]
            if kind in ('syntax',):
                out = call(model, [{"role": "user", "content": prompt}])
                code = extract_code(out)
                ok = bool(code) and parses(code)
                detail = '' if ok else f"code_len={len(code)} parse={parses(code) if code else 'no code'}"
            elif kind in ('logic', 'subtle'):
                test = task[2]
                out = call(model, [{"role": "user", "content": prompt}])
                code = extract_code(out)
                if not code:
                    ok, detail = False, "no code extracted"
                elif not parses(code):
                    ok, detail = False, "syntax error"
                else:
                    ok, detail = run_py(code, test)
            elif kind == 'red_herring':
                needles = task[2]
                out = call(model, [{"role": "user", "content": prompt}])
                ok = all(n in out for n in needles)
                detail = '' if ok else f"missing {[n for n in needles if n not in out]}"
            elif kind == 'multi_turn':
                test = task[2]
                messages, code = [], ""
                for t in prompt:
                    messages.append({"role": "user", "content": t})
                    out = call(model, messages)
                    messages.append({"role": "assistant", "content": out})
                    code = extract_code(out)
                if not code:
                    ok, detail = False, "no code extracted"
                elif not parses(code):
                    ok, detail = False, "syntax error"
                else:
                    ok, detail = run_py(code, test)
            print(f"  {tname:<18} {'PASS' if ok else 'FAIL':<6} {detail}")
            if not ok:
                # save the raw output for inspection
                print(f"    raw output: {out[:400]!r}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "subtle"):
        run_category("SUBTLE & TRICKY", SUBTLE_TASKS, "subtle")
    if which in ("all", "logic"):
        run_category("LOGIC & ALGORITHMIC", LOGIC_TASKS, "logic")
    if which in ("all", "syntax"):
        run_category("SYNTAX & SURFACE", SYNTAX_TASKS, "syntax")
    if which in ("all", "red_herring"):
        run_category("RED HERRING RESISTANCE", RED_HERRING_TASKS, "red_herring")
    if which in ("all", "multi_turn"):
        run_category("MULTI-TURN DEBUGGING", MULTI_TURN_TASKS, "multi_turn")
