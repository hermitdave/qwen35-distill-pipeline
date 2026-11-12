#!/usr/bin/env python3
"""Validate a LoRA adapter on Qwen3.5-2B: coding execution + general ability.

Usage:
  python validate_adapter.py <adapter_dir_or_none> <out_json>

If adapter_dir is "none", evaluates the base model.
"""
import sys, os, json, re, subprocess, tempfile
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import generate

MODEL = "/Users/hermit.dave/llm-models/Qwen3.5-2B-MLX-8bit"
CODING_TASKS = [
    ("lis", "Write a Python function `longest_increasing_subsequence(nums)` that returns the length of the longest strictly increasing subsequence. Return only the function, no extra text.",
     "def test():\n    f=longest_increasing_subsequence\n    assert f([10,9,2,5,3,7,101,18])==4\n    assert f([0,1,0,3,2,3])==4\n    assert f([7,7,7,7,7])==1\n    assert f([])==0\n    print('PASS')\n"),
    ("merge", "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list without using sort(). Return only the function.",
     "def test():\n    f=merge_sorted\n    assert f([1,3,5],[2,4,6])==[1,2,3,4,5,6]\n    assert f([],[1,2])==[1,2]\n    assert f([1,2],[])==[1,2]\n    assert f([1,1,2],[1,3])==[1,1,1,2,3]\n    print('PASS')\n"),
    ("fizzbuzz", "Write a Python function `fizzbuzz(n)` that returns a list of strings for numbers 1 to n, replacing multiples of 3 with 'Fizz', multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'. Return only the function.",
     "def test():\n    f=fizzbuzz\n    assert f(5)==['1','2','Fizz','4','Buzz']\n    assert f(15)[-1]=='FizzBuzz'\n    assert f(3)==['1','2','Fizz']\n    print('PASS')\n"),
    ("palindrome", "Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome (ignoring case and non-alphanumeric characters), False otherwise. Return only the function.",
     "def test():\n    f=is_palindrome\n    assert f('A man, a plan, a canal: Panama')==True\n    assert f('race a car')==False\n    assert f('')==True\n    assert f('a')==True\n    print('PASS')\n"),
    ("reverse", "Write a Python function `reverse_words(s)` that reverses the order of words in a string. Return only the function.",
     "def test():\n    f=reverse_words\n    assert f('the sky is blue')=='blue is sky the'\n    assert f('  hello world  ')=='world hello'\n    assert f('a')=='a'\n    print('PASS')\n"),
]
GENERAL_TASKS = [
    ("stack_queue", "Explain the difference between a stack and a queue in one paragraph."),
    ("capital", "What is the capital of France and what is it known for?"),
    ("bat_ball", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning."),
    ("formal", "Rewrite the following sentence in a formal tone: 'Hey, can you send me that file when you get a chance?'"),
    ("py_js", "What are the main differences between Python and JavaScript?"),
    ("red_herring", "Ignore the following irrelevant information: the sky is blue, grass is green, and water is wet. Now answer: what is 17 multiplied by 23?"),
]

def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    text = re.sub(r".*?", "", text, flags=re.DOTALL)
    return text.strip()

def run_code(code, test):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code + "\n" + test + "\ntry:\n    test()\nexcept Exception as e:\n    print('FAIL:', e)\n")
        path = f.name
    try:
        # clear PYTHONHOME/PYTHONPATH so the system python3 is not polluted by the oMLX framework
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONSTARTUP')}
        r = subprocess.run(['/usr/bin/python3', path], capture_output=True, text=True, timeout=20, env=env)
        return r.returncode, r.stdout.strip(), r.stderr.strip()[:300]
    finally:
        os.unlink(path)

def main():
    adapter_dir = sys.argv[1] if len(sys.argv) > 1 else "none"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/val_out.json"

    print(f"Loading model (adapter={adapter_dir})...", flush=True)
    model, tokenizer = load(MODEL)

    if adapter_dir != "none":
        cfg_path = os.path.join(adapter_dir, "adapter_config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        lp = cfg.get("lora_parameters", {"rank": 16, "dropout": 0.0, "scale": 16.0})
        num_layers = cfg.get("num_layers", 16)
        print(f"Applying LoRA rank={lp['rank']} scale={lp['scale']} num_layers={num_layers}", flush=True)
        linear_to_lora_layers(model, num_layers, lp)
        weights = mx.load(os.path.join(adapter_dir, "adapters.safetensors"))
        lora = {k: v for k, v in weights.items() if 'lora' in k}
        print(f"Loading {len(lora)} LoRA keys (of {len(weights)})...", flush=True)
        model.load_weights(lora, strict=False)
        mx.eval(model.parameters())

    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0)
    results = {"adapter": adapter_dir, "coding": [], "general": []}

    print("\n=== CODING TASKS ===", flush=True)
    for name, prompt, test in CODING_TASKS:
        out = generate(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=600, sampler=sampler)
        code = extract_code(out)
        rc, so, se = run_code(code, test)
        ok = (rc == 0 and 'PASS' in so)
        results["coding"].append({"task": name, "pass": ok, "code_len": len(code), "output": (so or se)[:200]})
        print(f"  {name:<12} {'PASS' if ok else 'FAIL':<6} len={len(code):<5} {(so or se).splitlines()[0][:60] if (so or se) else ''}", flush=True)

    print("\n=== GENERAL TASKS ===", flush=True)
    for name, prompt in GENERAL_TASKS:
        out = generate(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=400, sampler=sampler)
        results["general"].append({"task": name, "output": out[:500]})
        print(f"  {name:<14} len={len(out)}", flush=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {out_path}", flush=True)

if __name__ == "__main__":
    main()
