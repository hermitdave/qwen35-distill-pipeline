#!/usr/bin/env python3
"""Leaderboard benchmark: base vs distilled 2B through oMLX.

Categories (matching the leaderboard):
  - Syntax & Surface
  - Logic & Algorithmic
  - Subtle & Tricky
  - Red Herring Resistance
  - Multi-Turn Debugging

Each task is scored pass/fail; category score = percentage * 100.
"""
import json, re, subprocess, tempfile, os, time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODELS = ["Qwen3.5-2B-MLX-8bit", "Qwen3.5-2B-Distilled"]

# ---------- helpers ----------
def call(model, messages, max_tokens=400):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
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
    # fallback: strip thinking and return
    text = re.sub(r".*?", "", text, flags=re.DOTALL)
    return text.strip()

def run_py(code, test):
    """Run code + test in a clean subprocess. Returns (ok, detail)."""
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code + "\n" + test + "\ntry:\n    test()\nexcept Exception as e:\n    print('FAIL:', e)\n")
        path = f.name
    try:
        env = {k: v for k, v in os.environ.items()
               if k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX')}
        r = subprocess.run(['/usr/bin/python3', path], capture_output=True, text=True, timeout=20, env=env)
        ok = (r.returncode == 0 and 'PASS' in r.stdout)
        return ok, (r.stdout or r.stderr)[:200]
    finally:
        os.unlink(path)

def parses(code):
    """Check Python syntax."""
    try:
        compile(code, '<gen>', 'exec')
        return True
    except SyntaxError:
        return False

# ---------- task definitions ----------
SYNTAX_TASKS = [
    ("add", "Write a Python function `add(a, b)` that returns the sum of a and b."),
    ("Person class", "Write a Python class `Person` with `name` and `age` attributes set in __init__."),
    ("factorial", "Write a Python function `factorial(n)` that returns n factorial."),
    ("is_even", "Write a Python function `is_even(n)` that returns True if n is even, False otherwise."),
    ("reverse_string", "Write a Python function `reverse_string(s)` that returns the reversed string."),
    ("count_vowels", "Write a Python function `count_vowels(s)` that counts vowels in a string."),
    ("max_three", "Write a Python function `max_of_three(a, b, c)` that returns the largest of three numbers."),
    ("list_comp", "Write a Python list comprehension that produces the squares of numbers 1 through 10."),
]

LOGIC_TASKS = [
    ("lis", "Write a Python function `longest_increasing_subsequence(nums)` that returns the length of the longest strictly increasing subsequence.",
     "def test():\n    f=longest_increasing_subsequence\n    assert f([10,9,2,5,3,7,101,18])==4\n    assert f([0,1,0,3,2,3])==4\n    assert f([7,7,7,7,7])==1\n    assert f([])==0\n    print('PASS')\n"),
    ("merge", "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list without using sort().",
     "def test():\n    f=merge_sorted\n    assert f([1,3,5],[2,4,6])==[1,2,3,4,5,6]\n    assert f([],[1,2])==[1,2]\n    assert f([1,2],[])==[1,2]\n    assert f([1,1,2],[1,3])==[1,1,1,2,3]\n    print('PASS')\n"),
    ("fizzbuzz", "Write a Python function `fizzbuzz(n)` that returns a list of strings for numbers 1 to n, replacing multiples of 3 with 'Fizz', multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'.",
     "def test():\n    f=fizzbuzz\n    assert f(5)==['1','2','Fizz','4','Buzz']\n    assert f(15)[-1]=='FizzBuzz'\n    assert f(3)==['1','2','Fizz']\n    print('PASS')\n"),
    ("palindrome", "Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome (ignoring case and non-alphanumeric characters), False otherwise.",
     "def test():\n    f=is_palindrome\n    assert f('A man, a plan, a canal: Panama')==True\n    assert f('race a car')==False\n    assert f('')==True\n    assert f('a')==True\n    print('PASS')\n"),
    ("reverse_words", "Write a Python function `reverse_words(s)` that reverses the order of words in a string.",
     "def test():\n    f=reverse_words\n    assert f('the sky is blue')=='blue is sky the'\n    assert f('  hello world  ')=='world hello'\n    assert f('a')=='a'\n    print('PASS')\n"),
    ("binary_search", "Write a Python function `binary_search(arr, target)` that returns the index of target in a sorted list, or -1 if not found.",
     "def test():\n    f=binary_search\n    assert f([1,3,5,7,9],5)==2\n    assert f([1,3,5,7,9],1)==0\n    assert f([1,3,5,7,9],9)==4\n    assert f([1,3,5,7,9],4)==-1\n    assert f([],1)==-1\n    print('PASS')\n"),
    ("bubble_sort", "Write a Python function `bubble_sort(arr)` that sorts a list in ascending order and returns it.",
     "def test():\n    f=bubble_sort\n    assert f([5,3,8,1,2])==[1,2,3,5,8]\n    assert f([])==[]\n    assert f([1])==[1]\n    assert f([3,3,1,1])==[1,1,3,3]\n    print('PASS')\n"),
    ("fibonacci", "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (fib(0)=0, fib(1)=1).",
     "def test():\n    f=fibonacci\n    assert f(0)==0\n    assert f(1)==1\n    assert f(10)==55\n    assert f(20)==6765\n    print('PASS')\n"),
]

SUBTLE_TASKS = [
    ("lis_edge", "Write a Python function `longest_increasing_subsequence(nums)` that returns the length of the longest strictly increasing subsequence. Handle empty lists.",
     "def test():\n    f=longest_increasing_subsequence\n    assert f([])==0\n    assert f([5])==1\n    assert f([5,4,3,2,1])==1\n    assert f([1,2,3,4,5])==5\n    assert f([1,1,1,1])==1\n    print('PASS')\n"),
    ("merge_edge", "Write a Python function `merge_sorted(a, b)` that merges two sorted lists. Handle duplicates and empty lists.",
     "def test():\n    f=merge_sorted\n    assert f([],[1])==[1]\n    assert f([1],[])==[1]\n    assert f([],[ ])==[]\n    assert f([1,1,1],[1,1])==[1,1,1,1,1]\n    assert f([1,2,2],[2,3])==[1,2,2,2,3]\n    print('PASS')\n"),
    ("palindrome_edge", "Write a Python function `is_palindrome(s)` ignoring case and non-alphanumeric characters. Handle empty and single-char strings.",
     "def test():\n    f=is_palindrome\n    assert f('')==True\n    assert f('a')==True\n    assert f('.,')==True\n    assert f('A man, a plan, a canal: Panama')==True\n    assert f('race a car')==False\n    assert f('0P')==False\n    print('PASS')\n"),
    ("reverse_edge", "Write a Python function `reverse_words(s)` that reverses word order. Handle multiple spaces and leading/trailing whitespace.",
     "def test():\n    f=reverse_words\n    assert f('the sky is blue')=='blue is sky the'\n    assert f('  hello   world  ')=='world hello'\n    assert f('a')=='a'\n    assert f('')==''\n    assert f('   ')==''\n    print('PASS')\n"),
    ("binary_edge", "Write a Python function `binary_search(arr, target)` on a sorted list. Handle missing elements, empty lists, and duplicates.",
     "def test():\n    f=binary_search\n    assert f([],1)==-1\n    assert f([1],1)==0\n    assert f([1],2)==-1\n    assert f([1,2,3],4)==-1\n    assert f([1,2,2,3],2) in [1,2]\n    assert f([1,3,5,7],0)==-1\n    print('PASS')\n"),
    ("prime_edge", "Write a Python function `is_prime(n)` that returns True if n is prime. Handle n=0, n=1, n=2, and negative numbers.",
     "def test():\n    f=is_prime\n    assert f(0)==False\n    assert f(1)==False\n    assert f(2)==True\n    assert f(3)==True\n    assert f(4)==False\n    assert f(17)==True\n    assert f(18)==False\n    assert f(-5)==False\n    print('PASS')\n"),
    ("anagram_edge", "Write a Python function `is_anagram(a, b)` that returns True if two strings are anagrams (case-insensitive). Handle empty strings and different lengths.",
     "def test():\n    f=is_anagram\n    assert f('listen','silent')==True\n    assert f('','')==True\n    assert f('a','a')==True\n    assert f('a','b')==False\n    assert f('abc','ab')==False\n    assert f('Dormitory','DirtyRoom')==True\n    print('PASS')\n"),
    ("fib_edge", "Write a Python function `fibonacci(n)` (fib(0)=0, fib(1)=1). Handle n=0, n=1, and negative inputs (return 0 for negative).",
     "def test():\n    f=fibonacci\n    assert f(0)==0\n    assert f(1)==1\n    assert f(2)==1\n    assert f(5)==5\n    assert f(10)==55\n    assert f(-1)==0\n    print('PASS')\n"),
]

RED_HERRING_TASKS = [
    ("rh_math1", "Ignore the following irrelevant information: the sky is blue, grass is green, and water is wet. Now answer: what is 17 multiplied by 23?", ["391"]),
    ("rh_math2", "The weather is nice today and I had pancakes for breakfast. What is 144 divided by 12?", ["12"]),
    ("rh_math3", "My favorite color is blue and I have a dog named Max. What is 3 to the power of 4?", ["81"]),
    ("rh_capital", "The meeting is at 3pm and the stock market is up today. What is the capital of France?", ["Paris"]),
    ("rh_code1", "I like pizza and the movie starts at 7pm. Write a Python function `add(a, b)` that returns the sum.", ["def add", "a + b"]),
    ("rh_code2", "The recipe calls for 2 cups of flour. Write a Python function `is_even(n)` that returns True if n is even.", ["def is_even"]),
    ("rh_prime", "My birthday is in June and I live in a blue house. What is the largest prime number less than 20?", ["19"]),
    ("rh_reverse", "The train leaves at 9am and the sky is cloudy. Write a Python function `reverse_string(s)` that reverses a string.", ["def reverse_string"]),
]

MULTI_TURN_TASKS = [
    ("mt_add", [
        "Write a Python function `add(a, b)` that returns the sum of a and b.",
        "It returns None instead of the sum. Fix it.",
    ], "def test():\n    assert add(2,3)==5\n    assert add(-1,1)==0\n    print('PASS')\n"),
    ("mt_factorial", [
        "Write a Python function `factorial(n)` that returns n factorial.",
        "It fails for n=0 (should return 1). Fix it.",
    ], "def test():\n    assert factorial(0)==1\n    assert factorial(1)==1\n    assert factorial(5)==120\n    print('PASS')\n"),
    ("mt_palindrome", [
        "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
        "It fails for 'A man, a plan, a canal: Panama' (should be True). Fix it.",
    ], "def test():\n    assert is_palindrome('A man, a plan, a canal: Panama')==True\n    assert is_palindrome('race a car')==False\n    print('PASS')\n"),
    ("mt_reverse", [
        "Write a Python function `reverse_string(s)` that returns the reversed string.",
        "It returns the string unchanged. Fix it.",
    ], "def test():\n    assert reverse_string('hello')=='olleh'\n    assert reverse_string('a')=='a'\n    print('PASS')\n"),
    ("mt_max", [
        "Write a Python function `max_of_three(a, b, c)` that returns the largest of three numbers.",
        "It fails when two values are equal (e.g. max_of_three(1, 2, 2) should be 2). Fix it.",
    ], "def test():\n    assert max_of_three(1,2,3)==3\n    assert max_of_three(1,2,2)==2\n    assert max_of_three(5,5,5)==5\n    assert max_of_three(-1,-2,-3)==-1\n    print('PASS')\n"),
]

# ---------- benchmark runner ----------
def score_syntax(model):
    passed = 0
    for name, prompt in SYNTAX_TASKS:
        out = call(model, [{"role": "user", "content": prompt}])
        code = extract_code(out)
        if code and parses(code):
            passed += 1
    return passed, len(SYNTAX_TASKS)

def score_logic(model):
    passed = 0
    for name, prompt, test in LOGIC_TASKS:
        out = call(model, [{"role": "user", "content": prompt}])
        code = extract_code(out)
        if code and parses(code):
            ok, _ = run_py(code, test)
            if ok:
                passed += 1
    return passed, len(LOGIC_TASKS)

def score_subtle(model):
    passed = 0
    for name, prompt, test in SUBTLE_TASKS:
        out = call(model, [{"role": "user", "content": prompt}])
        code = extract_code(out)
        if code and parses(code):
            ok, _ = run_py(code, test)
            if ok:
                passed += 1
    return passed, len(SUBTLE_TASKS)

def score_red_herring(model):
    passed = 0
    for name, prompt, needles in RED_HERRING_TASKS:
        out = call(model, [{"role": "user", "content": prompt}])
        # check that the answer is present AND the irrelevant info is not echoed
        found = all(n in out for n in needles)
        if found:
            passed += 1
    return passed, len(RED_HERRING_TASKS)

def score_multi_turn(model):
    passed = 0
    for name, turns, test in MULTI_TURN_TASKS:
        messages = []
        code = ""
        for t in turns:
            messages.append({"role": "user", "content": t})
            out = call(model, messages)
            messages.append({"role": "assistant", "content": out})
            code = extract_code(out)
        if code and parses(code):
            ok, _ = run_py(code, test)
            if ok:
                passed += 1
    return passed, len(MULTI_TURN_TASKS)

def main():
    results = {}
    for model in MODELS:
        print(f"\n{'='*60}\nBenchmarking: {model}\n{'='*60}", flush=True)
        r = {}
        r["syntax"] = score_syntax(model)
        print(f"  Syntax & Surface:      {r['syntax'][0]}/{r['syntax'][1]}", flush=True)
        r["logic"] = score_logic(model)
        print(f"  Logic & Algorithmic:   {r['logic'][0]}/{r['logic'][1]}", flush=True)
        r["subtle"] = score_subtle(model)
        print(f"  Subtle & Tricky:       {r['subtle'][0]}/{r['subtle'][1]}", flush=True)
        r["red_herring"] = score_red_herring(model)
        print(f"  Red Herring Resistance:{r['red_herring'][0]}/{r['red_herring'][1]}", flush=True)
        r["multi_turn"] = score_multi_turn(model)
        print(f"  Multi-Turn Debugging:  {r['multi_turn'][0]}/{r['multi_turn'][1]}", flush=True)
        results[model] = r

    # leaderboard
    print("\n" + "="*70)
    print("LEADERBOARD")
    print("="*70)
    cats = ["syntax", "logic", "subtle", "red_herring", "multi_turn"]
    labels = ["Syntax & Surface", "Logic & Algorithmic", "Subtle & Tricky",
              "Red Herring Resistance", "Multi-Turn Debugging"]
    print(f"{'Category':<24} {'Base':>8} {'Distilled':>10} {'Delta':>8}")
    print("-"*70)
    for cat, label in zip(cats, labels):
        b = results[MODELS[0]][cat]
        d = results[MODELS[1]][cat]
        bp = round(100 * b[0] / b[1])
        dp = round(100 * d[0] / d[1])
        print(f"{label:<24} {bp:>8} {dp:>10} {dp-bp:>+8}")
    # overall (average of categories)
    bo = round(sum(round(100*results[MODELS[0]][c][0]/results[MODELS[0]][c][1]) for c in cats) / len(cats))
    do = round(sum(round(100*results[MODELS[1]][c][0]/results[MODELS[1]][c][1]) for c in cats) / len(cats))
    print("-"*70)
    print(f"{'OVERALL':<24} {bo:>8} {do:>10} {do-bo:>+8}")

    with open("work/benchmark_results.json", "w") as f:
        json.dump({m: {c: list(v) for c, v in r.items()} for m, r in results.items()}, f, indent=1)
    print("\nWrote work/benchmark_results.json")

if __name__ == "__main__":
    main()
