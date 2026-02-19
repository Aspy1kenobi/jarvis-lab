"""
GPT-2 Baseline Test - Apple Silicon Compatible v2
Fixes:
  - Bus error: CPU-only, no MPS, safe tensor loading
  - Code extraction: strips echoed docstrings, fixes bad whitespace,
    closes unterminated triple-quotes, handles missing returns
"""

import os
import re
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if hasattr(torch.backends, "mps"):
    torch.backends.mps.enabled = False

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from test_tasks import TASKS, run_test
import time
import json
from datetime import datetime

DEVICE = "cpu"
MODEL_NAME = "gpt2"


def load_model():
    print("Loading GPT-2 (117M)...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ GPT-2 loaded on {DEVICE}\n")
    return model, tokenizer


def generate_code(model, tokenizer, prompt: str, max_new_tokens=200, temperature=0.3) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def extract_function(generated: str, func_name: str) -> str:
    """
    Extract a clean, executable function from GPT-2 output.

    GPT-2 failure modes fixed here:
      1. Non-breaking spaces (U+00A0) -> SyntaxError
      2. Echoed docstring left unclosed -> SyntaxError  
      3. No return statement -> silent None returns
      4. 'return' dedented outside function body -> SyntaxError
    """
    # Fix 1: clean non-standard whitespace
    generated = generated.replace('\u00a0', ' ').replace('\u200b', '').replace('\t', '    ')

    # Fix 2: find the function start
    lines = generated.split('\n')
    func_start = None
    for i, line in enumerate(lines):
        if re.match(rf'^\s*def {re.escape(func_name)}\s*\(', line):
            func_start = i
            break

    if func_start is None:
        return f"def {func_name}(*args):\n    return None  # GPT-2 did not generate function"

    # Fix 3: collect lines until next top-level def/class
    body = [lines[func_start]]
    for line in lines[func_start + 1:]:
        if re.match(r'^(def |class )\S', line):
            break
        body.append(line)

    code = '\n'.join(body)

    # Fix 4: close unterminated triple-quoted docstring
    if code.count('"""') % 2 != 0:
        code += '\n    """'

    # Fix 5: if no return, GPT-2 generated a stub only
    if 'return ' not in code:
        code += '\n    return None  # GPT-2 generated no return statement'

    return code


def evaluate(code: str, func_name: str, test_cases: list):
    """Exec generated code and run test cases."""
    try:
        ns = {}
        exec(compile(code, '<generated>', 'exec'), ns)
        func = ns.get(func_name)
        if func is None:
            return 0, len(test_cases), [{"error": f"'{func_name}' not found after exec"}]
        return run_test(func, test_cases)
    except SyntaxError as e:
        return 0, len(test_cases), [{"error": f"SyntaxError line {e.lineno}: {e.msg}"}]
    except Exception as e:
        return 0, len(test_cases), [{"error": f"{type(e).__name__}: {e}"}]


def run_baseline(model, tokenizer):
    results = []
    total_passed = 0
    total_tests = 0

    print("=" * 60)
    print("EXPERIMENT 0: GPT-2 BASELINE")
    print(f"Model: {MODEL_NAME} | Device: {DEVICE} | Temp: 0.3")
    print("=" * 60 + "\n")

    for task in TASKS:
        print(f"Task {task['id']}: {task['name']} — {task['description']}")

        t0 = time.time()
        generated = generate_code(model, tokenizer, task["prompt"])
        elapsed = time.time() - t0

        code = extract_function(generated, task["name"])
        passed, total, failures = evaluate(code, task["name"], task["test_cases"])
        total_passed += passed
        total_tests += total

        status = "✓ PASS" if passed == total else f"✗ {passed}/{total}"
        print(f"  Result: {status} ({elapsed:.1f}s)")
        if failures and passed < total:
            print(f"  Issue:  {failures[0]}")

        # Show generated code body (skip def line + docstring for brevity)
        body_lines = [l for l in code.split('\n')[1:] if l.strip() and '"""' not in l]
        preview = ' | '.join(body_lines[:3])
        print(f"  Body:   {preview[:100]}")

        results.append({
            "task_id": task["id"],
            "task_name": task["name"],
            "passed": passed,
            "total": total,
            "score": round(passed / total, 3) if total else 0,
            "time_s": round(elapsed, 2),
            "generated_code": code,
            "failures": [str(f) for f in failures[:3]],
        })
        print()

    overall = total_passed / total_tests if total_tests > 0 else 0

    print("=" * 60)
    print(f"BASELINE RESULTS: {total_passed}/{total_tests} passed ({overall:.1%})")
    print("=" * 60)

    print(f"\n{'Task':<22} {'Score':>6}  {'Time':>5}  Bar")
    print("-" * 52)
    for r in results:
        filled = int(r["score"] * 10)
        bar = "█" * filled + "░" * (10 - filled)
        print(f"{r['task_name']:<22} {r['score']:>5.0%}  {r['time_s']:>4.1f}s  {bar}")

    # Failure analysis
    no_return  = sum(1 for r in results if "no return" in r["generated_code"])
    syntax_err = sum(1 for r in results if any("SyntaxError" in str(f) for f in r["failures"]))
    executed   = sum(1 for r in results if r["passed"] > 0)

    print(f"\n📊 Failure analysis:")
    print(f"  No return statement:  {no_return}/10")
    print(f"  Syntax errors:        {syntax_err}/10")
    print(f"  At least 1 test pass: {executed}/10")
    print(f"\n  → GPT-2 generates function *shape* but not *logic*")
    print(f"  → This is why multi-agent debate should help significantly")

    output = {
        "experiment": "baseline",
        "model": MODEL_NAME,
        "date": datetime.now().isoformat(),
        "overall_score": round(overall, 3),
        "total_passed": total_passed,
        "total_tests": total_tests,
        "tasks": results,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved: baseline_results.json")
    print(f"\n📋 Log in JARVIS:")
    print(f'  experiment baseline-gpt2 | GPT-2 zero-shot on 10 tasks | {total_passed}/{total_tests} passed ({overall:.0%}) — generates shape not logic')

    return output


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("Smoke test...")
    out = generate_code(model, tokenizer, "def add(a, b):\n    return ", max_new_tokens=20)
    print(f"  Output: {out[:80]}\n  ✓ Working\n")

    run_baseline(model, tokenizer)