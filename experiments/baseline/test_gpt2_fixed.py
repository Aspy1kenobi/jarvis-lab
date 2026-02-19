"""
GPT-2 Baseline Test - Apple Silicon Compatible
Fixes bus error by:
  1. Forcing CPU + disabling MPS entirely
  2. Using safe tensor loading (no memory mapping)
  3. Explicit attention masks on every generate() call
  4. Pinning compatible library versions
"""

import os
import torch

# ── Critical: disable MPS before any torch/transformers imports ──────────────
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force CPU globally — prevents generate() sneaking onto MPS
if hasattr(torch.backends, "mps"):
    torch.backends.mps.enabled = False

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from test_tasks import TASKS, run_test
import time
import json
from datetime import datetime

DEVICE = "cpu"
MODEL_NAME = "gpt2"


# ── Load model safely ─────────────────────────────────────────────────────────
def load_model():
    print("Loading GPT-2 (117M)...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    
    # low_cpu_mem_usage=False avoids memory-mapped tensors (bus error trigger)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=False,   # <-- key fix for bus error
        torch_dtype=torch.float32  # explicit dtype, no mixed precision
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(DEVICE)
    model.eval()

    print(f"✓ GPT-2 loaded on {DEVICE}\n")
    return model, tokenizer


# ── Code generation ───────────────────────────────────────────────────────────
def generate_code(model, tokenizer, prompt: str, max_new_tokens=150, temperature=0.3) -> str:
    """Generate code completion from a prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    # Ensure everything is on CPU
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],   # explicit mask — prevents warnings + occasional crashes
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ── Extract function body from generated text ─────────────────────────────────
def extract_function(generated: str, func_name: str) -> str:
    """Pull out just the function definition from GPT-2 output."""
    lines = generated.split("\n")
    in_func = False
    func_lines = []

    for line in lines:
        if f"def {func_name}" in line:
            in_func = True
        if in_func:
            func_lines.append(line)
            # Stop at next top-level definition (not the first line)
            if len(func_lines) > 1 and line.startswith("def "):
                func_lines.pop()
                break

    return "\n".join(func_lines) if func_lines else generated


# ── Run baseline on all 10 tasks ──────────────────────────────────────────────
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

        start = time.time()
        generated = generate_code(model, tokenizer, task["prompt"])
        elapsed = time.time() - start

        func_code = extract_function(generated, task["name"])
        print(f"  Generated ({elapsed:.1f}s):\n    {func_code[:120].strip()}...")

        # Try to execute and test
        passed, total, failures = 0, len(task["test_cases"]), []
        try:
            namespace = {}
            exec(func_code, namespace)
            func = namespace.get(task["name"])
            if func:
                passed, total, failures = run_test(func, task["test_cases"])
        except Exception as e:
            failures = [{"error": str(e)}]

        score = passed / total if total > 0 else 0
        total_passed += passed
        total_tests += total

        status = "✓" if passed == total else f"✗ {passed}/{total}"
        print(f"  Result: {status}")

        if failures:
            for f in failures[:2]:  # show max 2 failures
                print(f"    - {f}")

        results.append({
            "task_id": task["id"],
            "task_name": task["name"],
            "passed": passed,
            "total": total,
            "score": round(score, 3),
            "time_s": round(elapsed, 2),
            "generated_code": func_code,
        })
        print()

    # Summary
    overall = total_passed / total_tests if total_tests > 0 else 0
    print("=" * 60)
    print(f"BASELINE RESULTS: {total_passed}/{total_tests} tests passed ({overall:.1%})")
    print("=" * 60)

    # Per-task table
    print(f"\n{'Task':<20} {'Score':>6} {'Time':>6}")
    print("-" * 36)
    for r in results:
        bar = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
        print(f"{r['task_name']:<20} {r['score']:>5.0%}  {r['time_s']:>4.1f}s  {bar}")

    # Save results
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

    print(f"\n✓ Results saved to baseline_results.json")
    print(f"\n📋 JARVIS note to log:")
    print(f'  experiment baseline-gpt2 | GPT-2 zero-shot on 10 tasks | {total_passed}/{total_tests} tests passed ({overall:.0%})')

    return output


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer = load_model()

    # Quick smoke test first
    print("Smoke test...")
    test_out = generate_code(model, tokenizer, "def hello():\n    # returns greeting\n    ", max_new_tokens=20)
    print(f"  Output: {test_out[:80]}")
    print("  ✓ Generation working\n")

    run_baseline(model, tokenizer)
