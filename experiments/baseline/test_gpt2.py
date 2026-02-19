"""
Quick test to verify GPT-2 is working.
Optimized for Apple Silicon.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("Loading GPT-2...")
model_name = 'gpt2'  # 117M parameters
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# use CPU explicitly (safest for Apple Silicon with this version)
device = "cpu"
model = model.to(device)

print(f"✓ GPT-2 loaded successfully (Device: {device})\n")

# Test 1: Simple generation
print("Test 1: Basic code generation")
print("-" * 60)
prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate with attention mask
with torch.no_grad(): # disable gradient computation
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
print()

# Test 2: Multiple samples
print("Test 2: Generate 3 different completions")
print("-" * 60)
prompt = "def is_palindrome(s):"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        num_return_sequences=3,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

for i, output in enumerate(outputs, 1):
    print(f"\nSample {i}:")
    print(tokenizer.decode(output, skip_special_tokens=True))

print("\n" + "=" * 60)
print("✓ All tests passed! GPT-2 is ready to use.")