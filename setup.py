import importlib.util
import sys

REQUIRED = ["numpy", "pandas", "matplotlib"]

print("Checking dependencies...")
missing = []
for package in REQUIRED:
    if importlib.util.find_spec(package) is None:
        missing.append(package)

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    print(f"Run: pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("✓ All dependencies available.")