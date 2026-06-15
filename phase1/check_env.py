import os
import sys

# Bootstrap project root
def bootstrap_root():
    path = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(path, "phase1")):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    if path not in sys.path:
        sys.path.insert(0, path)
    return path

PROJECT_ROOT = bootstrap_root()
from config.paths import DATA_PHASE1

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Check available packages
for package in ["pandas", "openpyxl", "xlrd", "calamine", "pyarrow", "numpy", "matplotlib", "seaborn"]:
    try:
        __import__(package)
        print(f"Package '{package}' is available")
    except ImportError:
        print(f"Package '{package}' is NOT available")

# Inspect DataSet.xlsx size and existence
xlsx_path = os.path.join(DATA_PHASE1, "DataSet.xlsx")
if os.path.exists(xlsx_path):
    size = os.path.getsize(xlsx_path)
    print(f"DataSet.xlsx exists, size: {size} bytes ({size / (1024*1024):.2f} MB)")
else:
    print(f"DataSet.xlsx does NOT exist at {xlsx_path}")
