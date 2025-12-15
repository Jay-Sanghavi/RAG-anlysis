"""
Quick test script to verify the project setup.
Run this to check if all dependencies are installed correctly.
"""

import sys
import importlib


def check_module(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {str(e)}")
        return False


def main():
    print("=" * 80)
    print("RAG Analysis - Environment Check")
    print("=" * 80)

    print("\nChecking Python version...")
    python_version = sys.version_info
    print(
        f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    if python_version < (3, 8):
        print("⚠ Warning: Python 3.8 or higher is recommended")
    else:
        print("✓ Python version OK")

    print("\nChecking required packages...")

    required_modules = [
        ("datasets", "datasets (HuggingFace)"),
        ("transformers", "transformers"),
        ("torch", "PyTorch"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "FAISS"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("yaml", "PyYAML"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
    ]

    results = []
    for module_info in required_modules:
        if len(module_info) == 2:
            module, name = module_info
            results.append(check_module(module, name))
        else:
            module = module_info
            results.append(check_module(module))

    print("\n" + "=" * 80)

    if all(results):
        print("✓ All required packages are installed!")
        print("\nYou're ready to run the experiments:")
        print("  1. python main.py              # Run full pipeline")
        print("  2. jupyter notebook            # Open interactive notebooks")
    else:
        print("✗ Some packages are missing")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")

    print("=" * 80)

    # Check project structure
    print("\nChecking project structure...")

    import os

    required_dirs = ["src", "data", "notebooks", "results", "config"]
    required_files = [
        "src/data_loader.py",
        "src/rag_pipeline.py",
        "src/evaluator.py",
        "src/utils.py",
        "config/config.yaml",
        "main.py",
        "README.md",
    ]

    all_exist = True
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            all_exist = False

    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} (missing)")
            all_exist = False

    if all_exist:
        print("\n✓ Project structure is complete!")
    else:
        print("\n✗ Some project files are missing")

    print("=" * 80)


if __name__ == "__main__":
    main()
