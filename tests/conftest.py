"""
conftest.py
Pytest configuration — adds the project root to sys.path so all modules
are importable without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
