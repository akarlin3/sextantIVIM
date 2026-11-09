import os
import sys

# Make the sextant_core package importable without installation.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")

for _p in (_SRC, _ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
