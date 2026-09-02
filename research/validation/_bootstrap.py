"""Put the repo's local ``src/`` ahead of any installed ``entroptics`` on sys.path.

The validation is run against the TIP of the working tree, not the published wheel,
so a bump under development is exercised before it ships.  Import this module before
importing ``entroptics`` anywhere in the validation suite (it mirrors what
``src/tests/conftest.py`` does for pytest).  Side-effect import: no public API.
"""
import os
import sys

# _bootstrap.py lives in research/validation/ -> repo root is two levels up.
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
