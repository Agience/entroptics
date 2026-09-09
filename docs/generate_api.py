"""Generate docs/API.md from the library's own public surface.

    python docs/generate_api.py

Introspection rather than prose, so the reference cannot drift from the code: every name comes
from ``entroptics.__all__``, every signature from the object itself, and every summary from the
first sentence of its docstring.  Adding a public name puts it in the reference; renaming one
renames it here; deleting one removes it.  Nothing is maintained by hand.

Re-run it whenever the public API changes.  ``docs/API.md`` is committed, so a stale reference
shows up as a diff.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import entroptics as E  # noqa: E402

#: The modules, in the order the paper introduces them, with what each is for.  A name whose
#: module is not listed lands in "Other" -- so a new module is visible rather than silently
#: absorbed.
GROUPS = [
    ("entroptics.aperture", "The front door",
     "One object over a 2-D record; every read below is reachable from it."),
    ("entroptics.reads", "Optical reads",
     "The aperture quantities: fill fractions, etendue, Strehl, the decay and its "
     "diffraction limit, coherence, coupling, concentration."),
    ("entroptics.projection", "The screen and its noise floor",
     "Whiten, fold to the entropy-matched grid, and count what stands above the derived "
     "Tracy-Widom floor."),
    ("entroptics.null_providers", "Null providers",
     "The threshold a detection is taken against. The derived edge is the default; a caller "
     "may supply their own."),
    ("entroptics.dynamics", "The streaming operator",
     "Online DMD / Koopman: per-mode decay rates from a fixed-size sufficient statistic, "
     "splice-exact across segments."),
    ("entroptics.lift", "The observable lift",
     "Delay embedding, for trajectories with no linear one-step map on their own coordinates."),
    ("entroptics.screen", "The two-way screen",
     "Two or more sides meeting on one shared basis: lenses, beams, crossings, coupling."),
    ("entroptics.beam", "Beams", "A placed frame read as an aperture."),
    ("entroptics.batch", "Batched reads",
     "The same read over a stack of frames, on CPU or GPU, equal to the per-frame result."),
    ("entroptics.tensor", "N-D fields",
     "Reducing a higher-dimensional field to two axes: pool for ordered reads, plane-fold for "
     "feature reads."),
    ("entroptics.fields", "Field helpers", "Constructing and shaping the input record."),
    ("entroptics.environment", "Backend and precision",
     "numpy or torch, chosen by the input; determinism and working precision."),
]

HEADER = """# API reference

The public surface of `entroptics`, generated from the library itself by
[`docs/generate_api.py`](generate_api.py) -- every name, signature and summary is read from the
code, so this cannot drift from what is installed.

Everything listed here is importable directly from the package:

```python
from entroptics import Aperture, Projection, Screen
```

A name absent from this page is not public API, whatever its visibility in Python: it may change
without notice. The construction each read implements is defined in
[`research/PAPER.md`](../research/PAPER.md), which is the reference for what the numbers mean.

"""


def summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return ""
    # first sentence of the first paragraph, flattened
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    for stop in (". ", " -- "):
        if stop in first:
            first = first.split(stop)[0] + ("." if stop == ". " else "")
            break
    return " ".join(first.split())


def signature(name, obj) -> str:
    try:
        if inspect.isclass(obj):
            s = f"{name}{inspect.signature(obj.__init__)}"
            s = s.replace("(self, ", "(").replace("(self)", "()")
        elif callable(obj):
            s = f"{name}{inspect.signature(obj)}"
        else:
            return name
    except (TypeError, ValueError):
        return name
    # A sentinel default reprs as its memory address, which changes every run and would churn
    # the committed diff for no reason.  Render it as what it means instead.
    return re.sub(r"<object object at 0x[0-9a-fA-F]+>", "<unset>", s)


def main() -> int:
    by_mod: dict[str, list[str]] = {}
    for n in E.__all__:
        obj = getattr(E, n, None)
        by_mod.setdefault(getattr(obj, "__module__", "?") or "?", []).append(n)

    listed = {m for m, _, _ in GROUPS}
    other = sorted({n for m, names in by_mod.items() if m not in listed for n in names})

    out = [HEADER]
    total = 0
    for mod, title, blurb in GROUPS:
        names = sorted(by_mod.get(mod, []))
        if not names:
            continue
        out.append(f"## {title}\n\n{blurb}\n")
        out.append("| name | kind | summary |")
        out.append("| --- | --- | --- |")
        for n in names:
            obj = getattr(E, n)
            kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "value"
            sig = signature(n, obj).replace("|", "\\|")
            s = summary(obj).replace("|", "\\|")
            out.append(f"| `{sig}` | {kind} | {s} |")
            total += 1
        out.append("")

    if other:
        out.append("## Other\n")
        out.append("| name | kind | summary |")
        out.append("| --- | --- | --- |")
        for n in other:
            obj = getattr(E, n)
            kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "value"
            out.append(f"| `{signature(n, obj)}` | {kind} | {summary(obj)} |")
            total += 1
        out.append("")

    out.append(f"---\n\n{total} public names. Generated by `python docs/generate_api.py` from "
               f"`entroptics.__all__`; re-run it when the API changes.\n")

    dest = REPO / "docs/API.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {dest.relative_to(REPO)}  ({total} names)")
    missing = sorted(set(E.__all__) - {n for names in by_mod.values() for n in names})
    if missing:
        print(f"  ! not documented: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
