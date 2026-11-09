"""Confirm every repository's copy of the typesetting pipeline is byte-identical to this one.

    python tools/typeset/check_copies.py            # search the folder that holds this repository
    python tools/typeset/check_copies.py ROOT ...   # search these folders instead
    python tools/typeset/check_copies.py --self-test  # the heading rules on a part-divided paper

A copy is any folder named `typeset` that holds a `build_paper.py`. Prints the sha256 of each
canonical file and every copy that differs or is missing a file; exits 1 if any does.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("arxiv-twocolumn.latex", "arxiv-twocolumn.lua", "build_paper.py")
SKIP = {".git", ".venv", "venv", "node_modules", "_archive", "__pycache__", ".lake", "site-packages"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copies(roots: list[Path]) -> list[Path]:
    found = []
    for root in roots:
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP)
            p = Path(d)
            if p.name == "typeset" and "build_paper.py" in files and p.resolve() != HERE:
                found.append(p)
    return found


def self_test() -> int:
    """The heading rules of build_paper.py on the shape that once broke them: a title, numbered
    level-2 sections, and one level-1 part heading between them. The sections must stay level 2 and
    unnumbered (the paper numbers them itself); the part heading must not become the section level."""
    sys.path.insert(0, str(HERE))
    import build_paper as b
    doc = ("# Title\n\n## Abstract\n\nText.\n\n## 1. Introduction\n\n### 1.1 Scope\n\n## 2. Method\n\n"
           "# Further reads\n\n## 3. More\n\n## References\n")
    _, body = b.front_matter(doc)
    level, numbered = b.section_level(body), b.numbering(body)
    # DERIVED: 2 is the level of the "## " sections in the document above.
    ok = level == 2 and not numbered
    print(f"self-test: section level {level}, numbered {numbered}: {'ok' if ok else 'FAILED'}")
    core = HERE.parents[1] / "research" / "PAPER.md"
    if core.exists():
        _, body = b.front_matter(b.split_yaml(core.read_text(encoding="utf-8"))[1])
        level, numbered = b.section_level(body), b.numbering(body)
        print(f"core research paper: section level {level}, numbered {numbered}")
        ok = ok and not numbered
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if argv[:1] == ["--self-test"]:
        return self_test()
    roots = [Path(a) for a in argv] or [HERE.parents[2]]
    want = {f: sha(HERE / f) for f in FILES}
    for f, h in want.items():
        print(f"{h}  {f}")
    bad = 0
    found = copies(roots)
    for c in found:
        for f, h in want.items():
            p = c / f
            if not p.exists():
                print(f"MISSING  {p}")
                bad += 1
            elif sha(p) != h:
                print(f"DIFFERS  {p}")
                bad += 1
    print(f"{len(found)} copies, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
