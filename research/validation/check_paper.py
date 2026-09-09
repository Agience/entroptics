"""Check every number in PAPER.md against the artifacts that produce them.

    python research/validation/check_paper.py

Exits non-zero if a numeric literal anywhere in the paper's body is neither produced by a
committed artifact nor listed below as something that is not a measurement.

WHY THIS EXISTS.  The generated artifacts are trustworthy: ``RESULTS.md`` regenerates
byte-identically and every figure writes its table.  The paper is hand-written prose QUOTING
them, and nothing compared the two -- so a figure transcribed wrongly, or left behind when an
experiment changed, stayed in the paper indefinitely.  The 2026-09-08 audit found twelve: an
accuracy reported as 0.998 that the suite computes as 0.997, a cell attributed to the wrong
array, a floor quoted from the unquantized row, correlation lengths from a superseded run, a
per-channel median asserted to be exactly zero that is 3.9e-06.

It also found the shape of the underlying problem, which is worth stating because it recurs: a
test that COMPUTES a value and asserts only a BOUND on it leaves the value written down nowhere
but a docstring, and the paper then quotes the docstring.  ``test_extract.py`` carried "update
the paper if it moves" in a comment; ``test_dynamics.py`` carried section 9's rates the same
way.  Wherever that pattern appears the fix is to make something EMIT the value and to list the
emitting artifact here.

This is a COARSE check and deliberately so.  It asks "does this number appear in an artifact",
not "does it appear in the RIGHT place" -- a claim can still point at the wrong row and pass.
That second question is what the per-paper ``verify.py`` files answer, by binding each claim to
a named column.  Read the misses; do not read a pass as proof the prose is right.
"""
from __future__ import annotations

import re
import sys
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "research/PAPER.md"

#: Everything that emits a number the paper quotes.
ARTIFACTS = [
    REPO / "research/validation/RESULTS.md",              # the experiments
    REPO / "research/figures/calibration.csv",            # the read-side filter (Figure 1)
    REPO / "research/figures/frb_panel.csv",              # the four events (Figure 2)
    REPO / "research/figures/frb_spotcheck.csv",          # the 12-waterfall random draw
    REPO / "research/figures/frb_display.csv",            # the contrast-stretch diagnostics
    REPO / "research/supplemental/frb/tables/agreement.csv",
]

#: Constants that are DERIVED mathematics rather than measurements.  Each is tabulated with its
#: provenance in section 11.1 and none is fitted to anything.  Listed WITH THE REASON, so an
#: exemption cannot be added silently just to make a miss go away.
DERIVED = {
    Decimal("1.22"):    "first zero of Bessel J_1 -- the circular-aperture Rayleigh constant",
    Decimal("0.9793"):  "the universal Tracy-Widom_1 quantile at alpha = 0.05",
    Decimal("1.360"):   "influence-function variance of log MAD, 1/(16 f(D)^2 D^2)",
    Decimal("1.4826"):  "MAD-to-sigma consistency factor for a Gaussian",
    Decimal("0.05"):    "the default false-alarm level -- the reader's operating point",
    Decimal("0.023"):   "the N(0,1) one-sided target at z = 2, the level's own reference",
    Decimal("99.7"):    "the contrast-stretch percentile, a display choice stated in the caption",
    Decimal("16384"):   "the CHIME/FRB Catalog 1 recorded channel count, a property of the release",
    Decimal("21.0007"): "the CANFAR collection identifier, not a quantity",
}


def literals(text: str) -> set[Decimal]:
    out = set()
    for tok in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
        try:
            out.add(Decimal(tok))
        except InvalidOperation:
            pass
    return out


def citation_years(paper: str) -> set[Decimal]:
    """Years taken from the paper itself rather than a list kept by hand -- adding a citation
    should not create a spurious miss."""
    years = set()
    tail = paper.split("## References", 1)
    if len(tail) > 1:
        years |= {Decimal(m.group(1)) for m in re.finditer(r"\((1[89]\d\d|20\d\d)\)", tail[1])}
    # in-text citations carry the year too: [Kassner 1950], [Campbell 1960]
    years |= {Decimal(m.group(1))
              for m in re.finditer(r"\[[^\]]*?\b(1[89]\d\d|20\d\d)[a-z]?\b[^\]]*?\]", paper)}
    return years


def body(paper: str) -> str:
    """The argument: everything before the declaration and the reference list."""
    return paper.split("## Declaration of generative AI use")[0]


def quoted(sec: str):
    """Numeric literals in the prose, with context, with the machinery of a paper stripped --
    headings, cross-references, LaTeX commands and file names are not measurements."""
    s = sec
    # scientific notation first, before the latex-command strip removes the backslash-times
    s = re.sub(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{([-+]?\d+)\}", r"\1e\2", s)
    s = re.sub(r"^#{2,4}\s+(?:\d+(?:\.\d+)?|Appendix\s+[A-Z])\.?\s.*$", " ", s, flags=re.M)
    s = re.sub(r"`[^`]*`", " ", s)                                    # file and symbol names
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)                       # figure captions
    kinds = (r"(?:Definitions?|Defs?|Lemmas?|Theorems?|Thm|Propositions?|Props?"
             r"|Corollary|Corollaries|Cor|Remarks?)")
    s = re.sub(r"\(" + kinds + r"[^)]*\)", " ", s)
    # singular, plural, and "Lemmas 3.2 and 3.4"
    s = re.sub(kinds + r"\s+[A-Z]?\d+(?:\.\d+)?(?:\s+and\s+[A-Z]?\d+(?:\.\d+)?)*", " ", s)
    s = re.sub(r"§+\s*[A-Z]?[\d.]+(?:\s*,\s*[\d.]+)*", " ", s)        # section references
    s = re.sub(r"10\^\{[-+]?\d+\}", " ", s)                           # powers of ten
    s = re.sub(r"\\[a-zA-Z]+", " ", s)                                # latex commands
    s = re.sub(r"AR\(1\)", " ", s)
    s = re.sub(r"\$\((\d+),\s*(\d+)\)\$", r" \1 \2 ", s)              # shape pairs
    s = s.replace("{,}", "")                                          # 9{,}760 -> 9760
    out = []
    for m in re.finditer(r"(?<![\w.^{])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w}])", s):
        try:
            d = Decimal(m.group(1))
        except InvalidOperation:
            continue
        out.append((d, re.sub(r"\s+", " ", s[max(0, m.start() - 110):m.end() + 60])))
    return out


def sections(text: str):
    parts = re.split(r"^(## (?:\d+\.|Appendix [A-Z]\.).*)$", text, flags=re.M)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts), 2)]


def main() -> int:
    paper = PAPER.read_text(encoding="utf-8")
    have: set[Decimal] = set()
    absent = False
    for a in ARTIFACTS:
        if not a.is_file():
            print(f"  ! missing artifact {a.relative_to(REPO)} -- run its generator")
            absent = True
            continue
        have |= literals(a.read_text(encoding="utf-8"))
    exempt = set(DERIVED) | citation_years(paper)

    def found(d: Decimal) -> bool:
        if d in exempt or d in have:
            return True
        for h in have:                   # the paper rounds what the artifact prints
            try:
                if h.quantize(d) == d or abs(h - d) <= Decimal("1E-9"):
                    return True
                if h.quantize(d, rounding=ROUND_HALF_UP) == d:
                    return True
            except InvalidOperation:
                continue
        return False

    total, misses = 0, []
    print(f"{'section':46s} {'numbers':>8} {'no source':>10}")
    for name, text in sections(body(paper)):
        lits = quoted(text)
        total += len(lits)
        ctx = {d: c for d, c in lits}
        m = sorted({d for d, _ in lits if not found(d)})
        print(f"{name[:44]:46s} {len(lits):8d} {len(m):10d}{'' if not m else '   <--'}")
        misses += [(name, d, ctx[d]) for d in m]

    print(f"\n{total} numeric literals in the paper body, "
          f"{len(misses)} distinct value(s) with no source")
    if misses:
        print()
        for name, d, c in misses:
            print(f"  {name[:34]:36s} {str(d):>10}   ...{c[-110:]}")
        print("\nEach is a transcription error, a number no script emits, or a derived constant "
              "that belongs in DERIVED above -- with its reason.")
        return 1
    print("\nEvery number in the paper body traces to a committed artifact.")
    return 2 if absent else 0


if __name__ == "__main__":
    sys.exit(main())
