# Typesetting a paper

Every research repository ships its paper as a two-column arXiv-style PDF and its LaTeX source, both
built from the paper's Markdown by one shared pipeline. This folder holds the canonical copy:

- `arxiv-twocolumn.latex`: the pandoc template (10 pt, letter paper, two columns, title block and
  abstract across both columns, page numbers);
- `arxiv-twocolumn.lua`: the pandoc filter that lays out headings, figures, tables, code and display
  equations for two columns. It changes layout only, never text;
- `build_paper.py`: runs pandoc, then tectonic (Python standard library only);
- `check_copies.py`: confirms every repository's copy is byte-identical to this one, and with
  `--self-test` checks the heading rules on a paper divided into parts.

Each paper repository carries a byte-identical copy of the first three files in a `typeset/` folder
beside its paper, so it builds on its own. Change the canonical copy here, copy it out, and run
`python tools/typeset/check_copies.py` from this repository.

## Building

You need [pandoc](https://pandoc.org) 3.x and [tectonic](https://tectonic-typesetting.github.io)
on `PATH` (or set `PANDOC` and `TECTONIC` to their executables). Tectonic fetches the TeX packages
and fonts it needs on first use. Any XeLaTeX installation works instead with `--engine xelatex`.

```
python typeset/build_paper.py <paper>.md             # <paper>.tex and <paper>.pdf beside the Markdown
python typeset/build_paper.py <paper>.md --tex-only  # the .tex only (pandoc, no TeX)
python typeset/build_paper.py <paper>.md --keep-log  # also keep the TeX log
```

The build reports the page count, any line wider than its column by more than 20 pt, the smallest
scale applied to any display or table, anything too wide even at the floor size, and every page less
than a quarter full that is neither a float page nor the last page.

## What it reads from the Markdown

The Markdown is read as GitHub-flavoured Markdown with `$...$` math, as GitHub renders it, and is
not modified. The first level-1 heading is the title and a heading right under it the subtitle; a
line naming the author becomes the template's author block (with the ORCID iD when the line has
one); an italic line such as `*Pre-print. November 2026.*` is the date; the `Abstract` section is the
abstract. Anything else above the first section stays as the opening of the body. A YAML block at
the top of the Markdown overrides any of these, and can set `number-sections` and `figure-width`
(`auto`, `column` or `full`).

The section level is the shallowest heading level below the title that holds more than one heading;
a lone heading above it is a part of the paper and is set as a centred divider. Sections are numbered
by LaTeX only when most headings at the section level carry no number of their own.

## Layout rules

- The title block and abstract span both columns when they leave room for at least ten lines of each
  column on page 1; a longer abstract opens the two-column body instead, so no page is left
  near-empty.
- Nothing is set smaller than 8 pt, the template's `\footnotesize`. Tables and code are set at that
  size and never scaled; code wraps at the column edge. A display equation wider than the column is
  scaled to fit no smaller than 8 pt; if that is not enough it is set across both columns at that
  point of the text.
- A paragraph starting "Figure N" right after an image becomes its caption, and one starting
  "Table N" right before or after a full-width table travels with it. A `.pdf` beside a `.png`
  figure is used in its place. Landscape figures span both columns.
- Tables are sized from their cells: in the column when they fit, across both columns otherwise; a
  table too long for a page breaks across columns, or across pages at full width.

The `.tex` depends only on the inputs. The PDF carries the date in `SOURCE_DATE_EPOCH` when it is
set and 1970-01-01 otherwise, so two builds of the same inputs with the same tools are identical.
