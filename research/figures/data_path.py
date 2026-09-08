"""Where the CHIME/FRB waterfalls are -- resolved from configuration, never named in the source.

Every figure under ``research/figures`` runs on the CHIME/FRB Catalog 1 waterfall release, which is
NOT part of this repository: it is a separate public download with its own citation and its own
terms (CHIME/FRB Collaboration 2021, ApJS 257, 59; CANFAR CISTI.CANFAR/21.0007).  Each script took
the tree as ``argv[1]`` and refused without it, so running the figures meant retyping a UNC path or
a mount point on every invocation, and a machine-local path had nowhere to live except a shell
history.

Resolution order, most specific first:

  1. an explicit argument -- ``python frb_panel.py <root>`` still works and still wins, which is
     what a one-off read of a copied-out subset wants;
  2. the ``FRB_WATERFALLS`` environment variable -- a one-off override for a single run, and what
     CI or a batch job sets;
  3. ``research.local.env`` at the repository root -- git-ignored, written once per machine;
  4. nothing.  There is deliberately NO built-in default.  A default would have to name somebody's
     filesystem, and a path that exists on one workstation is a way to read nothing everywhere
     else while still exiting 0 -- these scripts write committed artifacts, so a silent empty read
     overwrites a good table with a header.

Nothing resolves at import, so a machine with no data can still import and test these modules; the
refusal happens at the read, where it can say what was being looked for.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Environment variable, checked before the local file.
ENV_VAR = "FRB_WATERFALLS"

#: Machine-local, git-ignored, at the repository root.  One line: ``FRB_WATERFALLS=<tree root>``.
LOCAL_FILENAME = "research.local.env"

#: Committed alongside it, documenting the key and where the data comes from.
EXAMPLE_FILENAME = "research.local.env.example"

#: The release, by the name it is published under -- named in full in every failure, because
#: someone who has just cloned this has no other way to know what is missing.
RELEASE_TITLE = "CHIME/FRB Catalog 1 waterfalls (CANFAR CISTI.CANFAR/21.0007)"


class WaterfallsNotConfigured(RuntimeError):
    """Neither an argument, the environment variable, nor the local file says where the tree is.

    A distinct type so a caller that genuinely wants to continue without data can catch exactly
    this, as opposed to the ``FileNotFoundError`` a wrong path produces further down -- which is
    indistinguishable from a corrupt or partial download.
    """


def repo_root() -> Path:
    """The repository root, from this file's own location -- not the working directory, since
    these scripts are run from the repo root and from ``research/figures`` alike."""
    return Path(__file__).resolve().parents[2]


def local_config_path() -> Path:
    """Full path of the git-ignored local config file, whether or not it exists."""
    return repo_root() / LOCAL_FILENAME


def local_value(key: str) -> str | None:
    """Any ``KEY`` from the machine-local config file, or None.

    A deliberately small parser -- ``KEY=VALUE``, ``#`` comments, an optional ``export`` prefix,
    and optional surrounding quotes.  Not a dotenv dependency: the file holds one or two paths,
    and a figure script should not pull a package in to read them."""
    path = local_config_path()
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == key:
            return value.strip().strip('"').strip("'") or None
    return None


def waterfall_root(explicit: str | None = None, *, required: bool = True) -> str | None:
    """The root of the waterfall tree: the directory whose subdirectories hold
    ``*_waterfall.h5``.

    ``explicit`` is a command-line argument if one was given, and wins.  ``required=False``
    returns None instead of raising, for a caller that wants to report the absence itself."""
    for candidate in (explicit, os.environ.get(ENV_VAR), local_value(ENV_VAR)):
        if candidate:
            return str(candidate)
    if not required:
        return None
    raise WaterfallsNotConfigured(
        f"The waterfall tree is not configured, so there is nothing to read.\n\n"
        f"  These figures run on: {RELEASE_TITLE}\n"
        f"  It is a separate public download and is not part of this repository.\n\n"
        f"Point this repository at your copy, any one of:\n"
        f"  1. pass it:   python <script>.py /path/to/waterfalls\n"
        f"  2. per run:   {ENV_VAR}=/path/to/waterfalls python <script>.py\n"
        f"  3. per machine: copy {EXAMPLE_FILENAME} to {LOCAL_FILENAME} at the repository root\n"
        f"     ({local_config_path()}) and set {ENV_VAR}.\n\n"
        f"The root is the directory that CONTAINS the per-event directories, so that\n"
        f"'<root>/*/*_waterfall.h5' matches.")
