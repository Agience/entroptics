"""Regenerate the golden optics contract (tests/golden/optics.json).

The golden pins the full ``Aperture(W).optics()`` dict plus the screen read for a
couple of fixed seeds, so any unintended numeric drift is caught.  Regenerate only
when a change to the reads is intended and reviewed:

    python tests/golden/_generate.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # src/ (tests live under src/tests)
import entroptics.aperture as A   # noqa: E402


def build_W(seed, T=80, F=48, amp=0.6):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, F)) + amp * np.sin(np.linspace(0, 25, T))[:, None]


def main():
    out = {}
    for seed in (7, 3):
        ap = A.Aperture(build_W(seed))
        sc = ap.projection()
        out[str(seed)] = {
            "optics": ap.optics(),
            "screen": {
                "coherence": float(sc.coherence),
                "K_signal": int(sc.K_signal),
                "noise_floor": float(sc.noise_floor),
                "sigma_top": float(sc.sigma_top),
                "H_screen": float(sc.H_screen),
                "delta_T": float(sc.delta_T),
                "delta_F": float(sc.delta_F),
            },
        }
    path = Path(__file__).with_name("optics.json")
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {path} ({len(out)} seeds)")


if __name__ == "__main__":
    main()
