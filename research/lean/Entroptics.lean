-- Entroptics: machine-checked core lemmas of research/PAPER.md, one module per paper section.
import Entroptics.Entropy        -- Sec 2-3: entropy bound, fill fraction, Strehl, fill-reach duality
import Entroptics.Diffraction    -- Sec 4: OTF PSD, peak at zero lag, Gabor scale-invariance
import Entroptics.Coherence      -- Sec 5: the permutation-null mean
import Entroptics.Spectrum       -- Sec 6: attenuation interval, certified count, separation
import Entroptics.Concentration  -- Sec 7: axial (focus) vs directional (resultant)
import Entroptics.Dynamics       -- Sec 9: exact recovery of the propagator; additive splicing
import Entroptics.Screen       -- Sec 5.5 + the crossing: conservation, brightness bound
import Entroptics.Coupling     -- Sec 5.5: the coupling null's algebra (Thm 5.6)
import Entroptics.Aperture     -- Sec 3 + 10: tr(P B P) = TW/N, spectrum in [0,1], the plunge count
import Entroptics.Plunge       -- Sec 3 + 10: the plunge is logarithmic; #{λ > 1/2} = TW/N ± (9 + log min(T,W))
