import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.Complex.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr

/-!
# The screen crossing (PAPER Sec 5.5; `entroptics.screen`)

Energy meeting a boundary does one of two things: it is ABSORBED (it stops propagating and
becomes structure on the other side) or it is TRANSMITTED (it keeps propagating, forward as
`bystanding` or backward as `reflected`).  `crossing_partition` is that conservation.

What crosses is bounded by phase space: the transmissible fraction is the etendue that fits,
`τ = min 1 (G_to / G_from)`.  `radiance_le` is the brightness theorem for that bound -- a
passive screen leaves the receiving side no brighter than the side that fed it -- and
`radiance_eq_of_concentrating` is its equality case, the ideal concentrator.

`real_inner_kernel` is the scalar identity that carries the coupling of Sec 5.5 from the
complex case to the real one, and `abs_alignment_le` its Cauchy-Schwarz bound.
-/

open scoped BigOperators

namespace Entroptics

/-- **The two root behaviours partition the energy.**
With `pertinent` the energy in the interaction and `τ` the transmissible fraction, the
absorbed part is `pertinent * τ` and the transmitted part is what stood by
(`energy - pertinent`) plus what was reflected (`pertinent * (1 - τ)`). -/
theorem crossing_partition (energy pertinent tau : ℝ) :
    energy = pertinent * tau + ((energy - pertinent) + pertinent * (1 - tau)) := by
  ring

/-- **The brightness theorem.**  Radiance is energy per unit etendue.  With the delivered
energy bounded by the etendue that fits, `τ = min 1 (G_to/G_from)`, the receiving side's
radiance never exceeds the sending side's: a passive screen cannot concentrate a signal
into less phase space than it arrived with. -/
theorem radiance_le {Gf Gt pertinent : ℝ} (hGf : 0 < Gf) (hGt : 0 < Gt)
    (hp : 0 ≤ pertinent) :
    pertinent * min 1 (Gt / Gf) / Gt ≤ pertinent / Gf := by
  rcases le_total (Gt / Gf) 1 with h | h
  · -- concentrating: τ = G_to/G_from, and the radiance is carried across exactly
    rw [min_eq_right h]
    have : pertinent * (Gt / Gf) / Gt = pertinent / Gf := by
      field_simp
    rw [this]
  · -- diluting: τ = 1, all the pertinent energy crosses into a larger phase space
    rw [min_eq_left h, mul_one]
    have hle : Gf ≤ Gt := by
      rwa [le_div_iff₀ hGf, one_mul] at h
    gcongr

/-- **The equality case: the ideal concentrator.**  Where the receiving side has the smaller
etendue, the bound is attained and radiance is conserved exactly -- concentrating costs
energy (`τ < 1`) and leaves brightness untouched. -/
theorem radiance_eq_of_concentrating {Gf Gt pertinent : ℝ} (hGf : 0 < Gf) (hGt : 0 < Gt)
    (hle : Gt ≤ Gf) :
    pertinent * min 1 (Gt / Gf) / Gt = pertinent / Gf := by
  have h : Gt / Gf ≤ 1 := by rwa [div_le_one hGf]
  rw [min_eq_right h]
  field_simp

/-- **The real-embedding kernel (Def 5.7).**  The real part of the Hermitian product is the
real inner product of the embedded pair, `x ↦ (Re x, Im x)`.  This is why the coupling needs
one estimator for real and complex sides alike. -/
theorem real_inner_kernel (a b : ℂ) :
    ((starRingEnd ℂ) a * b).re = a.re * b.re + a.im * b.im := by
  simp [Complex.mul_re]

/-- **The strength is a cosine (Def 5.7).**  Cauchy-Schwarz bounds the alignment by the
product of the norms, so the signed strength `⟪A, B⟫ / (‖A‖ ‖B‖)` lies in `[-1, 1]`. -/
theorem abs_alignment_le {n : ℕ} (A B : EuclideanSpace ℝ (Fin n)) :
    |inner ℝ A B| ≤ ‖A‖ * ‖B‖ :=
  abs_real_inner_le_norm A B

end Entroptics
