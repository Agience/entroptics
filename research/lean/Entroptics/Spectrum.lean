import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.LinearAlgebra.Matrix.Hermitian
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.NormNum

/-!
# The mode spectrum and certified intervals (PAPER Sec 6)

The attenuation constant `α = log(λ₁/r)` and its Weyl-certified interval, the certified resolved
count, and the concentration separation criterion.

* `attenuation_interval_bounds`, `weyl_top_of_rayleigh`, `attenuation_weyl_certified` (Lemma 6.2):
  the Weyl band encloses the true attenuation `[α_lo, α_hi]`.
* `resolved_count_certified` (Lemma 6.2, count form): the Weyl band encloses the true resolved count
  `K_lo ≤ K_true ≤ K_hi`.
* `separated_of_disjoint_intervals`: non-overlapping certified intervals order two ensemble means.
-/

open scoped BigOperators

namespace Entroptics

/-- **Lemma 6.2 (Weyl-certified attenuation), monotone-propagation step.**
Given the Weyl eigenvalue bounds `λ₁ ∈ [lo1, hi1]`, `r ∈ [lor, hir]` (all positive), the attenuation
`log λ₁ - log r`, increasing in `λ₁`, decreasing in `r`, lies in `[log lo1 - log hir, log hi1 - log
lor]`. Weyl's inequality supplies the eigenvalue bounds; this propagates them through the read. -/
theorem attenuation_interval_bounds {lo1 hi1 lor hir lam1 r : ℝ}
    (hlo1 : 0 < lo1) (hlor : 0 < lor)
    (h1 : lo1 ≤ lam1) (h1' : lam1 ≤ hi1) (hr : lor ≤ r) (hr' : r ≤ hir) :
    Real.log lo1 - Real.log hir ≤ Real.log lam1 - Real.log r ∧
      Real.log lam1 - Real.log r ≤ Real.log hi1 - Real.log lor := by
  have hr0 : 0 < r := lt_of_lt_of_le hlor hr
  have hlam0 : 0 < lam1 := lt_of_lt_of_le hlo1 h1
  refine ⟨?_, ?_⟩
  · have ha : Real.log lo1 ≤ Real.log lam1 := Real.log_le_log hlo1 h1
    have hb : Real.log r ≤ Real.log hir := Real.log_le_log hr0 hr'
    linarith
  · have ha : Real.log lam1 ≤ Real.log hi1 := Real.log_le_log hlam0 h1'
    have hb : Real.log lor ≤ Real.log r := Real.log_le_log hlor hr
    linarith

/-- **Lemma 6.2 (Weyl estimate for the top eigenvalue).**
The top of a supremum of Rayleigh quotients moves by at most the perturbation. In Courant-Fischer
form `λ₁ = ⨆ x, ⟪C x, x⟫` over the unit sphere; if `|⟪(Ĉ - C) x, x⟫| ≤ ε` at every test point (in
particular `ε = ‖Ĉ - C‖₂`), then `|λ̂₁ - λ₁| ≤ ε`. Proved at the level of the two suprema, over any
nonempty index of bounded Rayleigh values. -/
theorem weyl_top_of_rayleigh {ι : Type*} [Nonempty ι] (rhat r : ι → ℝ) (ε : ℝ)
    (hbr : BddAbove (Set.range r)) (hbrhat : BddAbove (Set.range rhat))
    (h : ∀ i, |rhat i - r i| ≤ ε) :
    |(⨆ i, rhat i) - (⨆ i, r i)| ≤ ε := by
  have hub : (⨆ i, rhat i) ≤ (⨆ i, r i) + ε :=
    ciSup_le fun i => by have := (abs_le.mp (h i)).2; have := le_ciSup hbr i; linarith
  have hlb : (⨆ i, r i) ≤ (⨆ i, rhat i) + ε :=
    ciSup_le fun i => by have := (abs_le.mp (h i)).1; have := le_ciSup hbrhat i; linarith
  rw [abs_le]
  exact ⟨by linarith, by linarith⟩

/-- **Lemma 6.2 (Weyl-certified attenuation), end-to-end.**
Composing the Weyl eigenvalue bound `|λ̂₁ - λ₁| ≤ ε`, `|r̂ - r| ≤ ε` with the monotone propagation
certifies the attenuation interval from the read eigenvalues: the true `log λ₁ - log r` lies in
`[log(λ̂₁ - ε) - log(r̂ + ε), log(λ̂₁ + ε) - log(r̂ - ε)]`, the interval of Lemma 6.2 (with
`r = max(λ₂, λ₊)` supplied by the caller). -/
theorem attenuation_weyl_certified {lam1hat rhat lam1 r ε : ℝ}
    (hlo1 : 0 < lam1hat - ε) (hlor : 0 < rhat - ε)
    (h1 : |lam1hat - lam1| ≤ ε) (hr : |rhat - r| ≤ ε) :
    Real.log (lam1hat - ε) - Real.log (rhat + ε) ≤ Real.log lam1 - Real.log r ∧
      Real.log lam1 - Real.log r ≤ Real.log (lam1hat + ε) - Real.log (rhat - ε) := by
  rw [abs_le] at h1 hr
  exact attenuation_interval_bounds hlo1 hlor (by linarith [h1.2]) (by linarith [h1.1])
    (by linarith [hr.2]) (by linarith [hr.1])

/-- **Lemma 6.2 (count form): the certified resolved dimension.**
The count analogue of `attenuation_weyl_certified`. Read eigenvalues `lamhat` differ from the true
`lam` by at most `ε` (Weyl), and the noise edge is a fixed function of the shape. Then the count of
TRUE eigenvalues above the edge is enclosed by the two read-side counts,
`#{lamhat k - ε > edge} ≤ #{lam k > edge} ≤ #{lamhat k + ε > edge}`, i.e. `K_lo ≤ K_true ≤ K_hi`.
This certifies `resolved_dimension_interval`; the proof is the monotonicity of counting under the
band. -/
theorem resolved_count_certified {ι : Type*} (s : Finset ι) (lam lamhat : ι → ℝ) (ε edge : ℝ)
    (hband : ∀ k ∈ s, |lamhat k - lam k| ≤ ε) :
    (s.filter (fun k => edge < lamhat k - ε)).card ≤ (s.filter (fun k => edge < lam k)).card ∧
      (s.filter (fun k => edge < lam k)).card ≤ (s.filter (fun k => edge < lamhat k + ε)).card := by
  refine ⟨Finset.card_le_card ?_, Finset.card_le_card ?_⟩
  · intro k hk
    rw [Finset.mem_filter] at hk ⊢
    obtain ⟨hks, hlt⟩ := hk
    have hb := abs_le.mp (hband k hks)
    exact ⟨hks, by linarith [hb.2]⟩
  · intro k hk
    rw [Finset.mem_filter] at hk ⊢
    obtain ⟨hks, hlt⟩ := hk
    have hb := abs_le.mp (hband k hks)
    exact ⟨hks, by linarith [hb.1]⟩

/-- **Certified separation (concentration form).**
Two ensemble means `Ec, Ed` lie in their certified intervals `[mc - tc, mc + tc]`,
`[md - td, md + td]` (a concentration guarantee, `|E - m| ≤ t`). If the intervals do not overlap,
`mc + tc < md - td`, the true means are ordered, `Ec < Ed`: the two populations are distinct. This
certifies the separation criterion the ensemble read (`k_signal_certificate`) reports. -/
theorem separated_of_disjoint_intervals {Ec Ed mc tc md td : ℝ}
    (hEc : |Ec - mc| ≤ tc) (hEd : |Ed - md| ≤ td) (hgap : mc + tc < md - td) :
    Ec < Ed := by
  have hc := abs_le.mp hEc
  have hd := abs_le.mp hEd
  linarith [hc.2, hd.1]

/-- **Section 8 (derived floor), monotone in significance.**
The Johnstone/Tracy-Widom noise floor `Φ(q) = sqrt(σ²·(μ + q·ς_J))` rises with the significance
quantile `q`: for `σ² ≥ 0` and `ς_J ≥ 0`, a stricter false-alarm rate (larger `q`) gives a higher
floor. This is monotonicity of the derived edge in its universal Tracy-Widom quantile. -/
theorem noise_floor_monotone {σ2 μ ςJ q₁ q₂ : ℝ}
    (hσ : 0 ≤ σ2) (hς : 0 ≤ ςJ) (hq : q₁ ≤ q₂) :
    Real.sqrt (σ2 * (μ + q₁ * ςJ)) ≤ Real.sqrt (σ2 * (μ + q₂ * ςJ)) := by
  apply Real.sqrt_le_sqrt
  have hinner : q₁ * ςJ ≤ q₂ * ςJ := mul_le_mul_of_nonneg_right hq hς
  exact mul_le_mul_of_nonneg_left (by linarith) hσ

/-- **Section 8 (resolved dimension), antitone in the floor.**
The resolved dimension `K_signal = #{k : sₖ > Φ}` is nonincreasing in the floor `Φ`: for `Φ₁ ≤ Φ₂` the count above `Φ₂` is at most the count above `Φ₁`,
by monotonicity of the counting filter under the threshold. -/
theorem resolved_dim_antitone {n : ℕ} (s : Fin n → ℝ) {Φ₁ Φ₂ : ℝ} (h : Φ₁ ≤ Φ₂) :
    (Finset.univ.filter (fun k => Φ₂ < s k)).card
      ≤ (Finset.univ.filter (fun k => Φ₁ < s k)).card := by
  classical
  apply Finset.card_le_card
  intro k hk
  rw [Finset.mem_filter] at hk ⊢
  exact ⟨hk.1, lt_of_le_of_lt h hk.2⟩

end Entroptics
