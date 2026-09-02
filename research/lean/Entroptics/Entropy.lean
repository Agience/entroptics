import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.LinearAlgebra.Matrix.Hermitian
import Mathlib.LinearAlgebra.Matrix.Charpoly.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.NormNum

/-!
# Entropy, fill fraction, and Strehl (PAPER Sec 2-3)

The entropy content that fixes the fill fraction `φ = 2^H/n`, the Strehl ratio, and the diffraction
limit `a_δ = 2^{-H_C}`, together with the fill-reach duality `φ·δ = 1`.

* `ratio_bounds` (Lemma 3.4): a dominating weight's share lies in `[1/n, 1]`.
* `entropy_nonneg_le_log`, `fill_fraction_entropy_bounds` (Lemma 3.2): `0 ≤ H(q) ≤ log n`, hence
  `2^H ∈ [1, n]` and `φ ∈ [1/n, 1]`.
* `aperture_duality` (Sec 2): `φ·δ = 1` for `φ = 2^H/n`, `δ = n/2^H`.
-/

open scoped BigOperators Matrix

namespace Entroptics

/-- **Lemma 3.2 / 3.4 (fill-fraction & Strehl bounds).**
For nonnegative weights `w` over `Fin n` (`n ≥ 1`) with a dominating index `k` and a positive total,
the ratio `w k / ∑ w` lies in `[1/n, 1]`. Taking `w` the singular-value power spectrum bounds the
fill fraction `φ ∈ [1/n, 1]` (Lemma 3.2); taking `w` the ordered correlation spectrum bounds the
Strehl ratio `𝒮 ∈ [1/T, 1]` (Lemma 3.4). -/
theorem ratio_bounds {n : ℕ} (hn : 0 < n) (w : Fin n → ℝ)
    (hw : ∀ i, 0 ≤ w i) (k : Fin n) (htop : ∀ i, w i ≤ w k)
    (hpos : 0 < ∑ i, w i) :
    1 / (n : ℝ) ≤ w k / (∑ i, w i) ∧ w k / (∑ i, w i) ≤ 1 := by
  have hconst : (∑ _i : Fin n, w k) = (n : ℝ) * w k := by
    simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hsum_le : (∑ i, w i) ≤ (n : ℝ) * w k :=
    calc (∑ i, w i) ≤ ∑ _i : Fin n, w k := Finset.sum_le_sum (fun i _ => htop i)
      _ = (n : ℝ) * w k := hconst
  have hk_le : w k ≤ ∑ i, w i := Finset.single_le_sum (fun i _ => hw i) (Finset.mem_univ k)
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have hn0 : (n : ℝ) ≠ 0 := ne_of_gt hnpos
  have hS0 : (∑ i, w i) ≠ 0 := ne_of_gt hpos
  refine ⟨?_, ?_⟩
  · have hden : 0 < (n : ℝ) * (∑ i, w i) := mul_pos hnpos hpos
    have hnum : 0 ≤ (n : ℝ) * w k - (∑ i, w i) := by linarith [hsum_le]
    have hid : w k / (∑ i, w i) - 1 / (n : ℝ)
        = ((n : ℝ) * w k - (∑ i, w i)) / ((n : ℝ) * (∑ i, w i)) := by
      field_simp
    have hdiff : 0 ≤ w k / (∑ i, w i) - 1 / (n : ℝ) := by
      rw [hid]; exact div_nonneg hnum (le_of_lt hden)
    linarith
  · rw [div_le_one hpos]
    exact hk_le

/-- **Lemma 3.2 (entropy bound).**
For a probability vector `q` on `Fin n` (`n ≥ 1`) the natural-log Shannon entropy
`H q = ∑ i, negMulLog (q i)` (with `negMulLog x = -x·log x`, so `H` is in nats) obeys
`0 ≤ H q ≤ log n`. The lower bound is termwise; the upper bound is Gibbs/Jensen, concavity of
`negMulLog` on `[0,∞)` with the uniform weights `1/n`. This gives the entropy-based fill fraction
`φ = 2^H/n` and diffraction limit `a_δ = 2^{-H_C}` their bounds. -/
theorem entropy_nonneg_le_log {n : ℕ} (hn : 0 < n) (q : Fin n → ℝ)
    (hq : ∀ i, 0 ≤ q i) (hsum : ∑ i, q i = 1) :
    0 ≤ ∑ i, Real.negMulLog (q i) ∧ ∑ i, Real.negMulLog (q i) ≤ Real.log n := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have hqle : ∀ i, q i ≤ 1 := fun i =>
    hsum ▸ Finset.single_le_sum (fun j _ => hq j) (Finset.mem_univ i)
  refine ⟨Finset.sum_nonneg fun i _ => Real.negMulLog_nonneg (hq i) (hqle i), ?_⟩
  have hw : ∀ i ∈ (Finset.univ : Finset (Fin n)), (0 : ℝ) ≤ (n : ℝ)⁻¹ :=
    fun _ _ => by positivity
  have hwsum : ∑ _i : Fin n, (n : ℝ)⁻¹ = 1 := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      mul_inv_cancel₀ (ne_of_gt hn0)]
  have hmem : ∀ i ∈ (Finset.univ : Finset (Fin n)), q i ∈ Set.Ici (0 : ℝ) :=
    fun i _ => hq i
  have hJ := Real.concaveOn_negMulLog.le_map_sum hw hwsum hmem
  simp only [smul_eq_mul] at hJ
  have hLHS : (∑ i, (n : ℝ)⁻¹ * Real.negMulLog (q i))
      = (n : ℝ)⁻¹ * ∑ i, Real.negMulLog (q i) := by rw [Finset.mul_sum]
  have hRHS : (∑ i, (n : ℝ)⁻¹ * q i) = (n : ℝ)⁻¹ := by rw [← Finset.mul_sum, hsum, mul_one]
  rw [hLHS, hRHS] at hJ
  have hval : Real.negMulLog ((n : ℝ)⁻¹) = (n : ℝ)⁻¹ * Real.log n := by
    rw [show Real.negMulLog ((n : ℝ)⁻¹) = -(n : ℝ)⁻¹ * Real.log ((n : ℝ)⁻¹) from rfl,
      Real.log_inv]; ring
  rw [hval] at hJ
  exact le_of_mul_le_mul_left hJ (by positivity)

/-- **Lemma 3.2, fill-fraction corollary.**
Exponentiating the entropy bound gives the fill fraction. With the bit-entropy `H₂ = H/log 2` (so
`2^{H₂}` is the paper's `2^{H(q)}`), the bound yields `2^{H₂} ∈ [1, n]` and `φ = 2^{H₂}/n ∈ [1/n, 1]`
(Definition 3.1); the same bound on the correlation spectrum bounds the Strehl ratio `𝒮 ∈ [1/T, 1]`,
and on the lag distribution bounds `a_δ = 2^{-H_C} ∈ [1/T, 1]` (Definition 4.4). -/
theorem fill_fraction_entropy_bounds {n : ℕ} (hn : 0 < n) (q : Fin n → ℝ)
    (hq : ∀ i, 0 ≤ q i) (hsum : ∑ i, q i = 1) :
    1 ≤ (2 : ℝ) ^ ((∑ i, Real.negMulLog (q i)) / Real.log 2) ∧
    (2 : ℝ) ^ ((∑ i, Real.negMulLog (q i)) / Real.log 2) ≤ (n : ℝ) ∧
    1 / (n : ℝ) ≤ (2 : ℝ) ^ ((∑ i, Real.negMulLog (q i)) / Real.log 2) / (n : ℝ) ∧
    (2 : ℝ) ^ ((∑ i, Real.negMulLog (q i)) / Real.log 2) / (n : ℝ) ≤ 1 := by
  obtain ⟨hH0, hHlog⟩ := entropy_nonneg_le_log hn q hq hsum
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  set H := ∑ i, Real.negMulLog (q i) with hHdef
  have hlog2 : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  have hbridge : (2 : ℝ) ^ (H / Real.log 2) = Real.exp H := by
    rw [Real.rpow_def_of_pos (by norm_num : (0 : ℝ) < 2)]
    congr 1
    field_simp
  rw [hbridge]
  have h1 : (1 : ℝ) ≤ Real.exp H := by
    rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr hH0
  have h2 : Real.exp H ≤ (n : ℝ) := by
    calc Real.exp H ≤ Real.exp (Real.log n) := Real.exp_le_exp.mpr hHlog
      _ = (n : ℝ) := Real.exp_log hn0
  refine ⟨h1, h2, ?_, (div_le_one hn0).mpr h2⟩
  gcongr

/-- **Fill-reach duality** (`φ·δ = 1`): with fill `φ = x/y` and reach `δ = y/x` (`x = 2^H`, `y = n`),
the product is `1`. -/
theorem aperture_duality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
    (x / y) * (y / x) = 1 := by
  field_simp

/-- **Section 2 (fold guard).**
The Miller-Madow uniform-null bias band `β_a = (L_a - 1)/(2·L_{a'}·log 2)` is nonnegative: with
`L_a ≥ 1` (at least one level) and `L_{a'} > 0`, the closed-form fold band is `≥ 0`. This is a
derived analytic guard, so §2's fold decision has a well-defined nonnegative tolerance. -/
theorem miller_madow_band_nonneg {La La' : ℝ} (hLa : 1 ≤ La) (hLa' : 0 < La') :
    0 ≤ (La - 1) / (2 * La' * Real.log 2) := by
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  exact div_nonneg (by linarith) (by positivity)

/-! ### Basis invariance of the spectral read (Proposition 3.5, spectral form)

Proposition 3.5 states the étendue `𝓔 = φ_F φ_T`, and each axis fill `φ_a`, are invariant under a
congruence of the correlation operator (a relabeling plus a per-variable phase). The invariance holds
for ANY congruence, not only permutations: a read that is a function of the operator's characteristic
polynomial (hence of its eigenvalue multiset) is unchanged by conjugation, the characteristic
polynomial being a similarity invariant. In particular an orthogonal congruence `C ↦ P C Pᵀ`
(`Pᵀ P = 1`), the action of a rotation, leaves every spectral read invariant, so the read is isotropic
under the full orthogonal group and its rotation subgroup. -/

/-- **Proposition 3.5, spectral form (congruence invariance).** A read `f` that depends only on the
correlation operator's characteristic polynomial (the fill fraction, étendue, Strehl ratio: functions
of the eigenvalue multiset) is invariant under conjugation `C ↦ P C Q` by an inverse pair
(`Q P = 1`). -/
theorem spectral_read_congruence {n : Type*} [Fintype n] [DecidableEq n]
    {R : Type*} [CommRing R] {α : Type*}
    (f : Polynomial R → α) (P C Q : Matrix n n R) (hQP : Q * P = 1) :
    f ((P * C * Q).charpoly) = f (C.charpoly) := by
  have hcong : (P * C * Q).charpoly = C.charpoly := by
    rw [Matrix.charpoly_mul_comm, ← Matrix.mul_assoc, hQP, Matrix.one_mul]
  rw [hcong]

/-- **Proposition 3.5 for a rotation (orthogonal congruence).** A rotation acts on the correlation
operator by an orthogonal congruence `C ↦ P C Pᵀ` (`Pᵀ P = 1`). Every spectral read is invariant, so
the read is isotropic under the orthogonal group `O(n)` and its rotation subgroup `SO(n)`; the
hypercubic point group (permutation matrices) is the special case. -/
theorem spectral_read_orthogonal {n : Type*} [Fintype n] [DecidableEq n]
    {R : Type*} [CommRing R] {α : Type*}
    (f : Polynomial R → α) (P C : Matrix n n R) (hP : Pᵀ * P = 1) :
    f ((P * C * Pᵀ).charpoly) = f (C.charpoly) :=
  spectral_read_congruence f P C Pᵀ hP

end Entroptics
