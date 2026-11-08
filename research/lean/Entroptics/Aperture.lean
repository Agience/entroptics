import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.Analysis.Matrix.PosDef
import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
import Mathlib.NumberTheory.LegendreSymbol.AddCharacter
import Mathlib.Analysis.Complex.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.NormNum

/-!
# A finite aperture holds a finite count (PAPER Sec 3, the aperture; Sec 10, the degrees of freedom)

The discrete time-frequency concentration operator. On the cyclic group `ℤ_N` take a set of sites
`S` (the aperture in time, `T = #S`) and a set of Fourier modes `Ω` (the band, `W = #Ω`). The
time-limit `P_S` is the diagonal projection onto `S`; the band-limit `B_Ω` is the projection onto the
span of the characters `j ↦ e(ω j)`, `ω ∈ Ω`, written entrywise as
`B_Ω j k = N⁻¹ ∑_{ω ∈ Ω} e(ω (j - k))` with `e = ZMod.stdAddChar`. The concentration operator is
`A = P_S B_Ω P_S`.

* `timeLimit_isHermitian`, `timeLimit_mul_self`, `bandLimit_isHermitian`, `bandLimit_mul_self`: both
  are orthogonal projections. The band-limit is idempotent by character orthogonality on `ZMod N`.
* `concentration_trace`: `tr A = T W / N`, the classical degrees-of-freedom count of the aperture.
* `compression_eigenvalues_mem_unit`: for any two orthogonal projections, the eigenvalues of `P B P`
  lie in `[0, 1]`, because `P B P - (P B P)² = Mᴴ M` with `M = (1 - P) B P`.
* `plunge_card_mul_le`, `count_half_sub_sum_le`, `collision_count_bounds`: the spectral counting,
  on any spectrum in `[0, 1]`. The number of eigenvalues in the plunge `(ε, 1 - ε)` is at most
  `δ / (ε (1 - ε))`, where `δ = ∑ λ (1 - λ) = tr A - tr A²`; the number above `1/2` is within `2 δ` of
  `tr A`; when `∑ μ² > 0`, the Rényi-2 (collision) count of the normalized spectrum is at least
  `tr A`, and at most `tr A + 2 δ` whenever `2 δ ≤ tr A`.
* `concentration_plunge_eq_leakage`: `δ = ∑_{j ∈ S} ∑_{k ∉ S} ‖B_Ω k j‖²`, the band-limit's leakage
  across the edge of the aperture.
* `aperture_holds_finite_count`: the assembled statement for `A = P_S B_Ω P_S`.

This module does not bound `δ`. For consecutive `S` and `Ω` the bound
`δ ≤ (9 + log (min T W)) / 2` (the plunge region is logarithmic) is `window_plunge_le_log` in
`Entroptics.Plunge`, the Dirichlet-kernel estimate of the leakage sum.
-/

open scoped BigOperators ComplexConjugate ComplexOrder
open Matrix

namespace Entroptics

/-! ### Two orthogonal projections and their compression -/

section Compression

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **The spectral form of `A - A²`.**
For a Hermitian `A` with eigenvector unitary `U` and eigenvalues `λ`, `A - A A` is the conjugate by
`U` of the diagonal matrix `diag (λ - λ²)`. -/
theorem sub_mul_self_eq_conj {A : Matrix n n ℂ} (hA : A.IsHermitian) :
    A - A * A = (hA.eigenvectorUnitary : Matrix n n ℂ) *
      diagonal (RCLike.ofReal ∘ fun i => hA.eigenvalues i - hA.eigenvalues i ^ 2) *
      star (hA.eigenvectorUnitary : Matrix n n ℂ) := by
  have hspec := hA.spectral_theorem
  rw [Unitary.conjStarAlgAut_apply] at hspec
  have hUU : star (hA.eigenvectorUnitary : Matrix n n ℂ) * (hA.eigenvectorUnitary : Matrix n n ℂ)
      = 1 := Unitary.coe_star_mul_self _
  set U : Matrix n n ℂ := (hA.eigenvectorUnitary : Matrix n n ℂ) with hU
  set D : Matrix n n ℂ := diagonal (RCLike.ofReal ∘ hA.eigenvalues) with hD
  have hdiag : diagonal (RCLike.ofReal ∘ fun i => hA.eigenvalues i - hA.eigenvalues i ^ 2)
      = D - D * D := by
    rw [hD, diagonal_mul_diagonal, diagonal_sub]
    congr 1
    funext i
    simp only [Function.comp_apply, RCLike.ofReal_sub, RCLike.ofReal_pow]
    ring
  have hsq : U * D * star U * (U * D * star U) = U * (D * D) * star U := by
    calc U * D * star U * (U * D * star U) = U * D * (star U * U) * D * star U := by
          simp only [Matrix.mul_assoc]
      _ = U * (D * D) * star U := by rw [hUU, Matrix.mul_one]; simp only [Matrix.mul_assoc]
  rw [hdiag]
  conv_lhs => rw [hspec]
  rw [hsq, Matrix.mul_sub, Matrix.sub_mul]

/-- **Eigenvalues in the unit interval.**
If `A` is Hermitian and `A - A²` is positive semidefinite, every eigenvalue of `A` lies in `[0, 1]`:
the conjugate `diag (λ - λ²)` is positive semidefinite, so `λ (1 - λ) ≥ 0`. -/
theorem eigenvalues_mem_unit_of_posSemidef {A : Matrix n n ℂ} (hA : A.IsHermitian)
    (h : (A - A * A).PosSemidef) (i : n) :
    0 ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ 1 := by
  rw [sub_mul_self_eq_conj hA,
    IsUnit.posSemidef_star_right_conjugate_iff Unitary.isUnit_coe, posSemidef_diagonal_iff] at h
  have hi := h i
  simp only [Function.comp_apply, RCLike.ofReal_nonneg] at hi
  have h0 : 0 ≤ hA.eigenvalues i := by nlinarith [sq_nonneg (hA.eigenvalues i)]
  exact ⟨h0, by nlinarith⟩

/-- **The trace of a Hermitian matrix is the sum of its eigenvalues** (real-part form). -/
theorem re_trace_eq_sum_eigenvalues {A : Matrix n n ℂ} (hA : A.IsHermitian) :
    A.trace.re = ∑ i, hA.eigenvalues i := by
  rw [hA.trace_eq_sum_eigenvalues, Complex.re_sum]
  simp

/-- **The plunge mass as a trace.**
`re tr (A - A²) = ∑ λ (1 - λ)` over the eigenvalues of a Hermitian `A`. -/
theorem re_trace_sub_mul_self {A : Matrix n n ℂ} (hA : A.IsHermitian) :
    (A - A * A).trace.re = ∑ i, hA.eigenvalues i * (1 - hA.eigenvalues i) := by
  rw [sub_mul_self_eq_conj hA, trace_mul_cycle, Unitary.coe_star_mul_self, Matrix.one_mul,
    trace_diagonal, Complex.re_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [Function.comp_apply, RCLike.ofReal_eq_complex_ofReal, Complex.ofReal_re]
  ring

omit [DecidableEq n] in
/-- **The trace of a Gram matrix.** `re tr (Mᴴ M) = ∑_j ∑_k ‖M k j‖²`. -/
theorem re_trace_conjTranspose_mul_self (M : Matrix n n ℂ) :
    (Mᴴ * M).trace.re = ∑ j, ∑ k, ‖M k j‖ ^ 2 := by
  rw [Matrix.trace, Complex.re_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Matrix.diag_apply, mul_apply, Complex.re_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [conjTranspose_apply, show star (M k j) = conj (M k j) from rfl, Complex.conj_mul',
    ← Complex.ofReal_pow, Complex.ofReal_re]

omit [DecidableEq n] in
/-- **The compression of a Hermitian matrix by a Hermitian matrix is Hermitian.** -/
theorem compression_isHermitian {P B : Matrix n n ℂ} (hP : P.IsHermitian) (hB : B.IsHermitian) :
    (P * B * P).IsHermitian := by
  show (P * B * P)ᴴ = P * B * P
  rw [conjTranspose_mul, conjTranspose_mul, hP.eq, hB.eq, Matrix.mul_assoc]

/-- **The plunge of a compression is a leakage.**
For orthogonal projections `P` and `B` (Hermitian and idempotent),
`P B P - (P B P)² = Mᴴ M` with `M = (1 - P) B P`: what `B` carries out of the range of `P`. -/
theorem compression_sub_sq {P B : Matrix n n ℂ} (hP : P.IsHermitian) (hB : B.IsHermitian)
    (hPP : P * P = P) (hBB : B * B = B) :
    P * B * P - P * B * P * (P * B * P) = ((1 - P) * B * P)ᴴ * ((1 - P) * B * P) := by
  have hQ : (1 - P) * (1 - P) = 1 - P := by
    rw [Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one,
      Matrix.one_mul, hPP, sub_self, sub_zero]
  have hM : ((1 - P) * B * P)ᴴ = P * B * (1 - P) := by
    rw [conjTranspose_mul, conjTranspose_mul, conjTranspose_sub, conjTranspose_one, hP.eq, hB.eq,
      Matrix.mul_assoc]
  rw [hM]
  calc P * B * P - P * B * P * (P * B * P) = P * B * P - P * B * (P * P) * B * P := by
        simp only [Matrix.mul_assoc]
    _ = P * (B * B) * P - P * B * P * B * P := by rw [hPP, hBB]
    _ = P * B * (1 - P) * B * P := by noncomm_ring
    _ = P * B * ((1 - P) * (1 - P)) * B * P := by rw [hQ]
    _ = P * B * (1 - P) * ((1 - P) * B * P) := by simp only [Matrix.mul_assoc]

/-- **A compression of one orthogonal projection by another has spectrum in `[0, 1]`.**
Every eigenvalue of `P B P` lies in `[0, 1]`: `P B P - (P B P)²` is a Gram matrix `Mᴴ M`. -/
theorem compression_eigenvalues_mem_unit {P B : Matrix n n ℂ} (hP : P.IsHermitian)
    (hB : B.IsHermitian) (hPP : P * P = P) (hBB : B * B = B) (i : n) :
    0 ≤ (compression_isHermitian hP hB).eigenvalues i ∧
      (compression_isHermitian hP hB).eigenvalues i ≤ 1 := by
  refine eigenvalues_mem_unit_of_posSemidef _ ?_ i
  rw [compression_sub_sq hP hB hPP hBB]
  exact posSemidef_conjTranspose_mul_self _

end Compression

/-! ### Counting a spectrum in `[0, 1]` -/

section Counting

variable {ι : Type*} [Fintype ι]

/-- **The plunge bound.**
For a spectrum `μ` in `[0, 1]` and `0 < ε < 1`, the eigenvalues strictly inside `(ε, 1 - ε)` number
at most `δ / (ε (1 - ε))`, with `δ = ∑ μ (1 - μ)`; in product form,
`#{ε < μ < 1 - ε} · ε (1 - ε) ≤ δ`. Each such eigenvalue carries at least `ε (1 - ε)` of `δ`, since
`μ (1 - μ) - ε (1 - ε) = (μ - ε)(1 - ε - μ)`. -/
theorem plunge_card_mul_le (μ : ι → ℝ) (h01 : ∀ i, 0 ≤ μ i ∧ μ i ≤ 1) (ε : ℝ) :
    ((Finset.univ.filter fun i => ε < μ i ∧ μ i < 1 - ε).card : ℝ) * (ε * (1 - ε))
      ≤ ∑ i, μ i * (1 - μ i) := by
  classical
  calc ((Finset.univ.filter fun i => ε < μ i ∧ μ i < 1 - ε).card : ℝ) * (ε * (1 - ε))
      = ∑ _i ∈ Finset.univ.filter (fun i => ε < μ i ∧ μ i < 1 - ε), ε * (1 - ε) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ ∑ i ∈ Finset.univ.filter (fun i => ε < μ i ∧ μ i < 1 - ε), μ i * (1 - μ i) := by
        refine Finset.sum_le_sum fun i hi => ?_
        obtain ⟨h1, h2⟩ := (Finset.mem_filter.mp hi).2
        nlinarith [mul_pos (sub_pos.mpr h1) (sub_pos.mpr h2)]
    _ ≤ ∑ i, μ i * (1 - μ i) :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
          fun i _ _ => mul_nonneg (h01 i).1 (by linarith [(h01 i).2])

/-- **The plunge bound, count form.**
`#{ε < μ < 1 - ε} ≤ δ / (ε (1 - ε))` for `0 < ε < 1`. -/
theorem plunge_card_le (μ : ι → ℝ) (h01 : ∀ i, 0 ≤ μ i ∧ μ i ≤ 1) {ε : ℝ} (hε : 0 < ε)
    (hε1 : ε < 1) :
    ((Finset.univ.filter fun i => ε < μ i ∧ μ i < 1 - ε).card : ℝ)
      ≤ (∑ i, μ i * (1 - μ i)) / (ε * (1 - ε)) := by
  rw [le_div_iff₀ (mul_pos hε (by linarith))]
  exact plunge_card_mul_le μ h01 ε

/-- **The count above one half is the trace, to within twice the plunge.**
For a spectrum `μ` in `[0, 1]`, `|#{μ > 1/2} - ∑ μ| ≤ 2 ∑ μ (1 - μ)`. Termwise, `1 - μ ≤ 2 μ (1 - μ)`
when `μ > 1/2` and `μ ≤ 2 μ (1 - μ)` when `μ ≤ 1/2`. No threshold enters but the midpoint. -/
theorem count_half_sub_sum_le (μ : ι → ℝ) (h01 : ∀ i, 0 ≤ μ i ∧ μ i ≤ 1) :
    |((Finset.univ.filter fun i => 1 / 2 < μ i).card : ℝ) - ∑ i, μ i|
      ≤ 2 * ∑ i, μ i * (1 - μ i) := by
  classical
  rw [Finset.natCast_card_filter, ← Finset.sum_sub_distrib, Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
  obtain ⟨h0, h1⟩ := h01 i
  split_ifs with h
  · rw [abs_of_nonneg (by linarith)]
    nlinarith
  · rw [zero_sub, abs_neg, abs_of_nonneg h0]
    nlinarith

/-- **The Rényi-2 (collision) count of a normalized spectrum.**
With `p = μ / ∑ μ`, the collision count `(∑ p²)⁻¹` equals `(∑ μ)² / ∑ μ²`. -/
theorem collision_count_eq (μ : ι → ℝ) :
    (∑ i, (μ i / ∑ j, μ j) ^ 2)⁻¹ = (∑ i, μ i) ^ 2 / ∑ i, μ i ^ 2 := by
  simp_rw [div_pow]
  rw [← Finset.sum_div, inv_div]

/-- **The collision count is bracketed by the trace and the plunge.**
For a spectrum `μ` in `[0, 1]` with `∑ μ² > 0`, write `s = ∑ μ` and `δ = ∑ μ (1 - μ)`. The Rényi-2
count `s² / ∑ μ²` of the normalized spectrum is at least `s`, and at most `s + 2 δ` whenever
`2 δ ≤ s`. With `s = T W / N` this is a finite aperture holding a finite count. The condition is
needed: a spectrum of many equal small eigenvalues has a large collision count and a small trace. -/
theorem collision_count_bounds (μ : ι → ℝ) (h01 : ∀ i, 0 ≤ μ i ∧ μ i ≤ 1)
    (hq : 0 < ∑ i, μ i ^ 2) :
    ∑ i, μ i ≤ (∑ i, μ i) ^ 2 / ∑ i, μ i ^ 2 ∧
      (2 * ∑ i, μ i * (1 - μ i) ≤ ∑ i, μ i →
        (∑ i, μ i) ^ 2 / ∑ i, μ i ^ 2 ≤ ∑ i, μ i + 2 * ∑ i, μ i * (1 - μ i)) := by
  have hδ : ∑ i, μ i * (1 - μ i) = ∑ i, μ i - ∑ i, μ i ^ 2 := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  have hqs : ∑ i, μ i ^ 2 ≤ ∑ i, μ i :=
    Finset.sum_le_sum fun i _ => by nlinarith [(h01 i).1, (h01 i).2]
  have hs0 : 0 ≤ ∑ i, μ i := Finset.sum_nonneg fun i _ => (h01 i).1
  rw [hδ]
  refine ⟨?_, fun h => ?_⟩
  · rw [le_div_iff₀ hq]
    nlinarith
  · rw [div_le_iff₀ hq]
    nlinarith [mul_nonneg (sub_nonneg.mpr hqs) (by linarith : (0 : ℝ) ≤ 2 * ∑ i, μ i ^ 2 - ∑ i, μ i)]

/-- **The Rényi-∞ count is at most the collision count.**
If `M` is an upper bound of the spectrum (in particular its largest eigenvalue) and `M > 0`, then
`∑ μ / M ≤ (∑ μ)² / ∑ μ²`, since `∑ μ² ≤ M ∑ μ`. -/
theorem minEntropy_count_le_collision (μ : ι → ℝ) (h0 : ∀ i, 0 ≤ μ i) {M : ℝ} (hM : 0 < M)
    (hle : ∀ i, μ i ≤ M) (hq : 0 < ∑ i, μ i ^ 2) :
    (∑ i, μ i) / M ≤ (∑ i, μ i) ^ 2 / ∑ i, μ i ^ 2 := by
  have hs0 : 0 ≤ ∑ i, μ i := Finset.sum_nonneg fun i _ => h0 i
  have hqM : ∑ i, μ i ^ 2 ≤ M * ∑ i, μ i := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => by nlinarith [h0 i, hle i]
  rw [div_le_div_iff₀ hM hq]
  nlinarith [mul_le_mul_of_nonneg_left hqM hs0]

end Counting

/-! ### The time-limit, the band-limit, and the concentration operator on `ℤ_N` -/

section Window

variable {N : ℕ} [NeZero N]

/-- The time-limit `P_S`: the diagonal projection onto the sites in `S`. -/
noncomputable def timeLimit (S : Finset (ZMod N)) : Matrix (ZMod N) (ZMod N) ℂ :=
  diagonal fun j => if j ∈ S then 1 else 0

/-- The band-limit `B_Ω`: the projection onto the characters `j ↦ e(ω j)`, `ω ∈ Ω`, with entries
`B_Ω j k = N⁻¹ ∑_{ω ∈ Ω} e(ω (j - k))` and `e = ZMod.stdAddChar`, `e(x) = exp(2πi x / N)`. -/
noncomputable def bandLimit (Ω : Finset (ZMod N)) : Matrix (ZMod N) (ZMod N) ℂ :=
  of fun j k => (N : ℂ)⁻¹ * ∑ ω ∈ Ω, ZMod.stdAddChar (ω * (j - k))

/-- The concentration operator `A = P_S B_Ω P_S` of the aperture `S` and the band `Ω`. -/
noncomputable def concentration (S Ω : Finset (ZMod N)) : Matrix (ZMod N) (ZMod N) ℂ :=
  timeLimit S * bandLimit Ω * timeLimit S

omit [NeZero N] in
/-- The time-limit `P_S` is Hermitian. -/
theorem timeLimit_isHermitian (S : Finset (ZMod N)) : (timeLimit S).IsHermitian := by
  show (timeLimit S)ᴴ = timeLimit S
  rw [timeLimit, diagonal_conjTranspose]
  congr 1
  funext j
  by_cases h : j ∈ S <;> simp [h]

/-- The time-limit `P_S` is idempotent. -/
theorem timeLimit_mul_self (S : Finset (ZMod N)) : timeLimit S * timeLimit S = timeLimit S := by
  rw [timeLimit, diagonal_mul_diagonal]
  congr 1
  funext j
  by_cases h : j ∈ S <;> simp [h]

/-- The band-limit `B_Ω` is Hermitian, since `conj e(x) = e(-x)`. -/
theorem bandLimit_isHermitian (Ω : Finset (ZMod N)) : (bandLimit Ω).IsHermitian := by
  show (bandLimit Ω)ᴴ = bandLimit Ω
  ext j k
  simp only [conjTranspose_apply, bandLimit, of_apply, star_mul', star_sum, star_inv₀,
    star_natCast]
  congr 1
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [show ω * (j - k) = -(ω * (k - j)) by ring, AddChar.map_neg_eq_conj]
  rfl

/-- **Character orthogonality, the step of `bandLimit_mul_self`.**
`∑_m e(ω (j - m)) e(ω' (m - k)) = N e(ω (j - k))` when `ω = ω'`, and `0` otherwise. -/
theorem sum_stdAddChar_mul (ω ω' j k : ZMod N) :
    ∑ m : ZMod N, ZMod.stdAddChar (ω * (j - m)) * ZMod.stdAddChar (ω' * (m - k))
      = if ω = ω' then (N : ℂ) * ZMod.stdAddChar (ω * (j - k)) else 0 := by
  have hsplit : ∀ m : ZMod N,
      ZMod.stdAddChar (ω * (j - m)) * ZMod.stdAddChar (ω' * (m - k))
        = ZMod.stdAddChar (ω * j - ω' * k) * ZMod.stdAddChar (m * (ω' - ω)) := by
    intro m
    rw [← AddChar.map_add_eq_mul, ← AddChar.map_add_eq_mul]
    congr 1
    ring
  rw [Finset.sum_congr rfl fun m _ => hsplit m, ← Finset.mul_sum,
    AddChar.sum_mulShift _ (ZMod.isPrimitive_stdAddChar N), ZMod.card]
  by_cases h : ω = ω'
  · subst h
    rw [if_pos (sub_self ω), if_pos rfl, ← mul_sub, mul_comm]
  · rw [if_neg (sub_ne_zero.mpr (Ne.symm h)), if_neg h, Nat.cast_zero, mul_zero]

/-- The band-limit `B_Ω` is idempotent, by character orthogonality (`sum_stdAddChar_mul`). -/
theorem bandLimit_mul_self (Ω : Finset (ZMod N)) : bandLimit Ω * bandLimit Ω = bandLimit Ω := by
  have hN : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne N)
  ext j k
  rw [mul_apply]
  simp only [bandLimit, of_apply]
  have hexp : ∀ m : ZMod N,
      ((N : ℂ)⁻¹ * ∑ ω ∈ Ω, ZMod.stdAddChar (ω * (j - m))) *
        ((N : ℂ)⁻¹ * ∑ ω' ∈ Ω, ZMod.stdAddChar (ω' * (m - k)))
      = (N : ℂ)⁻¹ * (N : ℂ)⁻¹ * ∑ ω ∈ Ω, ∑ ω' ∈ Ω,
          ZMod.stdAddChar (ω * (j - m)) * ZMod.stdAddChar (ω' * (m - k)) := by
    intro m
    rw [mul_mul_mul_comm, Finset.sum_mul_sum]
  rw [Finset.sum_congr rfl fun m _ => hexp m, ← Finset.mul_sum, Finset.sum_comm]
  rw [Finset.sum_congr rfl fun ω _ => Finset.sum_comm]
  rw [Finset.sum_congr rfl fun ω hω => by
    rw [Finset.sum_congr rfl fun ω' _ => sum_stdAddChar_mul ω ω' j k, Finset.sum_ite_eq,
      if_pos hω]]
  rw [← Finset.mul_sum]
  field_simp

/-- Each diagonal entry of the band-limit is `#Ω / N`. -/
theorem bandLimit_apply_self (Ω : Finset (ZMod N)) (j : ZMod N) :
    bandLimit Ω j j = (Ω.card : ℂ) / N := by
  simp [bandLimit, div_eq_inv_mul]

/-- The concentration operator `P_S B_Ω P_S` is Hermitian. -/
theorem concentration_isHermitian (S Ω : Finset (ZMod N)) : (concentration S Ω).IsHermitian :=
  compression_isHermitian (timeLimit_isHermitian S) (bandLimit_isHermitian Ω)

/-- **The degrees-of-freedom count of the aperture.**
`tr (P_S B_Ω P_S) = #S · #Ω / N`, the time-bandwidth product `T W / N`. -/
theorem concentration_trace (S Ω : Finset (ZMod N)) :
    (concentration S Ω).trace = (S.card * Ω.card : ℂ) / N := by
  have hterm : ∀ j : ZMod N, (concentration S Ω) j j
      = if j ∈ S then (Ω.card : ℂ) / N else 0 := by
    intro j
    simp only [concentration, timeLimit, mul_diagonal, diagonal_mul, bandLimit_apply_self]
    by_cases h : j ∈ S <;> simp [h]
  rw [trace]
  simp only [diag]
  rw [Finset.sum_congr rfl fun j _ => hterm j, ← Finset.sum_filter, Finset.filter_mem_eq_inter,
    Finset.univ_inter, Finset.sum_const, nsmul_eq_mul]
  ring

/-- **The spectrum of the concentration operator lies in `[0, 1]`.** -/
theorem concentration_eigenvalues_mem_unit (S Ω : Finset (ZMod N)) (i : ZMod N) :
    0 ≤ (concentration_isHermitian S Ω).eigenvalues i ∧
      (concentration_isHermitian S Ω).eigenvalues i ≤ 1 :=
  compression_eigenvalues_mem_unit (timeLimit_isHermitian S) (bandLimit_isHermitian Ω)
    (timeLimit_mul_self S) (bandLimit_mul_self Ω) i

/-- **The eigenvalues sum to `T W / N`.** -/
theorem concentration_sum_eigenvalues (S Ω : Finset (ZMod N)) :
    ∑ i, (concentration_isHermitian S Ω).eigenvalues i = (S.card * Ω.card : ℝ) / N := by
  rw [← re_trace_eq_sum_eigenvalues, concentration_trace]
  have hcast : ((S.card * Ω.card : ℂ) / N) = (((S.card * Ω.card : ℝ) / N : ℝ) : ℂ) := by
    push_cast
    ring
  rw [hcast, Complex.ofReal_re]

/-- **The plunge is the leakage across the aperture's edge.**
`∑ λ (1 - λ) = ∑_{j ∈ S} ∑_{k ∉ S} ‖B_Ω k j‖²` for `A = P_S B_Ω P_S`: the part of `B_Ω` that maps
the inside of the aperture to its outside. This sum is what the Dirichlet-kernel estimate bounds. -/
theorem concentration_plunge_eq_leakage (S Ω : Finset (ZMod N)) :
    ∑ i, (concentration_isHermitian S Ω).eigenvalues i *
        (1 - (concentration_isHermitian S Ω).eigenvalues i)
      = ∑ j ∈ S, ∑ k ∈ Sᶜ, ‖bandLimit Ω k j‖ ^ 2 := by
  rw [← re_trace_sub_mul_self]
  show (concentration S Ω - concentration S Ω * concentration S Ω).trace.re = _
  rw [concentration, compression_sub_sq (timeLimit_isHermitian S) (bandLimit_isHermitian Ω)
    (timeLimit_mul_self S) (bandLimit_mul_self Ω)]
  have hcompl : (1 : Matrix (ZMod N) (ZMod N) ℂ) - timeLimit S
      = diagonal fun k => if k ∈ Sᶜ then 1 else 0 := by
    rw [timeLimit, ← diagonal_one, diagonal_sub]
    congr 1
    funext k
    by_cases h : k ∈ S <;> simp [h]
  have hentry : ∀ k j : ZMod N, ((1 - timeLimit S) * bandLimit Ω * timeLimit S) k j
      = (if k ∈ Sᶜ then 1 else 0) * bandLimit Ω k j * (if j ∈ S then 1 else 0) := by
    intro k j
    rw [hcompl, timeLimit, mul_diagonal, diagonal_mul]
  have hnorm : ∀ k j : ZMod N,
      ‖((1 - timeLimit S) * bandLimit Ω * timeLimit S) k j‖ ^ 2
        = if j ∈ S then (if k ∈ Sᶜ then ‖bandLimit Ω k j‖ ^ 2 else 0) else 0 := by
    intro k j
    rw [hentry]
    by_cases hj : j ∈ S <;> by_cases hk : k ∈ Sᶜ <;> simp [hj, hk]
  have hind : ∀ (T : Finset (ZMod N)) (g : ZMod N → ℝ),
      ∑ k, (if k ∈ T then g k else 0) = ∑ k ∈ T, g k := fun T g => by
    rw [← Finset.sum_filter, Finset.filter_mem_eq_inter, Finset.univ_inter]
  rw [re_trace_conjTranspose_mul_self]
  simp only [hnorm]
  rw [← hind S]
  refine Finset.sum_congr rfl fun j _ => ?_
  by_cases hj : j ∈ S
  · simp only [hj, if_true]
    exact hind Sᶜ _
  · simp only [hj, if_false, Finset.sum_const_zero]

/-- **A finite aperture holds a finite count.**
For the concentration operator `A = P_S B_Ω P_S` on `ℤ_N`, with `T = #S`, `W = #Ω` and eigenvalues
`λ`, write `δ = ∑ λ (1 - λ) = tr A - tr A²`. Then
1. `∑ λ = T W / N`, and every `λ ∈ [0, 1]`;
2. `δ = ∑_{j ∈ S} ∑_{k ∉ S} ‖B_Ω k j‖²`, the leakage across the edge of the aperture;
3. the number of eigenvalues above `1/2` is within `2 δ` of `T W / N`;
4. for `0 < ε < 1`, at most `δ / (ε (1 - ε))` eigenvalues lie in the plunge `(ε, 1 - ε)`.
The count of the aperture is the time-bandwidth product, to within the plunge. This theorem does
not bound `δ`; for consecutive `S` and `Ω`, `window_plunge_le_log` in `Entroptics.Plunge` gives
`δ ≤ (9 + log (min T W)) / 2`, and `window_count_half` the resulting count. -/
theorem aperture_holds_finite_count (S Ω : Finset (ZMod N)) :
    (∑ i, (concentration_isHermitian S Ω).eigenvalues i = (S.card * Ω.card : ℝ) / N) ∧
    (∀ i, 0 ≤ (concentration_isHermitian S Ω).eigenvalues i ∧
      (concentration_isHermitian S Ω).eigenvalues i ≤ 1) ∧
    (∑ i, (concentration_isHermitian S Ω).eigenvalues i *
        (1 - (concentration_isHermitian S Ω).eigenvalues i)
      = ∑ j ∈ S, ∑ k ∈ Sᶜ, ‖bandLimit Ω k j‖ ^ 2) ∧
    |((Finset.univ.filter fun i => 1 / 2 < (concentration_isHermitian S Ω).eigenvalues i).card
        : ℝ) - (S.card * Ω.card : ℝ) / N|
      ≤ 2 * ∑ i, (concentration_isHermitian S Ω).eigenvalues i *
        (1 - (concentration_isHermitian S Ω).eigenvalues i) ∧
    (∀ ε : ℝ, 0 < ε → ε < 1 →
      ((Finset.univ.filter fun i => ε < (concentration_isHermitian S Ω).eigenvalues i ∧
          (concentration_isHermitian S Ω).eigenvalues i < 1 - ε).card : ℝ)
        ≤ (∑ i, (concentration_isHermitian S Ω).eigenvalues i *
            (1 - (concentration_isHermitian S Ω).eigenvalues i)) / (ε * (1 - ε))) := by
  have h01 := concentration_eigenvalues_mem_unit S Ω
  refine ⟨concentration_sum_eigenvalues S Ω, h01, concentration_plunge_eq_leakage S Ω, ?_,
    fun ε hε hε1 => plunge_card_le _ h01 hε hε1⟩
  rw [← concentration_sum_eigenvalues S Ω]
  exact count_half_sub_sum_le _ h01

/-- **The collision count of the aperture.**
For `A = P_S B_Ω P_S` with `tr A² > 0`, the Rényi-2 count of the normalized spectrum is at least
`T W / N`, and at most `T W / N + 2 δ` whenever `2 δ ≤ T W / N`. -/
theorem aperture_collision_count (S Ω : Finset (ZMod N))
    (hq : 0 < ∑ i, (concentration_isHermitian S Ω).eigenvalues i ^ 2) :
    (S.card * Ω.card : ℝ) / N
        ≤ (∑ i, (concentration_isHermitian S Ω).eigenvalues i) ^ 2 /
          ∑ i, (concentration_isHermitian S Ω).eigenvalues i ^ 2 ∧
      (2 * ∑ i, (concentration_isHermitian S Ω).eigenvalues i *
          (1 - (concentration_isHermitian S Ω).eigenvalues i) ≤ (S.card * Ω.card : ℝ) / N →
        (∑ i, (concentration_isHermitian S Ω).eigenvalues i) ^ 2 /
            ∑ i, (concentration_isHermitian S Ω).eigenvalues i ^ 2
          ≤ (S.card * Ω.card : ℝ) / N + 2 * ∑ i, (concentration_isHermitian S Ω).eigenvalues i *
            (1 - (concentration_isHermitian S Ω).eigenvalues i)) := by
  rw [← concentration_sum_eigenvalues S Ω]
  exact collision_count_bounds _ (concentration_eigenvalues_mem_unit S Ω) hq

end Window

end Entroptics
