import Mathlib

/-!
# The OTF, the diffraction limit, and scale-invariance (PAPER Sec 4)

The autocorrelation (OTF) is positive semidefinite, peaks at zero lag, and its shape factor
`g = ξ·a_δ` is exactly scale-invariant under integer block-rescale.

* `otf_posSemidef`, `toeplitz_acf_posSemidef` (Lemma 4.2): the Toeplitz matrix of the autocovariance
  is positive semidefinite (sum-of-squares).
* `peak_at_zero_lag`, `peak_at_zero_lag_toeplitz` (Cor 4.3): `|C τ| ≤ C 0`.
* `entropy_block_repeat`, `integral_length_block_repeat`, `gabor_product_scale_invariant`
  (Prop 4.5): the discrete scale-invariance of `g = ξ·a_δ`, via `H(block-repeat) = H + log s` and
  `ξ' = s·ξ`.
-/

open scoped BigOperators

namespace Entroptics

/-- **Corollary 4.3 (peak at zero lag).**
Positive-semidefiniteness of the 2×2 autocovariance minor, `C τ ^ 2 ≤ C 0 ^ 2` with `0 ≤ C 0`, makes
the OTF peak at zero lag: `|C τ| ≤ C 0`. -/
theorem peak_at_zero_lag {C0 Ctau : ℝ} (h0 : 0 ≤ C0) (hpsd : Ctau ^ 2 ≤ C0 ^ 2) :
    |Ctau| ≤ C0 := by
  rw [abs_le]
  constructor <;>
    nlinarith [hpsd, h0, sq_nonneg (C0 - Ctau), sq_nonneg (C0 + Ctau)]

/-- **Lemma 4.2 (OTF positive semidefiniteness), sum-of-squares core.**
The Toeplitz quadratic form of the (biased) autocovariance is a sum of squares, hence `≥ 0`. Here
`a : ℤ → ℝ` is the zero-extended signal, `S` the summation window, and `shift j` the lag of index
`j`; the inner sum `∑_{s∈S} a(s+shift j) a(s+shift k)` is the un-normalised Toeplitz entry
`T·C(shift j − shift k)`, so this is
`∑_{j,k} v_j v_k · T·C(j−k) = ∑_s (∑_j v_j a(s+shift j))² ≥ 0`. The finite, index-correct form of the
periodogram (Wiener–Khinchin) argument. -/
theorem otf_posSemidef {n : ℕ} (a : ℤ → ℝ) (S : Finset ℤ)
    (v : Fin n → ℝ) (shift : Fin n → ℤ) :
    0 ≤ ∑ j, ∑ k, v j * v k * ∑ s ∈ S, a (s + shift j) * a (s + shift k) := by
  have key : (∑ s ∈ S, (∑ j, v j * a (s + shift j)) ^ 2)
      = ∑ j, ∑ k, v j * v k * ∑ s ∈ S, a (s + shift j) * a (s + shift k) := by
    simp_rw [pow_two, Finset.sum_mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun s _ => ?_
    ring
  rw [← key]
  exact Finset.sum_nonneg fun s _ => sq_nonneg _

/-- **Lemma 4.2 (OTF positive semidefiniteness) as the autocovariance Toeplitz.**
For a finitely-supported signal `a : ℤ → ℝ`, the full-line autocovariance `C τ = ∑ᶠ s, a s · a (s+τ)`
depends only on the lag, so for shifts `shift j` the Toeplitz entry is `M j k = C (shift j - shift k)`
(taking `shift = id` gives `M j k = C (j - k)`). Its matrix is positive semidefinite:
`∀ v, 0 ≤ ∑ j k, v j · v k · C (shift j - shift k)`, via the sum-of-squares identity
`∑ j k v j v k C(shift j - shift k) = ∑ᶠ m, (∑ j v j a(m + shift j))²`. The inner sum is reindexed to
a function of the lag alone. -/
theorem toeplitz_acf_posSemidef {n : ℕ} (a : ℤ → ℝ) (ha : (Function.support a).Finite)
    (v : Fin n → ℝ) (shift : Fin n → ℤ) :
    0 ≤ ∑ j, ∑ k, v j * v k * (∑ᶠ s, a s * a (s + (shift j - shift k))) := by
  have hfs : ∀ p : ℤ, (Function.support (fun m => a (m + p))).Finite := by
    intro p
    apply Set.Finite.subset (ha.preimage ((add_left_injective p).injOn))
    intro m hm
    simp only [Function.mem_support, Set.mem_preimage] at hm ⊢
    exact hm
  have hC : ∀ p q : ℤ, (∑ᶠ m, a (m + p) * a (m + q)) = ∑ᶠ s, a s * a (s + (p - q)) := by
    intro p q
    rw [← finsum_comp_equiv (Equiv.addRight q) (f := fun s => a s * a (s + (p - q)))]
    refine finsum_congr (fun m => ?_)
    simp only [Equiv.coe_addRight]
    rw [show m + q + (p - q) = m + p from by ring, mul_comm]
  have hsupp2 : ∀ j k : Fin n,
      Function.HasFiniteSupport
        (fun m => v j * v k * (a (m + shift j) * a (m + shift k))) := by
    intro j k
    apply Set.Finite.subset (hfs (shift j))
    intro m hm
    simp only [Function.mem_support] at hm ⊢
    intro h0; apply hm; rw [h0]; ring
  have hsupp1 : ∀ j : Fin n,
      Function.HasFiniteSupport
        (fun m => ∑ k, v j * v k * (a (m + shift j) * a (m + shift k))) := by
    intro j
    apply Set.Finite.subset (hfs (shift j))
    intro m hm
    simp only [Function.mem_support] at hm ⊢
    intro h0; apply hm; exact Finset.sum_eq_zero (fun k _ => by rw [h0]; ring)
  have key : (∑ᶠ m, (∑ j, v j * a (m + shift j)) ^ 2)
      = ∑ j, ∑ k, v j * v k * (∑ᶠ s, a s * a (s + (shift j - shift k))) := by
    have expand : ∀ m : ℤ, (∑ j, v j * a (m + shift j)) ^ 2
        = ∑ j, ∑ k, v j * v k * (a (m + shift j) * a (m + shift k)) := by
      intro m
      rw [pow_two, Finset.sum_mul_sum]
      exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => by ring
    rw [finsum_congr expand, finsum_sum_comm _ _ (fun j _ => hsupp1 j)]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [finsum_sum_comm _ _ (fun k _ => hsupp2 j k)]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [← mul_finsum, hC (shift j) (shift k)]
  rw [← key]
  exact finsum_nonneg fun m => sq_nonneg _

/-- **Corollary 4.3 (peak at zero lag), from the Toeplitz PSD.**
Instantiating the positive-semidefinite quadratic form of `toeplitz_acf_posSemidef` on `![1, 1]` and
`![1, -1]` at shifts `![0, τ]` gives `2·C 0 ± 2·C τ ≥ 0`, hence `|C τ| ≤ C 0`. Here
`C 0 = ∑ᶠ s, (a s)²`. Uses the ACF symmetry `C (-τ) = C τ`. -/
theorem peak_at_zero_lag_toeplitz (a : ℤ → ℝ) (ha : (Function.support a).Finite) (τ : ℤ) :
    |∑ᶠ s, a s * a (s + τ)| ≤ ∑ᶠ s, a s * a s := by
  have hsym : (∑ᶠ s, a s * a (s + -τ)) = ∑ᶠ s, a s * a (s + τ) := by
    rw [← finsum_comp_equiv (Equiv.addRight τ) (f := fun s => a s * a (s + -τ))]
    exact finsum_congr fun m => by
      simp only [Equiv.coe_addRight]; rw [show m + τ + -τ = m from by ring, mul_comm]
  have hquad : ∀ w : Fin 2 → ℝ,
      (∑ j, ∑ k, w j * w k * (∑ᶠ s, a s * a (s + (![(0 : ℤ), τ] j - ![(0 : ℤ), τ] k))))
        = (w 0 ^ 2 + w 1 ^ 2) * (∑ᶠ s, a s * a s)
          + 2 * (w 0 * w 1) * (∑ᶠ s, a s * a (s + τ)) := by
    intro w
    simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
      sub_self, sub_zero, zero_sub, add_zero]
    rw [hsym]; ring
  have h1 := toeplitz_acf_posSemidef a ha ![1, -1] ![(0 : ℤ), τ]
  have h2 := toeplitz_acf_posSemidef a ha ![1, 1] ![(0 : ℤ), τ]
  rw [hquad ![1, -1]] at h1
  rw [hquad ![1, 1]] at h2
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one] at h1 h2
  rw [abs_le]
  exact ⟨by nlinarith [h1, h2], by nlinarith [h1, h2]⟩

/-! ### Proposition 4.5: discrete scale-invariance of the Gabor product.

The discrete statement that makes `g = ξ·a_δ` exactly scale-invariant is the block-repeat identity:
rescaling by an integer factor `s` splits each cell into `s` sub-cells, each carrying `1/s` of its
mass. Here `finProdFinEquiv : Fin n × Fin s ≃ Fin (n*s)` sends `(i, k) ↦ s·i + k`, so `q'` is `q`
block-repeated `s` times with mass split equally. -/

/-- **Proposition 4.5, entropy step.**
If `q' : Fin (n*s) → ℝ` is `q` block-repeated `s ≥ 1` times with each mass split equally
(`q' (s·i + k) = q i / s`), then the natural-log Shannon entropy gains exactly `log s`:
`H q' = H q + log s`. Dividing by `log 2` gives the bit-entropy form `H(q_s) = H(q) + log₂ s`. -/
theorem entropy_block_repeat {n s : ℕ} (hs : 0 < s)
    (q : Fin n → ℝ) (hsum : ∑ i, q i = 1)
    (q' : Fin (n * s) → ℝ)
    (hq' : ∀ (i : Fin n) (k : Fin s), q' (finProdFinEquiv (i, k)) = q i / (s : ℝ)) :
    (∑ j, Real.negMulLog (q' j)) = (∑ i, Real.negMulLog (q i)) + Real.log s := by
  have hs0 : (0 : ℝ) < s := by exact_mod_cast hs
  have hsne : (s : ℝ) ≠ 0 := ne_of_gt hs0
  rw [← Equiv.sum_comp finProdFinEquiv (fun j => Real.negMulLog (q' j)), Fintype.sum_prod_type]
  have hterm : ∀ i : Fin n,
      (∑ k : Fin s, Real.negMulLog (q' (finProdFinEquiv (i, k))))
        = Real.negMulLog (q i) + q i * Real.log s := by
    intro i
    have hnml : Real.negMulLog (q i / (s : ℝ))
        = (s : ℝ)⁻¹ * Real.negMulLog (q i) + q i * ((s : ℝ)⁻¹ * Real.log s) := by
      rw [div_eq_mul_inv, Real.negMulLog_mul]
      congr 1
      rw [show Real.negMulLog ((s : ℝ)⁻¹) = -(s : ℝ)⁻¹ * Real.log ((s : ℝ)⁻¹) from rfl,
        Real.log_inv]; ring
    rw [Finset.sum_congr rfl (fun k _ => by rw [hq' i k]), Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, hnml]
    field_simp
  rw [Finset.sum_congr rfl (fun i _ => hterm i), Finset.sum_add_distrib, ← Finset.sum_mul,
    hsum, one_mul]

/-- **Proposition 4.5, integral-length step.**
The integral correlation length `ξ = ∑_τ C(τ)/C(0)` scales linearly under the same block-rescaling:
if `f'` is the ratio profile block-repeated `s` times (`f' (s·i + k) = f i`), then
`ξ' = ∑ f' = s · ∑ f = s · ξ`. -/
theorem integral_length_block_repeat {n s : ℕ}
    (f : Fin n → ℝ) (f' : Fin (n * s) → ℝ)
    (hf' : ∀ (i : Fin n) (k : Fin s), f' (finProdFinEquiv (i, k)) = f i) :
    (∑ j, f' j) = (s : ℝ) * ∑ i, f i := by
  rw [← Equiv.sum_comp finProdFinEquiv (fun j => f' j), Fintype.sum_prod_type]
  have hterm : ∀ i : Fin n, (∑ k : Fin s, f' (finProdFinEquiv (i, k))) = (s : ℝ) * f i := by
    intro i
    rw [Finset.sum_congr rfl (fun k _ => hf' i k), Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
  rw [Finset.sum_congr rfl (fun i _ => hterm i), ← Finset.mul_sum]

/-- **Proposition 4.5: the Gabor product is exactly scale-invariant (discrete).**
With `a_δ = exp(-H)` (the natural-log form of `2^{-H}`) and `ξ = ∑ f`, the shape factor `g = ξ · a_δ`
is unchanged under block-rescaling: `ξ' · a_δ' = (s·ξ)·(a_δ / s) = ξ · a_δ`. The discrete block form
is exact. -/
theorem gabor_product_scale_invariant {n s : ℕ} (hs : 0 < s)
    (q : Fin n → ℝ) (hsum : ∑ i, q i = 1)
    (q' : Fin (n * s) → ℝ) (hq' : ∀ (i : Fin n) (k : Fin s), q' (finProdFinEquiv (i, k)) = q i / (s : ℝ))
    (f : Fin n → ℝ) (f' : Fin (n * s) → ℝ) (hf' : ∀ (i : Fin n) (k : Fin s), f' (finProdFinEquiv (i, k)) = f i) :
    (∑ j, f' j) * Real.exp (-(∑ j, Real.negMulLog (q' j)))
      = (∑ i, f i) * Real.exp (-(∑ i, Real.negMulLog (q i))) := by
  have hs0 : (0 : ℝ) < s := by exact_mod_cast hs
  have hsne : (s : ℝ) ≠ 0 := ne_of_gt hs0
  have hlog : Real.exp (-Real.log s) = (s : ℝ)⁻¹ := by rw [Real.exp_neg, Real.exp_log hs0]
  rw [entropy_block_repeat hs q hsum q' hq', integral_length_block_repeat f f' hf',
    neg_add, Real.exp_add, hlog]
  field_simp

end Entroptics
