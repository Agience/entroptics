import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.GroupTheory.Perm.Fin
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The coupling null (PAPER Sec 5.5, Thm 5.6)

The coupling of two frames on a shared basis is the alignment `S = ∑ₜ ⟨a̅ₜ, b̅ₜ⟩`, standardised by
its exact null over a uniform re-pairing of the two sides:

    Var_π[Re S] = tr(C_A C_B) / (T - 1),   C_A = A̅ᴴA̅,  C_B = B̅ᴴB̅.

The derivation has two halves.  The COMBINATORIAL half counts permutations by their action on one
and two indices; its two-point count `(T-2)!` is already machine-checked as `Coherence.fibre_card`,
and the same counting carries the coupling's moments.  The ALGEBRAIC half turns what that counting
produces into the closed form, and is what this module checks:

  `variance_coefficient`   the same-index and distinct-index terms combine to a single `1/(T-1)`
  `sum_hadamard_eq_trace`  the entrywise double sum they multiply is `tr(C_A C_Bᵀ)`
  `trace_mul_comm_symm`    and for the symmetric Gram matrices at hand, `tr(C_A C_B)`

The empirical check on the whole theorem is `research/validation/exp7_coupling.py`, which matches
the closed form against 20 000 brute-force re-pairings at three shapes.
-/

open scoped BigOperators
open Matrix

namespace Entroptics

/-- **Summing a function of `σ i` over every permutation.**
Each image is hit exactly `n!` times -- the permutations of the remaining places -- so the sum is
`n! * ∑ₐ f a`, whichever index `i` is.  Proved structurally rather than by counting:
`Equiv.Perm.decomposeFin` splits a permutation of `Fin (n+1)` into its value at `0` and a
permutation of the rest, and right multiplication by a transposition carries any index to `0`. -/
private lemma sum_perm_apply {n : ℕ} (f : Fin (n + 1) → ℝ) (i : Fin (n + 1)) :
    (∑ σ : Equiv.Perm (Fin (n + 1)), f (σ i)) = (Nat.factorial n : ℝ) * ∑ a, f a := by
  classical
  have hshift : (∑ σ : Equiv.Perm (Fin (n + 1)), f (σ i))
      = ∑ σ : Equiv.Perm (Fin (n + 1)), f (σ 0) := by
    rw [← Equiv.sum_comp (Equiv.mulRight (Equiv.swap (0 : Fin (n + 1)) i))
          (fun σ : Equiv.Perm (Fin (n + 1)) => f (σ 0))]
    exact Finset.sum_congr rfl fun σ _ => by simp [Equiv.swap_apply_left]
  rw [hshift, ← Equiv.sum_comp (Equiv.Perm.decomposeFin (n := n)).symm
        (fun σ : Equiv.Perm (Fin (n + 1)) => f (σ 0)), Fintype.sum_prod_type]
  simp [Finset.sum_const, Fintype.card_perm, nsmul_eq_mul, Finset.mul_sum, mul_comm]

/-- **Theorem 5.6 (first moment).**
Summed over every re-pairing of the two sides, the alignment of a CENTRED sending side vanishes:
each ordered index contributes `n!` copies of that column's total, and those totals are zero.
So `𝔼_π[S] = 0`, and the z-score of Definition 5.7 is centred. -/
theorem coupling_null_mean {n D : ℕ} (A B : Fin (n + 1) → Fin D → ℝ)
    (hA : ∀ d, (∑ t, A t d) = 0) :
    (∑ σ : Equiv.Perm (Fin (n + 1)), ∑ t, ∑ d, A (σ t) d * B t d) = 0 := by
  classical
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero fun t _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero fun d _ => ?_
  have hfac : (∑ σ : Equiv.Perm (Fin (n + 1)), A (σ t) d * B t d)
      = (∑ σ : Equiv.Perm (Fin (n + 1)), A (σ t) d) * B t d := by
    rw [← Finset.sum_mul]
  rw [hfac, sum_perm_apply (fun a => A a d) t, hA d, mul_zero, zero_mul]

/-- **Theorem 5.6 (second moment, same-index term).**
At a single ordered index the permutation sum of a product of two coordinates is `n!` copies of
that Gram entry: `∑_σ A(σ i)_d A(σ i)_e = n! (C_A)_{de}` -- the `1/T` term of the variance, and
the one-point lemma applied to the product. -/
theorem sum_perm_gram {n D : ℕ} (A : Fin (n + 1) → Fin D → ℝ) (i : Fin (n + 1)) (d e : Fin D) :
    (∑ σ : Equiv.Perm (Fin (n + 1)), A (σ i) d * A (σ i) e)
      = (Nat.factorial n : ℝ) * ∑ a, A a d * A a e :=
  sum_perm_apply (fun a => A a d * A a e) i

/-- A permutation carrying `0 ↦ t` and `1 ↦ s`, for any two distinct indices.  Built from two
transpositions, so it needs no group theory beyond `Equiv.swap`. -/
private lemma exists_carrier {n : ℕ} {t s : Fin (n + 2)} (hts : t ≠ s) :
    ∃ τ : Equiv.Perm (Fin (n + 2)), τ 0 = t ∧ τ 1 = s := by
  classical
  have hu : Equiv.swap (0 : Fin (n + 2)) t s ≠ 0 := by
    intro hc
    apply hts
    have h := congrArg (Equiv.swap (0 : Fin (n + 2)) t) hc
    simpa [Equiv.swap_apply_self, Equiv.swap_apply_left] using h.symm
  have h01 : (0 : Fin (n + 2)) ≠ 1 := by simp
  refine ⟨Equiv.swap 0 t * Equiv.swap 1 (Equiv.swap 0 t s), ?_, ?_⟩
  · simp [Equiv.Perm.mul_apply, Equiv.swap_apply_of_ne_of_ne h01 (Ne.symm hu),
          Equiv.swap_apply_left]
  · simp [Equiv.Perm.mul_apply, Equiv.swap_apply_self]

/-- **Theorem 5.6 (second moment, distinct-index term).**
At two DISTINCT ordered indices the permutation sum does not depend on WHICH pair:

    ∑_σ f(σ t) g(σ s) = ∑_σ f(σ 0) g(σ 1)   for any `t ≠ s`.

Right multiplication by a carrier permutation is a bijection of the group, so only the
distinctness of the indices matters.  This is the term the `1/(T(T-1))` correction multiplies. -/
theorem sum_perm_pair_eq {n : ℕ} (f g : Fin (n + 2) → ℝ) {t s : Fin (n + 2)} (hts : t ≠ s) :
    (∑ σ : Equiv.Perm (Fin (n + 2)), f (σ t) * g (σ s))
      = ∑ σ : Equiv.Perm (Fin (n + 2)), f (σ 0) * g (σ 1) := by
  classical
  obtain ⟨τ, hτ0, hτ1⟩ := exists_carrier hts
  rw [← Equiv.sum_comp (Equiv.mulRight τ)
        (fun σ : Equiv.Perm (Fin (n + 2)) => f (σ 0) * g (σ 1))]
  exact Finset.sum_congr rfl fun σ _ => by simp [Equiv.Perm.mul_apply, hτ0, hτ1]

/-- **Theorem 5.6 (second moment, the distinct-index VALUE).**

    (T)(T-1) * ∑_σ f(σ 0) g(σ 1) = T! * ( (∑ₐ f a)(∑_b g b) - ∑ₐ f a g a ),   T = n+2.

The route is the bijection itself.  A permutation reindexes each factor, so
`∑_σ (∑ₜ f(σ t))(∑ₛ g(σ s))` is one constant summed over the whole group.  Expanding the SAME
quantity by ordered pairs splits it into `T` diagonal terms (each `sum_perm_apply` on the
product) and `T(T-1)` off-diagonal ones, all equal by `sum_perm_pair_eq`.  Equating the two
expansions and solving gives the off-diagonal value -- the `(T-2)!` count the second moment
needs, with no counting argument anywhere. -/
theorem sum_perm_pair_value {n : ℕ} (f g : Fin (n + 2) → ℝ) :
    (((n : ℝ) + 2) * ((n : ℝ) + 1)) * (∑ σ : Equiv.Perm (Fin (n + 2)), f (σ 0) * g (σ 1))
      = (Nat.factorial (n + 2) : ℝ)
          * ((∑ a, f a) * (∑ b, g b) - ∑ a, f a * g a) := by
  classical
  set X := ∑ σ : Equiv.Perm (Fin (n + 2)), f (σ 0) * g (σ 1) with hXdef
  set P := ∑ a, f a * g a with hPdef
  set Q := (∑ a, f a) * (∑ b, g b) with hQdef
  -- one constant, summed over the whole group
  have hgroup : (∑ _σ : Equiv.Perm (Fin (n + 2)), Q) = (Nat.factorial (n + 2) : ℝ) * Q := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_perm, Fintype.card_fin, nsmul_eq_mul]
  have hre : ∀ σ : Equiv.Perm (Fin (n + 2)), (∑ t, f (σ t)) * (∑ s, g (σ s)) = Q := by
    intro σ; rw [hQdef, Equiv.sum_comp σ f, Equiv.sum_comp σ g]
  -- the same quantity, expanded by ordered pairs
  have hexp : (∑ σ : Equiv.Perm (Fin (n + 2)), (∑ t, f (σ t)) * (∑ s, g (σ s)))
      = ∑ t, ∑ s, ∑ σ : Equiv.Perm (Fin (n + 2)), f (σ t) * g (σ s) := by
    rw [Finset.sum_congr rfl fun σ _ => Finset.sum_mul_sum _ _ _ _, Finset.sum_comm]
    exact Finset.sum_congr rfl fun t _ => Finset.sum_comm
  -- each row: one diagonal term and (T-1) equal off-diagonal ones
  have hrow : ∀ t : Fin (n + 2),
      (∑ s, ∑ σ : Equiv.Perm (Fin (n + 2)), f (σ t) * g (σ s))
        = (Nat.factorial (n + 1) : ℝ) * P + ((n : ℝ) + 1) * X := by
    intro t
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ t)]
    congr 1
    · exact sum_perm_apply (fun a => f a * g a) t
    · rw [Finset.sum_congr rfl fun s hs =>
            sum_perm_pair_eq f g (Ne.symm (Finset.ne_of_mem_erase hs)),
          Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ t), Finset.card_univ,
          Fintype.card_fin, nsmul_eq_mul]
      push_cast
      ring
  have hsum : (Nat.factorial (n + 2) : ℝ) * Q
      = ((n : ℝ) + 2) * ((Nat.factorial (n + 1) : ℝ) * P + ((n : ℝ) + 1) * X) := by
    rw [← hgroup, ← Finset.sum_congr rfl fun σ (_ : σ ∈ Finset.univ) => hre σ, hexp,
        Finset.sum_congr rfl fun t _ => hrow t, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul]
    push_cast
    ring
  have hfac : (Nat.factorial (n + 2) : ℝ) = ((n : ℝ) + 2) * (Nat.factorial (n + 1) : ℝ) := by
    rw [Nat.factorial_succ]; push_cast; ring
  rw [hfac] at hsum ⊢
  nlinarith [hsum]

/-- **The variance coefficient (Thm 5.6).**
The same-index expectation contributes `1/T` and the distinct-index correction `1/(T(T-1))`;
they combine to a single factor `1/(T-1)`.  This is the step that leaves `tr(C_A C_B)/(T-1)`. -/
theorem variance_coefficient {T : ℝ} (h : 1 < T) :
    1 / T + 1 / (T * (T - 1)) = 1 / (T - 1) := by
  have h0 : T ≠ 0 := by linarith
  have h1 : T - 1 ≠ 0 := by linarith
  field_simp
  ring

/-- **The double sum is a trace (Thm 5.6).**
The second moment produces the entrywise sum of the two Gram matrices; it is the trace of their
product with one transposed. -/
theorem sum_hadamard_eq_trace {D : ℕ} (CA CB : Matrix (Fin D) (Fin D) ℝ) :
    (∑ d, ∑ e, CA d e * CB d e) = Matrix.trace (CA * CBᵀ) := by
  classical
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]

/-- **And the Gram matrices are symmetric**, so the transpose falls away and the coefficient
multiplies `tr(C_A C_B)` -- the form the read reports. -/
theorem sum_hadamard_eq_trace_of_symm {D : ℕ} (CA CB : Matrix (Fin D) (Fin D) ℝ)
    (hB : CBᵀ = CB) :
    (∑ d, ∑ e, CA d e * CB d e) = Matrix.trace (CA * CB) := by
  rw [sum_hadamard_eq_trace, hB]

/-- A Gram matrix `Xᵀ X` is symmetric, which is the hypothesis the previous theorem takes. -/
theorem gram_symm {T D : ℕ} (X : Matrix (Fin T) (Fin D) ℝ) : (Xᵀ * X)ᵀ = Xᵀ * X := by
  simp [Matrix.transpose_mul]

/-- **The standardisation is well posed.**  With at least two ordered-axis samples and a
non-degenerate pair, the variance the z-score divides by is positive. -/
theorem variance_pos {T : ℝ} (h : 1 < T) {q : ℝ} (hq : 0 < q) : 0 < q / (T - 1) := by
  have : (0:ℝ) < T - 1 := by linarith
  exact div_pos hq this

end Entroptics
