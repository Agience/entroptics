import Mathlib

/-!
# The ordered-axis coherence null (PAPER Sec 5)

The coherence z-score uses the mean of the row-permutation null. `coherence_null_mean` computes
that mean via the two-point stabilizer count `(N-2)!` (helpers `stab_card`, `exists_mover`,
`fibre_card`).
-/

open scoped BigOperators
open MulAction

namespace Entroptics

/-- The two-point stabilizer of the symmetric group has `(N-2)!` elements: the permutations of
`Fin N` fixing both `i` and `j` (`i ≠ j`) permute the remaining `N-2` points. Via
`index_of_fixingSubgroup_mul` for the 2-transitive `Perm` action; the load-bearing count for
Theorem 5.2. -/
private lemma stab_card {N : ℕ} (i j : Fin N) (hij : i ≠ j) :
    Nat.card (fixingSubgroup (Equiv.Perm (Fin N)) ({i, j} : Set (Fin N)))
      = Nat.factorial (N - 2) := by
  have hNc : Nat.card (Fin N) = N := by simp [Nat.card_eq_fintype_card]
  have hpair : ({i, j} : Set (Fin N)).ncard = 2 := Set.ncard_pair hij
  have hidx := (Equiv.Perm.isMultiplyPretransitive (Fin N) 2).index_of_fixingSubgroup_mul hpair
  rw [hNc] at hidx
  have hlag := Subgroup.index_mul_card
      (fixingSubgroup (Equiv.Perm (Fin N)) ({i, j} : Set (Fin N)))
  rw [Nat.card_perm, hNc] at hlag
  have hpos : 0 < (fixingSubgroup (Equiv.Perm (Fin N)) ({i, j} : Set (Fin N))).index :=
    Nat.pos_of_ne_zero Subgroup.index_ne_zero_of_finite
  apply Nat.eq_of_mul_eq_mul_left hpos
  rw [hlag]; exact hidx.symm

/-- A permutation sending `i ↦ a` and `j ↦ b` exists (`i ≠ j`, `a ≠ b`), by 2-transitivity of the
symmetric group. -/
private lemma exists_mover {N : ℕ} (i j a b : Fin N) (hij : i ≠ j) (hab : a ≠ b) :
    ∃ g : Equiv.Perm (Fin N), g i = a ∧ g j = b := by
  have hx : Function.Injective ![i, j] := by
    intro p q h; fin_cases p <;> fin_cases q <;> simp_all
  have hy : Function.Injective ![a, b] := by
    intro p q h; fin_cases p <;> fin_cases q <;> simp_all
  obtain ⟨g, hg⟩ := Equiv.Perm.exists_smul_eq_embedding
    (⟨![i, j], hx⟩ : Fin 2 ↪ Fin N) (⟨![a, b], hy⟩ : Fin 2 ↪ Fin N)
  refine ⟨g, ?_, ?_⟩
  · have h0 := DFunLike.congr_fun hg 0
    simpa [Function.Embedding.smul_apply, Equiv.Perm.smul_def,
      Matrix.cons_val_zero] using h0
  · have h1 := DFunLike.congr_fun hg 1
    simpa [Function.Embedding.smul_apply, Equiv.Perm.smul_def,
      Matrix.cons_val_one, Matrix.head_cons] using h1

/-- The two-point fibre count: for `i ≠ j` and `a ≠ b`, exactly `(N-2)!` permutations send `i ↦ a`
and `j ↦ b` (a left coset of the stabilizer, `stab_card`). -/
private lemma fibre_card {N : ℕ} (i j : Fin N) (hij : i ≠ j) (a b : Fin N) (hab : a ≠ b) :
    Fintype.card {σ : Equiv.Perm (Fin N) // σ i = a ∧ σ j = b} = Nat.factorial (N - 2) := by
  have hfix : Fintype.card {σ : Equiv.Perm (Fin N) // σ i = i ∧ σ j = j}
      = Nat.factorial (N - 2) := by
    rw [← Nat.card_eq_fintype_card, ← stab_card i j hij]
    apply Nat.card_congr
    refine Equiv.subtypeEquivRight (fun σ => ?_)
    rw [mem_fixingSubgroup_iff]
    constructor
    · rintro ⟨h1, h2⟩ y hy
      simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hy
      rcases hy with rfl | rfl
      · exact h1
      · exact h2
    · intro h
      exact ⟨h i (by simp), h j (by simp)⟩
  obtain ⟨g, hgi, hgj⟩ := exists_mover i j a b hij hab
  rw [← hfix]
  apply Fintype.card_congr
  refine Equiv.subtypeEquiv (Equiv.mulLeft g⁻¹) (fun σ => ?_)
  simp only [Equiv.coe_mulLeft, Equiv.Perm.mul_apply]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨by rw [h1, ← hgi]; simp, by rw [h2, ← hgj]; simp⟩
  · rintro ⟨h1, h2⟩
    refine ⟨?_, ?_⟩
    · have h := congrArg (⇑g) h1; simpa [hgi] using h
    · have h := congrArg (⇑g) h2; simpa [hgj] using h

/-- **Theorem 5.2 (permutation-null mean).**
The average of a single compared pair over all row permutations equals the mean of `R` over ordered
off-diagonal pairs. Summing this over the `N-ℓ` superdiagonal terms gives `𝔼_π[A] = μ`. The fibre
`{σ // σ i = a ∧ σ j = b}` has size `(N-2)!` for `a ≠ b` (`fibre_card`) and is empty on the diagonal;
`sum_fiberwise` over `σ ↦ (σ i, σ j)` gives `∑_σ = (N-2)! ∑_{a≠b} R a b`. -/
theorem coherence_null_mean {N : ℕ} (R : Fin N → Fin N → ℝ) (i j : Fin N) (hij : i ≠ j) :
    (∑ σ : Equiv.Perm (Fin N), R (σ i) (σ j))
      = (Nat.factorial (N - 2) : ℝ) * ∑ a, ∑ b, if a = b then 0 else R a b := by
  classical
  have hcard : ∀ a b : Fin N,
      ((Finset.univ.filter (fun σ : Equiv.Perm (Fin N) => (σ i, σ j) = (a, b))).card : ℝ)
        = if a = b then 0 else (Nat.factorial (N - 2) : ℝ) := by
    intro a b
    by_cases hab : a = b
    · rw [if_pos hab]
      have hemp : (Finset.univ.filter
          (fun σ : Equiv.Perm (Fin N) => (σ i, σ j) = (a, b))) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        rintro σ -
        rw [Prod.mk.injEq]
        rintro ⟨h1, h2⟩
        exact hij (σ.injective (h1.trans (hab ▸ h2).symm))
      rw [hemp, Finset.card_empty, Nat.cast_zero]
    · rw [if_neg hab]
      have hset : (Finset.univ.filter
          (fun σ : Equiv.Perm (Fin N) => (σ i, σ j) = (a, b))).card
            = Fintype.card {σ : Equiv.Perm (Fin N) // σ i = a ∧ σ j = b} := by
        rw [Fintype.card_subtype]
        congr 1
        ext σ
        simp [Prod.ext_iff]
      rw [hset, fibre_card i j hij a b hab]
  trans ∑ p : Fin N × Fin N,
      ∑ σ ∈ Finset.univ.filter (fun σ : Equiv.Perm (Fin N) => (σ i, σ j) = p), R (σ i) (σ j)
  · exact (Finset.sum_fiberwise_of_maps_to (fun σ _ => Finset.mem_univ _) _).symm
  rw [Fintype.sum_prod_type, Finset.mul_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun b _ => ?_
  rw [Finset.sum_congr rfl (g := fun _ => R a b) (fun σ hσ => by
        rw [Finset.mem_filter, Prod.mk.injEq] at hσ; rw [hσ.2.1, hσ.2.2]),
      Finset.sum_const, nsmul_eq_mul, hcard a b]
  by_cases hab : a = b
  · rw [if_pos hab, if_pos hab, zero_mul, mul_zero]
  · rw [if_neg hab, if_neg hab]

end Entroptics
