import Mathlib

/-!
# The dynamical operator: exact recovery and splicing (PAPER Sec 9)

The streaming accumulators `P_xx = ∑ xₜ xₜᴴ`, `P_yx = ∑ x_{t+1} xₜᴴ` add over a disjoint partition
(`accumulator_additive`), and under noise-free dynamics the read propagator `P_yx P_xx⁻¹` equals `A`,
so the read rates `αₖ = -log|μₖ|`, `βₖ = arg μₖ` are exact (`propagator_*`).
-/

open scoped BigOperators
open Matrix

namespace Entroptics

/-- **Theorem 9.3 (additive splicing).**
The streaming accumulators are `Finset` sums over transitions, so over a disjoint partition of the
transition set they add. This is the exactness of `merge` and of `state` → `from_state`. -/
theorem accumulator_additive {ι M : Type*} [DecidableEq ι] [AddCommMonoid M]
    (A B : Finset ι) (hAB : Disjoint A B) (f : ι → M) :
    ∑ i ∈ A ∪ B, f i = (∑ i ∈ A, f i) + ∑ i ∈ B, f i :=
  Finset.sum_union hAB

/-- **Theorem 9.2 (exact recovery, core identity).**
Under noise-free dynamics `x_{t+1} = A xₜ`, the cross accumulator factors through `A`:
`∑ (A xₜ) xₜᵀ = A · (∑ xₜ xₜᵀ)`, i.e. `P_yx = A · P_xx`. -/
theorem propagator_core {F n : ℕ} (A : Matrix (Fin F) (Fin F) ℝ)
    (x : Fin n → Matrix (Fin F) (Fin 1) ℝ) :
    (∑ t, (A * x t) * (x t)ᵀ) = A * ∑ t, x t * (x t)ᵀ := by
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun t _ => Matrix.mul_assoc A (x t) ((x t)ᵀ))

/-- **Theorem 9.2 (exact recovery, eigenvalue step).**
When `P_xx = ∑ xₜ xₜᵀ` is invertible, the read propagator is `A` exactly: `P_yx · P_xx⁻¹ = A`. From
`propagator_core`, `P_yx · P_xx⁻¹ = (A · P_xx) · P_xx⁻¹ = A · (P_xx · P_xx⁻¹) = A`. -/
theorem propagator_recovers {F n : ℕ} (A : Matrix (Fin F) (Fin F) ℝ)
    (x : Fin n → Matrix (Fin F) (Fin 1) ℝ)
    (hinv : IsUnit (∑ t, x t * (x t)ᵀ).det) :
    (∑ t, (A * x t) * (x t)ᵀ) * (∑ t, x t * (x t)ᵀ)⁻¹ = A := by
  rw [propagator_core A x, Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hinv, Matrix.mul_one]

/-- **Theorem 9.2 (exact recovery, spectrum).**
The recovered operator `P_yx · P_xx⁻¹` equals `A` on the full-rank state space, so it shares `A`'s
characteristic polynomial, hence its eigenvalues `μₖ` are those of `A`, and the read rates
`αₖ = -log|μₖ|`, `βₖ = arg μₖ` are exact. Certified for invertible `P_xx`; the rank-deficient
Moore-Penrose form needs a pseudoinverse theory this Mathlib does not provide. -/
theorem propagator_recovers_charpoly {F n : ℕ} (A : Matrix (Fin F) (Fin F) ℝ)
    (x : Fin n → Matrix (Fin F) (Fin 1) ℝ)
    (hinv : IsUnit (∑ t, x t * (x t)ᵀ).det) :
    ((∑ t, (A * x t) * (x t)ᵀ) * (∑ t, x t * (x t)ᵀ)⁻¹).charpoly = A.charpoly := by
  rw [propagator_recovers A x hinv]

end Entroptics
