import Mathlib

/-!
# Concentration: axial versus directional (PAPER Sec 7)

`focus` (axial, a spectral function of the second moment `∑ xᵢ xᵢᴴ`) and the `resultant` (directional,
`‖∑ xᵢ‖`) measure distinct quantities: the second moment is invariant under per-row phase, the
resultant is not. `antipodal_separates` exhibits a cloud where they diverge.
-/

open scoped BigOperators
open Matrix

namespace Entroptics

/-- **Proposition 7.2, real invariance step.**
The second-moment matrix `∑ xᵢ xᵢᵀ`, and therefore `focus` (a spectral function of it), is invariant
under a per-row sign flip `xᵢ ↦ -xᵢ`. -/
theorem second_moment_sign_invariant {F n : ℕ}
    (x : Fin n → Matrix (Fin F) (Fin 1) ℝ) :
    (∑ i, (-x i) * (-x i)ᵀ) = ∑ i, x i * (x i)ᵀ := by
  refine Finset.sum_congr rfl (fun i _ => ?_)
  simp [Matrix.transpose_neg, Matrix.neg_mul, Matrix.mul_neg, neg_neg]

/-- **Proposition 7.2, directional step.**
The resultant `∑ xᵢ` is negated by the same sign flip. With `second_moment_sign_invariant`, `focus`
is blind to per-row sign (axial) while the resultant is not (directional). -/
theorem resultant_sign_flip {F n : ℕ}
    (x : Fin n → Matrix (Fin F) (Fin 1) ℝ) :
    (∑ i, (-x i)) = - ∑ i, x i := by
  simp

/-- **Proposition 7.2 (axial ≠ directional), complex phase-invariance.**
The second-moment matrix `Σ = ∑ᵢ xᵢ xᵢᴴ` is invariant under any per-row unit-modulus phase
`xᵢ ↦ cᵢ xᵢ` with `‖cᵢ‖ = 1`, because `(cᵢ • xᵢ)(cᵢ • xᵢ)ᴴ = ‖cᵢ‖² • (xᵢ xᵢᴴ) = xᵢ xᵢᴴ`. Hence
`focus`, a spectral function of `Σ`, is phase-blind (axial). -/
theorem second_moment_phase_invariant {F n : ℕ}
    (x : Fin n → Matrix (Fin F) (Fin 1) ℂ) (c : Fin n → ℂ) (hc : ∀ i, ‖c i‖ = 1) :
    (∑ i, (c i • x i) * (c i • x i)ᴴ) = ∑ i, x i * (x i)ᴴ := by
  refine Finset.sum_congr rfl (fun i _ => ?_)
  have hcc : c i * star (c i) = 1 := by
    rw [show star (c i) = (starRingEnd ℂ) (c i) from rfl, RCLike.mul_conj, hc i]; norm_num
  rw [Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul, smul_smul, hcc, one_smul]

/-- **Proposition 7.2 (axial ≠ directional), the antipodal separation.**
The antipodal cloud `x = ![u, u]` (`u ≠ 0`) with phases `c = ![1, -1]`: the phased resultant
collapses, `∑ᵢ cᵢ • xᵢ = 0`, while the resultant `∑ᵢ xᵢ = 2u ≠ 0`, and the second moment
`∑ᵢ (cᵢ • xᵢ)(cᵢ • xᵢ)ᴴ` is unchanged. So `focus` (axial, phase-blind) and the resultant
(directional) measure distinct quantities. A uniform real sign-flip does not exhibit this, since it
leaves the resultant's norm unchanged. -/
theorem antipodal_separates {F : ℕ}
    (u : Matrix (Fin F) (Fin 1) ℂ) (hu : u ≠ 0) :
    (∑ i : Fin 2, (![(1 : ℂ), -1] i) • (![u, u] i)) = 0 ∧
    (∑ i : Fin 2, (![u, u] i)) ≠ 0 ∧
    (∑ i : Fin 2, (![(1 : ℂ), -1] i • ![u, u] i) * (![(1 : ℂ), -1] i • ![u, u] i)ᴴ)
      = ∑ i : Fin 2, (![u, u] i) * (![u, u] i)ᴴ := by
  refine ⟨?_, ?_, ?_⟩
  · rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, one_smul, neg_smul]
    rw [add_neg_cancel]
  · rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [← two_smul ℂ u]
    exact smul_ne_zero two_ne_zero hu
  · exact second_moment_phase_invariant ![u, u] ![1, -1] (by
      intro i; fin_cases i <;> simp)

end Entroptics
