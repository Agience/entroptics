import Entroptics.Aperture
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.Normed.Ring.Finite
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.Field.GeomSum
import Mathlib.Data.ZMod.ValMinAbs

/-!
# The plunge region is logarithmic (PAPER Sec 3, the aperture; Sec 10, the degrees of freedom)

For a window of `T` consecutive sites and a band of `W` consecutive modes on `ℤ_N`, the plunge mass
`δ = tr A - tr A² = ∑ λ (1 - λ)` of the concentration operator `A = P_S B_Ω P_S` is at most
`(9 + log (min T W)) / 2`. Hence the number of eigenvalues above `1/2` is `T W / N` to within
`9 + log (min T W)`: a finite aperture holds a finite count, with a logarithmic correction.

The route starts from the leakage form `concentration_plunge_eq_leakage`.

* `leakage_eq_shift_sum`: the leakage is `∑_m c(m) κ(m)`, with `κ(m) = ‖B_Ω m 0‖²` the kernel at
  lag `m` and `c(m) = #{j ∈ S : j + m ∉ S}`.
* `window_shift_card_le`: for a window, `c(m) ≤ min (T, |r|)` with `r = valMinAbs m`.
* `norm_bandLimit_sq_le`: the Dirichlet bound `κ(m) ≤ min (W²/N², 1/(4 r²))` for `m ≠ 0`, from the
  geometric sum and Jordan's inequality `sin x ≥ 2x/π` (`norm_stdAddChar_sub_one_ge`).
* `sum_valMinAbs_le`: each `|r| = d` is taken by at most two lags.
* `profile_sum_le_log_T`, `profile_sum_le_log_W`: the harmonic sums, `≤ (2 + log T)/4` and
  `≤ (9 + log W)/4`.
* `window_plunge_le_log`, `window_count_half`: the assembled bounds.

The constants are computed, not chosen. The `2` of `(2 + log T)/4` is `1 + 1`: the `1` of
`H_T ≤ 1 + log T` and the `T · (1/T)` of the `1/d²` tail. The `9` of `(9 + log W)/4` is
`4 · 9/4 = 4 · (3/2)²`: the head sum of `profile_sum_le_log_W` is at most `(K W / N)²`, and
`K W / N ≤ (N/(2W) + 1) W / N = 1/2 + W/N ≤ 3/2` since `W ≤ N`. The one choice is the crossover
`K = ⌊N/(2W)⌋ + 1` of that proof.
-/

open scoped BigOperators ComplexConjugate
open Matrix Finset

namespace Entroptics

/-! ### Harmonic sums -/

/-- **The harmonic tail.** `∑_{a < d ≤ b} 1/d ≤ log b - log a` for `1 ≤ a ≤ b`, each term being at
most `log d - log (d - 1)`. -/
theorem sum_Ioc_inv_le_log {a b : ℕ} (ha : 1 ≤ a) (hab : a ≤ b) :
    ∑ d ∈ Ioc a b, (1 : ℝ) / d ≤ Real.log b - Real.log a := by
  induction b, hab using Nat.le_induction with
  | base => simp
  | succ b hab ih =>
    rw [sum_Ioc_succ_top hab]
    have hb : (0 : ℝ) < b := by exact_mod_cast lt_of_lt_of_le ha hab
    have key : (1 : ℝ) / ((b : ℝ) + 1) ≤ Real.log ((b : ℝ) + 1) - Real.log b := by
      have h := Real.log_le_sub_one_of_pos (x := (b : ℝ) / ((b : ℝ) + 1)) (by positivity)
      rw [Real.log_div hb.ne' (by positivity)] at h
      have h' : (b : ℝ) / ((b : ℝ) + 1) - 1 = -(1 / ((b : ℝ) + 1)) := by
        field_simp
        ring
      linarith
    push_cast
    linarith

/-- **The harmonic sum.** `∑_{0 < d ≤ T} 1/d ≤ 1 + log T` for `T ≥ 1`. -/
theorem sum_Ioc_zero_inv_le {T : ℕ} (hT : 1 ≤ T) :
    ∑ d ∈ Ioc 0 T, (1 : ℝ) / d ≤ 1 + Real.log T := by
  rw [← sum_Ioc_consecutive _ (Nat.zero_le 1) hT]
  have h01 : Ioc 0 1 = ({1} : Finset ℕ) := by
    ext x
    simp only [mem_Ioc, mem_singleton]
    omega
  have h := sum_Ioc_inv_le_log (le_refl 1) hT
  rw [h01, sum_singleton, Nat.cast_one, div_one]
  simp only [Nat.cast_one, Real.log_one, sub_zero] at h
  linarith

/-- **The telescoping tail of `1/d²`.** `∑_{a < d ≤ b} 1/d² ≤ 1/a - 1/b` for `1 ≤ a ≤ b`. -/
theorem sum_Ioc_inv_sq_le {a b : ℕ} (ha : 1 ≤ a) (hab : a ≤ b) :
    ∑ d ∈ Ioc a b, (1 : ℝ) / (d : ℝ) ^ 2 ≤ 1 / a - 1 / b := by
  induction b, hab using Nat.le_induction with
  | base => simp
  | succ b hab ih =>
    rw [sum_Ioc_succ_top hab]
    have hb : (0 : ℝ) < b := by exact_mod_cast lt_of_lt_of_le ha hab
    have key : (1 : ℝ) / ((b : ℝ) + 1) ^ 2 ≤ 1 / b - 1 / ((b : ℝ) + 1) := by
      have hid : (1 : ℝ) / b - 1 / ((b : ℝ) + 1) = 1 / ((b : ℝ) * ((b : ℝ) + 1)) := by
        field_simp
        ring
      rw [hid]
      exact one_div_le_one_div_of_le (by positivity) (by nlinarith)
    push_cast
    linarith

/-! ### The radial profile and its two harmonic bounds -/

/-- The radial profile of the leakage at `|r| = d`: at most `min (T, d)` pairs, each carrying at
most `min (W²/N², 1/(4 d²))`. -/
noncomputable def plungeProfile (T W N d : ℕ) : ℝ :=
  min (T : ℝ) d * min ((W : ℝ) ^ 2 / (N : ℝ) ^ 2) (1 / (4 * (d : ℝ) ^ 2))

/-- The radial profile is nonnegative. -/
theorem plungeProfile_nonneg (T W N d : ℕ) : 0 ≤ plungeProfile T W N d :=
  mul_nonneg (le_min (Nat.cast_nonneg _) (Nat.cast_nonneg _)) (le_min (by positivity) (by positivity))

/-- **The profile sum, in the aperture `T`.** `∑_{0 < d ≤ D} profile ≤ (2 + log T)/4` for `T ≥ 1`
and every `D`: `1/(4d)` up to `T`, then `T/(4d²)`. COMPUTED: the `2` is `1 + 1`, the `1` of
`H_T ≤ 1 + log T` plus `T · (1/T)` from the `1/d²` tail (`sum_Ioc_inv_sq_le`). -/
theorem profile_sum_le_log_T {T : ℕ} (W N : ℕ) (hT : 1 ≤ T) (D : ℕ) :
    ∑ d ∈ Ioc 0 D, plungeProfile T W N d ≤ (2 + Real.log T) / 4 := by
  have hTpos : (0 : ℝ) < T := by exact_mod_cast hT
  set g : ℕ → ℝ := fun d => min (T : ℝ) d / (4 * (d : ℝ) ^ 2) with hg
  have hgnn : ∀ d, 0 ≤ g d := fun d =>
    div_nonneg (le_min (Nat.cast_nonneg _) (Nat.cast_nonneg _)) (by positivity)
  have hle : ∀ d ∈ Ioc 0 D, plungeProfile T W N d ≤ g d := fun d _ => by
    calc plungeProfile T W N d ≤ min (T : ℝ) d * (1 / (4 * (d : ℝ) ^ 2)) :=
          mul_le_mul_of_nonneg_left (min_le_right _ _)
            (le_min (Nat.cast_nonneg _) (Nat.cast_nonneg _))
      _ = g d := by rw [hg, mul_one_div]
  have h1 : ∑ d ∈ Ioc 0 T, g d ≤ (1 + Real.log T) / 4 := by
    have heq : ∑ d ∈ Ioc 0 T, g d = (1 / 4) * ∑ d ∈ Ioc 0 T, (1 : ℝ) / d := by
      rw [mul_sum]
      refine sum_congr rfl fun d hd => ?_
      obtain ⟨hd0, hdT⟩ := mem_Ioc.mp hd
      have hdpos : (0 : ℝ) < d := by exact_mod_cast hd0
      rw [hg]
      simp only
      rw [min_eq_right (by exact_mod_cast hdT)]
      field_simp
    rw [heq]
    have := sum_Ioc_zero_inv_le hT
    linarith
  have h2 : ∑ d ∈ Ioc T (max D T), g d ≤ 1 / 4 := by
    have heq : ∑ d ∈ Ioc T (max D T), g d
        = ((T : ℝ) / 4) * ∑ d ∈ Ioc T (max D T), (1 : ℝ) / (d : ℝ) ^ 2 := by
      rw [mul_sum]
      refine sum_congr rfl fun d hd => ?_
      obtain ⟨hTd, _⟩ := mem_Ioc.mp hd
      have hdpos : (0 : ℝ) < d := by exact_mod_cast lt_of_le_of_lt (Nat.zero_le T) hTd
      rw [hg]
      simp only
      rw [min_eq_left (by exact_mod_cast hTd.le)]
      field_simp
    rw [heq]
    have hs := sum_Ioc_inv_sq_le hT (le_max_right D T)
    have hM : (0 : ℝ) ≤ 1 / ((max D T : ℕ) : ℝ) := by positivity
    calc (T : ℝ) / 4 * ∑ d ∈ Ioc T (max D T), (1 : ℝ) / (d : ℝ) ^ 2
        ≤ (T : ℝ) / 4 * (1 / T) :=
          mul_le_mul_of_nonneg_left (by linarith) (by positivity)
      _ = 1 / 4 := by field_simp
  calc ∑ d ∈ Ioc 0 D, plungeProfile T W N d ≤ ∑ d ∈ Ioc 0 D, g d := sum_le_sum hle
    _ ≤ ∑ d ∈ Ioc 0 (max D T), g d :=
        sum_le_sum_of_subset_of_nonneg (Ioc_subset_Ioc_right (le_max_left _ _))
          fun d _ _ => hgnn d
    _ = ∑ d ∈ Ioc 0 T, g d + ∑ d ∈ Ioc T (max D T), g d :=
        (sum_Ioc_consecutive _ (Nat.zero_le T) (le_max_right _ _)).symm
    _ ≤ (1 + Real.log T) / 4 + 1 / 4 := add_le_add h1 h2
    _ = (2 + Real.log T) / 4 := by ring

/-- **The profile sum, in the band `W`.** `∑_{0 < d ≤ D} profile ≤ (9 + log W)/4` for
`1 ≤ W ≤ N` and `D ≤ N/2`: `d W²/N²` up to `K = ⌊N/(2W)⌋ + 1`, then `1/(4d)` from `K` to `D`.
CHOSEN: `K` is the kernel crossover `d = N/(2W)`, where `W²/N² = 1/(4d²)`, plus `1` so that
`N < K · 2W`. COMPUTED: the `9` is `4 · 9/4 = 4 · (3/2)²`, the head sum being at most `(K W / N)²`
with `K W / N ≤ 1/2 + W/N ≤ 3/2` by `W ≤ N`; the tail gives the `log W`. -/
theorem profile_sum_le_log_W (T : ℕ) {W N : ℕ} (hW : 1 ≤ W) (hWN : W ≤ N) {D : ℕ}
    (hD : D ≤ N / 2) :
    ∑ d ∈ Ioc 0 D, plungeProfile T W N d ≤ (9 + Real.log W) / 4 := by
  have hWpos : (0 : ℝ) < W := by exact_mod_cast hW
  have hNpos : (0 : ℝ) < N := by exact_mod_cast lt_of_lt_of_le hW hWN
  have hWNr : (W : ℝ) ≤ N := by exact_mod_cast hWN
  -- CHOSEN: the kernel crossover `d = N/(2W)` (where `W²/N² = 1/(4d²)`), plus 1 so `N < K · 2W`.
  set K : ℕ := N / (2 * W) + 1 with hK
  have hK1 : 1 ≤ K := by rw [hK]; exact Nat.le_add_left 1 _
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK1
  have hKlo : (N : ℝ) < K * (2 * W) := by
    have h := Nat.lt_div_mul_add (a := N) (show 0 < 2 * W by omega)
    have h' : N < K * (2 * W) := by rw [hK, Nat.add_mul, one_mul]; exact h
    exact_mod_cast h'
  have hKhi : (K : ℝ) ≤ (N : ℝ) / (2 * W) + 1 := by
    rw [hK]
    push_cast
    have := Nat.cast_div_le (α := ℝ) (m := N) (n := 2 * W)
    push_cast at this
    linarith
  have hDr : (D : ℝ) ≤ (N : ℝ) / 2 := by
    have h2 : D * 2 ≤ N := (Nat.le_div_iff_mul_le (by norm_num)).mp hD
    have h2' : (D : ℝ) * 2 ≤ N := by exact_mod_cast h2
    linarith
  set g : ℕ → ℝ := fun d => (d : ℝ) * min ((W : ℝ) ^ 2 / (N : ℝ) ^ 2) (1 / (4 * (d : ℝ) ^ 2))
    with hg
  have hgnn : ∀ d, 0 ≤ g d := fun d =>
    mul_nonneg (Nat.cast_nonneg _) (le_min (by positivity) (by positivity))
  have hle : ∀ d ∈ Ioc 0 D, plungeProfile T W N d ≤ g d := fun d _ =>
    mul_le_mul_of_nonneg_right (min_le_right _ _) (le_min (by positivity) (by positivity))
  set M : ℕ := max D K with hM
  -- COMPUTED: `9/4 = (3/2)²`, the head sum being at most `(K W / N)²` with `K W / N ≤ 3/2`.
  have h1 : ∑ d ∈ Ioc 0 K, g d ≤ 9 / 4 := by
    have hterm : ∀ d ∈ Ioc 0 K, g d ≤ (K : ℝ) * ((W : ℝ) ^ 2 / (N : ℝ) ^ 2) := fun d hd => by
      have hdK : (d : ℝ) ≤ K := by exact_mod_cast (mem_Ioc.mp hd).2
      exact mul_le_mul hdK (min_le_left _ _) (le_min (by positivity) (by positivity))
        (Nat.cast_nonneg _)
    have hsum := sum_le_card_nsmul _ _ _ hterm
    rw [Nat.card_Ioc, Nat.sub_zero, nsmul_eq_mul] at hsum
    -- COMPUTED: `K W / N ≤ (N/(2W) + 1) W / N = 1/2 + W/N ≤ 3/2`, using `W ≤ N`.
    have hKW : (K : ℝ) * W / N ≤ 3 / 2 := by
      rw [div_le_iff₀ hNpos]
      have : (K : ℝ) * W ≤ ((N : ℝ) / (2 * W) + 1) * W :=
        mul_le_mul_of_nonneg_right hKhi hWpos.le
      have h' : ((N : ℝ) / (2 * W) + 1) * W = N / 2 + W := by field_simp
      linarith
    have hsq : ((K : ℝ) * W / N) ^ 2 ≤ (3 / 2) ^ 2 :=
      pow_le_pow_left₀ (by positivity) hKW 2
    have hid : (K : ℝ) * ((K : ℝ) * ((W : ℝ) ^ 2 / (N : ℝ) ^ 2)) = ((K : ℝ) * W / N) ^ 2 := by
      field_simp
    linarith
  have h2 : ∑ d ∈ Ioc K M, g d ≤ Real.log W / 4 := by
    have hterm : ∀ d ∈ Ioc K M, g d ≤ (1 / 4) * ((1 : ℝ) / d) := fun d hd => by
      have hdpos : (0 : ℝ) < d := by
        exact_mod_cast lt_of_le_of_lt (Nat.zero_le K) (mem_Ioc.mp hd).1
      calc g d ≤ (d : ℝ) * (1 / (4 * (d : ℝ) ^ 2)) :=
            mul_le_mul_of_nonneg_left (min_le_right _ _) (Nat.cast_nonneg _)
        _ = (1 / 4) * ((1 : ℝ) / d) := by field_simp
    have hKM : K ≤ M := le_max_right _ _
    have hlog := sum_Ioc_inv_le_log hK1 hKM
    have hMWK : (M : ℝ) ≤ W * K := by
      rw [hM, Nat.cast_max]
      refine max_le ?_ ?_
      · have : (N : ℝ) / 2 < W * K := by
          rw [div_lt_iff₀ (by norm_num : (0 : ℝ) < 2)]
          linarith
        linarith
      · exact le_mul_of_one_le_left hKpos.le (by exact_mod_cast hW)
    have hMpos : (0 : ℝ) < M := by exact_mod_cast lt_of_lt_of_le hK1 hKM
    have hlogM : Real.log M ≤ Real.log W + Real.log K := by
      rw [← Real.log_mul hWpos.ne' hKpos.ne']
      exact Real.log_le_log hMpos hMWK
    calc ∑ d ∈ Ioc K M, g d ≤ ∑ d ∈ Ioc K M, (1 / 4) * ((1 : ℝ) / d) := sum_le_sum hterm
      _ = (1 / 4) * ∑ d ∈ Ioc K M, (1 : ℝ) / d := by rw [mul_sum]
      _ ≤ (1 / 4) * (Real.log M - Real.log K) :=
          mul_le_mul_of_nonneg_left hlog (by norm_num)
      _ ≤ Real.log W / 4 := by linarith
  calc ∑ d ∈ Ioc 0 D, plungeProfile T W N d ≤ ∑ d ∈ Ioc 0 D, g d := sum_le_sum hle
    _ ≤ ∑ d ∈ Ioc 0 M, g d :=
        sum_le_sum_of_subset_of_nonneg (Ioc_subset_Ioc_right (le_max_left _ _))
          fun d _ _ => hgnn d
    _ = ∑ d ∈ Ioc 0 K, g d + ∑ d ∈ Ioc K M, g d :=
        (sum_Ioc_consecutive _ (Nat.zero_le K) (le_max_right _ _)).symm
    _ ≤ 9 / 4 + Real.log W / 4 := add_le_add h1 h2
    _ = (9 + Real.log W) / 4 := by ring

/-- **The profile sum, in the smaller side.** `∑_{0 < d ≤ D} profile ≤ (9 + log (min T W))/4`.
The `T` branch's constant `2` (`profile_sum_le_log_T`) is loosened to the `W` branch's `9`, so that
one constant covers both branches. -/
theorem profile_sum_le_log {T W N : ℕ} (hT : 1 ≤ T) (hW : 1 ≤ W) (hWN : W ≤ N) {D : ℕ}
    (hD : D ≤ N / 2) :
    ∑ d ∈ Ioc 0 D, plungeProfile T W N d ≤ (9 + Real.log (min T W : ℕ)) / 4 := by
  rcases le_total T W with h | h
  · rw [min_eq_left h]
    have := profile_sum_le_log_T W N hT D
    linarith
  · rw [min_eq_right h]
    exact profile_sum_le_log_W T hW hWN hD

/-! ### Windows on `ℤ_N` -/

section Window

variable {N : ℕ} [NeZero N]

/-- The window of `T` consecutive sites (or modes) starting at `b`: `{b, b + 1, …, b + T - 1}`. -/
noncomputable def window (b : ZMod N) (T : ℕ) : Finset (ZMod N) :=
  (range T).image fun t : ℕ => b + (t : ZMod N)

omit [NeZero N] in
/-- A window of length at most `N` has no repeated sites. -/
theorem window_injOn (b : ZMod N) {T : ℕ} (hT : T ≤ N) :
    Set.InjOn (fun t : ℕ => b + (t : ZMod N)) (range T) := by
  intro t ht t' ht' h
  simp only [coe_range, Set.mem_Iio] at ht ht'
  have h' : (t : ZMod N) = t' := add_left_cancel h
  have hv := congrArg ZMod.val h'
  rwa [ZMod.val_natCast, ZMod.val_natCast, Nat.mod_eq_of_lt (by omega),
    Nat.mod_eq_of_lt (by omega)] at hv

omit [NeZero N] in
/-- A window of length `T ≤ N` has `T` sites. -/
theorem window_card (b : ZMod N) {T : ℕ} (hT : T ≤ N) : (window b T).card = T := by
  rw [window, card_image_of_injOn (window_injOn b hT), card_range]

/-- **The shift count of a window.** At lag `m`, the sites `j` of the window with `j + m` outside it
number at most `min (T, |r|)`, with `r = valMinAbs m`: only the last `r` (or first `|r|`) sites
leave. -/
theorem window_shift_card_le (b m : ZMod N) (T : ℕ) :
    ((window b T).filter fun j => j + m ∈ (window b T)ᶜ).card
      ≤ min T m.valMinAbs.natAbs := by
  rw [window, filter_image]
  refine card_image_le.trans ?_
  have hm : m = (m.valMinAbs : ZMod N) := (ZMod.coe_valMinAbs m).symm
  refine le_min ((card_filter_le _ _).trans (card_range T).le) ?_
  rcases Int.natAbs_eq m.valMinAbs with h | h
  · have hmn : m = (m.valMinAbs.natAbs : ZMod N) := by
      conv_lhs => rw [hm, h]
      rw [Int.cast_natCast]
    calc _ ≤ (Ico (T - m.valMinAbs.natAbs) T).card := card_le_card fun t ht => by
          simp only [mem_filter, mem_range, mem_compl] at ht
          rw [mem_Ico]
          refine ⟨?_, ht.1⟩
          by_contra hlt
          rw [not_le] at hlt
          apply ht.2
          rw [mem_image]
          refine ⟨t + m.valMinAbs.natAbs, mem_range.mpr (by omega), ?_⟩
          show b + ((t + m.valMinAbs.natAbs : ℕ) : ZMod N) = b + (t : ZMod N) + m
          conv_rhs => rw [hmn]
          push_cast
          ring
      _ ≤ m.valMinAbs.natAbs := by rw [Nat.card_Ico]; omega
  · have hmn : m = -(m.valMinAbs.natAbs : ZMod N) := by
      conv_lhs => rw [hm, h]
      rw [Int.cast_neg, Int.cast_natCast]
    calc _ ≤ (range m.valMinAbs.natAbs).card := card_le_card fun t ht => by
          simp only [mem_filter, mem_range, mem_compl] at ht
          rw [mem_range]
          by_contra hlt
          rw [not_lt] at hlt
          apply ht.2
          rw [mem_image]
          refine ⟨t - m.valMinAbs.natAbs, mem_range.mpr (by omega), ?_⟩
          show b + ((t - m.valMinAbs.natAbs : ℕ) : ZMod N) = b + (t : ZMod N) + m
          rw [Nat.cast_sub hlt]
          conv_rhs => rw [hmn]
          ring
      _ = m.valMinAbs.natAbs := card_range _

/-- **Jordan's inequality for the character.** `‖e(m) - 1‖ ≥ 4 |r| / N`, with `r = valMinAbs m`:
`‖e(m) - 1‖ = 2 |sin (π r / N)|` and `sin x ≥ 2x/π` on `[0, π/2]`, where `|r| ≤ N/2`. -/
theorem norm_stdAddChar_sub_one_ge (m : ZMod N) :
    4 * (m.valMinAbs.natAbs : ℝ) / N ≤ ‖ZMod.stdAddChar m - 1‖ := by
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hexp : ZMod.stdAddChar m
      = Complex.exp (Complex.I * ((2 * Real.pi * (m.valMinAbs : ℝ) / N : ℝ) : ℂ)) := by
    conv_lhs => rw [← ZMod.coe_valMinAbs m]
    rw [ZMod.stdAddChar_coe]
    congr 1
    push_cast
    ring
  rw [hexp, Complex.norm_exp_I_mul_ofReal_sub_one, Real.norm_eq_abs, abs_mul, abs_two]
  have hθ : 2 * Real.pi * (m.valMinAbs : ℝ) / N / 2 = Real.pi * (m.valMinAbs : ℝ) / N := by ring
  rw [hθ]
  have hcast : ((m.valMinAbs.natAbs : ℕ) : ℝ) = |(m.valMinAbs : ℝ)| := by
    rw [Nat.cast_natAbs, Int.cast_abs]
  have habs : Real.pi * |(m.valMinAbs : ℝ)| / N = |Real.pi * (m.valMinAbs : ℝ) / N| := by
    rw [abs_div, abs_mul, abs_of_pos Real.pi_pos, abs_of_pos hN]
  have hsin : ∀ y : ℝ, Real.sin |y| ≤ |Real.sin y| := fun y => by
    rcases abs_cases y with ⟨hy, _⟩ | ⟨hy, _⟩
    · rw [hy]; exact le_abs_self _
    · rw [hy, Real.sin_neg]; exact neg_le_abs _
  have hhalf : (m.valMinAbs.natAbs : ℝ) * 2 ≤ N := by
    have h := (Nat.le_div_iff_mul_le (by norm_num : 0 < 2)).mp (ZMod.natAbs_valMinAbs_le m)
    exact_mod_cast h
  have hle : Real.pi * (m.valMinAbs.natAbs : ℝ) / N ≤ Real.pi / 2 := by
    rw [div_le_iff₀ hN]
    nlinarith [Real.pi_pos]
  have hjordan := Real.mul_le_sin (by positivity) hle
  have hval : 2 / Real.pi * (Real.pi * (m.valMinAbs.natAbs : ℝ) / N)
      = 2 * (m.valMinAbs.natAbs : ℝ) / N := by
    field_simp
  rw [hval] at hjordan
  have hs := hsin (Real.pi * (m.valMinAbs : ℝ) / N)
  rw [← habs, ← hcast] at hs
  have h4 : 4 * (m.valMinAbs.natAbs : ℝ) / N = 2 * (2 * (m.valMinAbs.natAbs : ℝ) / N) := by ring
  linarith

/-- **The Dirichlet bound.** For a band of `W ≤ N` consecutive modes and a lag `m ≠ 0`,
`‖∑_{ω ∈ Ω} e(ω m)‖ ≤ N / (2 |r|)`, by the geometric sum `|x^W - 1| / |x - 1| ≤ 2 / |x - 1|`. -/
theorem norm_sum_window_le (a m : ZMod N) {W : ℕ} (hWN : W ≤ N) (hm : m ≠ 0) :
    ‖∑ ω ∈ window a W, ZMod.stdAddChar (ω * m)‖ ≤ N / (2 * (m.valMinAbs.natAbs : ℝ)) := by
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hn : (0 : ℝ) < m.valMinAbs.natAbs := by
    exact_mod_cast Int.natAbs_pos.mpr ((ZMod.valMinAbs_eq_zero m).not.mpr hm)
  have hlow := norm_stdAddChar_sub_one_ge m
  have hlowpos : 0 < 4 * (m.valMinAbs.natAbs : ℝ) / N := by positivity
  have hx1 : ZMod.stdAddChar m ≠ 1 := by
    intro h
    rw [h, sub_self, norm_zero] at hlow
    linarith
  have hsum : ∑ ω ∈ window a W, ZMod.stdAddChar (ω * m)
      = ZMod.stdAddChar (a * m) * ∑ s ∈ range W, ZMod.stdAddChar m ^ s := by
    rw [window, sum_image (window_injOn a hWN), mul_sum]
    refine sum_congr rfl fun s _ => ?_
    rw [add_mul, AddChar.map_add_eq_mul, ← nsmul_eq_mul, AddChar.map_nsmul_eq_pow]
  rw [hsum, norm_mul, AddChar.norm_apply, one_mul, geom_sum_eq hx1, norm_div]
  have hnum : ‖ZMod.stdAddChar m ^ W - 1‖ ≤ 2 := by
    calc ‖ZMod.stdAddChar m ^ W - 1‖ ≤ ‖ZMod.stdAddChar m ^ W‖ + ‖(1 : ℂ)‖ := norm_sub_le _ _
      _ = 2 := by rw [norm_pow, AddChar.norm_apply, one_pow, norm_one]; norm_num
  calc ‖ZMod.stdAddChar m ^ W - 1‖ / ‖ZMod.stdAddChar m - 1‖
      ≤ 2 / ‖ZMod.stdAddChar m - 1‖ :=
        div_le_div_of_nonneg_right hnum (norm_nonneg _)
    _ ≤ 2 / (4 * (m.valMinAbs.natAbs : ℝ) / N) :=
        div_le_div_of_nonneg_left (by norm_num) hlowpos hlow
    _ = N / (2 * (m.valMinAbs.natAbs : ℝ)) := by
        field_simp
        ring

/-- **The kernel bound.** For a band of `W ≤ N` consecutive modes and `k ≠ j`,
`‖B_Ω k j‖² ≤ min (W²/N², 1/(4 r²))`, with `r = valMinAbs (k - j)`. -/
theorem norm_bandLimit_sq_le (a : ZMod N) {W : ℕ} (hWN : W ≤ N) (k j : ZMod N)
    (hkj : k - j ≠ 0) :
    ‖bandLimit (window a W) k j‖ ^ 2
      ≤ min ((W : ℝ) ^ 2 / (N : ℝ) ^ 2) (1 / (4 * ((k - j).valMinAbs.natAbs : ℝ) ^ 2)) := by
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hn : (0 : ℝ) < (k - j).valMinAbs.natAbs := by
    exact_mod_cast Int.natAbs_pos.mpr ((ZMod.valMinAbs_eq_zero _).not.mpr hkj)
  have hB : ‖bandLimit (window a W) k j‖
      = (N : ℝ)⁻¹ * ‖∑ ω ∈ window a W, ZMod.stdAddChar (ω * (k - j))‖ := by
    simp only [bandLimit, of_apply, norm_mul, norm_inv, Complex.norm_natCast]
  have hcard : ((window a W).card : ℝ) ≤ W := by
    exact_mod_cast card_image_le.trans (card_range W).le
  have hW : ‖bandLimit (window a W) k j‖ ≤ (W : ℝ) / N := by
    rw [hB, inv_mul_eq_div]
    refine div_le_div_of_nonneg_right ?_ hN.le
    calc ‖∑ ω ∈ window a W, ZMod.stdAddChar (ω * (k - j))‖
        ≤ ∑ ω ∈ window a W, ‖ZMod.stdAddChar (ω * (k - j))‖ := norm_sum_le _ _
      _ = (window a W).card := by simp
      _ ≤ W := hcard
  have hr : ‖bandLimit (window a W) k j‖ ≤ 1 / (2 * ((k - j).valMinAbs.natAbs : ℝ)) := by
    rw [hB]
    calc (N : ℝ)⁻¹ * ‖∑ ω ∈ window a W, ZMod.stdAddChar (ω * (k - j))‖
        ≤ (N : ℝ)⁻¹ * (N / (2 * ((k - j).valMinAbs.natAbs : ℝ))) :=
          mul_le_mul_of_nonneg_left (norm_sum_window_le a (k - j) hWN hkj) (by positivity)
      _ = 1 / (2 * ((k - j).valMinAbs.natAbs : ℝ)) := by field_simp
  refine le_min ?_ ?_
  · calc ‖bandLimit (window a W) k j‖ ^ 2 ≤ ((W : ℝ) / N) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hW 2
      _ = (W : ℝ) ^ 2 / (N : ℝ) ^ 2 := div_pow _ _ _
  · calc ‖bandLimit (window a W) k j‖ ^ 2 ≤ (1 / (2 * ((k - j).valMinAbs.natAbs : ℝ))) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hr 2
      _ = 1 / (4 * ((k - j).valMinAbs.natAbs : ℝ) ^ 2) := by field_simp; ring

/-- **The leakage as a sum over lags.** `∑_{j ∈ S} ∑_{k ∉ S} ‖B_Ω k j‖² = ∑_m c(m) ‖B_Ω m 0‖²`, with
`c(m) = #{j ∈ S : j + m ∉ S}`: `B_Ω k j` depends on `k - j` alone. -/
theorem leakage_eq_shift_sum (S Ω : Finset (ZMod N)) :
    ∑ j ∈ S, ∑ k ∈ Sᶜ, ‖bandLimit Ω k j‖ ^ 2
      = ∑ m : ZMod N, ((S.filter fun j => j + m ∈ Sᶜ).card : ℝ) * ‖bandLimit Ω m 0‖ ^ 2 := by
  have hshift : ∀ k j : ZMod N, bandLimit Ω k j = bandLimit Ω (k - j) 0 := by
    intro k j
    simp only [bandLimit, of_apply, sub_zero]
  have hind : ∀ (T : Finset (ZMod N)) (g : ZMod N → ℝ),
      ∑ k, (if k ∈ T then g k else 0) = ∑ k ∈ T, g k := fun T g => by
    rw [← sum_filter, filter_mem_eq_inter, univ_inter]
  calc ∑ j ∈ S, ∑ k ∈ Sᶜ, ‖bandLimit Ω k j‖ ^ 2
      = ∑ j ∈ S, ∑ m : ZMod N, (if j + m ∈ Sᶜ then ‖bandLimit Ω m 0‖ ^ 2 else 0) := by
        refine sum_congr rfl fun j _ => ?_
        rw [← hind Sᶜ]
        refine (Fintype.sum_equiv (Equiv.addLeft j) _ _ fun m => ?_).symm
        simp only [Equiv.coe_addLeft]
        rw [hshift (j + m) j, add_sub_cancel_left]
    _ = ∑ m : ZMod N, ∑ j ∈ S, (if j + m ∈ Sᶜ then ‖bandLimit Ω m 0‖ ^ 2 else 0) := sum_comm
    _ = ∑ m : ZMod N, ((S.filter fun j => j + m ∈ Sᶜ).card : ℝ) * ‖bandLimit Ω m 0‖ ^ 2 :=
        sum_congr rfl fun m _ => by rw [← sum_filter, sum_const, nsmul_eq_mul]

/-- **Two lags per radius.** For `f ≥ 0` with `f 0 = 0`, `∑_m f (|valMinAbs m|) ≤ 2 ∑_{0 < d ≤ N/2} f d`:
the lags with `|r| = d` are among `d` and `-d`. -/
theorem sum_valMinAbs_le (f : ℕ → ℝ) (hf : ∀ d, 0 ≤ f d) (hf0 : f 0 = 0) :
    ∑ m : ZMod N, f m.valMinAbs.natAbs ≤ 2 * ∑ d ∈ Ioc 0 (N / 2), f d := by
  rw [← sum_fiberwise_of_maps_to' (s := univ) (t := Icc 0 (N / 2))
    (g := fun m : ZMod N => m.valMinAbs.natAbs)
    (fun m _ => mem_Icc.mpr ⟨Nat.zero_le _, ZMod.natAbs_valMinAbs_le m⟩) f,
    Icc_eq_cons_Ioc (Nat.zero_le _), sum_cons, hf0, sum_const_zero, zero_add, mul_sum]
  refine sum_le_sum fun d _ => ?_
  rw [sum_const, nsmul_eq_mul]
  refine mul_le_mul_of_nonneg_right ?_ (hf d)
  have hsub : (univ.filter fun m : ZMod N => m.valMinAbs.natAbs = d)
      ⊆ {(d : ZMod N), -(d : ZMod N)} := by
    intro m hm
    have hmd : m.valMinAbs.natAbs = d := (mem_filter.mp hm).2
    have hm' : m = (m.valMinAbs : ZMod N) := (ZMod.coe_valMinAbs m).symm
    rcases Int.natAbs_eq m.valMinAbs with h | h
    · rw [hmd] at h
      rw [hm', h, Int.cast_natCast]
      exact mem_insert_self _ _
    · rw [hmd] at h
      rw [hm', h, Int.cast_neg, Int.cast_natCast]
      exact mem_insert_of_mem (mem_singleton_self _)
  have hc : (univ.filter fun m : ZMod N => m.valMinAbs.natAbs = d).card ≤ 2 :=
    (card_le_card hsub).trans ((card_insert_le _ _).trans (by rw [card_singleton]))
  exact_mod_cast hc

/-- **The plunge of two windows is bounded by its radial profile.**
For `S` a window of `T` sites and `Ω` a window of `W ≤ N` modes,
`δ ≤ 2 ∑_{0 < d ≤ N/2} profile(d)`. -/
theorem window_plunge_le_profile (a b : ZMod N) {T W : ℕ} (hWN : W ≤ N) :
    ∑ i, (concentration_isHermitian (window b T) (window a W)).eigenvalues i *
        (1 - (concentration_isHermitian (window b T) (window a W)).eigenvalues i)
      ≤ 2 * ∑ d ∈ Ioc 0 (N / 2), plungeProfile T W N d := by
  rw [concentration_plunge_eq_leakage, leakage_eq_shift_sum]
  have hterm : ∀ m : ZMod N,
      (((window b T).filter fun j => j + m ∈ (window b T)ᶜ).card : ℝ) *
          ‖bandLimit (window a W) m 0‖ ^ 2
        ≤ plungeProfile T W N m.valMinAbs.natAbs := by
    intro m
    have hc := window_shift_card_le b m T
    by_cases hm : m = 0
    · subst hm
      rw [ZMod.valMinAbs_zero, Int.natAbs_zero, Nat.min_zero] at hc
      rw [Nat.le_zero.mp hc, Nat.cast_zero, zero_mul]
      exact plungeProfile_nonneg _ _ _ _
    · have hκ := norm_bandLimit_sq_le a hWN m 0 (by rwa [sub_zero])
      rw [sub_zero] at hκ
      have hc' : (((window b T).filter fun j => j + m ∈ (window b T)ᶜ).card : ℝ)
          ≤ min (T : ℝ) (m.valMinAbs.natAbs : ℝ) := by
        rw [← Nat.cast_min]
        exact_mod_cast hc
      exact mul_le_mul hc' hκ (sq_nonneg _) (le_min (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  calc _ ≤ ∑ m : ZMod N, plungeProfile T W N m.valMinAbs.natAbs := sum_le_sum fun m _ => hterm m
    _ ≤ 2 * ∑ d ∈ Ioc 0 (N / 2), plungeProfile T W N d :=
        sum_valMinAbs_le _ (plungeProfile_nonneg T W N) (by simp [plungeProfile])

/-- **The plunge region is logarithmic.**
For `S` a window of `T ≥ 1` consecutive sites and `Ω` a window of `1 ≤ W ≤ N` consecutive modes on
`ℤ_N`, the plunge mass of `A = P_S B_Ω P_S` obeys
`δ = ∑ λ (1 - λ) = tr A - tr A² ≤ (9 + log (min T W)) / 2`. The `9` is computed (the module
docstring and `profile_sum_le_log_W`). `T > N` is allowed: the window is then all of `ℤ_N`, nothing
leaks, and `δ = 0`. -/
theorem window_plunge_le_log (a b : ZMod N) {T W : ℕ} (hT : 1 ≤ T) (hW : 1 ≤ W) (hWN : W ≤ N) :
    ∑ i, (concentration_isHermitian (window b T) (window a W)).eigenvalues i *
        (1 - (concentration_isHermitian (window b T) (window a W)).eigenvalues i)
      ≤ (9 + Real.log (min T W : ℕ)) / 2 := by
  have h1 := window_plunge_le_profile a b (T := T) hWN
  have h2 := profile_sum_le_log (N := N) hT hW hWN (le_refl (N / 2))
  linarith

/-- **A finite aperture holds a finite count, with a logarithmic correction.**
For `S` a window of `1 ≤ T ≤ N` consecutive sites and `Ω` a window of `1 ≤ W ≤ N` consecutive modes
on `ℤ_N`, the number of eigenvalues of `P_S B_Ω P_S` above `1/2` is the time-bandwidth product to
within a logarithm: `|#{λ > 1/2} - T W / N| ≤ 9 + log (min T W)`. The `9` is computed (the module
docstring and `profile_sum_le_log_W`). -/
theorem window_count_half (a b : ZMod N) {T W : ℕ} (hT : 1 ≤ T) (hTN : T ≤ N) (hW : 1 ≤ W)
    (hWN : W ≤ N) :
    |((univ.filter fun i =>
          1 / 2 < (concentration_isHermitian (window b T) (window a W)).eigenvalues i).card : ℝ)
        - (T * W : ℝ) / N|
      ≤ 9 + Real.log (min T W : ℕ) := by
  have h := (aperture_holds_finite_count (window b T) (window a W)).2.2.2.1
  rw [window_card b hTN, window_card a hWN] at h
  have := window_plunge_le_log a b hT hW hWN
  linarith

/-- **The plunge count of two windows.** For `0 < ε < 1`, at most
`(9 + log (min T W)) / (2 ε (1 - ε))` eigenvalues of `P_S B_Ω P_S` lie in `(ε, 1 - ε)`. The `9` is
computed (the module docstring and `profile_sum_le_log_W`). -/
theorem window_plunge_card_le (a b : ZMod N) {T W : ℕ} (hT : 1 ≤ T) (hW : 1 ≤ W) (hWN : W ≤ N)
    {ε : ℝ} (hε : 0 < ε) (hε1 : ε < 1) :
    ((univ.filter fun i =>
          ε < (concentration_isHermitian (window b T) (window a W)).eigenvalues i ∧
          (concentration_isHermitian (window b T) (window a W)).eigenvalues i < 1 - ε).card : ℝ)
      ≤ (9 + Real.log (min T W : ℕ)) / (2 * (ε * (1 - ε))) := by
  have h := (aperture_holds_finite_count (window b T) (window a W)).2.2.2.2 ε hε hε1
  have hδ := window_plunge_le_log a b hT hW hWN
  have hpos : 0 < ε * (1 - ε) := mul_pos hε (by linarith)
  calc _ ≤ _ := h
    _ ≤ (9 + Real.log (min T W : ℕ)) / 2 / (ε * (1 - ε)) :=
        div_le_div_of_nonneg_right hδ hpos.le
    _ = (9 + Real.log (min T W : ℕ)) / (2 * (ε * (1 - ε))) := by rw [div_div]

end Window

end Entroptics
