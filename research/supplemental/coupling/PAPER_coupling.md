# Exact permutation-null moments for a signed bilinear coupling statistic on a shared basis

### With a real-embedding construction that makes the standardisation exact for complex frames.

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`  
ORCID [0009-0002-0150-4027](https://orcid.org/0009-0002-0150-4027)

*Pre-print. September 2026.*

---

## Abstract

Given two frames $A,B\in\mathbb{C}^{T\times D}$ sampled on a shared ordered axis and expressed in one shared feature basis, we consider the signed bilinear alignment $S=\langle\tilde A,\tilde B\rangle_F$ of their column-centred forms, and test it against the null that re-pairs the two sides — a uniformly random permutation of one side's rows, which destroys the correspondence between the sides while leaving each side's own internal structure entirely intact. We give the first two moments of that null in closed form: $\mathbb E_\pi[S]=0$ and $\operatorname{Var}_\pi[\operatorname{Re}S]=\operatorname{tr}(C_AC_B)/(T-1)$, where $C_A,C_B$ are the Gram matrices of the frames **under the real embedding** $\iota(x)=(\operatorname{Re}x,\operatorname{Im}x)$. The embedding is not a convenience: on the Hermitian Grams the same sum computes $\mathbb E_\pi[|S|^2]=\operatorname{Var}_\pi[\operatorname{Re}S]+\operatorname{Var}_\pi[\operatorname{Im}S]$, which is the variance of $\operatorname{Re}S$ only when $S$ is real, and we exhibit a $T=2$, $D=1$ pair for which the true variance is $0$ while the Hermitian form reports $4$. The resulting standardisation requires no sampling. It differs from the established permutation moments of Kazi-Aoual et al. (1995) and the test of Josse et al. (2008) in the statistic it applies to: those concern $\operatorname{tr}(W_xW_y)$, the RV coefficient, which is quadratic in each side, rotation-invariant and non-negative, and therefore cannot report a sign; the statistic here is linear in each side on a shared basis and can. The quantity $\operatorname{tr}(C_AC_B)$ appears in our result as the *variance* of a bilinear form rather than as the statistic itself. We verify the closed form against brute-force re-pairing at five shapes, real and complex, agreeing to within $2.0\%$ at $100{,}000$ draws and converging to $1.009$ at $200{,}000$; the planted sign is recovered in every draw; and over $1600$ independent pairs the standardised statistic has mean $0.011$ and standard deviation $1.018$.

**We are explicit about what is not established.** The moments are exact at every $T$; the *tail* is not. The permutation distribution of $\operatorname{Re}S$ is mildly skewed at small $T$, and the level is the Pitman–Hoeffding combinatorial CLT limit, approached as $T$ grows. Measured over $1600$ independent pairs the statistic fires at $0.049$ against a nominal two-sided $0.05$. A third permutation moment with a Pearson type III tail — exactly the correction Josse et al. apply to the same kind of skew in the RV setting — is the natural fix, and **it is not done here.**

---

## 1. Introduction

Two multivariate records are sampled together. Each is a $T\times D$ frame whose rows are indexed by a shared ordered axis, so that row $t$ of one is the same instant, or the same place, as row $t$ of the other. The question is whether the two are *coupled*: whether the correspondence between them carries information, over and above whatever structure each has internally.

The standard tools for this question — the RV coefficient [Robert & Escoufier 1976], linear CKA [Kornblith et al. 2019], and the wider Hilbert–Schmidt independence family — answer a slightly different one. Each is built from $\|\tilde A^{\mathsf H}\tilde B\|_F^2$, is invariant to a rotation of either configuration, and is non-negative. Non-negativity is not incidental; it is forced by rotation invariance, since a rotation of one side can flip the sign of any linear functional of it. These statistics therefore report the *magnitude* of an association and cannot report its direction.

For many problems that is the right answer, because the two sides carry unrelated coordinates and there is no meaningful sense in which they could agree or disagree in direction. But when the two sides are expressed in **one shared basis** — the same coordinates, in the same order, meaning the same thing on both sides — direction is exactly the interesting quantity. Two systems placed in one representation can be aligned or anti-aligned, and a statistic that returns only $|{\cdot}|$ discards the distinction.

**What this paper contributes.**

1. The exact first two permutation moments of the signed bilinear alignment $\langle\tilde A,\tilde B\rangle_F$ under the re-pairing null (Theorem 2.2), giving a closed-form standardisation with no resampling.
2. A real-embedding construction (§2.3) that makes the standardisation exact for complex frames, with a counterexample showing the Hermitian form is not merely less convenient but wrong.
3. A precise account (§3) of how this relates to the existing exact-moment literature, which concerns a different statistic.
4. An explicit statement of the tail's status (§4) and the correction that would close it.

The statistic and its null arise as one read in a larger construction for signal analysis [Sessford 2026], where two systems are carried into one shared representation and the sign of their coupling is a reported quantity. This paper is self-contained and makes no use of that setting.

---

## 2. The statistic and its exact null

### 2.1 The alignment

Let $A,B\in\mathbb{C}^{T\times D}$ be sampled on a shared ordered axis and expressed in one shared feature basis, and let $\tilde A,\tilde B$ denote their column-centred forms.

**Definition 2.1 (Alignment)** — $\displaystyle S=\langle\tilde A,\tilde B\rangle_F=\sum_{t}\langle\tilde a_t,\tilde b_t\rangle$, the Hermitian inner product of the two frames.

$S$ is linear in each side separately — a bilinear form, not a quadratic one — which is what allows it to carry a sign.

### 2.2 The null re-pairs the two sides

The null permutes the rows of one side uniformly at random. Each side then keeps its own internal structure *entirely*, and the only thing that varies is the correspondence between them. This isolates the question — are $A$ and $B$ coupled? — from the question of whether either has structure of its own, which a null that destroyed one side's internal structure would confound with it.

### 2.3 The real embedding

Write $\iota:\mathbb{C}^D\to\mathbb{R}^{2D}$, $x\mapsto(\operatorname{Re}x,\operatorname{Im}x)$, and $\check A=\iota\tilde A$, $\check B=\iota\tilde B\in\mathbb{R}^{T\times 2D}$. The statistic is unchanged by the embedding, since

$$\operatorname{Re}\langle a,b\rangle_{\mathbb C}=\langle\iota a,\iota b\rangle_{\mathbb R},\qquad\text{so}\qquad \operatorname{Re}S=\langle\check A,\check B\rangle_F .$$

The moments below are stated on the embedded frames, and §2.5 shows why that is necessary rather than cosmetic.

### 2.4 The moments

**Theorem 2.2 (Permutation-null moments)** — With $C_A=\check A^{\mathsf T}\check A$ and $C_B=\check B^{\mathsf T}\check B$, and $\pi$ uniform on the symmetric group $\mathfrak S_T$,

$$\mathbb{E}_\pi[S]=0,\qquad \operatorname{Var}_\pi\!\big[\operatorname{Re}S\big]=\frac{\operatorname{tr}(C_AC_B)}{T-1}.$$

*Proof.* Both embedded frames are real, and $\operatorname{Re}S=\langle\check A,\check B\rangle_F$. Write $x_{td}$ for the entries of $\check A$ and $y_{td}$ for those of $\check B$.

Centring gives $\mathbb{E}_\pi[\check a_{\pi(t)}]=0$, hence $\mathbb{E}_\pi[S]=0$.

For the second moment, write $M_{de}=(C_A)_{de}$. For a uniform $\pi$, the index $\pi(t)$ is uniform on $\{1,\dots,T\}$, so

$$\mathbb{E}\big[x_{\pi(t)d}\,x_{\pi(t)e}\big]=\frac1T\sum_i x_{id}x_{ie}=\frac{M_{de}}{T}.$$

For $t\ne s$ the pair $(\pi(t),\pi(s))$ is uniform over the $T(T-1)$ ordered pairs of distinct indices, so by the centring identity $\sum_i x_{id}=0$,

$$\mathbb{E}\big[x_{\pi(t)d}\,x_{\pi(s)e}\big]=\frac{1}{T(T-1)}\sum_{i\ne j}x_{id}x_{je} =\frac{1}{T(T-1)}\Big(\underbrace{\textstyle\sum_i x_{id}\sum_j x_{je}}_{=0}-M_{de}\Big) =-\frac{M_{de}}{T(T-1)} .$$

Now expand. Using $\sum_t y_{td}y_{te}=(C_B)_{ed}$ and, again by centring on the $B$ side, $\sum_{t\ne s}y_{td}y_{se}=-(C_B)_{ed}$,

$$\operatorname{Var}_\pi[\operatorname{Re}S] =\sum_{d,e}\Big(\frac{M_{de}}{T}(C_B)_{ed}+\Big(-\frac{M_{de}}{T(T-1)}\Big)\big(-(C_B)_{ed}\big)\Big) =\Big(\frac1T+\frac1{T(T-1)}\Big)\sum_{d,e}(C_A)_{de}(C_B)_{ed}.$$

The coefficient is $\frac1T+\frac1{T(T-1)}=\frac{(T-1)+1}{T(T-1)}=\frac{1}{T-1}$, and the double sum is $\operatorname{tr}(C_AC_B)$ for the symmetric Gram matrices $C_A,C_B$. ∎

Both moments are exact at every $T$, with no asymptotics and no assumption on the frames beyond centring.

The combinatorial core of the proof — that summing a function of $\sigma(i)$ over $\mathfrak S_T$ counts each image exactly $(T-1)!$ times whichever index $i$ is taken, that the distinct-index term does not depend on *which* pair of indices is taken, the coefficient identity $\frac1T+\frac1{T(T-1)}=\frac{1}{T-1}$, and the reduction of the entrywise double sum to $\operatorname{tr}(C_AC_B)$ for symmetric Grams — is machine-checked in Lean 4 / Mathlib [Sessford 2026, §13], by explicit bijections rather than by a counting argument. The assembly into the displayed form is done on paper, above.

### 2.5 The embedding carries the theorem

On the Hermitian Grams $\tilde A^{\mathsf H}\tilde A$ and $\tilde B^{\mathsf H}\tilde B$ the same sum computes

$$\mathbb{E}_\pi\big[|S|^2\big]=\operatorname{Var}_\pi[\operatorname{Re}S]+\operatorname{Var}_\pi[\operatorname{Im}S],$$

which equals $\operatorname{Var}_\pi[\operatorname{Re}S]$ only when $S$ is real. Off the real axis the Hermitian form is not a loose bound but a different quantity, and it can exceed the truth without limit.

**A counterexample.** Take $T=2$, $D=1$, $\tilde a=(1,-1)^{\mathsf T}$ and $\tilde b=(i,-i)^{\mathsf T}$. There are exactly two re-pairings, and both give $\operatorname{Re}S=0$; the true permutation variance is therefore **exactly $0$**. The Hermitian form gives

$$\frac{\operatorname{tr}\big(\tilde A^{\mathsf H}\tilde A\,\tilde B^{\mathsf H}\tilde B\big)}{T-1}=\frac{2\cdot 2}{1}=4 .$$

The real embedding gives $0$, in agreement with the enumeration. A standardisation built on the Hermitian form would divide a statistic that is identically zero by $2$, and would report the wrong scale on any pair whose alignment carries a phase. The counterexample is enumerated in `tables/embedding.csv`.

The embedding is therefore what makes the standardisation of Definition 2.3 exact, and it is what an implementation must form.

### 2.6 The standardised coupling

**Definition 2.3 (Coupling)** — $\displaystyle z_{AB}=\frac{\operatorname{Re}S}{\sqrt{\operatorname{tr}(C_AC_B)/(T-1)}}$, with signed strength $\displaystyle \rho_{AB}=\frac{\operatorname{Re}S}{\|\tilde A\|_F\,\|\tilde B\|_F}$.

**Proposition 2.4 (Bounds and invariance)** — $|\rho_{AB}|\le1$, with equality iff $\tilde B=\lambda\tilde A$ for a real $\lambda$. Moreover $\rho_{AB}$ is invariant to a separate positive scale on either side.

*Proof.* The bound is Cauchy–Schwarz applied to $\operatorname{Re}\langle\tilde A,\tilde B\rangle_F\le|\langle\tilde A,\tilde B\rangle_F|\le\|\tilde A\|_F\|\tilde B\|_F$, with equality in the first step iff the inner product is real and in the second iff the frames are proportional; together, iff $\tilde B=\lambda\tilde A$ with $\lambda\in\mathbb R$. Scaling $\tilde A\mapsto c\tilde A$ for $c>0$ multiplies numerator and denominator alike. ∎

The invariance matters for interpretation: a side's choice of units cannot move either the sign or the strength.

### 2.7 The sign requires a shared basis

$\rho_{AB}$ is a statement about the two sides' coordinates being *the same* coordinates, and it is defined only on a shared basis.

Across two bases ($D_A\ne D_B$) the only basis-free statistic is $\|\tilde A^{\mathsf H}\tilde B\|_F^2$, which is non-negative by construction and so reports magnitude alone. A sign recovered from the leading singular vectors of the cross-covariance would follow their arbitrary global sign, making it a convention of the eigensolver rather than a property of the data. The read is therefore *defined* on a shared basis and raises without one, rather than returning a number whose meaning depends on a factorisation's sign convention.

This is a restriction on applicability and it should be read as one. Where the two sides genuinely carry unrelated coordinates, the rotation-invariant family is the correct tool and this statistic does not apply.

---

## 3. Relation to existing work

**Closed-form permutation moments are an established technique, on a different statistic.** The first three exact moments of $\operatorname{tr}(W_xW_y)$ under a uniform row permutation are given by Kazi-Aoual et al. [1995], and Josse et al. [2008] build a test on them, matching a Pearson type III distribution to the mean, variance and skewness rather than resampling. That work is the direct methodological precedent for what is done here, and the technique — exact combinatorial moments in place of a permutation sample — is theirs.

The statistic is not the same one. $\operatorname{tr}(W_xW_y)$ is the RV coefficient's numerator: it is **quadratic** in each side, it is normalised by $\|\tilde A^{\mathsf H}\tilde A\|_F\|\tilde B^{\mathsf H}\tilde B\|_F$ to a similarity in $[0,1]$, it is invariant to a rotation of either configuration, and it is non-negative. It is the same object as linear CKA [Kornblith et al. 2019], the linear case of the Hilbert–Schmidt independence criterion.

Theorem 2.2's statistic is the **signed bilinear** form $\langle\tilde A,\tilde B\rangle_F$ on a shared basis: linear in each side, and defined only where the two sides share coordinates. The sign is precisely what the rotation-invariant family cannot report, and it is the reason for a separate result rather than an application of an existing one.

**Three differences follow, and they are worth separating.**

*The distributions are different objects.* The moments of Kazi-Aoual et al. describe the permutation law of a quadratic, non-negative statistic, which is supported on $[0,\infty)$ and right-skewed. The law here is that of a signed bilinear form, symmetric about $0$ to first order. Neither set of moments implies the other, and the exact-moment result for the RV coefficient does not specialise to Theorem 2.2 by any substitution.

*The role of $\operatorname{tr}(C_AC_B)$ is inverted.* In the RV setting the trace of a product of Gram matrices **is** the statistic. Here the same shape of quantity appears as the **variance** of a different statistic. The coincidence of form is genuine and is worth naming, because a reader who knows the RV literature will recognise the expression and may take the two results to be the same; they are not, and the trace occupies opposite positions in them.

*The invariance groups differ.* The RV family is invariant under a rotation of either configuration, which is what makes it applicable across unrelated bases and what forces non-negativity. Definition 2.3 is invariant under a separate positive scale on either side (Proposition 2.4) and nothing more. The smaller invariance group is what buys the sign, and it is also what restricts the statistic to a shared basis. The trade is exact and it goes in both directions.

**On the wider permutation literature.** The combinatorial central limit theorem for statistics of this form is Hoeffding's [1951], and it supplies the asymptotic normality that §4 relies on and qualifies. What Theorem 2.2 adds is the exact finite-$T$ second moment, which removes the need to estimate a scale by resampling; the limit theorem still governs the tail.

---

## 4. What is not established: the tail

**The moments are exact; the tail is not.** This section states the limitation plainly because a reader will otherwise find it, and because the size of the gap is measurable and we measure it.

Theorem 2.2 gives $\mathbb E_\pi[S]$ and $\operatorname{Var}_\pi[\operatorname{Re}S]$ exactly, at every $T$, with no approximation. The standardisation $z_{AB}$ therefore has mean $0$ and unit variance under the null exactly. **The approximation is in the tail's shape alone, not in the standardisation.**

Converting $z_{AB}$ to a level requires a distribution, and what is available is the Pitman–Hoeffding combinatorial CLT limit [Hoeffding 1951]: $z_{AB}$ is asymptotically standard normal as $T\to\infty$. At finite $T$ the permutation distribution of $\operatorname{Re}S$ is mildly skewed, so the normal quantile is not the exact permutation quantile and the realised level departs from the nominal one.

**Measured.** Over $1600$ independent pairs across four shapes, the statistic fires at $0.0488$ against a nominal two-sided $0.05$ (`tables/null.csv`, and §5.3). The departure is small and in the conservative direction at these shapes, but it is a departure, and nothing here bounds it as a function of $T$ and $D$.

**The correction that would close it.** A third permutation moment, with a Pearson type III distribution matched to mean, variance and skewness, is the natural fix. It is precisely what Josse et al. [2008] apply to the analogous skew in the RV setting, using the third moment of Kazi-Aoual et al. [1995]. Carrying that programme to the bilinear statistic of Definition 2.1 would require deriving $\mathbb E_\pi[(\operatorname{Re}S)^3]$ in closed form — a three-index combinatorial sum of the same character as the two-index sum in the proof of Theorem 2.2, and plausibly tractable by the same method.

**It is not done here.** The result of this paper is the exact second moment and the embedding that makes it correct for complex frames. A reader needing a calibrated level at small $T$ should either derive the third moment, or sample the permutation distribution directly — which is always available, and against which the closed form's role is to remove the need to estimate the *scale* by sampling, not to remove sampling entirely.

---

## 5. Validation

Every number below is written by `reproduce.py`; §6 says which table holds which.

### 5.1 The closed form against brute force

This is the paper's main empirical claim: that $\operatorname{tr}(C_AC_B)/(T-1)$ is the permutation variance and not merely close to it.

For each of five shapes, two independent frames are drawn, the closed form is evaluated, and the empirical variance of $\operatorname{Re}S$ is taken over uniform row re-pairings drawn directly — not through the library, since the sample is the reference the closed form is being checked against. Three of the shapes are real and **two are complex**, and the complex cases are the ones that exercise the real embedding of §2.3: a real-only check cannot distinguish Theorem 2.2 from the Hermitian form, because the two agree when $S$ is real.

| shape $(T,D)$ | kind | closed form | ratio at 20k | at 100k | at 200k |
|---|---|---|---|---|---|
| $(40,5)$ | real | 173.47 | 1.0054 | 1.0001 | 1.0046 |
| $(64,8)$ | real | 486.79 | 1.0094 | 0.9987 | 0.9985 |
| $(120,3)$ | real | 440.69 | 1.0044 | 1.0012 | 1.0012 |
| $(64,6)$ | complex | 791.68 | 1.0056 | 0.9994 | 0.9995 |
| $(96,4)$ | complex | 780.68 | 1.0291 | 1.0197 | 1.0088 |

*Table 1. Empirical permutation variance over the closed form, from `tables/brute_force.csv`. A ratio of 1 is exact agreement. The three columns are nested cut points of one draw stream per shape, so they show a single estimate settling rather than three independent runs.*

At $100{,}000$ re-pairings the agreement is within $2.0\%$ at every shape tested. **The residual is sampling error in the reference, not error in the closed form**, and the worst case across the five shapes falls monotonically with the draw count: $1.029$ at $20{,}000$, $1.020$ at $100{,}000$ and $1.009$ at $200{,}000$. That convergence is the substantive check. A fixed discrepancy that did not shrink would indicate the closed form was wrong; a discrepancy that falls as the sample grows is the sample approaching the exact value the theorem states.

Individual shapes do not fall monotonically — $(40,5)$ reads $1.0054$, $1.0001$, $1.0046$ — and should not be expected to, since each cut point is one realisation of an estimator whose own error is random. It is the worst case over shapes, and the direction of travel, that carries the claim.

The variance of an empirical variance is governed by the fourth moment of the distribution being sampled, and the permutation law here is not normal (§4), so convergence is slower than a normal-theory rule of thumb would suggest. The $(96,4)$ complex case is the slowest of the five and is reported as such rather than averaged away.

### 5.2 Sign recovery

Two sides share a planted carrier at signed strength $\rho$ in one shared basis, at $(T,D)=(96,6)$, $60$ draws per strength.

| planted $\rho$ | expected sign | sign agreement | mean strength | resolved rate |
|---|---|---|---|---|
| $+1.0$ | $+$ | 1.000 | $+0.693$ | 1.000 |
| $+0.5$ | $+$ | 1.000 | $+0.508$ | 1.000 |
| $0.0$ | none | 0.967 | $+0.003$ | 0.033 |
| $-0.5$ | $-$ | 1.000 | $-0.497$ | 1.000 |
| $-1.0$ | $-$ | 1.000 | $-0.678$ | 1.000 |

*Table 2. From `tables/sign.csv`.*

The sign is recovered in every draw at $|\rho|\in\{0.5,1\}$. At $\rho=0$ the statistic resolves nothing in $96.7\%$ of draws, which is the nominal level of the test and not a failure: at a two-sided $\alpha=0.05$ a firing rate near $0.033$–$0.05$ on independent sides is what a calibrated test does.

The recovered strength is not the planted $\rho$ and is not intended to be — the planted quantity is a carrier amplitude ratio and $\rho_{AB}$ is a normalised frame alignment, so the two agree in sign and ordering, not in value. What is claimed is that the sign is a measurement, and that it returns nothing rather than a fitted value when nothing resolves.

### 5.3 Null calibration

Over $1600$ independent pairs across four shapes — $(64,4)$, $(128,6)$, $(96,12)$, $(200,5)$, $400$ draws each — the standardised statistic has

$$\text{mean }0.011,\qquad \text{standard deviation }1.018,\qquad \text{firing rate }0.0488\ \text{against a nominal two-sided }0.05 .$$

The mean and standard deviation are the direct check on Theorem 2.2: a standardisation built on a wrong variance would show a standard deviation away from $1$, and it does not. The firing rate is the check on the *tail*, and it is the number §4 is about.

---

## 6. Reproduction

```
python research/supplemental/coupling/reproduce.py
```

The brute-force check of §5.1 is the paper's main empirical claim and it is one command. The full run draws $200{,}000$ re-pairings at five shapes and takes a few minutes; `--quick` cuts it to $20{,}000$, which is a smoke test and not what the paper reports.

| table | holds | used in |
|---|---|---|
| `tables/brute_force.csv` | closed form against re-pairing, three draw counts | Table 1, §5.1 |
| `tables/sign.csv` | planted-sign recovery | Table 2, §5.2 |
| `tables/null.csv` | null calibration over 1600 pairs | §§4, 5.3 |
| `tables/embedding.csv` | the $T=2$, $D=1$ counterexample | §2.5 |

`verify.py` beside it checks every number quoted in this paper against those tables and exits non-zero on any drift.

The statistic is computed through the public API of the `entroptics` library [Sessford 2026] — `Screen.register`, `Screen.place`, `Screen.coupling` — which forms the real embedding and evaluates Definition 2.3. The permutation sample of §5.1 is drawn in the reproduction script rather than in the library, because it is the reference the library's closed form is being checked against. Every seed is fixed and the tables are byte-reproducible across runs.

---

## Declaration of generative AI use

The author used Anthropic's Claude Opus (versions 4.8 and 5) in the preparation of this work. Its
contribution was to write code, and to generate and validate content. The ideas, the construction
and the claims are the author's. No other generative AI tool was used. The author reviewed and
edited all output and takes full responsibility for the content of this publication.

---

## References

- W. Hoeffding, *A combinatorial central limit theorem*, Ann. Math. Statist. **22** (1951) 558–566.
- J. Josse, J. Pagès, F. Husson, *Testing the significance of the RV coefficient*, Comput. Statist. Data Anal. **53** (2008) 82–91.
- F. Kazi-Aoual, S. Hitier, R. Sabatier, J.-D. Lebreton, *Refined approximations to permutation tests for multivariate inference*, Comput. Statist. Data Anal. **20** (1995) 643–656.
- S. Kornblith, M. Norouzi, H. Lee, G. Hinton, *Similarity of neural network representations revisited*, Proc. 36th Int. Conf. Machine Learning (ICML), PMLR **97** (2019), arXiv:1905.00414.
- P. Robert, Y. Escoufier, *A unifying tool for linear multivariate statistical methods: the RV-coefficient*, J. R. Statist. Soc. C (Applied Statistics) **25** (1976) 257–265.
- I. J. Sessford, *Entroptics: reading a 2-D signal as a finite optical aperture at its own entropy-matched resolution*, pre-print, 2026. https://github.com/Agience/entroptics
