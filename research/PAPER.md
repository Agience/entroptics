# Entroptics

### Reading any 2-D signal as a finite optical aperture at its own entropy-matched resolution.

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`

*Pre-print. July 2026.*

---

## Abstract

We treat any two-dimensional signal $W\in\mathbb{C}^{T\times F}$, with one **ordered** axis and one **feature** axis, as a finite optical aperture whose resolution the signal's own Shannon entropy sets. From the entropy of the power marginals we read a matched scale; from the singular and correlation spectra, the standard optical quantities (fill fraction, étendue, Strehl ratio, space–bandwidth product); from the signal's own autocorrelation (a direct lag average, via Wiener–Khinchin), the optical transfer function (OTF) and, through the Abbe/Rayleigh relation, a diffraction limit. A streaming online dynamic mode decomposition (DMD) operator supplies the per-mode decay rates that the entropy width approximates, and a read-side filter projects the field onto its own resolved modes to recover the signal, synthesising nothing. Each read is a standard optical quantity and each governing claim a theorem specialised to the finite, discrete setting. The construction is *parameter-free*; the only external input is a decision-risk level, and the only distributional assumption is the independent and identically distributed (i.i.d.) Gaussian bulk null of the noise-floor read (§§6, 8). The provenance of every constant is tabulated in §11.1. We give precise definitions and prove the governing lemmas — positive semidefiniteness of the biased autocovariance (hence a peak-at-zero-lag OTF), the fill-fraction and Strehl bounds, the scale-invariance of the Rayleigh shape factor, the permutation-null mean of the coherence statistic, a Weyl-certified interval for the attenuation constant, the recovery of the decay rates and the additivity behind stream splicing, and the axial/directional dichotomy of the concentration reads — and tie each read to a named theorem in a term-by-term dictionary. The construction is realised in a small, backend-agnostic (numpy or torch), deterministic Python library whose test suite pins these identities, and is accompanied by a Lean 4 / Mathlib certification of the governing lemmas.

---

## 1. Introduction

Describing a field through the language of optics is an old experimental practice: free-space microwave measurements characterise a physical *screen* by its reflection, attenuation, and diffraction, read at varying angles and distances, in optical terms [Kassner 1950]. The optical picture we take literally for data is the finite aperture: classically, it cannot resolve to infinitely fine detail — its size fixes a diffraction limit, and the image is the object convolved with the aperture's point-spread function [Goodman 2005]. Given a two-dimensional array $W$ indexed by one **ordered** axis (time, evolution, depth) and one **feature** axis (channels, frequency bins, coordinates), we ask: what is the aperture through which this particular signal is observed, and what is its diffraction limit? The signal sets its own aperture: the resolution is read from its Shannon entropy, the reads are the classical optical invariants, and nothing is fitted to the signal. Entroptics is the study of the optics read from the signal itself, rather than from an apparatus.

Each read is defined from $W$ alone and each governing claim is a finite, discrete specialisation of a standard result: Shannon's source entropy, Wiener–Khinchin, the Fourier-optics autocorrelation theorem, Abbe/Rayleigh, conservation of étendue (the Lagrange invariant), the Gabor–Lukosz degrees-of-freedom invariance [Gabor 1946; Lukosz 1966], the coherent-mode decomposition, Mercer's theorem, the Marchenko–Pastur correlation edge, the Tracy–Widom singular-value edge, Weyl's inequality, and the exact dynamic-mode decomposition of a linear map. The two detection tests reference explicit closed-form nulls: the row-permutation null (§5) and the Marchenko–Pastur i.i.d.-bulk null (§§6, 8).

**Convention.** Throughout, $W\in\mathbb{C}^{T\times F}$ has rows indexed by the ordered axis $t\in\{1,\dots,T\}$ (subscript $T$) and columns by the feature axis $f\in\{1,\dots,F\}$ (subscript $F$). "Time" and "features" are *roles*, not physical quantities: any array with a single ordered axis qualifies. Entropy $\mathrm{H}(\cdot)$ is in bits, $\mathrm{H}(p)=-\sum_i p_i\log_2 p_i$ for a probability vector $p$, with $0\log_2 0:=0$. We write $x^{\mathsf H}$ for the conjugate transpose, $M^{+}$ for the Moore–Penrose pseudoinverse, and $\operatorname{Re}$ for the real part. A few symbols carry a role that is fixed per section and disambiguated by subscript or context: bare $\alpha$ is the false-alarm level (§§8, 11), while a subscripted $\alpha$ is an attenuation or decay rate ($\alpha_{\mathrm p}$ in §6, $\alpha_k$ in §9); $\mu$ is the coherence null mean (§5), the Johnstone centre (§§6, 8), or — as $\mu_k$ — a DMD eigenvalue (§9); and $\rho$ is the Mercer ratio (§4.3) or a correlation length (§12).

---

## 2. The entropy-matched scale

Let $P=|W|^2$ elementwise (nonfinite entries and masked cells set to $0$), and $S_{\!P}=\sum_{t,f}P_{tf}$ the total power.

**Definition 2.1 (Power marginals and matched scale)** — When $S_{\!P}>0$ the *ordered* and *feature* power marginals are the probability vectors

$$
  p^{T}_t=\frac1{S_{\!P}}\sum_f P_{tf},\qquad
  p^{F}_f=\frac1{S_{\!P}}\sum_t P_{tf},
$$

with entropies $\mathrm{H}_T=\mathrm{H}(p^{T})$, $\mathrm{H}_F=\mathrm{H}(p^{F})$ [Shannon 1948]. The *effective mode counts* and *matched cell scales* are

$$
  n_a=\operatorname{round}\!\big(2^{\mathrm{H}_a}\big),\qquad
  \delta_a=\frac{L_a}{2^{\mathrm{H}_a}}, \qquad a\in\{T,F\},
$$

with $L_T=T$, $L_F=F$. When $S_{\!P}=0$ we set $\mathrm{H}_a=\log_2 L_a$ (maximal spread, $\delta_a=1$).

Power-weighting lets bright, on-signal rows dominate irrespective of how small a fraction of the record they occupy. The scale $\delta_a=L_a/2^{\mathrm{H}_a}$ is the reciprocal of the occupied fraction $2^{\mathrm{H}_a}/L_a$: a signal that concentrates its power in $2^{\mathrm{H}_a}\ll L_a$ effective cells is oversampled by $\delta_a>1$ along that axis.

**Only the feature axis folds.** The fold (§8) coarsens the *feature* axis to its own resolution. The *ordered* axis keeps native spacing, which the ordered reads require: the coherence of §5 (adjacent-row similarity), the decay/OTF of §4 (lag structure), and the rates of §9 (the ordered trajectory) are all lag statistics. So $\delta_T:=1,\ n_T:=T$ always, and only the feature scale carries a noise guard.

**The finite-sample noise guard.** A structureless (uniform-power) feature marginal has $\mathrm{H}_F=\log_2 F$ in the population but, by the plug-in bias, sits below that maximum on any finite record — a spurious $\delta_F>1$. The guard holds $\delta_F=1$ whenever $\mathrm{H}_F$ lies within the uniform-null bias band.

**Definition 2.2 (Noise-guarded scale)** — The ordered axis keeps native resolution ($\delta_T:=1,\ n_T:=T$). For the feature axis, with the capped uniform-null band $\beta_F=\min\!\big((F-1)/(2\,T\ln 2),\ \tfrac12\log_2 F\big)$, set $\delta_F:=1,\ n_F:=F$ if $\mathrm{H}_F\ge \log_2 F-\beta_F$, otherwise as in Definition 2.1.

The inner term $(F-1)/(2\,T\ln 2)$ is the Miller–Madow bias expression for $F$ bins at $T$ samples [Miller 1955], a *conservative* uniform-null guard: it exceeds the marginal's mean finite-sample deficit — which, for the power marginal, is smaller by a factor $F$, its effective sample size being the total cell count $TF$ — so both structureless noise and a genuine low-rank signal's *delocalised* feature marginal stay at native resolution. The inner band grows without bound in $F/T$ and *exceeds* $\log_2 F$ for wide-short data ($F\gtrsim 2T\ln F$); capping it at $\tfrac12\log_2 F$ keeps the guard active, folding once power concentrates below $\sqrt{F}$ effective channels ($\mathrm{H}_F<\tfrac12\log_2 F$, deficit $>\tfrac12\log_2 F$). For all tall/square shapes the inner band is already below the cap and the behaviour is unchanged, so noise and delocalised low-rank signals stay at native resolution (§12) and the coherence null of §5 stays calibrated.

The band is closed-form, fixed by the axis lengths alone. The one choice it embeds — the $\sqrt{F}$ concentration floor of the cap — is a stated criterion, not a fitted constant; the provenance of every constant is tabulated in §11.1.

---

## 3. Aperture reads: fill fraction, étendue, Strehl

### 3.1 The fill fraction and its duality

For a block $M\in\mathbb{C}^{m\times k}$ with singular values $s_1\ge\cdots\ge s_n\ge 0$, $n=\min(m,k)$, let $q_i=s_i^2/\sum_j s_j^2$ be the normalised power spectrum.

**Definition 3.1 (Fill fraction and magnification)**

$$
  \varphi(M)=\frac{2^{\mathrm{H}(q)}}{n}\in[1/n,\,1],\qquad
  \operatorname{mag}(M)=\frac1{\varphi(M)}\in[1,\,n].
$$

$\varphi$ is the aperture's fill fraction (the fraction of singular modes that are active, the *bounded* face), and $\operatorname{mag}$ its reciprocal *reach* (oversampling), the *unbounded* face. The two are one continuum meeting at $\varphi=\operatorname{mag}=1$, the critically sampled diffraction limit.

**Lemma 3.2 (Fill-fraction bounds)** — For any $M$ with at least one nonzero singular value, $\varphi(M)\in[1/n,\,1]$. Moreover $\varphi(M)=1/n$ iff $\operatorname{rank}(M)=1$, and $\varphi(M)=1$ iff all $n$ singular values are nonzero and equal.

*Proof.* $q$ is a probability vector of length $n$, so $\mathrm{H}(q)\in[0,\log_2 n]$, whence $2^{\mathrm{H}(q)}\in[1,n]$ and $\varphi=2^{\mathrm{H}(q)}/n\in[1/n,1]$. $\mathrm{H}(q)=0$ iff $q$ is a point mass (a single nonzero singular value, i.e. rank $1$); $\mathrm{H}(q)=\log_2 n$ iff $q$ is uniform over all $n$ entries (every singular value nonzero and equal). ∎

### 3.2 Per-axis correlation spectra

For axis $a$ let $M_a$ be $W$ (feature axis, columns as variables) or $W^{\mathsf\top}$ (ordered axis), so the variables are the $L_a$ coordinates of that axis and the samples are those of the other. Let $R_a$ be the (Hermitian) *correlation* matrix of $M_a$ (columns centred, then rescaled to unit diagonal, each coordinate carrying nonzero variance), with eigenvalues $\lambda^{(a)}_1\ge\cdots\ge\lambda^{(a)}_{L_a}\ge 0$.

**Definition 3.3 (Axis reads)** — With $\bar\lambda^{(a)}_i=\lambda^{(a)}_i/\sum_j\lambda^{(a)}_j$,

$$
  \varphi_a=\frac{2^{\mathrm{H}(\bar\lambda^{(a)})}}{L_a},\qquad
  \sigma_a=\sqrt{\lambda^{(a)}_1},\qquad
  \mathcal E=\varphi_F\varphi_T,\qquad
  \mathrm{SBW}=n_F\,n_T,\qquad
  \mathcal S=\frac{\lambda^{(T)}_1}{\sum_j\lambda^{(T)}_j}.
$$

$\mathcal E$ is the *étendue* (the joint 2-D aperture area), $\mathrm{SBW}$ the *space–bandwidth product* (the invariant count of degrees of freedom the screen carries [Lukosz 1966]), and $\mathcal S$ the *Strehl ratio*: the fraction of power in the dominant coherent mode of the ordered-axis correlation operator, its leading mode in the coherent-mode (Mercer) decomposition [Wolf 1982; Mercer 1909]. Because $R_a$ has unit diagonal, $\operatorname{tr}R_a=\sum_j\lambda^{(a)}_j=L_a$; the correlation spectrum needs no separate normalisation.

**Lemma 3.4 (Strehl bounds)** — $\mathcal S\in[1/T,\,1]$, with $\mathcal S=1$ iff the ordered-axis correlation has rank $1$ (a single coherent mode) and $\mathcal S=1/T$ iff its spectrum is flat.

*Proof.* $\operatorname{tr}R_T=T$, so the top eigenvalue $\lambda^{(T)}_1\in[1,T]$ (at least the mean $1$, at most the trace); hence $\mathcal S=\lambda^{(T)}_1/T\in[1/T,1]$, with the extremes at the rank-one and flat spectra. ∎

**Proposition 3.5 (Étendue is a bounded, basis-invariant area)** — $\mathcal E=\varphi_F\varphi_T\in(0,1]$, the product of the two axis fill fractions, the phase-space area the finite screen carries. Each $\varphi_a$, hence $\mathcal E$, is invariant under relabeling of axis $a$'s variables and under a per-variable phase, the discrete Smith–Helmholtz/Lagrange invariance. A system that decouples the axes (one fill $\to 0$ while the other stays finite) sends $\mathcal E\to0$: the scale-free limit.

*Proof.* $\varphi_a\in(0,1]$ by Lemma 3.2 applied to $R_a$. A permutation $P$ and per-variable phase $\Theta=\operatorname{diag}(e^{i\theta})$ send $R_a\mapsto P\Theta R_a\Theta^{\mathsf H}P^{\mathsf\top}$, a unitary congruence preserving the unit diagonal and hence the spectrum $\lambda^{(a)}$; since $\varphi_a=2^{\mathrm H(\bar\lambda^{(a)})}/L_a$ depends only on that spectrum, it is unchanged. ∎

**Spectral form.** Because $\varphi_a$ depends only on the eigenvalue multiset of $R_a$, it is a similarity invariant of that *fixed* operator: any congruence $C\mapsto PCQ$ with $QP=1$ (in particular an orthogonal conjugation $P^{\mathsf\top}P=1$) leaves the characteristic polynomial, hence every spectral read, unchanged. This is what §13 certifies — an abstract matrix, no unit-diagonal or data hypothesis. The invariance the *data* carries is the sub-case preserving the unit diagonal of $R_a$: variable permutation and per-variable phase (Proposition 3.5), the group $S_{L_a}\ltimes U(1)^{L_a}$ (for real data the signed-permutation subgroup $B_{L_a}$), not the full orthogonal group. A rotation of the data mixes coordinates and re-normalises the correlation's diagonal, which is not a similarity of $R_a$, so $\varphi_a$ moves: the two axes carry distinct coordinates (a feature channel, a time step) and $\varphi_a$ reports how power distributes across *those*.

---

## 4. The decay, the optical transfer function, and the diffraction limit

The signal's own autocorrelation along the ordered axis is read as an optical transfer function [Goodman 2005].

### 4.1 The connected autocovariance

Let $x(t)\in\mathbb{C}^F$ be the $t$-th row of the field: $x(t)=W_{t,\cdot}$ for signed or complex $W$ (a coherent field read) or $x(t)=|W_{t,\cdot}|^2$ for nonnegative $W$ (an incoherent intensity read). Let $x_c(t)=x(t)-\frac1T\sum_s x(s)$ be the centred (connected) field, with $x_c(t)=0$ for $t\notin\{1,\dots,T\}$.

**Definition 4.1 (Decay / OTF)** — The pooled biased autocovariance is, for lags $\tau=0,\dots,T-1$,

$$
  C(\tau)=\frac1T\sum_{t=1}^{T-\tau}\operatorname{Re}\big\langle x_c(t),\,x_c(t+\tau)\big\rangle
        =\frac1T\sum_{t}\operatorname{Re}\!\sum_f \overline{x_{c,f}(t)}\,x_{c,f}(t+\tau),
$$

extended by $C(-\tau)=C(\tau)$.

The normalisation by $T$ (not $T-\tau$) is the *biased* estimator, and it is essential: it makes $C$ a positive-definite sequence (Lemma 4.2). By Wiener–Khinchin [Wiener 1930; Khinchin 1934], $C$ equals the inverse transform of the nonnegative periodogram $S(\omega)=\tfrac1T\sum_f|\hat x_{c,f}(\omega)|^2$; this identity ties the decay-entropy width to the spectral-width read of §4.3. As an aperture read $C$ is the point-spread / coherence function: by the Fourier-optics autocorrelation theorem the OTF is the autocorrelation of the pupil [Goodman 2005].

**Lemma 4.2 (The OTF is positive semidefinite)** — The $T\times T$ Toeplitz matrix $M$ with $M_{jk}=C(j-k)$ is positive semidefinite (PSD).

*Proof.* Take $W$ real (the complex case follows from the same periodogram argument, or by the real embedding). For $v\in\mathbb{R}^T$, writing $C=\sum_f r_f$ with $r_f(\tau)=\tfrac1T\sum_t x_{c,f}(t)x_{c,f}(t+\tau)$ the per-channel biased autocovariance,

$$
  v^{\mathsf\top} M v
  =\sum_{j,k}v_j v_k\,C(j-k)
  =\frac1T\sum_f\sum_{m\in\mathbb{Z}}\Big(\sum_j v_j\,x_{c,f}(m+j)\Big)^{\!2}\ \ge 0 .
$$

The middle equality is the convolution identity $\sum_m\big(\sum_j v_j x(m+j)\big)^2=T\sum_{j,k}v_jv_k\,r(j-k)$, valid because $x_{c,f}$ has finite support; summing over channels $f$ preserves the sign. ∎

**Corollary 4.3 (Peak at zero lag)** — $C(0)\ge|C(\tau)|$ for every $\tau$, and $C(0)=\tfrac1T\sum_t\|x_c(t)\|^2\ge 0$.

*Proof.* The $2\times2$ principal submatrix $\left(\begin{smallmatrix}C(0)&C(\tau)\\ C(\tau)&C(0)\end{smallmatrix}\right)$ of $M$ is PSD (Lemma 4.2), so its determinant $C(0)^2-C(\tau)^2\ge0$ and $C(0)\ge0$; hence $C(0)\ge|C(\tau)|$. The value of $C(0)$ is immediate from Definition 4.1. ∎

### 4.2 The entropy-width diffraction limit

Classically the resolution is the reciprocal of the OTF bandwidth, equivalently the reciprocal correlation length. We estimate that length by the entropy width of the decay, which is robust to noise in the tail.

**Definition 4.4 (Diffraction limit)** — Let $q(\tau)=C(\tau)^2/\sum_{\tau'}C(\tau')^2$ be the power-weighted lag distribution and $\mathrm{H}_C=\mathrm{H}(q)$ its entropy. The (primary) entropy-width limit $a_\delta$, the integral correlation length $\xi$, and the (secondary, classical) Abbe limit $1/\xi$ are

$$
  a_\delta=2^{-\mathrm{H}_C}\in[1/T,\,1],\qquad
  \xi=\sum_{\tau=0}^{\tau^\star-1}\frac{C(\tau)}{C(0)},\qquad
  a_\delta^{\mathrm{Abbe}}=1/\xi,
$$

where $\tau^\star$ is the first nonpositive lag of $C$ (its first zero crossing).

By the Abbe/Rayleigh criterion [Abbe 1873] the resolution is a length times a constant fixed by the aperture's *shape* (the "$1.22$" of a circular aperture is the first zero of $J_1$). Here the two length reads (the entropy width $a_\delta^{-1}=2^{\mathrm{H}_C}$ and the integral length $\xi$) differ by such a shape constant.

**Proposition 4.5 (Shape factor is scale-invariant)** — The Rayleigh shape factor $g=\xi\cdot a_\delta$ depends only on the *shape* of the decay profile, not its width: under an integer block-rescaling by $s\in\mathbb N$ (each lag replaced by $s$ copies, $C_s(s\tau+r)=C(\tau)$ for $0\le r<s$), the entropy width and the integral length scale reciprocally and $g$ is *exactly* unchanged.

*Proof.* Block-rescaling splits each mass $q(\tau)$ of the lag distribution into $s$ equal parts $q(\tau)/s$, so
$$
  \mathrm H(q_s)=-\sum_{\tau}\sum_{r=0}^{s-1}\frac{q(\tau)}s\log_2\frac{q(\tau)}s
  =-\sum_{\tau}q(\tau)\big(\log_2 q(\tau)-\log_2 s\big)=\mathrm H(q)+\log_2 s,
$$
hence $a_{\delta,s}=2^{-\mathrm H(q_s)}=a_\delta/s$. The first zero crossing moves to $s\tau^\star$, so $\xi_s=\sum_{\tau'=0}^{s\tau^\star-1}C_s(\tau')/C(0)=s\sum_{\tau=0}^{\tau^\star-1}C(\tau)/C(0)=s\,\xi$. Therefore $g_s=\xi_s\,a_{\delta,s}=(s\xi)(a_\delta/s)=g$. ∎

For an exponential decay the two length reads agree up to the profile's shape factor; for a white signal ($C(\tau)=0$, $\tau\ne0$) the lag distribution is a point mass and $a_\delta=1$ (critically sampled, nothing to resolve). Small $a_\delta$ means a long correlation length and a finely resolved aperture. The *Abbe factor* $a_\delta/\varphi_F$ divides the diffraction limit by the feature-axis fill $\varphi_F$ (§3.2): the resolution per unit aperture the feature axis carries, distinct from the dimensionless shape factor $g=\xi\,a_\delta$. The Fresnel number $N_F\sim(\text{window})\cdot\varphi_T$ places the read in the near ($N_F\gg1$) or far ($N_F\ll1$) field and is monotone in the probe scale.

### 4.3 The Mercer certificate

The same reciprocal length can be read a second, independent way, from the spectrum of the stationary correlation operator, and by Mercer's theorem [Mercer 1909] the two readings are tied: for a stationary, clean signal their ratio is constant, a built-in, model-free consistency check.

**Definition 4.6 (Spectral width and Mercer ratio)** — Let $\{\lambda_k\}$ be the eigenvalues of the Toeplitz operator $M$ of Lemma 4.2 (the stationary correlation operator built from $C$) and $n_{\mathrm{dof}}=2^{\mathrm{H}(\bar\lambda)}$ its effective mode count. The spectral limit and Mercer ratio are $a_\delta^{\mathrm{spec}}=n_{\mathrm{dof}}/T$ and $\rho=a_\delta^{\mathrm{spec}}/a_\delta$.

**Proposition 4.7 (Internal certificate)** — $M\succeq0$ (Lemma 4.2), so $\{\lambda_k\}\subset[0,\infty)$ and $n_{\mathrm{dof}},a_\delta^{\mathrm{spec}},\rho$ are well defined: the eigenvalues are nonnegative in exact arithmetic, and the implementation clips only floating-point round-off at zero. For a wide-sense-stationary process the temporal width $a_\delta$ and the spectral width $a_\delta^{\mathrm{spec}}$ are both functionals of the one spectral density, so $\rho$ is constant along the record; a drift of $\rho$ localises nonstationarity. Its value is fixed by the decay's shape (Proposition 4.5); the certificate is the constancy of $\rho$, demonstrated on stationary and regime-switching signals in §12.

---

## 5. Ordered-axis coherence: a closed-form permutation null

The coherence $z$-score tests for ordered structure against the row-permutation null in closed form, without sampling.

Let $S\in\mathbb{C}^{N\times F_{\mathrm{eff}}}$ be the projected screen (§8) with rows $s_1,\dots,s_N$, and $R=\big(\operatorname{Re}(SS^{\mathsf H})\big)^{\odot2}$ the elementwise square of the real Gram, so $R_{ab}=\operatorname{Re}\langle s_a,s_b\rangle^2\ge0$ and $R$ is symmetric. Fix a lag $\ell\ge1$.

**Definition 5.1 (Coherence statistic)** — $\displaystyle A=\frac1{N-\ell}\sum_{i=1}^{N-\ell}R_{i,\,i+\ell}$, the mean of the $\ell$-th superdiagonal of $R$.

Squaring the inner products makes $A$ invariant to a common row scale. Under the null the rows are exchangeable; we compare $A$ to the same statistic under a uniformly random row permutation $\pi$.

**Theorem 5.2 (Permutation-null mean)** — Let $\mu=\dfrac1{N(N-1)}\sum_{a\ne b}R_{ab}$ be the mean of $R$ over ordered off-diagonal pairs. Then $\mathbb{E}_\pi[A]=\mu$.

*Proof.* $A(\pi)=\frac1{N-\ell}\sum_i R_{\pi(i)\pi(i+\ell)}$. For each fixed $i$ and uniform $\pi$, the pair $(\pi(i),\pi(i+\ell))$ is uniform over the $N(N-1)$ ordered pairs of distinct indices (since $i\ne i+\ell$), so $\mathbb{E}_\pi[R_{\pi(i)\pi(i+\ell)}]=\mu$. Linearity over the $N-\ell$ terms gives $\mathbb{E}_\pi[A]=\mu$. ∎

**Definition 5.3 (Coherence $z$-score)** — Standardise $A$ by its permutation standard deviation,

$$
  z=\frac{A-\mu}{\sqrt{\operatorname{Var}_\pi[A]}} .
$$

The $M=N-\ell$ superdiagonal terms are not independent — at $\ell=1$ consecutive terms share a row — so $\operatorname{Var}_\pi[A]$ is not the naive $\varsigma^2/M$ (with $\varsigma^2=\operatorname{Var}_{a\ne b}(R_{ab})$ the off-diagonal population variance); it is the Cliff–Ord/Mantel second moment [Cliff & Ord 1981], assembled in closed form from the graph moments $S_1=\sum_{a\ne b}R_{ab}$, $S_2=\sum_{a\ne b}R_{ab}^2$, and $U=\sum_a\big(\sum_{b\ne a}R_{ab}\big)^2$. With $\mu_2=S_2/(N(N-1))$, the two-point expectations $E_{\mathrm{sh}}=(U-S_2)/(N(N-1)(N-2))$ for term-pairs that share one index and $E_{\mathrm{dis}}=(S_1^2-4U+2S_2)/(N(N-1)(N-2)(N-3))$ for disjoint pairs, and the counts $n_{\mathrm{sh}}=2\max(0,N-2\ell)$, $n_{\mathrm{dis}}=M(M-1)-n_{\mathrm{sh}}$,

$$
  \operatorname{Var}_\pi[A]=\frac1{M^2}\Big[M(\mu_2-\mu^2)+n_{\mathrm{sh}}(E_{\mathrm{sh}}-\mu^2)+n_{\mathrm{dis}}(E_{\mathrm{dis}}-\mu^2)\Big].
$$

**Remark 5.4 (Calibration)** — Theorem 5.2 fixes the null mean $\mu$ (§13) and Definition 5.3 the null variance in closed form, so $z$ has mean $0$ and unit variance under the permutation null at every shape at the operative lag $\ell=1$ (validated against brute-force permutation, §12); the closed form holds at any $\ell$ for which the disjoint-pair count is defined ($N\ge 2\ell+2$). The permutation distribution of $A$ is itself mildly right-skewed at small $N$, so the one-sided tail is only approximately normal — near the nominal rate, and approaching it as $N$ grows (§12); the approximation is in the tail *shape* alone, not in the standardisation.

---

## 6. The mode spectrum and a certified propagation constant

Read the feature-correlation eigenspectrum $\lambda_1\ge\lambda_2\ge\cdots\ge0$ of the $N$-sample, $F$-variable correlation matrix (aspect $\gamma=F/N$). The dominant mode's separation from the noise sea is an optical propagation constant.

**Definition 6.1 (Noise edge and propagation constant)** — The upper noise edge for the largest eigenvalue of a unit-diagonal correlation matrix is the derived finite-size Tracy–Widom edge (the §8 construction in correlation units, the Wishart edge divided by the sample count): with the Johnstone centre $\mu=(\sqrt{N-1}+\sqrt F)^2$ and scale $\varsigma_J=(\sqrt{N-1}+\sqrt F)\big(\tfrac1{\sqrt{N-1}}+\tfrac1{\sqrt F}\big)^{1/3}$ and the universal Tracy–Widom$_1$ quantile $q_\alpha$ at false-alarm rate $\alpha$ ($q_{0.05}=0.9793$), $\lambda_+=(\mu+q_\alpha\varsigma_J)/N$. It reduces to the asymptotic Marchenko–Pastur bulk edge $(1+\sqrt\gamma)^2$ as $N,F\to\infty$ [Marchenko & Pastur 1967; Jiang 2004] and, unlike the bare bulk edge, holds the top eigenvalue at finite size in the wide $F>N$ regime. Writing $r=\max(\lambda_2,\lambda_+)$, the dominant mode's propagation constant $\gamma_{\mathrm p}=\alpha_{\mathrm p}+i\beta_{\mathrm p}$ has

$$
  \alpha_{\mathrm p}=\log\frac{\lambda_1}{r}\ (\ge0),\qquad
  \beta_{\mathrm p}=\arg\!\Big(\textstyle\sum_{j\ge2} v_{1,j}\,\overline{v_{1,j-1}}\Big),
$$

with $v_1$ the dominant eigenvector; $\alpha_{\mathrm p}$ is the attenuation (a spectral separation) and $\beta_{\mathrm p}$ the per-step phase advance (a carrier frequency, well defined because it is invariant to the eigenvector's global phase). The *resolved-sector power* $\Pi=\sum_{\lambda_k>\lambda_+}(\lambda_k-\lambda_+)\ge0$ is the total eigenvalue excess above the bulk edge (the power the aperture resolves), and the *dominance* $(\lambda_1-1)/(F-1)\in[0,1]$ is the leading mode's excess over the mean eigenvalue ($1$ for a unit-diagonal correlation), normalised ($F$ variables, so $\lambda_1\in[1,F]$). For heavy-tailed or correlated spectra the caller can supply a deterministic data-derived edge, the Tukey upper fence [Tukey 1977] of $\{\lambda_k\}$, in place of $\lambda_+$.

**Lemma 6.2 (Weyl-certified attenuation)** — Suppose the read correlation matrix $\hat C$ differs from the truth $C$ by $\|\hat C-C\|_2\le\varepsilon$ (both Hermitian). Then, with read eigenvalues $\hat\lambda_1,\hat\lambda_2$ and $\hat\alpha_{\mathrm p}=\log(\hat\lambda_1/\max(\hat\lambda_2,\lambda_+))$,

$$
  \alpha_{\mathrm p}\in\Big[\ \log\frac{(\hat\lambda_1-\varepsilon)_+}{\max(\hat\lambda_2+\varepsilon,\ \lambda_+)},
  \ \ \log\frac{\hat\lambda_1+\varepsilon}{\max(\hat\lambda_2-\varepsilon,\ \lambda_+)}\ \Big],
$$

and a certified positive attenuation is the event that the lower endpoint exceeds $0$.

*Proof.* By Weyl's inequality [Weyl 1912] $|\hat\lambda_k-\lambda_k|\le\varepsilon$ for all $k$. The map $(\lambda_1,r)\mapsto\log\lambda_1-\log r$ is increasing in $\lambda_1$ and decreasing in $r$, and $r=\max(\lambda_2,\lambda_+)$ is nondecreasing in $\lambda_2$. Substituting the extreme admissible values of $\lambda_1,\lambda_2$ permitted by Weyl gives the interval. ∎

For empirical correlations from $N$ i.i.d. samples of an $F$-vector, matrix concentration [Vershynin 2018] supplies an admissible band $\varepsilon\lesssim\|C\|_2(\sqrt{F/N}+F/N)$, so Lemma 6.2 certifies how many samples make the read tight. The edge $\lambda_+$ is the finite-size Tracy–Widom edge of the correlation *eigenvalues*, a different object from the screen's singular-value noise floor (§8), which is the derived Tracy–Widom edge of the (row-scaled) screen's *singular values*; both are the same finite-size construction applied to their respective spectra.

---

## 7. Concentration: axial versus directional

For a stack of $M$ row-vectors $x_1,\dots,x_M\in\mathbb{C}^D$ (optionally unit-normalised), two notions of how concentrated the cloud is are distinct.

**Definition 7.1 (Concentration reads)** — With the second-moment matrix $\Sigma=\sum_i x_ix_i^{\mathsf H}$ and top eigenvalue $\sigma_1^2$,

$$
  \text{intensity}=\sigma_1^2,\qquad
  \text{focus}=\frac{\sigma_1^2}{M},\qquad
  \text{resultant}=\Big\lVert\tfrac1M\textstyle\sum_i x_i\Big\rVert .
$$

**Proposition 7.2 (Axial $\ne$ directional)** — For unit rows, $\text{focus}\in(0,1]$ is invariant under any per-row phase $x_i\mapsto e^{i\theta_i}x_i$, whereas the resultant is not. Hence the two measure distinct quantities: an antipodal cloud $\{u,\dots,u,-u,\dots,-u\}$ (equal halves) has $\text{focus}=1$ but $\text{resultant}=0$.

*Proof.* $(e^{i\theta_i}x_i)(e^{i\theta_i}x_i)^{\mathsf H}=x_ix_i^{\mathsf H}$, so $\Sigma$ and hence $\text{focus}=\sigma_1^2/M$ are unchanged by per-row phases; but the mean $\frac1M\sum_i e^{i\theta_i}x_i$ is not. For the antipodal cloud $\Sigma=\sum_i x_ix_i^{\mathsf H}=M\,uu^{\mathsf H}$ gives $\sigma_1^2=M$ (unit rows) and $\text{focus}=\sigma_1^2/M=1$, while the halves cancel and the resultant is $0$. ∎

Focus is *axial* concentration (power on the leading principal axis, blind to sign); the resultant is *directional* (the von Mises–Fisher sufficient statistic [Mardia & Jupp 2000]); the construction reports both.

---

## 8. The projection screen

The reads above are information *about* the structure. The screen holds the information *within* it: the signal folded onto its own entropy-matched grid.

**Definition 8.1 (Whiten, then project)** — *Whitening* rescales each feature channel to a common robust median absolute deviation (MAD) noise scale at native resolution, each channel's scale shrunk toward the pooled cross-channel scale by a data-derived James–Stein weight [James & Stein 1961], stabilising small-sample per-channel estimates while equalising genuinely different channels; masked or nonfinite cells are marked missing. A fully-dead row or column (every cell missing) carries no information and is *dropped* from the read; missing data is ignored, so the singular value decomposition (SVD) reads only observed cells. *Projection* then folds the feature axis to the matched grid $(N,F_{\mathrm{eff}})=\big(T,\ \operatorname{round}(F/\delta_F)\big)$ by an area-weighted fold that excludes the remaining scattered missing cells; the ordered axis is kept at native resolution ($\delta_T=1$, §2), so $N=T$.

The projected screen $S$ has singular values $s_1\ge s_2\ge\cdots$; the count above the noise floor is the resolved signal dimension.

**Definition 8.2 (Noise floor and resolved dimension)** — The largest eigenvalue $s_1^2$ of an $N\times F_{\mathrm{eff}}$ noise screen concentrates at the Johnstone centre $\mu=(\sqrt{N-1}+\sqrt{F_{\mathrm{eff}}})^2$ with fluctuation scale $\varsigma_J=(\sqrt{N-1}+\sqrt{F_{\mathrm{eff}}})\big(\tfrac1{\sqrt{N-1}}+\tfrac1{\sqrt{F_{\mathrm{eff}}}}\big)^{1/3}$, and $(s_1^2-\mu)/\varsigma_J$ follows the real ($\beta=1$) Tracy–Widom law [Tracy & Widom 1996; Johnstone 2001] (the Airy-kernel $\beta=2$ origin is [Tracy & Widom 1994]). The per-cell noise variance is read robustly from the median row energy, de-biased for the two finite-size effects the amplification $\mu/\varsigma_J$ (large at *either* aspect extreme) magnifies into a false-alarm shift:

$$
  \hat\sigma^2=\frac{\operatorname{median}_t\|S_{t,\cdot}\|^2}{F_{\mathrm{eff}}\;c_{F_{\mathrm{eff}}}\;(N-1)/N},
  \qquad c_F=\frac{\operatorname{median}(\chi^2_F)}{F}=\Big(1-\tfrac{2}{9F}\Big)^3 ,
$$

where $c_F$ corrects the median-versus-mean gap of the per-row energy $\|S_{t,\cdot}\|^2\sim\chi^2_{F_{\mathrm{eff}}}$ (a small-$F$ bias) and $(N-1)/N$ is the degree-of-freedom deflation from the per-channel centring of Def 8.1 (a small-$N$ bias). Both are *derived*: $c_F$ is the Wilson–Hilferty $\chi^2$ median [Wilson & Hilferty 1931], and $(N-1)/N$ is the mean-centring factor, applied here as a conservative correction for the median centring the screen actually uses (whose true deflation is marginally smaller, so the floor is slightly conservative at small $N$, in the safe direction). With the *universal* Tracy–Widom$_1$ quantile $q_\alpha$ at false-alarm rate $\alpha$ ($q_{0.05}=0.9793$), the singular-value floor and resolved dimension are

$$
  \Phi=\sqrt{\hat\sigma^2\,\big(\mu+q_\alpha\,\varsigma_J\big)},
  \qquad K_{\mathrm{signal}}=\#\{k:s_k>\Phi\}.
$$

Given the false-alarm level $\alpha$, the floor is fixed by the shape $(N,F_{\mathrm{eff}})$: $\mu,\varsigma_J,c_F,(N-1)/N$ depend on it alone and $q_\alpha$ is a universal Tracy–Widom constant. The read is the per-mode evidence — the tail probability $p_k=P\big(\mathrm{TW}_1>(s_k^2/\hat\sigma^2-\mu)/\varsigma_J\big)$ of each singular value against the null — and $\alpha$ selects only the count $K_{\mathrm{signal}}=\#\{k:p_k<\alpha\}$; it is the reader's operating point, external by the Neyman–Pearson lemma (§11.1), and the measured false-alarm rate holds near $\alpha$ across aspect ratios from $N\gg F$ to $N\ll F$ (§12).

The law is the null of the largest eigenvalue under an i.i.d.-Gaussian bulk, so the floor is conditional on that null: noise that is *correlated across channels* (common-mode drift, narrowband interference) or *heavy-tailed* concentrates its variance into a few modes and is, on the singular spectrum alone, indistinguishable from signal, so the floor counts it as resolved. Separating structured noise from signal there needs a noise reference (prewhiten from a signal-free window) or the mode *shape* read of Def 8.3, rather than the scalar floor. Because the correct null is a modelling choice the library cannot make for every substrate, the floor is a **null provider**: a callback returning the threshold for one screen, evaluated *locally* on each block (per plane, per window, or streaming). The library ships the derived default ($\Phi$ above) and a small closed-form set; a caller can pass their own — a permutation surrogate (§5's philosophy: preserve each channel's marginal, destroy the cross-channel correlation), a signal-free reference, or a substrate-specific null. The provider owns both the threshold and the $\alpha$ it is drawn at (the derived edge serves an arbitrary $\alpha$ by inverting the Tracy–Widom survival function), so $\alpha$ and the null are the two *declared* external inputs the instrument cannot derive.

**Definition 8.3 (Mode footprint)** — Each of the $K_{\mathrm{signal}}$ resolved modes carries a left (ordered) singular vector $u_k$ and a right (feature) singular vector $v_k$. Its *footprint* is the pair of fill fractions of those vectors together with their product,

$$
  \varphi^{(k)}_T=\frac{2^{\mathrm H(|u_k|^2)}}{N},\qquad
  \varphi^{(k)}_F=\frac{2^{\mathrm H(|v_k|^2)}}{F_{\mathrm{eff}}},\qquad
  \mathcal E_k=\varphi^{(k)}_T\,\varphi^{(k)}_F ,
$$

the étendue of §3 resolved *per mode*: the phase-space area the mode occupies. The footprint separates modes that share a singular value — a broadband transient ($\varphi^{(k)}_F\to1$ with $\varphi^{(k)}_T$ small), narrowband persistent interference ($\varphi^{(k)}_F$ small with $\varphi^{(k)}_T\to1$), a compact blob (both small) — reading that shape directly from the same entropic fill as the axis reads, where the scalar floor sees only $s_k$. The floor decides *whether* a mode stands above the noise; the footprint reads *what shape* it has, where the labelling of structured noise versus signal (Def 8.2) is made.

**Definition 8.4 (The read-side filter)** — The footprint labels each mode; the filter acts on that label to recover the signal itself. `extract` returns the field reconstructed from its resolved modes alone,

$$
  \widehat S = U\,\operatorname{diag}(\tilde s)\,V^{\mathsf H},
$$

with $U,V$ the screen's own left and right singular vectors and $\tilde s$ the shrunk singular spectrum. Because $U,V$ are read from the data, $\widehat S$ is a two-sided orthogonal *projection* of the measured screen onto its resolved modes: it synthesises nothing — a planted burst is recovered to machine precision in the noise-free limit and the map is idempotent (§12). The shrinkage $\tilde s$ is the Frobenius-optimal (minimum-mean-squared-error) singular-value nonlinearity against the derived floor [Gavish & Donoho 2017]: modes at or below the bulk edge map to $0$ and the survivors are de-biased toward it. The transient/persistent dichotomy of Def 8.3 then gates the reconstruction — a persistent narrowband mode ($\varphi^{(k)}_F\le\varphi^{(k)}_T$, the signature of channelised interference) is dropped, a broadband transient ($\varphi^{(k)}_F>\varphi^{(k)}_T$) kept — so the filter removes noise and RFI modes while preserving the burst morphology, with no template and no tuning.

The screen is invertible: upsampling both axes back to native resolution reconstructs the waterfall (the inverse fold), and a delay-embedded Tucker (higher-order SVD, HOSVD) [Tucker 1966; De Lathauwer et al. 2000] of the whitened native-resolution field, computed by randomised range-finding [Halko et al. 2011], exposes the within-window fine structure the averaged screen discards.

**N-D fields: the geometry-preserving reduction.** The screen fold (Definition 8.1) and every read above act on a 2-D $W\in\mathbb{C}^{T\times F}$. A higher-dimensional field (a video $T\times H\times W$, a spatial volume, a multichannel record) is first reduced to two axes, and *which* reduction is correct depends on the read: a feature read needs the within-plane correlation that a single flattened feature axis discards.

**Definition 8.5 (Pool versus plane-fold)** — Fix the ordered axis of an N-D field. *Pool* moves it first and flattens every other axis into the feature axis, so each off-axis site is an exchangeable sample of the one ordered process: the correct reduction for the ORDERED reads (the decay $C(\tau)$ and diffraction limit $a_\delta$ of §4, the rates of §9, and the ordered fill $\varphi_T$). *Plane-fold* applies a scalar 2-D read to each intact $(a,b)$ plane, iterating the remaining axes and averaging, so each plane is its own screen and within-plane correlation is preserved: the correct reduction for the FEATURE and plane reads (the feature fill $\varphi_F$ of §3, the mode spectrum of §6, the concentration of §7, and the resolved dimension $K_{\mathrm{signal}}$).

The two reductions are duals: pooling treats the off-axis structure as exchangeable samples of the ordered process, while plane-fold treats each plane's own structure as the object of the read, and getting them backwards changes the answer. The choice is therefore fixed by the read. Both reductions are backend-agnostic.

---

## 9. The dynamical operator: per-mode decay rates

The entropy-width $a_\delta$ estimates a single correlation length; the ordered axis is a state trajectory, and its one-step propagator gives the per-mode rates. We estimate the propagator recursively from the first frame (online DMD / Koopman [Tu et al. 2014]).

**Definition 9.1 (Streaming accumulators and reduced propagator)** — For states $x_1,\dots,x_T\in\mathbb{C}^F$ and forgetting $\lambda\in(0,1]$ maintain $P_{xx}=\sum_t \lambda^{\,\cdot}\,x_t x_t^{\mathsf H}$ and $P_{yx}=\sum_t \lambda^{\,\cdot}\,x_{t+1} x_t^{\mathsf H}$. The one-step propagator is $A=P_{yx}P_{xx}^{+}$, and $x\mapsto Ax$ is the one-step forecast. In the leading proper orthogonal decomposition (POD) subspace $V_r$ (top eigenvectors of $P_{xx}$ with eigenvalues $w_r$), the reduced propagator is $\tilde A=(V_r^{\mathsf H} P_{yx}V_r)\operatorname{diag}(w_r)^{-1}$, and

$$
  \alpha_k=-\log|\mu_k|,\qquad \beta_k=\arg\mu_k,\qquad \{\mu_k\}=\operatorname{eig}(\tilde A),
$$

with long- and short-range rates $\min_k\alpha_k$ and $\max_k\alpha_k$. The forgetting $\lambda$ is the sole optional input of the construction: $\lambda=1$ (no forgetting) is the stationary default used throughout, under which the accumulators are additive (Theorem 9.3) and the recovery is exact for noise-free linear dynamics (Theorem 9.2); $\lambda<1$ is an explicit choice of memory horizon for a nonstationary stream, a stated modelling input rather than a tuned constant, and it enters no other read (§11.1).

**Well-posed truncation.** The reduced rank $r$ is the numerical rank of $P_{xx}$ when the stream over-determines it ($n_{\mathrm{pairs}}\ge 2F$), and the resolved signal dimension (§8) when it does not. In the under-sampled regime ($T<F$, e.g. a short cutout), truncating to the resolved rank keeps the fit over-determined, so the recovered spectrum — and the forgetting margin read from it — stays well-posed. The resolved rank is a *detection* decision (which feature modes are signal), so its operating point is the caller's declared $(\alpha,\text{null})$ — the same null-provider contract as §8 — defaulting to the derived edge at $\alpha=0.05$. On a well-sampled record the truncation stays inactive, and the exact recovery of Theorem 9.2 holds.

**The operator carries the reads.** The accumulators $(P_{xx},P_{yx})$ are a fixed $F\times F$ sufficient statistic for a stream of any length, so the reads are functions of them and require no stored history. Three are read straight off the operator, each incremental (O($F^2$) per frame, O($F^3$) per read): the **forgetting margin** $\max_k|\mu_k|$ (below $1$ iff the screen forgets: a spectral margin below one forces every weight-carrying mode to decay, so the finite exponential sum $C(\tau)$ has no persistent term); the **feature spectrum** as the unit-diagonal correlation eigenvalues of $P_{xx}$, whose count above the derived floor (§8, cut point "bulk") is the resolved dimension $K_{\mathrm{signal}}$ and whose per-mode Tracy–Widom tail probabilities $p_k$ are the streaming form of the evidence of §8; and the **decay** $C(\tau)=\sum_k P_k\mu_k^{\tau}$ reconstructed from the spectrum, extrapolatable to any lag. A wide low-rank feature axis is read in O($F^2 k$) by a randomized range-finder [Halko et al. 2011] for the top-$k$ modes. Every operation is a matmul or a symmetric eigendecomposition and runs identically on CPU (numpy) or GPU (torch, on-device).

**Theorem 9.2 (Exact recovery)** — If $x_{t+1}=Ax_t$ for all $t$ (noise-free linear dynamics), then $P_{yx}P_{xx}^{+}=A$ on $\operatorname{span}\{x_t\}$, and the recovered eigenvalues are those of $A$ restricted to that subspace; hence $\alpha_k=-\log|\mu_k|$ and $\beta_k=\arg\mu_k$ are exact.

*Proof.* $P_{yx}=\sum_t x_{t+1}x_t^{\mathsf H}=\sum_t (Ax_t)x_t^{\mathsf H}=A\sum_t x_tx_t^{\mathsf H}=A\,P_{xx}$. Therefore $P_{yx}P_{xx}^{+}=A\,P_{xx}P_{xx}^{+}=A\,\Pi$, where $\Pi=P_{xx}P_{xx}^{+}$ is the orthogonal projector onto $\operatorname{span}\{x_t\}=\operatorname{range}(P_{xx})$. On that subspace $\Pi$ is the identity, so the operator equals $A$ there and shares its spectrum on it. ∎

**Theorem 9.3 (Additivity and exact splicing)** — At $\lambda=1$ the accumulators are additive over any partition of the stream: for a split at index $m$,

$$
  P_{xx}^{[1,T]}=P_{xx}^{[1,m]}+P_{xx}^{[m,T]},\qquad
  P_{yx}^{[1,T]}=P_{yx}^{[1,m]}+P_{yx}^{[m,T]},
$$

provided the boundary transition $(x_m,x_{m+1})$ is counted once. Consequently the accumulators of a concatenated stream are the sums of the segment accumulators (plus the boundary pair); the operator $A=P_{yx}P_{xx}^{+}$ and every read are fixed functions of them, so splicing and the export/import of $(P_{xx},P_{yx})$ resume a stream bit-for-bit.

*Proof.* Each accumulator is a sum over the one-step transitions $t\mapsto t+1$. The $T-1$ transitions of $[1,T]$ are the disjoint union of the $m-1$ transitions of $[1,m]$ (namely $1\mapsto2,\dots,(m-1)\mapsto m$) and the $T-m$ transitions of $[m,T]$ (namely $m\mapsto m+1,\dots,(T-1)\mapsto T$), with the boundary transition $m\mapsto m+1$ belonging to the second segment and so counted once. Summation splits accordingly. The pseudoinverse and eigendecomposition are functions of the accumulators alone, so equal accumulators yield an identical operator. ∎

The recovered $\alpha_k=-\log|\mu_k|,\ \beta_k=\arg\mu_k$ are the spectrum of the fitted propagator; Theorem 9.2 gives exact recovery of the generating rates when the ordered axis is a linear trajectory. The full complex spectrum $\{\mu_k\}$ reconstructs and extrapolates the decay as $C(\tau)=\sum_k P_k\,\mu_k^{\tau}$, so a short observation yields the long-range values, and Theorem 9.3 lets long streams and multi-session records splice exactly.

**Definition 9.4 (The forgetting aperture: adaptive locality)** — The construction is streaming-first: the operator $(P_{xx},P_{yx})$ accumulates over the *whole* stream and carries the global reads (rates, forgetting margin, feature spectrum, decay), while the frame-level reads that need the raw samples (the ordered fill $\varphi_T$, the screen snapshot) act on a *local* window of recent frames whose length is set by the signal, not a clock. With margin $m=\max_k|\mu_k|$, a mode of magnitude $m$ has decayed to a fraction $\varepsilon$ by the lag $\ell(m)=\lceil\ln(1/\varepsilon)/(-\ln m)\rceil$ (its correlation length), and the window retains

$$
  w=\begin{cases}\max\{w_{\min},\ \ell(m)\} & K_{\mathrm{signal}}\ge 1,\\[2pt] w_{\min} & K_{\mathrm{signal}}=0,\end{cases}
  \qquad w\to\infty\ \text{as}\ m\to 1 .
$$

A minimum $w_{\min}$ (default $128$) is always kept, more while a mode is still coherent, and a *persistent active mode is retained in full*; structureless noise forgets to $w_{\min}$. This is the forgetting axiom as a runtime policy — the memory horizon is the signal's own correlation length, read incrementally off the operator. The two constants $w_{\min},\varepsilon$ are LOCALITY choices: they bound the frame-level window (rates, margin, $K_{\mathrm{signal}}$, decay are computed from all frames, independent of $w_{\min},\varepsilon$).

**Proposition 9.5 (Complexity)** — Let a stream have $T$ frames of $F$ features, resolved dimension $k=K_{\mathrm{signal}}$, and coherence window $w$ (Def 9.4). Then:

| stage | cost | note |
|---|---|---|
| accumulate, per frame | $O(F^2)$ | one rank-one outer product into $(P_{xx},P_{yx})$ |
| accumulate, whole stream | $O(T\,F^2)$ | **linear in $T$** (a block is one matmul, same total) |
| read (rates / margin / spectrum / decay) | $O(F^3)$ | one symmetric eigendecomposition, **independent of $T$** |
| read, top-$k$ only | $O(F^2 k)$ | randomized range-finder [Halko et al. 2011] |
| frame-level ordered read ($\varphi_T$, screen) | $O(w^2)$ | bounded by the coherence window |
| memory | $O(F^2 + wF)$ | fixed statistic + window, **independent of $T$** |

*Proof.* The accumulator update $P\mapsto \lambda P + x\,x^{\mathsf H}$ is a rank-one outer product, $O(F^2)$; summed over $T$ frames (equivalently one $F\times T$ by $T\times F$ matmul) it is $O(TF^2)$, and the state is the two $F\times F$ matrices, $O(F^2)$. Every operator read is a function of $(P_{xx},P_{yx})$ alone (Thm 9.3), obtained by a single $F\times F$ symmetric eigendecomposition ($O(F^3)$, or $O(F^2k)$ restricted to the top-$k$ range), with no dependence on $T$. The only ordered-axis (length-wise) work is the frame-level reads, which act on the length-$w$ window, $O(w^2)$. ∎

The ordered axis is read through the operator, whose per-read cost is fixed at $O(F^3)$ and does not grow with the stream: the optics are recovered from a fixed-size sufficient statistic in one online pass, $O(TF^2)$ time and $O(F^2+wF)$ memory. Missing data (RFI, dropped cells) contributes nothing to the accumulators, so a coherent signal on the valid cells is still resolved — *sparsity preserves coherence* — and the coherence estimate degrades gracefully with the fraction observed (the Cramér–Rao floor on the rates).

---

## 10. The optics dictionary

Every read maps to a standard optical quantity, and each governing step instantiates a named theorem; the governing mathematical facts are, for the load-bearing reads, certified (§13, with the optical naming itself a definitional dictionary rather than a theorem). $W$ is *read as* a wave field through a finite aperture whose two axes are the aperture's two coordinates, whose autocorrelation is the point-spread function, and whose diffraction limit is its finest resolvable scale.

| read | symbol | optical quantity | governing theorem |
|---|---|---|---|
| axis fill $\varphi_F,\varphi_T$ | $2^{\mathrm H_a}/L_a$ | aperture fill on each axis | Whittaker–Shannon sampling; Slepian–Pollak–Landau concentration |
| screen fill $\varphi$ | $2^{\mathrm H_{\mathrm{sv}}}/n$ | fraction of active SVD modes | band-limit $\Rightarrow$ finite active modes |
| magnification | $1/\varphi$ | reciprocal reach / oversampling | fill–reach duality ($\varphi\,\delta=1$) |
| étendue $\mathcal E$ | $\varphi_F\varphi_T$ | joint 2-D aperture area | étendue (Lagrange / Smith–Helmholtz invariant) |
| space–bandwidth | $n_F\,n_T$ | resolvable-spot count | degrees-of-freedom invariance (Lukosz) |
| Strehl $\mathcal S$ | $\lambda_1/\!\sum\lambda$ | dominant coherent-mode fraction | coherent-mode decomposition (Wolf; Mercer) |
| ordered coherence $z$ | $(A-\mu)/\sqrt{\operatorname{Var}_\pi[A]}$ | ordered-axis coherence | permutation null (Cliff–Ord) |
| decay $C(\tau)$ | biased autocovariance | OTF / point-spread | Wiener–Khinchin; autocorrelation theorem |
| diffraction limit $a_\delta$ | $2^{-\mathrm H(C^2)}$ | minimum resolvable spacing | Abbe / Rayleigh |
| shape factor $g$ | $\xi\,a_\delta$ | dimensionless shape factor (Rayleigh) | scale-invariance (Prop 4.5) |
| Abbe factor | $a_\delta/\varphi_F$ | resolution per feature fill | Abbe / Rayleigh |
| Fresnel number | $\sim\!\text{win}\cdot\varphi_T$ | near/far-field coordinate | Fresnel scaling |
| Mercer ratio $\rho$ | $a_\delta^{\mathrm{spec}}/a_\delta$ | temporal–vs–spectral width | Mercer's theorem |
| propagation constant | $\alpha_{\mathrm p}+i\beta_{\mathrm p}$ | mode contrast + carrier | Tracy–Widom finite-size edge (Marchenko–Pastur / Jiang limit); Weyl |
| resolved power $\Pi$; dominance | $\sum_{\lambda_k>\lambda_+}\!(\lambda_k-\lambda_+)$; $\tfrac{\lambda_1-1}{F-1}$ | resolved spectral power; mode dominance | Tracy–Widom finite-size edge |
| concentration | $\sigma_1^2/M$; $\lVert\bar x\rVert$ | axial focus vs directional resultant | von Mises–Fisher (axial $\ne$ directional) |
| decay rates | $-\log\lvert\mu_k\rvert,\ \arg\mu_k$ | per-mode attenuation + frequency | exact DMD / Koopman |
| clean field | $U\operatorname{diag}(\tilde s)V^{\mathsf H}$ | resolved-mode projection (denoise) | optimal singular-value shrinkage (Gavish–Donoho) |

**The limit is over-determined.** Independent reads each pin a strictly positive resolution limit for a filled finite aperture, from *distinct* objects: the decay-lag entropy gives $a_\delta$ (Abbe/Rayleigh); the stationary correlation operator's spectrum gives $a_\delta^{\mathrm{spec}}$ (Mercer, §4.3), which agrees with $a_\delta$; and the power marginals give a finite space–bandwidth product $\mathrm{SBW}$. These draw on different entropies (the decay-lag entropy $\mathrm H_C$, the operator spectrum, and the marginal entropies $\mathrm H_T,\mathrm H_F$), so their agreement is a cross-check.

---

## 11. Reproducibility

The construction is realised as a small Python package with a single backend-agnostic code path: every kernel is written once against the array namespace of its input, so a numpy array runs on CPU and a torch tensor runs on its device (GPU), staying on-device; torch is imported lazily only when a tensor is present. All reads are deterministic (the coherence is the closed-form $z$-score of §5), so a numpy and a torch evaluation agree to floating-point round-off and repeated evaluations are bit-identical. Complex inputs are supported throughout, with the coherent/incoherent dispatch of §4.

A test suite pins the identities of this paper: the full optics read as a regression contract; numpy/torch parity; the invariants $\mathcal E=\varphi_F\varphi_T$, $\operatorname{mag}=1/\varphi$, the peak-at-zero-lag and PSD-autocovariance of §4, the permutation-null mean and variance of the coherence $z$ (§5, against a full enumeration of all $N!$ permutations), the rate recovery of Theorem 9.2, the additive splice of Theorem 9.3, the axial/directional dichotomy of Proposition 7.2, and the projection and idempotence of the read-side filter (§8, Def 8.4); the round-trips (screen reconstruction, tensor HOSVD, factor serialisation); and degenerate-input robustness. The package, its test and validation suites, and the Lean development are available at https://github.com/Agience/agience-entroptics.

### 11.1 Provenance of every constant and choice

The construction is parameter-free in a precise, verifiable sense, tabulated here. Every fixed number it uses is a *derived* mathematical quantity; every structural choice is the canonical realisation of a *stated requirement*; the only quantity the instrument does not itself supply is the reader's decision risk.

| Quantity | Role | Kind | Provenance |
|---|---|---|---|
| $1/\Phi_{\mathcal N}^{-1}(3/4)$ | MAD $\to$ robust $\sigma$ | derived | Gaussian consistency of the MAD ($\Phi_{\mathcal N}$ the standard-normal CDF, distinct from the floor $\Phi$ of §8) |
| $\mathrm{Var}(\log\widehat{\mathrm{MAD}})\!\cdot\!N\!=\!1.360$ | whitening shrinkage weight | derived | influence-function variance $1/(16 f(D)^2 D^2)$ |
| $c_F=(1-\tfrac{2}{9F})^3$ | noise-level de-bias | derived | Wilson–Hilferty $\chi^2_F$ median |
| $(N-1)/N$ | noise-level de-bias | derived | centring degree of freedom |
| $q_\alpha$ (Tracy–Widom$_1$) | floor threshold shape | derived | universal $\mathrm{TW}_1$ quantile |
| $\beta_F=\min\!\big(\tfrac{F-1}{2T\ln 2},\ \tfrac12\log_2 F\big)$ | feature-fold guard | derived + cap | conservative uniform-null deficit bound, capped for non-vacuity (Def 2.2) |
| biased autocovariance ($/T$) | OTF estimator | forced | positive-semidefiniteness of the OTF (Lemma 4.2) |
| feature-only fold | resolution | forced | the ordered reads (§4, §5, §9) are lag/shift statistics needing native spacing |
| MAD whitening + shrinkage | noise scale | forced | maximal breakdown-point robust scale; empirical-Bayes weight |
| $|W|^2$ power marginal | matched-scale marginal | definitional | the intensity (the incoherent read of §4) |
| $2^{\mathrm H}$ effective count | fill fraction | definitional | Shannon entropy $\to$ participation number |
| $\alpha$ (false-alarm level) | decision risk | **external input** | the reader's operating point (Neyman–Pearson); enters no read, only the count |
| $\lambda$ (DMD forgetting) | memory horizon | **optional input** | $\lambda=1$ is the stationary default (Theorem 9.2); $\lambda<1$ is an explicit nonstationarity choice |
| $w_{\min},\varepsilon$ | frame-window extent | locality | bound only the frame-level window (§9.4); enter no global operator read |
| numerical-rank tol. | rank of $P_{xx}$ | implementation | drops null directions below $10^{-10}$ of the top energy |
| DMD signal-rank cut ($\alpha$, null) | well-posed truncation at $T<F$ | declared input | the §8 floor at the caller's operating point (default derived edge at $\alpha=0.05$); acts only in the under-sampled regime |
| $K=\lfloor\sqrt{N F_{\mathrm{eff}}}\rfloor$ | SVD-embedding size | implementation | geometric-mean stability size; sizes the embedding output only, enters no read or count |

(i) No constant is fitted to data or calibrated to a substrate. (ii) Every constant is *derived*: a robustness consistency factor, an influence-function variance, a $\chi^2$ median, a universal Tracy–Widom quantile. (iii) Every structural choice is *forced* by a requirement, the biased autocovariance by the positivity of the OTF, the feature-only fold by the native lag spacing the ordered reads require, the robust whitening by a maximal breakdown point. (iv) The sole *decision* input external to the instrument is the false-alarm level $\alpha$, the operating point at which a continuous read is called a detection. By the Neyman–Pearson lemma $\alpha$ cannot be derived from the data, since it encodes the reader's relative cost of a false alarm against a miss; so it enters no read. The reads expose, per mode, the standardised deviate $g_k$ and its tail probability $p_k=P(\mathrm{TW}_1>g_k)$ against the universal null (the evidence, §8); $\alpha$ selects only how many modes a reader calls resolved, $K_{\mathrm{signal}}=\#\{k:p_k<\alpha\}$. (v) Beyond $\alpha$ and the optional forgetting $\lambda$, the table lists a few *locality and implementation* constants — the frame-window extent $w_{\min},\varepsilon$, a numerical-rank tolerance, the SVD-embedding size, and the $\tfrac12\log_2 F$ fold cap; each bounds a window, drops null directions, sizes an output, or keeps the fold non-vacuous, none is fitted to data or a substrate, and none enters a read value.

The finite-size Tracy–Widom null of §8 and the stationarity behind the rates of §9 are the assumptions the classical results being specialised already make; the construction introduces none beyond them.

---

## 12. Empirical validation

Every read is checked against a planted ground truth on seeded synthetic signals (`research/validation/`, deterministic, regenerated by `run_all.py`). Each experiment recovers the quantity it claims to.

*Diffraction limit tracks correlation length.* For first-order autoregressive AR(1) fields of known correlation length $\rho$ (48 channels, $T=4000$), the entropy-width limit $a_\delta$ and the integral length $\xi$ are strictly monotone in $1/\rho$ (Spearman $1.000$); $a_\delta\sim 1/\rho$ with log-log slope $0.994$ ($R^2=0.9992$), and $\xi$ recovers $\rho$ to a constant shape factor.

*Decay-rate recovery (Theorem 9.2).* For a linear trajectory $x_{t+1}=Ax_t$ with known eigenvalues, `rates()` recovers $\alpha_k,\beta_k$ to machine precision at zero noise (max $|\alpha\text{ error}|=1.5\times10^{-15}$), degrading smoothly as the signal-to-noise ratio falls ($2\times10^{-2}$ at $20$ dB).

*Resolved dimension recovers planted rank.* With $K\in\{0,1,3,5\}$ orthogonal modes planted at signal-to-edge ratio $\ge 1$ across four aspect ratios (including $600\times40$ and $40\times600$), $K_{\mathrm{signal}}$ recovers the exact count with mean accuracy $1.000$ at ratio $1$ and $0.994$ at ratio $2$, at a $K=0$ specificity of $0.95$ overall. At the highest tested ratio ($4$) the tall $600\times40$ array over-resolves — a planted $K=1$ reads mean $K_{\mathrm{signal}}=2.3$ — the finite-size floor under-estimating the edge for a strongly super-critical mode at that extreme aspect; the calibrated operating range is ratio $1$–$2$, where the derived floor is false-alarm calibrated across aspect ratios, tall to wide.

*Coherence separates order from noise and is null-calibrated.* A smooth ordered signal reads $z=26.1$; its own row permutation reads $z=-0.16$. Over $2000$ i.i.d.-noise draws across five shapes the closed-form-variance null $z$ (Def 5.3) has mean $0.01$ and standard deviation $1.00$; the one-sided rate $P(z>2)=0.027$ sits near the $\mathcal N(0,1)$ target $0.023$, the small residual being the permutation distribution's own tail skew, not the standardisation.

*The Mercer ratio flags nonstationarity.* Across sliding windows $\rho$ is near-constant for a stationary AR(1) record (mean coefficient of variation $0.06$ over $12$ seeds) and drifts markedly more at a regime switch (mean CV $0.58$), a $9\times$ separation.

*Étendue and space–bandwidth read the aperture size.* Both rise strictly monotonically (Spearman $1.000$) with a signal's planted effective rank and feature bandwidth; $\varphi_F$ tracks the occupied fraction $R/F$ and $n_F$ recovers the bandwidth $B$.

*The read-side filter recovers the signal and synthesises nothing (Def 8.4).* On a planted broadband burst the filter's output is a two-sided projection of the data onto its own resolved modes: at zero noise it recovers the input to machine precision (relative error $<10^{-12}$) and is idempotent (a true projection); under observation noise its fidelity rises monotonically with signal-to-noise and beats the raw field wherever noise is non-trivial (image correlation $>0.95$ at S/N $=10$), and its recovery of the surviving channels holds above $97\%$ through roughly $60\%$ random channel dropout before the read collapses. A persistent modulated narrowband tone added alongside the burst is dropped by the $\varphi_F>\varphi_T$ geometry cut (correlation with the tone $<0.2$) while the burst is preserved (correlation $>0.9$). These checks are pinned in `test_extract.py`; the recovery-versus-dropout curve behind Figure 1 is committed alongside it (`calibration.csv`).

![Figure 1. The read-side filter (Def 8.4) calibrated on a synthetic burst: the injected burst, the same field plus noise, and the recovery (top); recovery versus the fraction of channels dropped, the field plus noise and random channel dropout, and the recovery of the surviving channels (bottom).](figures/calibration.png)

### 12.1 Real data: fast radio bursts

The synthetic checks plant a known truth; a real substrate tests whether the *same untuned instrument* transfers. We apply the plain library (zero tuning) to the publicly released CHIME/FRB Catalog-1 waterfalls [CHIME/FRB Collaboration 2021; data at the CANFAR archive, CISTI.CANFAR/21.0007] at native 16384-channel resolution through the read-side filter (`research/figures/frb_panel.py`), which supplies only observer facts: dead channels (the RFI mask) are dropped and the surviving channels are passed to the aperture front door. The catalog waterfalls arrive already dedispersed; the library adds no dedispersion of its own, and any residual sweep shows as diagonal structure the read catches directly. Everything downstream is the library.

For each burst the filter resolves the signal above the derived noise floor ($K_{\mathrm{signal}}\ge 1$) and reconstructs it as the projection onto its resolved modes (Def 8.4): the Gavish–Donoho shrinkage attenuates the noise sea and the $\varphi_F>\varphi_T$ geometry cut removes persistent narrowband interference, leaving the burst morphology intact. The per-burst reads annotated on each panel — the resolved count $K_{\mathrm{signal}}$, the bounded *contrast* $\sigma_1/\Phi$ (the leading singular value over the screen floor), and the coherence $z$ of §5 — report the contrast rather than a per-mode tail probability, whose Tracy–Widom approximation [Chiani 2014] is far outside its calibrated range at these deviates. The figure routine writes these per-burst values, with the live-channel and RFI counts, to a companion table (`frb_panel.csv`) regenerated from the CANFAR waterfalls. Figure 2 places the entroptics reconstruction beside CHIME's own *fitburst* forward model and the raw waterfall on four bright events: with no per-substrate tuning the recovered burst tracks the forward-model morphology while the noise and RFI modes are gone.

![Figure 2. Entroptics on real CHIME/FRB Catalog-1 bursts at native 16384-channel resolution (zero tuning), three panels per event: CHIME's *fitburst* forward model (left), the raw dedispersed waterfall (middle), and the entroptics reconstruction `Aperture(raw).extract()` (right) — the Gavish–Donoho projection onto the resolved modes with the persistent-structure cut. Per-burst $K_{\mathrm{signal}}$, contrast $\sigma_1/\Phi$, and coherence $z$ annotate each row.](figures/frb_panel.png)

---

## 13. Formal verification

The governing lemmas are verified in Lean 4 / Mathlib: thirty theorems that compile with no `sorry`, each depending only on the standard axioms `propext`, `Classical.choice`, and `Quot.sound`. Each certified statement is the finite, discrete fact the paper asserts.

- The entropy bound $\mathrm H(q)\in[0,\log_2 n]$, hence the fill-fraction range $\varphi=2^{\mathrm H(q)}/n\in[1/n,1]$ (Lemma 3.2), and the Strehl bound $\mathcal S\in[1/T,1]$ (Lemma 3.4).
- The congruence and orthogonal-conjugation invariance of a fixed correlation operator's spectrum (its characteristic polynomial a similarity invariant), whose data sub-case is the fill fraction's invariance under variable permutation and per-variable phase (Proposition 3.5).
- The positive semidefiniteness of the biased-autocovariance Toeplitz matrix whose entry depends on the lag alone (Lemma 4.2), with the peak-at-zero-lag corollary from its $2\times2$ minor (Corollary 4.3).
- The discrete scale-invariance of the shape factor through the block-repeat identities $\mathrm H'=\mathrm H+\log_2 s$ and $\xi'=s\,\xi$ (Proposition 4.5).
- The permutation-null mean via the two-point stabilizer count $(N-2)!$ (Theorem 5.2).
- The Weyl perturbation bound for the top eigenvalue composed into the certified attenuation interval (Lemma 6.2).
- The per-row phase invariance of the second moment together with an antipodal cloud that separates focus from resultant (Proposition 7.2).
- The monotonicity of the derived noise floor in its significance quantile and the antitone dependence of the resolved dimension on the floor (§8).
- The propagator identity $P_{yx}=A\,P_{xx}$ with operator recovery $P_{yx}P_{xx}^{-1}=A$ and shared spectrum (Theorem 9.2), and the accumulator additivity behind splicing (Theorem 9.3).

Two certified statements are narrower than their headline. Theorem 9.2 is verified for an invertible accumulator (ordinary inverse $P_{xx}^{-1}$); the rank-deficient pseudoinverse-on-a-subspace form is proved on paper (§9). The noise-floor monotonicity certifies the floor formula in its quantile argument; the Tracy–Widom law it invokes is cited. The classical limit laws (Tracy–Widom, Marchenko–Pastur, Weyl) are cited, not re-proved. The optical identification of §10 — that these entropic and spectral quantities are the aperture's fill fraction, étendue, Strehl ratio, OTF, and carrier — is a definitional dictionary, not a theorem; the certified lemmas bound and relate the quantities without asserting the naming.

---

## 14. Conclusion

Entroptics reads a signal at the resolution the signal itself dictates. From the entropy of the power marginals we obtain a matched scale; from the singular and correlation spectra, the classical optical invariants; from the biased autocorrelation, an OTF whose entropy width is the diffraction limit, with the Mercer ratio as an internal consistency check that doubles as a nonstationarity flag; and from an online linear operator, the decay rates the entropy width approximates. Each governing claim is a standard theorem specialised to the finite setting, proved above and verified (§13), and the whole is deterministic and backend-portable. The aperture, the resolution, the fold: the signal supplies each for itself.

---

## References

- E. Abbe, *Beiträge zur Theorie des Mikroskops und der mikroskopischen Wahrnehmung*, Arch. mikroskop. Anat. **9** (1873) 413–468.
- CHIME/FRB Collaboration, *The First CHIME/FRB Fast Radio Burst Catalog*, Astrophys. J. Suppl. Ser. **257** (2021) 59, arXiv:2106.04352. Data (public): CANFAR archive, CISTI.CANFAR/21.0007, https://www.canfar.net/.
- M. Chiani, *Distribution of the largest eigenvalue for real Wishart and Gaussian random matrices and a simple approximation for the Tracy–Widom distribution*, J. Multivariate Anal. **129** (2014) 69–81.
- A. D. Cliff, J. K. Ord, *Spatial Processes: Models and Applications*, Pion, 1981.
- L. De Lathauwer, B. De Moor, J. Vandewalle, *A multilinear singular value decomposition*, SIAM J. Matrix Anal. Appl. **21** (2000) 1253–1278.
- D. Gabor, *Theory of communication*, J. IEE **93** (1946) 429–457.
- M. Gavish, D. L. Donoho, *Optimal shrinkage of singular values*, IEEE Trans. Inform. Theory **63** (2017) 2137–2152.
- J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts & Co., 2005.
- N. Halko, P. G. Martinsson, J. A. Tropp, *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions*, SIAM Rev. **53** (2011) 217–288.
- W. James, C. Stein, *Estimation with quadratic loss*, in *Proc. Fourth Berkeley Symp. Math. Statist. Prob.* **1** (1961) 361–379.
- T. Jiang, *The limiting distributions of eigenvalues of sample correlation matrices*, Sankhyā A **66** (2004) 35–48.
- I. M. Johnstone, *On the distribution of the largest eigenvalue in principal components analysis*, Ann. Statist. **29** (2001) 295–327.
- M. H. Kassner, *Screens for the absorption of micro-wave radiation*, M.Sc. thesis, Dept. of Physics, McGill University, 1950. https://doi.org/10.82308/34546
- A. Khinchin, *Korrelationstheorie der stationären stochastischen Prozesse*, Math. Ann. **109** (1934) 604–615.
- H. J. Landau, H. O. Pollak, *Prolate spheroidal wave functions, Fourier analysis and uncertainty II*, Bell Syst. Tech. J. **40** (1961) 65–84.
- W. Lukosz, *Optical systems with resolving powers exceeding the classical limit*, J. Opt. Soc. Am. **56** (1966) 1463–1472.
- V. A. Marchenko, L. A. Pastur, *Distribution of eigenvalues for some sets of random matrices*, Math. USSR-Sb. **1** (1967) 457–483.
- K. V. Mardia, P. E. Jupp, *Directional Statistics*, Wiley, 2000.
- J. Mercer, *Functions of positive and negative type, and their connection with the theory of integral equations*, Phil. Trans. R. Soc. Lond. A **209** (1909) 415–446.
- G. A. Miller, *Note on the bias of information estimates*, in *Information Theory in Psychology* (1955) 95–100.
- C. E. Shannon, *A Mathematical Theory of Communication*, Bell Syst. Tech. J. **27** (1948) 379–423, 623–656.
- D. Slepian, H. O. Pollak, *Prolate spheroidal wave functions, Fourier analysis and uncertainty I*, Bell Syst. Tech. J. **40** (1961) 43–63.
- C. A. Tracy, H. Widom, *Level-spacing distributions and the Airy kernel*, Comm. Math. Phys. **159** (1994) 151–174.
- C. A. Tracy, H. Widom, *On orthogonal and symplectic matrix ensembles*, Comm. Math. Phys. **177** (1996) 727–754.
- J. H. Tu, C. W. Rowley, D. M. Luchtenburg, S. L. Brunton, J. N. Kutz, *On dynamic mode decomposition: theory and applications*, J. Comput. Dyn. **1** (2014) 391–421.
- L. R. Tucker, *Some mathematical notes on three-mode factor analysis*, Psychometrika **31** (1966) 279–311.
- J. W. Tukey, *Exploratory Data Analysis*, Addison-Wesley, 1977.
- R. Vershynin, *High-Dimensional Probability*, Cambridge Univ. Press, 2018.
- H. Weyl, *Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen*, Math. Ann. **71** (1912) 441–479.
- E. T. Whittaker, *On the functions which are represented by the expansions of the interpolation-theory*, Proc. R. Soc. Edinburgh **35** (1915) 181–194.
- N. Wiener, *Generalized harmonic analysis*, Acta Math. **55** (1930) 117–258.
- E. B. Wilson, M. M. Hilferty, *The distribution of chi-square*, Proc. Natl. Acad. Sci. USA **17** (1931) 684–688.
- E. Wolf, *New theory of partial coherence in the space–frequency domain. Part I: spectra and cross spectra of steady-state sources*, J. Opt. Soc. Am. **72** (1982) 343–351.