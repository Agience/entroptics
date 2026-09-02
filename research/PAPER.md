# Entroptics

### Reading a 2-D signal as a finite optical aperture at its own entropy-matched resolution.

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`

*Pre-print. September 2026.*

---

## Abstract

We treat a two-dimensional signal $W\in\mathbb{C}^{T\times F}$, with one **ordered** axis and one **feature** axis whose channels are commensurate, as a finite optical aperture whose resolution the signal's own Shannon entropy sets. From the entropy of the power marginals we read a matched scale; from the singular and correlation spectra, the standard optical quantities (fill fraction, étendue, Strehl ratio, space–bandwidth product); from the signal's own autocorrelation (a direct lag average, via Wiener–Khinchin), the optical transfer function (OTF) and, through the Abbe/Rayleigh relation, a diffraction limit. A streaming online dynamic mode decomposition (DMD) operator supplies the per-mode decay rates that the entropy width approximates, and a read-side filter — a rank cut at the derived floor with Gavish–Donoho shrinkage and a geometric persistent-structure cut — returns the field on its own resolved modes, introducing no direction the data does not carry. Every read is defined from $W$ alone and the construction is **parameter-free**: no constant anywhere is fitted to data or calibrated to a substrate, each being a derived mathematical quantity or a stated criterion, tabulated in §13.1. What the reader supplies is an operating point rather than a tuning — a decision-risk level $\alpha$, which by Neyman–Pearson encodes the cost of a false alarm against a miss and cannot be derived from the data, and the null it is taken against — together with two declarations about the record itself: which axis is ordered, and that the feature channels are commensurate. The distributional assumptions are the i.i.d. Gaussian bulk null of the noise-floor read (§§6, 8), row exchangeability for the permutation nulls (§5), channel independence for the estimator spread of Definition 4.8, and wide-sense stationarity for the Mercer ratio (§4.3) and the rates (§9). The provenance of every constant is tabulated in §13.1. We give precise definitions and prove the governing lemmas — positive semidefiniteness of the biased autocovariance (hence a peak-at-zero-lag OTF), the fill-fraction and Strehl bounds, the scale-invariance of the Rayleigh shape factor, the permutation-null mean of the coherence statistic, a Weyl-certified interval for the attenuation constant, the recovery of the decay rates and the additivity behind stream splicing, and the axial/directional dichotomy of the concentration reads — and tie each read to a named theorem in a term-by-term dictionary. The construction is realised in a small, backend-agnostic (numpy or torch), deterministic Python library whose test suite pins these identities, and is accompanied by a Lean 4 / Mathlib certification of the governing lemmas.

---

## 1. Introduction

Describing a field through the language of optics is an old experimental practice: free-space microwave measurements characterise a physical *screen* by its reflection, attenuation, and diffraction, read at varying angles and distances, in optical terms [Kassner 1950]. The optical picture we take literally for data is the finite aperture: classically, it cannot resolve to infinitely fine detail — its size fixes a diffraction limit, and the image is the object convolved with the aperture's point-spread function [Goodman 2005]. Given a two-dimensional array $W$ indexed by one **ordered** axis (time, evolution, depth) and one **feature** axis (channels, frequency bins, coordinates), we ask: what is the aperture through which this particular signal is observed, and what is its diffraction limit? The signal sets its own aperture: the resolution is read from its Shannon entropy, the reads are the classical optical invariants, and nothing is fitted to the signal. Entroptics is the study of the optics a signal carries in itself.

Each read is defined from $W$ alone and each governing claim is a finite, discrete specialisation of a standard result: Shannon's source entropy, Wiener–Khinchin, the Fourier-optics autocorrelation theorem, Abbe/Rayleigh, conservation of étendue (the Lagrange invariant), the Gabor–Lukosz degrees-of-freedom invariance [Gabor 1946; Lukosz 1966], the coherent-mode decomposition, Mercer's theorem, the Marchenko–Pastur correlation edge, the Tracy–Widom singular-value edge, Weyl's inequality, and the exact dynamic-mode decomposition of a linear map. The two detection tests reference explicit closed-form nulls: the row-permutation null (§5) and the Marchenko–Pastur i.i.d.-bulk null (§§6, 8).

**Convention.** Throughout, $W\in\mathbb{C}^{T\times F}$ has rows indexed by the ordered axis $t\in\{1,\dots,T\}$ (subscript $T$) and columns by the feature axis $f\in\{1,\dots,F\}$ (subscript $F$). "Time" and "features" are *roles*, not physical quantities: any array with a single ordered axis qualifies. Entropy $\mathrm{H}(\cdot)$ is in bits, $\mathrm{H}(p)=-\sum_i p_i\log_2 p_i$ for a probability vector $p$, with $0\log_2 0:=0$. We write $x^{\mathsf H}$ for the conjugate transpose, $M^{+}$ for the Moore–Penrose pseudoinverse, and $\operatorname{Re}$ for the real part. A few symbols carry a role that is fixed per section and disambiguated by subscript or context: bare $\alpha$ is the false-alarm level (§§8, 11), while a subscripted $\alpha$ is an attenuation or decay rate ($\alpha_{\mathrm p}$ in §6, $\alpha_k$ in §9); $\mu$ is the coherence null mean (§5), the Johnstone centre (§§6, 8), or — as $\mu_k$ — a DMD eigenvalue (§9); and $\rho$ is the Mercer ratio (§4.3) or a correlation length (§14).

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

with $L_a$ the *measured extent* of axis $a$: the number of coordinates carrying at least one finite, unmasked cell, written $T_{\mathrm{eff}}$ and $F_{\mathrm{eff}}$. It is the extent and not the array's shape because every quantity below compares an entropy against $L_a$ — the no-signal maximum $\log_2 L_a$, the matched scale $\delta_a$, the Miller–Madow band of Definition 2.2, and the fills of Definition 3.3 — and a coordinate nothing was observed in would raise the bar the signal has to clear while contributing no signal. A channel that was measured and read zero is an observation of no power and counts; a channel that was never measured is absent, and absence is not an observation of zero. When $S_{\!P}=0$ we set $\mathrm{H}_a=\log_2 L_a$ (maximal spread, $\delta_a=1$).

**Provenance of the exponential.** The exponential of an entropy counts effective modes. For a stationary process of unit variance with normalised spectral density $S$, the minimum *coefficient rate* — the numbers per unit time that determine the process — is $\exp\!\big(-\!\int S(f)\log S(f)\,df\big)$, which for a density flat on a band of width $2W$ and zero outside is the Nyquist count $2W$ [Campbell 1960]; $2^{\mathrm H}$ read as the *extent* of a distribution is [Campbell 1966]. Definition 2.1 takes that functional in base 2, on the two power marginals of a finite record, and uses it to set the matched scale $\delta_a$. The same functional measures information content and effective component count in the time–frequency plane, as a Rényi entropy of a quadratic representation [Baraniuk et al. 2001].

**The marginals are read before any whitening, and the channels must be commensurate.** The projection of §8 whitens each feature channel to a common noise scale, and that step *follows* this one: $\mathrm{H}_F$ is read on the raw $W$. The order is forced, because the two steps are opposites — this read measures how unevenly raw amplitude is spread across the channels, while whitening divides each channel by its own scale and so removes exactly that unevenness. Taken the other way a channel carrying only noise is lifted to the amplitude of one carrying signal: on a planted line of known width, $n_F$ tracks the line width in 9/9 cases read on the raw frame and returns $n_F=F$ in 9/9 read after whitening, the on-line channels holding $88.2\%$ of the power before and $9.6\%$ after (§14). The consequence is a *precondition* on the record rather than a parameter of the read: $2^{\mathrm{H}_F}$ counts the channels in play only where the feature channels are already in the same units, since a frame of mixed units reads its units. It is not detectable from one frame — scaling a column by $c$ is indistinguishable from that channel being $c$ times louder — so, like the identity of the ordered axis, it is declared by the caller (§13.1).

Power-weighting lets bright, on-signal rows dominate irrespective of how small a fraction of the record they occupy. The scale $\delta_a=L_a/2^{\mathrm{H}_a}$ is the reciprocal of the occupied fraction $2^{\mathrm{H}_a}/L_a$: a signal that concentrates its power in $2^{\mathrm{H}_a}\ll L_a$ effective cells is oversampled by $\delta_a>1$ along that axis.

**Only the feature axis folds.** The fold (§8) coarsens the *feature* axis to its own resolution. The *ordered* axis keeps native spacing, which the ordered reads require: the coherence of §5 (adjacent-row similarity), the decay/OTF of §4 (lag structure), and the rates of §9 (the ordered trajectory) are all lag statistics. So $\delta_T:=1,\ n_T:=T$ always, and only the feature scale carries a noise guard.

**The finite-sample noise guard.** A structureless (uniform-power) feature marginal has $\mathrm{H}_F=\log_2 F$ in the population but, by the plug-in bias, sits below that maximum on any finite record — a spurious $\delta_F>1$. The guard holds $\delta_F=1$ whenever $\mathrm{H}_F$ lies within the uniform-null bias band.

**Continuity licenses a fold.** The fold of §8 replaces adjacent feature cells by their area mean, which preserves the signal wherever it varies *continuously* across the merged cells. Concentration and continuity are independent properties of a record. A narrow line on a frequency axis is concentrated *and* continuous, and folds to its own width without loss. A record of a few active but mutually unrelated channels is concentrated to the same degree and belongs at native resolution — its feature axis is *nominal*, where adjacency is an accident of labelling. Separating the two therefore takes a second measurement: $\mathrm{H}_F$ reads the power marginal, which is invariant under relabelling of the channels, so both records give it the same value.

Continuity is measured by the statistic of §5, read across the feature axis. The coherence $z$-score of Definition 5.3 applied to $W^{\mathsf T}$ compares neighbouring *channels* against the exact permutation null over channels: $z_F>0$ means neighbours are more alike than a random relabelling of the channels, which is precisely the property an area mean requires. The null is closed form, so the test adds one decision input — the level $\alpha$ already carried by every other decision in the instrument.

**Definition 2.2 (Noise-guarded scale)** — The ordered axis keeps native resolution ($\delta_T:=1,\ n_T:=T$). For the feature axis, with the band $\beta_F$ of Definition 2.3 and the feature-axis adjacency score $z_F$ of Definition 5.3 applied to $W^{\mathsf T}$, set $\delta_F:=1,\ n_F:=F$ if **either**

$$\mathrm{H}_F\ \ge\ \log_2 F-\beta_F \qquad\text{(uniform within the band)} \qquad\text{or}\qquad z_F\ \le\ z_{1-\alpha} \qquad\text{(the feature axis is nominal)},$$

and otherwise as in Definition 2.1.

**Definition 2.3 (The band)** — A fold has to clear two bars, and each is a bound on a null the instrument already carries.

*Significance — is the concentration real?* Under an i.i.d. Gaussian null the feature marginal is a symmetric $\mathrm{Dirichlet}(a)$ over $F$ components, with $a=T/2$ for real cells and $a=T$ for complex ones (a real square is $\chi^2_1$, a complex modulus-square is $\mathrm{Exp}(1)$); $a=T/2$ is taken, carrying exactly twice the spread and so conservative for either input. The deficit $D=\log_2F-\mathrm H_F$ then has the exact Wolpert–Wolf moments

$$\mathbb E[\mathrm H]=\psi(Fa{+}1)-\psi(a{+}1),\qquad \operatorname{Var}[\mathrm H]=\tfrac{a+1}{Fa+1}\big[(\psi(a{+}2)-\psi(Fa{+}2))^2+\psi'(a{+}2)-\psi'(Fa{+}2)\big]+\tfrac{a(F-1)}{Fa+1}\big[(\psi(a{+}1)-\psi(Fa{+}2))^2-\psi'(Fa{+}2)\big]-\mathbb E[\mathrm H]^2,$$

and the bar is $\mathbb E[D]+k\,\mathrm{sd}(D)$ with $k=\sqrt{1/\alpha-1}$ from Cantelli's inequality — a distribution-free bound, so a pure-noise record clears it with probability at most $\alpha$ and nothing is assumed about the deficit's shape.

*Sufficiency — is the fold worth making?* A fold that moves the width by a fraction of a percent gains nothing and perturbs the floor, which depends on the shape. The floor of §8 sits at the Marchenko–Pastur edge $\mu=(\sqrt N+\sqrt F)^2$ with a Tracy–Widom margin $q_\alpha\varsigma_J$, so a width change earns its place only when it moves that edge by more than the margin: $|\mathrm d\mu/\mathrm dF|\,\Delta F>q_\alpha\varsigma_J$, i.e. $\Delta F>q_\alpha\sqrt F\,(1/\sqrt N+1/\sqrt F)^{1/3}$, which as a band on $\mathrm H_F$ is $-\log_2(1-\Delta F/F)$.

$\beta_F$ is the larger of the two: a fold must be both real and worth making. The first term governs wide-short frames, where the null deficit is large; the second governs square and tall ones, where it is not. **Neither can reach $\log_2F$**, since one is a tail bound on a deficit confined to $[0,\log_2F]$ and the other is $-\log_2(1-\Delta F/F)$ with $\Delta F<F$ — so unlike a band built from a mean, this one needs no cap to stay inside the entropy range. Measured, it sits below $1\%$ of $\log_2F$ at every shape from $64\times64$ to $19\times16384$.

The two conditions are independent and both operative: concentration says *how far* a fold may go ($2^{\mathrm{H}_F}$ effective channels), continuity says *whether* folding is licensed at all. The continuity test runs only where concentration has already admitted a fold, so structureless noise, delocalised low-rank signals, and the streaming path keep their existing behaviour at their existing cost. Continuity is a stated criterion for when an area mean is faithful; §5 supplies the calibrated null against which it is decided.

The band supersedes a capped Miller–Madow form, $\min\!\big((F-1)/(2T\ln2),\tfrac12\log_2F\big)$, whose inner term is the Miller–Madow bias expression for $F$ bins at $T$ samples [Miller 1955] — a *conservative* uniform-null guard: it exceeds the marginal's mean finite-sample deficit — which, for the power marginal, is smaller by a factor $F$, its effective sample size being the total cell count $TF$ — so both structureless noise and a genuine low-rank signal's *delocalised* feature marginal stay at native resolution. The inner band grows without bound in $F/T$ and *exceeds* $\log_2 F$ for wide-short data ($F\gtrsim 2T\ln F$); capping it at $\tfrac12\log_2 F$ keeps the guard active, folding once power concentrates below $\sqrt{F}$ effective channels ($\mathrm{H}_F<\tfrac12\log_2 F$, deficit $>\tfrac12\log_2 F$). For all tall/square shapes the inner band is already below the cap and the behaviour is unchanged, so noise and delocalised low-rank signals stay at native resolution (§14) and the coherence null of §5 stays calibrated.

The band is closed-form, fixed by the axis lengths alone. The one choice it embeds — the $\sqrt{F}$ concentration floor of the cap — is a stated criterion, not a fitted constant; the provenance of every constant is tabulated in §13.1.

---

## 3. Aperture reads: fill fraction, étendue, Strehl

### 3.1 The fill fraction and its duality

For a block $M\in\mathbb{C}^{m\times k}$ with singular values $s_1\ge\cdots\ge s_n\ge 0$, $n=\min(m,k)$, let $q_i=s_i^2/\sum_j s_j^2$ be the normalised power spectrum. $M$ is column-centred first, as the correlation reads of §3.2 are: a constant baseline is not structure, and on an uncentred block it lands in a leading singular value, so adding a single global constant would move the fill. Centring is idempotent, so a block that arrives centred — a projected screen (§8) — is unchanged by it.

**Definition 3.1 (Fill fraction and magnification)**

$$
  \varphi(M)=\frac{2^{\mathrm{H}(q)}}{n}\in[1/n,\,1],\qquad
  \operatorname{mag}(M)=\frac1{\varphi(M)}\in[1,\,n].
$$

$\varphi$ is the aperture's fill fraction (the fraction of singular modes that are active, the *bounded* face), and $\operatorname{mag}$ its reciprocal *reach* (oversampling), the *unbounded* face. The two are one continuum meeting at $\varphi=\operatorname{mag}=1$, the critically sampled diffraction limit.

**Lemma 3.2 (Fill-fraction bounds)** — For any $M$ with at least one nonzero singular value, $\varphi(M)\in[1/n,\,1]$. Moreover $\varphi(M)=1/n$ iff $\operatorname{rank}(M)=1$, and $\varphi(M)=1$ iff all $n$ singular values are nonzero and equal.

*Proof.* $q$ is a probability vector of length $n$, so $\mathrm{H}(q)\in[0,\log_2 n]$, whence $2^{\mathrm{H}(q)}\in[1,n]$ and $\varphi=2^{\mathrm{H}(q)}/n\in[1/n,1]$. $\mathrm{H}(q)=0$ iff $q$ is a point mass (a single nonzero singular value, i.e. rank $1$); $\mathrm{H}(q)=\log_2 n$ iff $q$ is uniform over all $n$ entries (every singular value nonzero and equal). ∎

$\varphi$ applies the functional of §2 to the singular-value power spectrum. Lemma 3.2's two extremes are its discrete face: a uniform spectrum fills the aperture ($\varphi=1$, critical sampling), as a flat spectral density gives the Nyquist rate, and a rank-1 block occupies one mode of $n$. The exponential entropy of a matrix's normalised singular values, $\operatorname{erank}=e^{\mathrm H(\sigma)}$ with $\sigma_i=s_i/\sum_j s_j$, is the *effective rank* [Roy & Vetterli 2007]. $\varphi$ takes the *power* spectrum $q_i=s_i^2/\sum_j s_j^2$, the same normalisation as the intensity read of §4 and the marginals of §2, and divides by $n$ to give a fill fraction bounded in $[1/n,1]$.

### 3.2 Per-axis correlation spectra

Both axis spectra are read off the *connected* screen $W_c$: each channel's time-mean removed. A component constant in time carries no order, so it must not enter the ordered-axis correlation, where it arrives as one vector added to every time point's sample — a rank-1 mode the read would otherwise score as coherence. On white noise plus a static bandpass the Strehl ratio reached $0.99$, above what a real periodic signal scores, and did not move when the time samples were shuffled. For axis $a$ let $M_a$ be $W_c$ (feature axis, columns as variables) or $W_c^{\mathsf\top}$ (ordered axis), so the variables are the $L_a$ coordinates of that axis and the samples are those of the other. The feature axis is unaffected by the connecting step, which is idempotent there. Let $R_a$ be the (Hermitian) *correlation* matrix of $M_a$ (columns centred, then rescaled to unit diagonal, each coordinate carrying nonzero variance), with eigenvalues $\lambda^{(a)}_1\ge\cdots\ge\lambda^{(a)}_{L_a}\ge 0$.

**Definition 3.3 (Axis reads)** — With $\bar\lambda^{(a)}_i=\lambda^{(a)}_i/\sum_j\lambda^{(a)}_j$,

$$
  \varphi_a=\frac{2^{\mathrm{H}(\bar\lambda^{(a)})}}{L_a},\qquad
  \sigma_a=\sqrt{\lambda^{(a)}_1},\qquad
  \mathcal E=\varphi_F\varphi_T,\qquad
  \mathrm{SBW}=n_F\,n_T,\qquad
  \mathcal S=\frac{\lambda^{(T)}_1}{\sum_j\lambda^{(T)}_j}.
$$

$\mathcal E$ is the *étendue* (the joint 2-D aperture area), $\mathrm{SBW}$ the *space–bandwidth product* (the invariant count of degrees of freedom the screen carries [Lukosz 1966; Toraldo di Francia 1969]), and $\mathcal S$ the *Strehl ratio*: the fraction of power in the dominant coherent mode of the ordered-axis correlation operator, its leading mode in the coherent-mode (Mercer) decomposition [Wolf 1982; Mercer 1909]. Because $R_a$ has unit diagonal, $\operatorname{tr}R_a=\sum_j\lambda^{(a)}_j=L_a$; the correlation spectrum needs no separate normalisation. $\mathrm{SBW}$ is a capacity and not a content: $n_a$ is the matched-grid width of Definition 2.2, so a screen the fold leaves at native resolution reports $TF$ whatever sits on it. The spots actually filled are $\mathcal E\cdot\mathrm{SBW}=2^{\mathrm H(\bar\lambda^{(T)})}2^{\mathrm H(\bar\lambda^{(F)})}$, which is $1$ for a single mode and grows with the modes present.

**Lemma 3.4 (Strehl bounds)** — $\mathcal S\in[1/T,\,1]$, with $\mathcal S=1$ iff the ordered-axis correlation has rank $1$ (a single coherent mode) and $\mathcal S=1/T$ iff its spectrum is flat.

*Proof.* $\operatorname{tr}R_T=T$, so the top eigenvalue $\lambda^{(T)}_1\in[1,T]$ (at least the mean $1$, at most the trace); hence $\mathcal S=\lambda^{(T)}_1/T\in[1/T,1]$, with the extremes at the rank-one and flat spectra. ∎

**Proposition 3.5 (Étendue is a bounded, basis-invariant area)** — $\mathcal E=\varphi_F\varphi_T\in(0,1]$, the product of the two axis fill fractions, the phase-space area the finite screen carries. Each $\varphi_a$, hence $\mathcal E$, is invariant under relabeling of either axis's variables, and $\varphi_F$ additionally under a per-channel phase — the discrete Smith–Helmholtz/Lagrange invariance, carried by the gauge the data actually has. Channel polarity is an instrument convention; the sign of an individual time sample is not, and $R_T$ is formed on the connected screen (§3.2). A system that decouples the axes (one fill $\to 0$ while the other stays finite) sends $\mathcal E\to0$: the scale-free limit.

*Proof.* $\varphi_a\in(0,1]$ by Lemma 3.2 applied to $R_a$. A permutation $P$ and per-variable phase $\Theta=\operatorname{diag}(e^{i\theta})$ send $R_a\mapsto P\Theta R_a\Theta^{\mathsf H}P^{\mathsf\top}$, a unitary congruence preserving the unit diagonal and hence the spectrum $\lambda^{(a)}$; since $\varphi_a=2^{\mathrm H(\bar\lambda^{(a)})}/L_a$ depends only on that spectrum, it is unchanged. The connecting step of §3.2 subtracts each channel's time-mean, which commutes with a permutation of either axis and with a per-channel phase, so those clauses stand; it does not commute with a per-time phase, under which no component of the record is constant in time and there is nothing for the step to remove. ∎

**Spectral form.** Because $\varphi_a$ depends only on the eigenvalue multiset of $R_a$, it is a similarity invariant of that *fixed* operator: any congruence $C\mapsto PCQ$ with $QP=1$ (in particular an orthogonal conjugation $P^{\mathsf\top}P=1$) leaves the characteristic polynomial, hence every spectral read, unchanged. This is what §15 certifies — an abstract matrix, no unit-diagonal or data hypothesis. The invariance the *data* carries is the sub-case preserving the unit diagonal of $R_a$: variable permutation and per-variable phase (Proposition 3.5), the group $S_{L_a}\ltimes U(1)^{L_a}$ (for real data the signed-permutation subgroup $B_{L_a}$), not the full orthogonal group. A rotation of the data mixes coordinates and re-normalises the correlation's diagonal, which is not a similarity of $R_a$, so $\varphi_a$ moves: the two axes carry distinct coordinates (a feature channel, a time step) and $\varphi_a$ reports how power distributes across *those*.

**The aperture is finite, and its size is a variable.** Every read above is taken through one aperture. Two questions follow: how the reads move as the aperture grows, and what to do when the field is wider than any aperture that can be resolved in a single decomposition.

**Definition 3.6 (Scale profile)** — Sweep trailing ordered-axis windows of increasing length over $W$ and read each one. The profile records, per window, the resolved dimension of Definition 8.2, the coherence $z$ of §5, the diffraction limit $a_\delta$ of §4 and the ordered fill $\varphi_T$. The *resolved window* is the smallest at which a mode first stands above the floor; the *dominant window* is the one of maximal coherence; the transitions are the windows at which the resolved count changes.

Windows are ordered-axis cells, not seconds: the aperture has no physical clock, and a caller maps cells to its own units. The profile is structure as a function of observation window — resolution against aperture size — and it reads the scale at which structure appears.

**Definition 3.7 (Swept aperture)** — For a field whose feature axis exceeds one aperture, fix a bounded patch width and step the aperture across the feature axis. The coherence of §5 gates each patch: a patch that resolves no ordered structure is not decomposed further. The reads reported per coherent band are its column span and the on-pulse extent and decay of §4.

Two properties make this a measurement. The gate is the level $\alpha$ of §13.1 corrected for the number of patches the sweep examines, so the field-wide false-alarm rate stays bounded by $\alpha$ however wide the field is. And the on-pulse extent is taken where the profile stands above its own robust noise scale, so it does not depend on the brightness of the peak — a decay rate must not.

---

## 4. The decay, the optical transfer function, and the diffraction limit

The signal's own autocorrelation along the ordered axis is read as an optical transfer function [Goodman 2005].

### 4.1 The connected autocovariance

Let $x(t)=W_{t,\cdot}\in\mathbb{C}^F$ be the $t$-th row of the record as given. Whether a record is a field or an intensity is a fact about the instrument that produced it and not a property of its samples — an intensity and an amplitude are both nonnegative, and subtracting a baseline from an intensity makes it signed without changing any physics — so no read here infers it. The incoherent read of an amplitude record is $C$ of $|W|^2$, formed by the caller. Let $x_c(t)=x(t)-\frac1T\sum_s x(s)$ be the centred (connected) field, with $x_c(t)=0$ for $t\notin\{1,\dots,T\}$.

**Definition 4.1 (Decay / OTF)** — The pooled biased autocovariance is, for lags $\tau=0,\dots,T-1$,

$$
  C(\tau)=\frac1T\sum_{t=1}^{T-\tau}\operatorname{Re}\big\langle x_c(t),\,x_c(t+\tau)\big\rangle
        =\frac1T\sum_{t}\operatorname{Re}\!\sum_f \overline{x_{c,f}(t)}\,x_{c,f}(t+\tau),
$$

extended by $C(-\tau)=C(\tau)$.

The normalisation by $T$ (not $T-\tau$) is the *biased* estimator, and it is essential: it makes $C$ a positive-definite sequence (Lemma 4.2). By Wiener–Khinchin [Wiener 1930; Khinchin 1934], $C$ equals the inverse transform of the nonnegative periodogram $S(\omega)=\tfrac1T\sum_f|\hat x_{c,f}(\omega)|^2$; this identity ties the decay-entropy width to the spectral-width read of §4.3. As an aperture read $C$ is the **optical transfer function**, and only that: by Wiener–Khinchin it is the transform of the nonnegative power spectrum, which is what an OTF is. The Fourier-optics autocorrelation theorem — the OTF is the autocorrelation of the pupil [Goodman 2005] — is the source of the name, and taking it further would require nominating the ordered axis as a pupil coordinate, which nothing here does. $C$ is *not* the point-spread function: the two are Fourier duals, and reading $C$ as a PSF reverses the sense of $a_\delta$, under which a long correlation is a narrow transfer width and so a finer limit.

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

**What $\xi$ measures, and what it cannot.** $\xi$ is summed over the decay's *first positive lobe*, and the truncation is not a convenience — it is the whole read. Because each channel is mean-centred, $\sum_t x_c(t)=0$, so the full two-sided lag sum of $C$ is $\tfrac1T\big|\sum_t x_c(t)\big|^2=0$ **exactly**, for every record. Hence $C(0)+2\sum_{\tau\ge1}C(\tau)=0$ and the one-sided sum to its natural limit is the constant $\tfrac12$, whatever the signal. Measured on the AR(1) family of §14 at $\rho\in\{2,8,32,128\}$: the two-sided sum is $0$ to $5\times10^{-15}$ and the one-sided sum is $0.500000000$ at every $\rho$. What varies with $\rho$ is where $C$ first crosses zero — $\xi=2.45,\,8.08,\,29.26,\,102.86$ across those four — so $\xi$ reports the location of that crossing and not a classical integral correlation length, which for this estimator does not exist. The Abbe/Rayleigh reading of $1/\xi$ should be taken in that light, and the residual curvature §14 reports in $\xi$ against $\rho$ is the crossing's own behaviour, not an estimator bias.

**Proposition 4.5 (Shape factor is invariant under block-repetition)** — The Rayleigh shape factor $g=\xi\cdot a_\delta$ is unchanged by an integer block-rescaling of the decay profile: under a rescaling by $s\in\mathbb N$ (each lag replaced by $s$ copies, $C_s(s\tau+r)=C(\tau)$ for $0\le r<s$), the entropy width and the integral length scale reciprocally and $g$ is *exactly* unchanged. Block-repetition is not a dilation: it replaces the profile by a staircase, which is a different shape and is not the autocovariance of any record of the same length. Invariance under a genuine dilation — an AR(1) at $\rho$ against one at $2\rho$ on the same grid — is not established here, and §14 tests one profile family, which cannot separate the two readings.

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

**Proposition 4.7 (Internal certificate)** — $M\succeq0$ (Lemma 4.2), so $\{\lambda_k\}\subset[0,\infty)$ and $n_{\mathrm{dof}},a_\delta^{\mathrm{spec}},\rho$ are well defined: the eigenvalues are nonnegative in exact arithmetic, and the implementation clips only floating-point round-off at zero. For a wide-sense-stationary process the temporal width $a_\delta$ and the spectral width $a_\delta^{\mathrm{spec}}$ are both functionals of the one spectral density, so $\rho$ is constant along the record; a drift of $\rho$ localises nonstationarity. Its value is fixed by the decay's shape (Proposition 4.5); the certificate is the constancy of $\rho$, demonstrated on stationary and regime-switching signals in §14.

---

**Definition 4.8 (Decay scatter)** — $C$ of Definition 4.1 is a sum of one biased autocovariance per feature channel, so the channels are independent replicates of the same decay and their spread is the estimator's own uncertainty — measured, with no null assumed and nothing subtracted. Writing $C_f$ for the channel terms, so that $C=\sum_f C_f$, the read returns two shares of the decay's power: the *noise share* $\sum_\tau F\operatorname{Var}_f C_f(\tau)\big/\sum_\tau C(\tau)^2$ and the *tail share* $\sum_{\tau>0}C(\tau)^2\big/\sum_\tau C(\tau)^2$.

The pair separates the width that is structure from the width that is disagreement. Sampling noise left in $C$ falls as $1/F$; it is spread across lags, it widens the entropy of Definition 4.4, and a wider entropy is a longer correlation — so a narrow record OVERSTATES the correlation length. When the two shares coincide, the width away from zero lag is scatter the channels do not agree on; when the noise share falls far below the tail share, it is structure they do. Neither is removed from $a_\delta$: what the scatter contributes could only be subtracted by assuming what the decay would have been, and Definition 4.4 reports the decay it measured. The remedy for a wide scatter is more channels, and the read is consistent in that direction (§14).

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

**Remark 5.4 (Calibration)** — Theorem 5.2 fixes the null mean $\mu$ (§15) and Definition 5.3 the null variance in closed form, so $z$ has mean $0$ and unit variance under the permutation null at every shape at the operative lag $\ell=1$ (validated against brute-force permutation, §14); the closed form holds at any $\ell$ for which the disjoint-pair count is defined ($N\ge 2\ell+2$). The permutation distribution of $A$ is itself mildly right-skewed at small $N$, so the one-sided tail is only approximately normal — near the nominal rate, and approaching it as $N$ grows (§14); the approximation is in the tail *shape* alone, not in the standardisation.

### 5.5 Two frames: the coupling of a shared basis

Definition 5.1 compares a frame's rows to its own lagged rows. The same null answers a second question — whether *two* frames, sampled together on one ordered axis, carry a common structure — and the estimator is the alignment of the two connected frames.

Let $A,B\in\mathbb{C}^{T\times D}$ be sampled on a shared ordered axis and expressed in one shared feature basis, and let $\tilde A,\tilde B$ be their column-centred forms.

**Definition 5.5 (Alignment)** — $S=\langle\tilde A,\tilde B\rangle_F=\sum_{t}\langle\tilde a_t,\tilde b_t\rangle$, the Hermitian inner product of the two frames.

The null re-pairs the two sides: permute the rows of one uniformly at random, so each side keeps its own internal structure entirely and the correspondence between them is what varies. This isolates the question — whether $A$ and $B$ are coupled — from whether either has structure of its own.

Write $\iota:\mathbb{C}^D\to\mathbb{R}^{2D}$, $x\mapsto(\operatorname{Re}x,\operatorname{Im}x)$ for the real embedding, and $\check A=\iota\tilde A$, $\check B=\iota\tilde B\in\mathbb{R}^{T\times 2D}$ for the embedded frames. The statistic is unchanged by it: $\operatorname{Re}\langle a,b\rangle_{\mathbb C}=\langle\iota a,\iota b\rangle_{\mathbb R}$, so $\operatorname{Re}S=\langle\check A,\check B\rangle_F$.

**Theorem 5.6 (Permutation-null moments)** — With $C_A=\check A^{\mathsf T}\check A$ and $C_B=\check B^{\mathsf T}\check B$, and $\pi$ uniform on $\mathfrak S_T$,

$$\mathbb{E}_\pi[S]=0,\qquad \operatorname{Var}_\pi\!\big[\operatorname{Re}S\big]=\frac{\operatorname{tr}(C_AC_B)}{T-1}.$$

*Proof.* Both frames are real, and $\operatorname{Re}S=\langle\check A,\check B\rangle_F$. Centring gives $\mathbb{E}_\pi[\check a_{\pi(t)}]=0$, hence $\mathbb{E}_\pi[S]=0$. For the second moment, write $M_{de}=(C_A)_{de}$. For a uniform $\pi$, $\mathbb{E}[x_{\pi(t)d}x_{\pi(t)e}]=M_{de}/T$, and for $t\ne s$ the pair $(\pi(t),\pi(s))$ is uniform over ordered pairs of distinct indices, so $\mathbb{E}[x_{\pi(t)d}x_{\pi(s)e}]=-M_{de}/(T(T-1))$ by the centring identity $\sum_i x_{id}=0$. Summing over $t,s$ and $d,e$, and using $\sum_t y_{td}y_{te}=(C_B)_{ed}$ with $\sum_{t\ne s}y_{td}y_{se}=-(C_B)_{ed}$, gives $\operatorname{Var}_\pi[\operatorname{Re}S]=\big(\tfrac1T+\tfrac1{T(T-1)}\big)\sum_{d,e}(C_A)_{de}(C_B)_{ed}=\operatorname{tr}(C_AC_B)/(T-1)$. ∎

**The embedding carries the theorem.** On the Hermitian Grams $\tilde A^{\mathsf H}\tilde A$ and $\tilde B^{\mathsf H}\tilde B$ the same sum is $\mathbb{E}_\pi[|S|^2]=\operatorname{Var}_\pi[\operatorname{Re}S]+\operatorname{Var}_\pi[\operatorname{Im}S]$, which is the variance of $\operatorname{Re}S$ only when $S$ is real. Taking $T=2$, $D=1$, $\tilde a=(1,-1)$ and $\tilde b=(i,-i)$: both re-pairings give $\operatorname{Re}S=0$, so the variance is $0$ while $\operatorname{tr}(\tilde A^{\mathsf H}\tilde A\,\tilde B^{\mathsf H}\tilde B)/(T-1)=4$. The embedding is what makes the standardisation of Definition 5.7 exact, and it is what the implementation forms.

**Definition 5.7 (Coupling)** — $\displaystyle z_{AB}=\frac{\operatorname{Re}S}{\sqrt{\operatorname{tr}(C_AC_B)/(T-1)}}$, with signed strength $\rho_{AB}=\operatorname{Re}S/(\|\tilde A\|_F\|\tilde B\|_F)$.

By Cauchy–Schwarz $|\rho_{AB}|\le1$, with equality iff $\tilde B=\lambda\tilde A$ for a real $\lambda$; $\rho_{AB}$ is invariant to a separate positive scale on either side, so a side's choice of units cannot move the sign or the strength. As in Remark 5.4 the moments are exact and the tail is the Pitman–Hoeffding limit [Hoeffding 1951], so the level is approached as $T$ grows.

**The sign requires a shared basis.** $\rho_{AB}$ is a statement about the two sides' coordinates being *the same* coordinates. Across two bases ($D_A\ne D_B$) the only basis-free statistic is $\|\tilde A^{\mathsf H}\tilde B\|_F^2$, which is non-negative by construction and so reports magnitude alone; the read is therefore defined on a shared basis, and raises without one. A sign taken from the leading singular vectors would follow their arbitrary global sign, making it a convention of the eigensolver. That basis-free quantity is the numerator of the RV-coefficient [Robert & Escoufier 1976] and of linear CKA [Kornblith et al. 2019], the linear case of the Hilbert–Schmidt independence criterion; both normalise it by $\|\tilde A^{\mathsf H}\tilde A\|_F\|\tilde B^{\mathsf H}\tilde B\|_F$ to a similarity in $[0,1]$. Definition 5.7 standardises the shared-basis inner product by the exact permutation null of Theorem 5.6.

**Closed-form permutation moments are an established technique, on a different statistic.** The first three exact moments of $\operatorname{tr}(W_xW_y)$ under a uniform row permutation are given by [Kazi-Aoual et al. 1995], and [Josse et al. 2008] build a test on them, matching a Pearson type III to mean, variance and skewness rather than resampling. That statistic is the RV coefficient: quadratic in each side, normalised, invariant to a rotation of either configuration, and non-negative — the same object as linear CKA. Theorem 5.6's statistic is the *signed bilinear* form $\langle\tilde A,\tilde B\rangle_F$ on a shared basis, linear in each side, and the sign is precisely what the rotation-invariant family cannot report. The two moments are therefore statements about different distributions, and $\operatorname{tr}(C_AC_B)$ appears here as the *variance* of the bilinear form rather than as the statistic itself.

The precedent does supply something this construction lacks. Remark 5.4 concedes the tail is only asymptotically normal, and §14 measures a one-sided rate $P(z>2)=0.027$ against the $\mathcal N(0,1)$ target $0.023$; a third moment with a Pearson type III tail is exactly the correction that literature applies to the same skew, and adopting it here is open work.

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

The step below equalises each channel's scale and performs **no decorrelation**: it is a per-channel divide, not $\Sigma^{-1/2}$. Noise correlated *across* channels survives it intact, which is why the floor below cannot separate such noise from signal on the singular spectrum alone. Correlation along the **ordered** axis is not addressed by it either, and moves the bulk itself: §14 measures a $100\%$ false-alarm rate on pure AR(1) rows at every correlation length tested, against $0$–$5\%$ on i.i.d. rows. That is a property of the i.i.d. null rather than of this floor — the standard singular-value and model-order selectors over-read the same records by two to four times as much (§14). "Whitening" is used here in that narrower sense.

**Definition 8.1 (Whiten, then project)** — *Whitening* rescales each feature channel to a common robust median absolute deviation (MAD) noise scale at native resolution, each channel's scale shrunk toward the pooled cross-channel scale by a data-derived James–Stein weight [James & Stein 1961], stabilising small-sample per-channel estimates while equalising genuinely different channels; masked or nonfinite cells are marked missing. A channel is *unresolvable* when its MAD falls at or below the frame's pooled MAD times the working arithmetic's machine epsilon $\varepsilon$; it is then given no scale at all. The test has to be relative on both sides: a MAD carries the record's units, so a fixed cut is a statement about units and would call every channel of a record kept in small ones dead, while a channel that never left its own median has a MAD of exactly zero, and dividing by any manufactured stand-in lifts round-off to unit amplitude — which the projection below would then resolve as signal. A fully-dead row or column (every cell missing) carries no information and is *dropped* from the read; missing data is ignored, so the singular value decomposition (SVD) reads only observed cells. *Projection* then folds the feature axis to the matched grid $(N,F_{\mathrm{eff}})=\big(T,\ \operatorname{round}(F/\delta_F)\big)$ by an area-weighted fold that excludes the remaining scattered missing cells; the ordered axis is kept at native resolution ($\delta_T=1$, §2), so $N=T$.

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

Given the false-alarm level $\alpha$, the floor is fixed by the shape $(N,F_{\mathrm{eff}})$: $\mu,\varsigma_J,c_F,(N-1)/N$ depend on it alone and $q_\alpha$ is a universal Tracy–Widom constant. The read is the per-mode evidence — the tail probability $p_k=P\big(\mathrm{TW}_1>(s_k^2/\hat\sigma^2-\mu)/\varsigma_J\big)$ of each singular value against the null — and $\alpha$ selects only the count $K_{\mathrm{signal}}=\#\{k:p_k<\alpha\}$; it is the reader's operating point, external by the Neyman–Pearson lemma (§13.1), and the measured false-alarm rate holds near $\alpha$ across aspect ratios from $N\gg F$ to $N\ll F$ (§14).

The law is the null of the largest eigenvalue under an i.i.d.-Gaussian bulk, so the floor is conditional on that null: noise that is *correlated across channels* (common-mode drift, narrowband interference) or *heavy-tailed* concentrates its variance into a few modes and is, on the singular spectrum alone, indistinguishable from signal, so the floor counts it as resolved. Separating structured noise from signal there needs a noise reference (prewhiten from a signal-free window) or the mode *shape* read of Def 8.3, in place of the scalar floor. Because the correct null is a modelling choice the library cannot make for every substrate, the floor is a **null provider**: a callback returning the threshold for one screen, evaluated *locally* on each block (per plane, per window, or streaming). The library ships the derived default ($\Phi$ above) and a small closed-form set; a caller can pass their own — a permutation surrogate (§5's philosophy: preserve each channel's marginal, destroy the cross-channel correlation), a signal-free reference, or a substrate-specific null. The provider owns both the threshold and the $\alpha$ it is drawn at (the derived edge serves an arbitrary $\alpha$ by inverting the Tracy–Widom survival function), so $\alpha$ and the null are the two *declared* external inputs the instrument cannot derive.

**Definition 8.3 (Mode footprint)** — Each of the $K_{\mathrm{signal}}$ resolved modes carries a left (ordered) singular vector $u_k$ and a right (feature) singular vector $v_k$. Its *footprint* is the pair of fill fractions of those vectors together with their product,

$$
  \varphi^{(k)}_T=\frac{2^{\mathrm H(|u_k|^2)}}{N},\qquad
  \varphi^{(k)}_F=\frac{2^{\mathrm H(|v_k|^2)}}{F_{\mathrm{eff}}},\qquad
  \mathcal E_k=\varphi^{(k)}_T\,\varphi^{(k)}_F ,
$$

the étendue of §3 resolved *per mode*: the phase-space area the mode occupies. The footprint separates modes that share a singular value — a broadband transient ($\varphi^{(k)}_F\to1$ with $\varphi^{(k)}_T$ small), narrowband persistent interference ($\varphi^{(k)}_F$ small with $\varphi^{(k)}_T\to1$), a compact blob (both small) — reading that shape directly from the same entropic fill as the axis reads, where the scalar floor sees only $s_k$. The floor decides *whether* a mode stands above the noise; the footprint reads *what shape* it has, where the labelling of structured noise versus signal (Def 8.2) is made.

**Definition 8.4 (The read-side filter)** — The footprint labels each mode; the filter acts on that label to recover the signal itself. The filter returns the field reconstructed from its resolved modes alone,

$$
  \widehat S = U\,\operatorname{diag}(\tilde s)\,V^{\mathsf H},
$$

with $U,V$ the screen's own left and right singular vectors and $\tilde s$ the shrunk singular spectrum. Because $U,V$ are read from the data, $\widehat S$ introduces no direction the screen does not carry: it is supported on the measured screen's own resolved modes and synthesises nothing. The shrinkage $\tilde s$ is a singular-value nonlinearity against the derived floor, after [Gavish & Donoho 2017]: modes at or below the floor map to $0$ and the survivors are de-biased toward it.

Two maps have to be kept apart here, and only one of them is a projection. With a **hard cut** at the floor ($\tilde s_k=s_k$ above, $0$ below) $\widehat S=U_KU_K^{\mathsf H}SV_KV_K^{\mathsf H}$ is a two-sided orthogonal projection, recovers a noise-free planted burst to machine precision, and is idempotent (§14). With **shrinkage** it is neither: the survivors are de-biased, so a second pass re-estimates the floor from an already-shrunk field and shrinks again. The shrinker's per-entry $\sigma$ is also backed out of $\Phi$ rather than the bulk edge, so it carries the Tracy–Widom margin and with it $\alpha$ (§13.1), and the Frobenius optimality of [Gavish & Donoho 2017] — an asymptotic statement for i.i.d. noise at a known $\sigma$, cut at the bulk edge — is not claimed for it. Because the read is taken on the scale-equalised screen, the output carries the burst's morphology and not the input's amplitude scale; §14 measures both. The transient/persistent dichotomy of Def 8.3 then gates the reconstruction — a persistent narrowband mode ($\varphi^{(k)}_F\le\varphi^{(k)}_T$, the signature of channelised interference) is dropped, a broadband transient ($\varphi^{(k)}_F>\varphi^{(k)}_T$) kept — so the filter removes noise and persistent narrowband modes while preserving the burst morphology, with no template and no fitted constant. The cut is a *shape* test and not a detection: it carries no null and no level, and it drops persistent structure whether or not that structure is interference. It is listed as a stated criterion in §13.1.

The screen is invertible: upsampling both axes back to native resolution reconstructs the waterfall (the inverse fold), and a delay-embedded Tucker (higher-order SVD, HOSVD) [Tucker 1966; De Lathauwer et al. 2000] of the whitened native-resolution field, computed by randomised range-finding [Halko et al. 2011], exposes the within-window fine structure the averaged screen discards.

**N-D fields: the geometry-preserving reduction.** The screen fold (Definition 8.1) and every read above act on a 2-D $W\in\mathbb{C}^{T\times F}$. A higher-dimensional field (a video $T\times H\times W$, a spatial volume, a multichannel record) is first reduced to two axes, and *which* reduction is correct depends on the read: a feature read needs the within-plane correlation that a single flattened feature axis discards.

**Definition 8.5 (Pool versus plane-fold)** — Fix the ordered axis of an N-D field. *Pool* moves it first and flattens every other axis into the feature axis, so each off-axis site is an exchangeable sample of the one ordered process: the correct reduction for the ORDERED reads (the decay $C(\tau)$ and diffraction limit $a_\delta$ of §4, the rates of §9, and the ordered fill $\varphi_T$). *Plane-fold* applies a scalar 2-D read to each intact $(a,b)$ plane, iterating the remaining axes and averaging, so each plane is its own screen and within-plane correlation is preserved: the correct reduction for the FEATURE and plane reads (the feature fill $\varphi_F$ of §3, the mode spectrum of §6, the concentration of §7, and the resolved dimension $K_{\mathrm{signal}}$).

The two reductions are duals: pooling treats the off-axis structure as exchangeable samples of the ordered process, while plane-fold treats each plane's own structure as the object of the read, and getting them backwards changes the answer. The choice is therefore fixed by the read. Both reductions are backend-agnostic.

**The spectrum as a profile.** Definition 8.2 reads the spectrum against the floor and returns a count: how many modes stand above it. Comparing two frames by their spectra asks a different question, and the answer it wants is the whole profile — how far each mode stands from where the frame's own noise law puts it.

**Definition 8.6 (Spectral deviation)** — For a frame of shape $(N,F)$ with noise level $\sigma^2$, let $s_{\mathrm{MP}}(k)$ be the position the Marchenko–Pastur law predicts for the $k$-th largest singular value of a pure-noise frame of that shape and level. The *deviation digest* is

$$\delta_k \;=\; \frac{s_k-s_{\mathrm{MP}}(k)}{\sqrt N},\qquad k=1,\dots,\min(N,F),$$

the whole profile, every mode kept and reported.

Each mode is compared to its own prediction, so every mode carries a reading: a mode below the edge reports its deviation from the noise it is made of. Above the bulk the prediction is bounded by the edge and converges to it as $k/N\to 0$, so the digest agrees with Definition 8.2 exactly where a count is meaningful and keeps reading where a count is not.

The digest carries magnitude and needs no width: it is a function of the frame alone, with no rank, rate, tolerance, grid or fitted constant. Two digests are comparable only when produced by the same instrument, so each carries the identity of the construction that made it and a comparison across two constructions is refused.

---

## 9. The dynamical operator: per-mode decay rates

The entropy-width $a_\delta$ estimates a single correlation length; the ordered axis is a state trajectory, and its one-step propagator gives the per-mode rates. We estimate the propagator recursively from the first frame (online DMD / Koopman [Tu et al. 2014]).

**Definition 9.1 (Streaming accumulators and reduced propagator)** — For states $x_1,\dots,x_T\in\mathbb{C}^F$ and forgetting $\lambda\in(0,1]$ maintain $P_{xx}=\sum_t \lambda^{\,\cdot}\,x_t x_t^{\mathsf H}$ and $P_{yx}=\sum_t \lambda^{\,\cdot}\,x_{t+1} x_t^{\mathsf H}$. The one-step propagator is $A=P_{yx}P_{xx}^{+}$, and $x\mapsto Ax$ is the one-step forecast. In the leading proper orthogonal decomposition (POD) subspace $V_r$ (top eigenvectors of $P_{xx}$ with eigenvalues $w_r$), the reduced propagator is $\tilde A=(V_r^{\mathsf H} P_{yx}V_r)\operatorname{diag}(w_r)^{-1}$, and

$$
  \alpha_k=-\log|\mu_k|,\qquad \beta_k=\arg\mu_k,\qquad \{\mu_k\}=\operatorname{eig}(\tilde A),
$$

with long- and short-range rates $\min_k\alpha_k$ and $\max_k\alpha_k$. The forgetting $\lambda$ is the sole optional input of the construction: $\lambda=1$ (no forgetting) is the stationary default used throughout, under which the accumulators are additive (Theorem 9.3) and the recovery is exact for noise-free linear dynamics (Theorem 9.2); $\lambda<1$ is an explicit choice of memory horizon for a nonstationary stream, a stated modelling input, and it enters no other read (§13.1).

**Well-posed truncation.** The reduced rank $r$ is the numerical rank of $P_{xx}$ when the stream over-determines it ($n_{\mathrm{pairs}}\ge 2F$), and the resolved signal dimension (§8) when it does not. In the under-sampled regime ($T<F$, e.g. a short cutout), truncating to the resolved rank keeps the fit over-determined, so the recovered spectrum — and the forgetting margin read from it — stays well-posed. The resolved rank is a *detection* decision (which feature modes are signal), so its operating point is the caller's declared $(\alpha,\text{null})$ — the same null-provider contract as §8 — defaulting to the derived edge at $\alpha=0.05$. On a well-sampled record the truncation stays inactive, and the exact recovery of Theorem 9.2 holds.

**A state with a hole in it is not a state.** Every other read here treats an unobserved cell as absent and reads what is there. An operator cannot: it is read off *pairs* $(x_t,x_{t+1})$, and a state missing a component is not the state the system was in. Substituting $0$ makes the transition into that component look like decay toward zero, so every rate reads faster than it is — on a planted operator the slowest rate rose by a factor of $23$ as dropout went from $0$ to $35\%$, always in the same direction. No reweighting of the accumulated sums repairs this, because it is the pairs that are wrong and not the weights; entry-wise correction by the number of terms each accumulator entry received also destroys the positive-semidefiniteness of $P_{xx}$ and returns modes outside the unit disc. What the hole can be filled with is what the record itself says belongs there: $x_t=Ax_{t-1}$, so the operator read off the observed cells predicts the missing ones, and re-reading it from the completed record gives a better predictor — iterated to a fixed point. The rate then does not depend on how much was dropped ($0.0195$ at none, $0.0194$ at half), and nothing is invented: a record with no operator acquires none at any dropout, and a record with no gaps is returned untouched.

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

**The observable lift.** Theorem 9.2 recovers a linear operator when one exists on the frame's own coordinates. A trajectory whose one-step map is nonlinear has none, and its screen resolves no modes: near-orthogonal successive states admit no linear map between them. The dynamics are then linear on a larger space of observables [Koopman 1931].

**Definition 9.6 (Delay-embedded observable)** — For a trajectory $W\in\mathbb{C}^{T\times F}$ and a depth $d\ge 1$, the delay embedding stacks $d$ consecutive frames as one observable,

$$Z\in\mathbb{C}^{(T-d+1)\times dF},\qquad Z_{t}= \big(W_{t},\,W_{t+1},\dots,W_{t+d-1}\big),$$

and the lifted operator is the operator of Definition 9.1 fitted on $Z$. This is the delay-coordinate construction of Takens [1981] carried into the streaming operator, and the same stacking that Definition 8.6 gives to a within-window decomposition — one construction, pointed at the operator.

The reads of §9 apply unchanged in the lift: the per-mode rates of Theorem 9.2 are read off the lifted spectrum, and the $h$-step forecast is $\sum_k \phi_k\mu_k^{h}b_k$ — exact eigenvalue powers, so a decaying spectrum stays numerically stable where $A^{h}$ would not.

Takens' condition is $d$ at least twice the intrinsic dimension, which a signal does not announce, so the depth is a stated input. It is not a constant of the construction: a different depth builds a different frame, and the reads of §9 act on the frame they are given. Choosing a depth is choosing which signal to submit, in the same sense as choosing which channels to include, and what the lift is for is the forecast: the reads of §9 act in the delay coordinates, and §14 separates a lifted trajectory from the same trajectory in a random order by forecast. Resolved dimension rises with $d\cdot F$ whether or not the order carries anything.


## 10. The two-way screen: lenses meeting on a shared basis

Sections 2–9 read one signal. Two signals meet only if they are carried into one set of coordinates, and the reads that follow are about the crossing.

**Definition 10.1 (Lens and screen)** — A *lens* is a pair of maps $(\iota_g,\iota_g^{-1})$ for a side $g$, where $\iota_g$ carries that side's surface $S_g\in\mathbb{C}^{T_g\times F_g}$ to a frame $X_g=\iota_g(S_g)\in\mathbb{C}^{T_g\times D}$ on a shared basis of width $D$, and $\iota_g^{-1}$ carries a frame back out. A *screen* is a set of lenses whose entries land on one $D$, together with the frames they have placed. Each side supplies its own energy law, zero and null; the screen supplies none. With $N$ sides there are $N$ conversions and no pairwise table, because every pair meets through the basis.

The ordered axis is each side's own. Reads that pair rows — the coupling of §5, the joint read — require a common ordering and say so; reads that meet on the basis alone do not.

**Definition 10.2 (Beam and étendue)** — The *beam* a side carries is its placed frame read as an aperture: energy $E_g=\lVert X_g\rVert_F^2$, basis the resolved directions of §8, and étendue $G_g=\varphi_T\varphi_F$ — the phase-space area of §3, invariant under the congruences of Lemma 3.4. A beam decomposes into modes, each itself a beam of one, so the same three quantities describe a mode, a bundle and a side.

**Definition 10.3 (Crossing)** — For an ordered pair $(g\to h)$, the *pertinent* energy $P$ is the part of $g$'s beam that $h$'s basis resolves. The *transmissible fraction* is the étendue that fits,

$$\tau=\min\!\left(1,\ \frac{G_h}{G_g}\right),$$

and the crossing is written

$$\underbrace{P\tau}_{\text{absorbed}}\ +\ \underbrace{(E_g-P)}_{\text{bystanding}}\ +\ \underbrace{P(1-\tau)}_{\text{reflected}}\ =\ E_g .$$

**Proposition 10.4 (The partition closes)** — The three parts of Definition 10.3 sum to the beam's energy identically, for every $E_g$, $P$ and $\tau$. This is a statement about the bookkeeping, not a conservation law: the three parts are *defined* as a split of $E_g$, and the proposition records that the split is exhaustive and leaves nothing unassigned.

*Proof.* Ring identity in $(E_g,P,\tau)$; verified in Lean (`crossing_partition`). $\square$

Energy that meets a side resolving nothing matching it is *bystanding*: real, present, and pertinent to some other pairing. It is not loss, and it remains available to a further screen.

**Proposition 10.5 (The radiance bound)** — Let radiance be pertinent energy per unit étendue, $L_g=P/G_g$ and $L_h=P\tau/G_h$. For $G_g,G_h>0$ and $P\ge 0$,

$$L_h\ \le\ L_g,$$

with equality whenever $G_h\le G_g$, and strictly $L_h<L_g$ when $G_h>G_g$ and $P>0$.

*Proof.* If $G_h\le G_g$ then $\tau=G_h/G_g$ and $L_h=P/G_g=L_g$. Otherwise $\tau=1$ and $L_h=P/G_h\le P/G_g$, strictly when $P>0$. Verified in Lean (`radiance_le`, `radiance_eq_of_concentrating`); the strict case is not certified. The equality is not a characterisation — at $P=0$ both radiances vanish for every pair of étendues. $\square$

The bound follows from how $\tau$ is defined — $\tau=\min(1,G_h/G_g)$ caps what crosses at the receiving side's étendue — so it is a consequence of the accounting rather than a physical law recovered from it. The classical brightness theorem it is named for rests on measure conservation under a Hamiltonian flow [Born & Wolf 1999]; nothing here supplies such a flow, and $G=\varphi_T\varphi_F$ is a product of entropic fill fractions rather than a phase-space volume. The reading is a deliberate analogy: the accounting is arranged so that a crossing cannot report more pertinent energy per unit étendue than it received.

**Definition 10.6 (Condensation)** — The absorbed energy itemised over the receiving side's resolved directions. For a quadratic energy law the itemisation is an orthogonal decomposition, so the parts sum to $P$ and each is the energy of its own frame. The étendue bound fixes how much crosses; it leaves open which directions take it.

**Definition 10.7 (Certificates)** — Three properties of a lens are read, each on the side's own signal. *Linearity* compares converting a frame's modes together against converting them apart, and converting a scaled frame against scaling the conversion; a departure that survives balancing means the modes interact inside the conversion. *Losslessness* measures the round-trip residual $\iota_g^{-1}(\iota_g(S))-S$ against the derived floor of §8. A residual below the floor is one this record cannot resolve, which is not the same as one that is not there: the read reports the largest residual it could have missed at the observed $\hat\sigma$ and shape, and a shorter or noisier record admits a larger one. *Realisation* compares what a crossing delivers through the receiving lens against the étendue bound of Definition 10.3, separating conversion loss from the phase-space loss $\tau$ already accounts for.

**Definition 10.8 (Balance)** — Each side is brought to its own zero before any read. The residual of the screen's accounting is scored against the same null as §8, so closure is a calibrated decision at the reader's level — a failure to resolve a residual, reported with the bound it was taken at, and not a demonstration that none exists.

The signed coupling of Theorem 5.6 is the crossing's alignment read, and it is defined only on a shared basis: across two bases the sole basis-free statistic is non-negative and has no exact null, so the screen refuses the pairing.

---

## 11. Symbol sequences: the ordered axis without amplitudes

Sections 2–10 read an ordered axis of amplitudes. When the ordered object is a string of symbols from a finite alphabet the question changes: not how many modes stand above the floor, but how much of the next symbol the past already fixes.

**Definition 11.1 (Block entropies)** — For a sequence over an alphabet $\mathcal X$, let $\mathrm H_n$ be the Shannon entropy of the distribution of $n$-words. Then

$$h_n=\mathrm H_{n+1}-\mathrm H_n=\mathrm H(X_{n+1}\mid X_1..X_n),\qquad h_n^{\mathrm{av}}=\mathrm H_n/n,$$

the conditional entropy of the next symbol and the per-symbol average. $\mathrm H_1$ sees only symbol frequencies; $\mathrm H_n$ carries every correlation shorter than $n$.

In the infinite-data limit both descend to the entropy rate $h$ and bound it from above, $h_n^{\mathrm{av}}\ge h_n\ge h$ [Lesne 2014], with $h<\mathrm H_1$ exactly when correlations are present.

**Proposition 11.2 (Finite-sequence saturation)** — On a sequence of length $N$ the ladder of Definition 11.1 does not reach $h$. Once the number of possible $n$-words exceeds the number of observed windows, $\mathrm H_n$ saturates at $\log_2(N-n+1)$, and $h_n$ turns over and descends below $h$.

*Proof.* At most $N-n+1$ windows are observed, so the empirical $n$-word distribution has support at most $N-n+1$ and $\mathrm H_n\le\log_2(N-n+1)$. Beyond that point increments of $\mathrm H_n$ are bounded by the growth of $\log_2(N-n+1)$, which tends to zero. $\square$

The consequence is stated outright: no single number $h$ is reported from the ladder. The estimate that a single finite sequence does support is the Lempel–Ziv rate, which converges to $h$ for a stationary ergodic source [Ziv & Lempel 1978].

**Definition 11.3 (Order surrogate)** — Compare the block entropies of the sequence against an ensemble drawn by permuting it. A permutation preserves the symbol frequencies exactly and destroys every correlation, so $\mathrm H_1$ is invariant and $\mathrm H_n$ for $n\ge 2$ is not. The standardised departure of $\mathrm H_n$ from that ensemble is the evidence that order carries information, at the reader's level of §12.1 and with no entropy rate estimated anywhere in the comparison.

This is the coherence of §5 in its symbolic form: the same permutation null, standing in the same relation to the same question — is the ordering doing work — read on a distribution over words instead of on a correlation over lags.

---

## 12. The optics dictionary

Every read carries an optical name, and the last column names the result **in this paper** that governs it. Where a classical theorem supplies the vocabulary rather than the proof — Abbe, the Lagrange invariant, the Gabor–Lukosz degrees-of-freedom count — it is named as the source of the name and the governing result is given beside it. Two rows govern nothing and say so. The reads marked ⊢ are certified in Lean (§15).

The aperture-fill reading descends from the classical accounts of how many degrees of freedom a band-limited record carries — the sampling expansion [Whittaker 1915; Shannon 1948] and the prolate-spheroidal concentration results [Slepian & Pollak 1961; Landau & Pollak 1961]. Those fix a count from a *declared* band and interval; §2 and §3 read an effective count from the record's own entropy instead, and no result below is derived from them.

| read | symbol | optical quantity | governing result |
|---|---|---|---|
| axis fill $\varphi_F,\varphi_T$ | $2^{\mathrm{H}(\bar\lambda^{(a)})}/L_a$ | aperture fill on each axis | Lemma 3.2 ⊢ (range); Prop 3.5 ⊢ (invariance). Attainable range narrows to $[1/\min(T,F-1),1]$ by the centring rank bound |
| screen fill $\varphi$ | $2^{\mathrm H_{\mathrm{sv}}}/n$ | fraction of active SVD modes | Lemma 3.2 ⊢. The exponential entropy of a normalised spectrum as an effective count is [Campbell 1960]; on singular values it is the effective rank [Roy & Vetterli 2007] |
| magnification | $1/\varphi$ | reciprocal reach / oversampling | Definition 3.1 — the reciprocal, by definition; no theorem is invoked |
| étendue $\mathcal E$ | $\varphi_F\varphi_T$ | joint 2-D aperture area | Prop 3.5 ⊢ — invariance under relabelling and per-variable phase, which is *not* the Lagrange invariant; the optical name is by analogy |
| space–bandwidth | $n_F\,n_T$ | resolvable-spot count | Definition 3.3. The classical count is a Gabor–Lukosz invariant [Lukosz 1966; Toraldo di Francia 1969]; $\mathrm{SBW}$ here is not invariant — it reads $TF$ on any screen the fold leaves at native resolution |
| Strehl $\mathcal S$ | $\lambda_1/\!\sum\lambda$ | dominant coherent-mode fraction | Lemma 3.4 ⊢ (range). The leading share of the ordered-axis correlation spectrum; the coherent-mode reading is by analogy [Wolf 1982] |
| ordered coherence $z$ | $(A-\mu)/\sqrt{\operatorname{Var}_\pi[A]}$ | ordered-axis coherence | Theorem 5.2 ⊢ (mean); Definition 5.3 (variance, assembled after [Cliff & Ord 1981]; not certified). A standardised deviate, not a bounded coherence |
| decay $C(\tau)$ | biased autocovariance | OTF (Wiener–Khinchin) | Lemma 4.2 ⊢ (positive semidefinite); Cor 4.3 ⊢ (peak at zero lag) |
| diffraction limit $a_\delta$ | $2^{-\mathrm H(C^2)}$ | inverse resolvable spacing (a cutoff) | Definition 4.4. Abbe/Rayleigh supplies the name; no result here is derived from it |
| shape factor $g$ | $\xi\,a_\delta$ | dimensionless shape factor (Rayleigh) | Prop 4.5 ⊢ — invariance under integer block-repetition |
| Abbe factor | $a_\delta/\varphi_F$ | resolution per feature fill | §4.2, definitional. **No read depends on it** |
| Fresnel number | $\sim\!\text{win}\cdot\varphi_T$ | near/far-field coordinate | §4.2, a scaling by analogy. **No definition and no read depends on it** |
| Mercer ratio $\rho$ | $a_\delta^{\mathrm{spec}}/a_\delta$ | temporal–vs–spectral width | Prop 4.7, resting on Lemma 4.2 ⊢ for positivity. Mercer's theorem is not invoked; the population statement is that both are functionals of one spectral density |
| propagation constant | $\alpha_{\mathrm p}+i\beta_{\mathrm p}$ | mode contrast + carrier | Definition 6.1 (the finite-size edge); Lemma 6.2 ⊢ (Weyl interval). $\beta_{\mathrm p}$ needs a complex record and a declared continuous feature axis |
| resolved power $\Pi$; dominance | $\sum_{\lambda_k>\lambda_+}\!(\lambda_k-\lambda_+)$; $\tfrac{\lambda_1-1}{F-1}$ | resolved spectral power; mode dominance | Definition 6.1, against the Tracy–Widom finite-size edge |
| concentration | $\sigma_1^2/M$; $\lVert\bar x\rVert$ | axial focus vs directional resultant | Prop 7.2 ⊢ — the axial/directional dichotomy under per-row phase |
| decay rates | $-\log\lvert\mu_k\rvert,\ \arg\mu_k$ | per-mode attenuation + frequency | Theorem 9.2 ⊢ — exact recovery for a noise-free linear map; exact DMD / Koopman |
| clean field | $U\operatorname{diag}(\tilde s)V^{\mathsf H}$ | resolved-mode reconstruction | Definition 8.4. The hard-cut form is a two-sided projection; with shrinkage after [Gavish & Donoho 2017] it is not |
| crossing partition | $E_g=P\tau+(E_g-P)+P(1-\tau)$ | absorbed / bystanding / reflected | Prop 10.4 ⊢ — an identity in $(E_g,P,\tau)$; the split is definitional |
| radiance | $L=P/G$ | radiance across a crossing | Prop 10.5 ⊢ — bounded by construction of $\tau=\min(1,G_h/G_g)$; the brightness reading is by analogy |

Nothing in §11's symbolic reads carries an optical name, and they are absent from this table for that reason.

**The limit is over-determined.** Independent reads each pin a strictly positive resolution limit for a filled finite aperture, from *distinct* objects: the decay-lag entropy gives $a_\delta$ (Abbe/Rayleigh); the stationary correlation operator's spectrum gives $a_\delta^{\mathrm{spec}}$ (Mercer, §4.3), which agrees with $a_\delta$; and the power marginals give a finite space–bandwidth product $\mathrm{SBW}$. These draw on different entropies (the decay-lag entropy $\mathrm H_C$, the operator spectrum, and the marginal entropies $\mathrm H_T,\mathrm H_F$), so their agreement is a cross-check.

---

## 13. Reproducibility

The construction is realised as a small Python package with a single backend-agnostic code path: every kernel is written once against the array namespace of its input, so a numpy array runs on CPU and a torch tensor runs on its device (GPU), staying on-device; torch is imported lazily only when a tensor is present. All reads are deterministic (the coherence is the closed-form $z$-score of §5), so a numpy and a torch evaluation agree to floating-point round-off and repeated evaluations are bit-identical. Complex inputs are supported throughout, with the coherent/incoherent dispatch of §4.

A stack of frames is read by the same construction at once. The resolved sector of §8 is the spectral projector onto the states above the noise edge, $P=\mathbb 1(C>t)$, so the reads are $K_{\mathrm{signal}}=\operatorname{tr}P$ and the per-row resolved energy $sPs^{\mathsf T}$. Two exact realisations compute it, chosen by where the data lives: a LAPACK singular value decomposition off CUDA, and the matrix sign function on CUDA, where batched dense factorisations are the bottleneck. Both are exact, so the integer count $K_{\mathrm{signal}}$ is identical across the two and the continuous energies agree to floating-point round-off. The fold decision (Definition 2.2) is the same call the per-frame read makes, so a stack and a frame resolve alike. A caller may bound threads, memory and device use; the work is then chunked, and chunking does not change the result.

A test suite pins the identities of this paper: the full optics read as a regression contract; numpy/torch parity; the invariants $\mathcal E=\varphi_F\varphi_T$, $\operatorname{mag}=1/\varphi$, the peak-at-zero-lag and PSD-autocovariance of §4, the permutation-null mean and variance of the coherence $z$ (§5, against a full enumeration of all $N!$ permutations), the rate recovery of Theorem 9.2, the additive splice of Theorem 9.3, the axial/directional dichotomy of Proposition 7.2, and the projection and idempotence of the read-side filter (§8, Def 8.4); the round-trips (screen reconstruction, tensor HOSVD, factor serialisation); degenerate-input robustness; and the equality of the batched read with the per-frame one, on both backends. The package, its test and validation suites, and the Lean development are available at https://github.com/Agience/entroptics.

### 13.1 Provenance of every constant and choice

The construction is parameter-free in a precise, verifiable sense, tabulated here. Every fixed number it uses is a *derived* mathematical quantity; every structural choice is the canonical realisation of a *stated requirement*; the only quantity the instrument does not itself supply is the reader's decision risk.

| Quantity | Role | Kind | Provenance |
|---|---|---|---|
| $1/\Phi_{\mathcal N}^{-1}(3/4)$ | MAD $\to$ robust $\sigma$ | derived | Gaussian consistency of the MAD ($\Phi_{\mathcal N}$ the standard-normal CDF, distinct from the floor $\Phi$ of §8) |
| $\mathrm{Var}(\log\widehat{\mathrm{MAD}})\!\cdot\!N\!=\!1.360$ | whitening shrinkage weight | derived | influence-function variance $1/(16 f(D)^2 D^2)$ |
| $c_F=(1-\tfrac{2}{9F})^3$ | noise-level de-bias | derived | Wilson–Hilferty $\chi^2_F$ median |
| $(N-1)/N$ | noise-level de-bias | derived | centring degree of freedom |
| $q_\alpha$ (Tracy–Widom$_1$) | floor threshold shape | derived | universal $\mathrm{TW}_1$ quantile |
| $\beta_F$ (feature-fold guard) | whether to fold at all | derived | the larger of a Cantelli bound on the exact Dirichlet null deficit and the width change that moves the Marchenko–Pastur edge past its Tracy–Widom margin (Def 2.3). No cap: neither term can reach $\log_2F$ |
| biased autocovariance ($/T$) | OTF estimator | forced | positive-semidefiniteness of the OTF (Lemma 4.2) |
| feature-only fold | resolution | forced | the ordered reads (§4, §5, §9) are lag/shift statistics needing native spacing |
| $z_F>z_{1-\alpha}$ (feature adjacency) | fold licence | derived + stated | area-meaning adjacent cells is faithful only on a continuous axis; null from §5 (Def 2.2) |
| MAD whitening + shrinkage | noise scale | forced | maximal breakdown-point robust scale; empirical-Bayes weight |
| $|W|^2$ power marginal | matched-scale marginal | definitional | the intensity (the incoherent read of §4) |
| $2^{\mathrm H}$ effective count | fill fraction | definitional | minimum coefficient rate of a normalised spectrum [Campbell 1960]; exponential entropy as extent [Campbell 1966] |
| sweep gate $z_{1-\alpha/n}$ | per-patch level over $n$ patches | derived | multiplicity correction of $\alpha$ (Definition 3.7) |
| on-pulse cut | extent of a swept band | derived | the profile's own MAD scale at $\alpha$ (Definition 3.7) |
| $s_{\mathrm{MP}}(k)$ | per-mode noise prediction | derived | Marchenko-Pastur quantile at the frame's shape (Definition 8.6) |
| resolution floor | dead-channel test in the whiten | derived | the frame's pooled MAD $\times$ the working dtype's $\varepsilon$ (Definition 8.1) |
| round-off bound $n\varepsilon$ | residual vs arithmetic | derived | backward error of a product contracted over $n$ [Higham 2002] (Definition 10.7) |
| commensurate feature channels | the marginal of §2 | **declared precondition** | the read is taken on raw amplitudes, so a frame of mixed units reads its units; not detectable from one frame, and not repairable by pre-standardising the channels — that is the whiten-first order (§14) |
| $\alpha$ (false-alarm level) | decision risk | **external input** | the reader's operating point (Neyman–Pearson); selects $K_{\mathrm{signal}}$, and enters two read *values* — the shrinkage scale of Def 8.4 and the fold gate of Def 2.2 |
| $\lambda$ (DMD forgetting) | memory horizon | **optional input** | $\lambda=1$ is the stationary default (Theorem 9.2); $\lambda<1$ is an explicit nonstationarity choice |
| $w_{\min},\varepsilon$ | frame-window extent | locality | bound only the frame-level window (§9.4); enter no global operator read |
| numerical-rank tol. | rank of $P_{xx}$ | implementation | drops null directions below $10^{-10}$ of the top energy |
| DMD signal-rank cut ($\alpha$, null) | well-posed truncation at $T<F$ | declared input | the §8 floor at the caller's operating point (default derived edge at $\alpha=0.05$); acts only in the under-sampled regime |
| $\varphi_F>\varphi_T$ geometry cut | drops persistent narrowband modes | stated criterion | a mode wider in feature than in order is persistent structure, not a transient (Def 8.4). No null and no level: it is a shape test, not a detection |
| shrinkage vs hard cut | survivor amplitudes | stated criterion | Gavish–Donoho shrinkage is the default; the hard floor cut is the alternative, and only the latter is a projection (Def 8.4) |
| $\sigma=\Phi/(\sqrt T+\sqrt F)$ | per-entry noise for the shrinker | derived + approximation | the Marchenko–Pastur edge inverted for $\sigma$; $\Phi$ carries the Tracy–Widom margin, so the $\sigma$ handed to the shrinker is the edge value times $\sqrt{1+q_\alpha\varsigma_J/\mu}$ |
| Chiani Gamma approximation | $\mathrm{TW}_1$ tail at any $\alpha$ | derived + approximation | [Chiani 2014]; supplies $q_\alpha$ where no tabulated quantile exists, with a stated maximum CDF error of $\approx7\times10^{-3}$ |
| lag $\ell$ | which rows the coherence compares | stated criterion | $\ell=1$ is adjacency (Def 5.1); any $\ell$ has the same closed-form null |
| patch width and step | the swept aperture | locality | bound the sweep's window (Def 3.7); the multiplicity correction is applied over whatever count they produce |
| window schedule | the scale profile | locality | start length and step of the increasing sweep (Def 3.6) |
| delay depth $d$ | the Takens lift | declared input | raises resolved dimension with $d\cdot F$ whether or not the order carries anything (§9.6); not derivable from one record |
| which axis is ordered | the role assignment | **declared precondition** | not detectable from one frame: every ordered read (§§4, 5, 9) is a lag statistic and presumes it |

(i) No constant is fitted to data or calibrated to a substrate. (ii) Every constant is *derived*: a robustness consistency factor, an influence-function variance, a $\chi^2$ median, a universal Tracy–Widom quantile. (iii) Every structural choice is *forced* by a requirement, the biased autocovariance by the positivity of the OTF, the feature-only fold by the native lag spacing the ordered reads require, the robust whitening by a maximal breakdown point. (iv) The sole *decision* input external to the instrument is the false-alarm level $\alpha$, the operating point at which a continuous read is called a detection. By the Neyman–Pearson lemma $\alpha$ cannot be derived from the data, since it encodes the reader's relative cost of a false alarm against a miss. The reads expose, per mode, the standardised deviate $g_k$ and its tail probability $p_k=P(\mathrm{TW}_1>g_k)$ against the universal null (the evidence, §8), and $\alpha$ selects how many modes a reader calls resolved, $K_{\mathrm{signal}}=\#\{k:p_k<\alpha\}$. It also enters two read *values*, and a reader lowering it should know both. The floor $\Phi$ carries the Tracy–Widom margin, and Definition 8.4 backs the shrinker's $\sigma$ out of $\Phi$, so every surviving amplitude moves with $\alpha$ — on the §14 calibration burst, by $0.25\%$ in Frobenius norm between $\alpha=0.2$ and $\alpha=0.001$ at a fixed $K_{\mathrm{signal}}$. And Definition 2.2's continuity gate is taken at $z_{1-\alpha}$, so $\alpha$ decides whether the feature axis is folded at all — an irreversible area mean, not a miscount. The two roles are independent and a reader wanting them separate should set the fold gate's level explicitly. (v) Beyond $\alpha$ and the optional forgetting $\lambda$, the table lists a few *locality and implementation* constants — the frame-window extent $w_{\min},\varepsilon$, a numerical-rank tolerance, and the $\tfrac12\log_2 F$ fold cap; each bounds a window, drops null directions, or keeps the fold non-vacuous, and none is fitted to data or a substrate. The fold cap is the exception to the last clause: where it binds ($F\gtrsim 2T\ln F$, which includes the wide frames of §14.1) it *is* the guard, and the condition reduces to $\mathrm H_F<\tfrac12\log_2F$ — more than $\sqrt F$ effective channels — with no finite-sample content. The exponent is a stated criterion, not a derived one.

The finite-size Tracy–Widom null of §8 and the stationarity behind the rates of §9 are the assumptions the classical results being specialised already make; the construction introduces none beyond them.

---

## 14. Empirical validation

Every read is checked against a planted ground truth on seeded synthetic signals (`research/validation/`, deterministic, regenerated by `run_all.py`). Each experiment recovers the quantity it claims to.

*Diffraction limit tracks correlation length.* For first-order autoregressive AR(1) fields of known correlation length $\rho$ (48 channels, $T=4000$), the entropy-width limit $a_\delta$ and the integral length $\xi$ are strictly monotone in $1/\rho$ (Spearman $1.000$); $a_\delta\sim 1/\rho$ with log-log slope $0.994$ ($R^2=0.9992$), and $\xi$ recovers $\rho$ to a constant shape factor.

*Decay-rate recovery (Theorem 9.2).* For a linear trajectory $x_{t+1}=Ax_t$ with known eigenvalues, the per-mode rates are recovered to machine precision at zero noise (max $|\alpha\text{ error}|=4.5\times10^{-16}$), degrading smoothly as the signal-to-noise ratio falls ($2\times10^{-2}$ at $20$ dB).

*Resolved dimension recovers planted rank.* With $K\in\{0,1,3,5\}$ orthogonal modes planted at signal-to-edge ratio $\ge 1$ across four aspect ratios (including $600\times40$ and $40\times600$), $K_{\mathrm{signal}}$ recovers the exact count with mean accuracy $1.000$ at ratio $1$ and $1.000$ at ratio $2$, at a $K=0$ specificity of $0.95$ overall. One cell departs from exact recovery above that range: at ratio $4$ the tall $600\times40$ array reads a planted $K=1$ as mean $K_{\mathrm{signal}}=1.13$ (accuracy $0.933$), every other shape and count recovering exactly. The calibrated operating range is ratio $1$–$2$, where the derived floor is false-alarm calibrated across aspect ratios, tall to wide. Below the edge nothing is resolved: at ratio $0.5$ the read returns mean $K_{\mathrm{signal}}$ of $0.3$–$1.3$ against planted counts of $3$ and $5$. That is the floor declining to certify modes that sit beneath it, which is what a floor is for — a sub-edge mode the read *did* resolve would mean the floor was wrong.

*The resolved count against the standard selectors.* On the same planted signals at the same seeds, three established rank selectors run in their standard form with nothing tuned: the Gavish–Donoho optimal hard threshold in its unknown-noise form [Gavish & Donoho 2017], and Wax–Kailath MDL and AIC. Exact-count accuracy over the 36 cells at ratio $\ge1$: $K_{\mathrm{signal}}$ $0.998$, AIC $0.946$, Gavish–Donoho $0.935$, MDL $0.889$. The margin is modest and the comparison is close, which is the useful reading — the derived floor is not doing something the field cannot, it is doing it without a fitted constant and without a known $\sigma$. Two of the four are also narrower in application: AIC and MDL are derived for $n$ snapshots of $p$ variables with $n>p$, are undefined on $9$ of the $36$ cells, and are reported as inapplicable there rather than guessed.

*An i.i.d. null cannot cover a correlated ordered axis, and this is how far each method is wrong.* The floor's null is an i.i.d. bulk, and the ordered axis of a real record is a correlated process — which is what §4 exists to measure. On pure AR(1) rows at unit marginal variance with **no planted signal**, the i.i.d. control ($\rho=1$) behaves as designed: a mode is resolved in $0$–$5\%$ of draws across three shapes. At every $\rho>1$ tested ($2$ to $32$) a mode is resolved in **$100\%$** of draws, mean $K_{\mathrm{signal}}$ between $5$ and $15$. The effect is in the spectrum and not in any estimator: at $(200,200)$ and $\rho=2$ the leading singular value of the raw field is $36.9$ against a Bai–Yin i.i.d. edge of $28.3$, with $13$ values above that edge. Serial correlation genuinely moves the bulk, so **no** method reading against that edge can be right on such a record.

What separates them is how far each is wrong. Run on the identical records, the mean spurious count at $\rho>1$ is $K_{\mathrm{signal}}$ $10.1$, MDL $19.0$, Gavish–Donoho $22.9$, AIC $44.7$ — the derived floor over-reads the least of the four, by a factor of two against the nearest singular-value threshold and four against AIC. §8 states the same caution for noise correlated across channels; it holds across rows as well, and this is its size and its context. The coherence read of §5 fires on the same records, which is correct — adjacent rows genuinely are more alike than a re-ordering — and the two must not be read as one number. What this bounds is a record whose *noise* is correlated along the ordered axis: a dedispersed waterfall, whose noise is close to white in time, is not affected, and the reads of §14.1 are not. A record carrying $1/f$ drift or a common-mode gain is, and the correction the floor would need is an effective row count below $N$ — a quantity §4 already measures as $\xi$, which is not pursued here.

*Coherence separates order from noise and is null-calibrated.* A smooth ordered signal reads $z=26.7$; its own row permutation reads $z=-0.15$. Over $2000$ i.i.d.-noise draws across five shapes the closed-form-variance null $z$ (Def 5.3) has mean $0.01$ and standard deviation $1.00$; the one-sided rate $P(z>2)=0.027$ sits near the $\mathcal N(0,1)$ target $0.023$, the small residual being the permutation distribution's own tail skew, not the standardisation.

*The Mercer ratio flags nonstationarity.* Across sliding windows $\rho$ is near-constant for a stationary AR(1) record (mean coefficient of variation $0.06$ over $12$ seeds) and drifts markedly more at a regime switch (mean CV $0.58$), a $9\times$ separation.

*Étendue and space–bandwidth read the aperture size.* Both rise strictly monotonically (Spearman $1.000$) with a signal's planted effective rank and feature bandwidth; $\varphi_F$ tracks the occupied fraction $R/F$ and $n_F$ recovers the bandwidth $B$.

*The coupling recovers a planted sign against an exact null (Thm 5.6).* Two sides sharing a planted carrier at signed strength $\rho$ in one shared basis recover the sign in every draw at $|\rho|\in\{0.5,1\}$, while independent sides stay below the level. The closed-form permutation variance $\operatorname{tr}(C_AC_B)/(T-1)$ matches $100\,000$ brute-force uniform row re-pairings to within $2.0\%$ at every shape tested — $(40,5)$, $(64,8)$, $(120,3)$ real and $(64,6)$, $(96,4)$ complex — so the standardisation is exact and the tail shape alone is asymptotic. The residual is the sampling error of a variance taken from a distribution that is not normal (Remark 5.4), and shrinks with the draw count: the worst case reads $1.029$ at $20\,000$ re-pairings and $1.006$ at $200\,000$. The complex shapes are the ones that exercise the real embedding, and a real-only check cannot see it. Over $1600$ independent pairs across four shapes the null has mean $0.011$, standard deviation $1.018$, and fires at $0.049$ against a nominal $\alpha=0.05$.

*A fold needs continuity, not just concentration (Def 2.2).* A narrow line on a continuous frequency axis and a few mutually unrelated active channels concentrate comparably ($9.7$ and $3.1$ effective channels of $64$), so a guard on $\mathrm{H}_F$ alone folds both. The feature-axis adjacency score separates them ($z_F=8.4$ against $0.03$), and the direct test of what an area mean claims — fold to the concentration-chosen width, unfold, measure the residual — gives $0.44$ on the continuous axis against $0.98$ on the nominal one, which recovers $2\%$ of the signal. The residual falls monotonically as the line is widened ($0.65\to0.44\to0.25$ at widths $8,16,40$), tracking continuity as claimed. Under the repaired guard the continuous family folds in $40/40$ draws and the nominal family in $2/40$ — the $5\%$ false-alarm rate of the level the test is taken at.

*The read-side filter recovers the signal's morphology (Def 8.4).* Two maps have to be kept apart. The hard-threshold form — truncate at the derived floor, no shrinkage — is a two-sided projection: on a planted broadband burst at zero noise it recovers the input to machine precision (relative error $<10^{-12}$) and is idempotent. The front door composes that with per-channel whitening and Gavish–Donoho shrinkage, and the composition is neither: shrinkage de-biases the surviving singular values, and the read is taken on the whitened screen, so the output carries the burst's *morphology* and not its amplitude scale.

Measured through the front door on the calibration burst, the image correlation is $0.983$ at S/N $=10$, $0.992$ at $50$ and $0.987$ at $1000$, against a raw-field correlation of $0.887$, $0.995$ and $1.000$ — so the filter helps where noise is non-trivial and the relative error against the input stays large throughout, which is the scale statement above. Fidelity is *not* monotone in signal-to-noise: it peaks in the mid band and falls toward the noiseless limit ($0.291$ at S/N $=10^6$, $0.154$ at zero noise), where whitening each channel by its own MAD divides by a vanishing scale. That limit is a property of the whitening, not of the threshold, and it is pinned in `test_extract.py` so it cannot move unremarked.

Under random channel dropout — channels *dropped*, as the read-side path drops them, not zeroed — the filter resolves the burst in all $10$ draws at every fraction through $78\%$, recovering the surviving channels at $98.3$–$98.6\%$ throughout; at $81\%$ it resolves in $7$ draws of $10$ and at $84\%$ in $3$, still at $98\%$ recovery where it resolves at all. Resolving nothing and recovering poorly are different outcomes and Figure 1 plots them as separate series. A persistent modulated narrowband tone added alongside the burst is dropped by the $\varphi_F>\varphi_T$ geometry cut (correlation with the tone $<0.2$) while the burst is preserved (correlation $>0.9$). These checks are pinned in `test_extract.py`; both series behind Figure 1 are committed alongside it (`calibration.csv`).

![Figure 1. The read-side filter (Def 8.4) calibrated on a synthetic burst: the injected burst, the same field plus noise, and the recovery (top); recovery versus the fraction of channels dropped — as two series, the fraction of draws that resolve anything and the recovery among those that do — the field plus noise and random channel dropout, and the recovery of the surviving channels (bottom).](figures/calibration.png)

*A crossing conserves energy and cannot brighten a beam.* Two sides placed on one shared basis through lenses of differing surface width, read in both directions across six étendue ratios. The partition of Proposition 10.4 closes to machine precision on all twelve crossings, which is a check on the arithmetic rather than evidence for the proposition: it is an identity in $(E_g,P,\tau)$ and holds for any three numbers. Radiance never rises (12/12), and in every concentrating crossing it is carried across exactly (6/6), which is the equality case of Proposition 10.5.

*The lift buys a forecast, and a shuffled control is what shows it.* A logistic map and an amplitude-modulated oscillator, $T=600$, have no linear one-step map on their own coordinates. In delay coordinates at $d=24$ the operator forecasts a held-out tail at $0.074$ and $0.040$ of persistence, against $0.492$ and $0.527$ for the same trajectories in a random order — a separation of $6.6\times$ and $13.4 imes$. The resolved dimension does not carry this claim: the shuffled logistic trajectory resolves 15 modes against the real one's 5, because the count rises with the observable's dimension $d\cdot F$ and with whiteness. Order is what the forecast reads, and disorder is what it refuses.

*Order is detected by the permutation null, not by an entropy rate.* Repeat-probability chains over an alphabet of 4, $N=4000$: an i.i.d. sequence reads $|z|=2.1$ against its own permutation ensemble and is not called ordered, while the same read at repeat probability $0.95$ gives $|z|=2219$, monotonically through $140$, $604$ and $1235$ at intermediate strengths. The first-order entropy is $2.00$ bits throughout, so the departure is carried entirely by $n\ge 2$ — a permutation preserves symbol frequencies exactly, and only the ordering is destroyed. The saturation of Proposition 11.2 is exhibited alongside: $\mathrm H_n$ turns toward $\log_2(N-n+1)$ as the word count passes the window count, at $n=6$ for this alphabet and length.

*Structure is reported at the scale that contains it.* A single ordered mode of known period over a unit noise floor, $T=512$, swept over trailing windows. The resolved window of Definition 3.6 tracks the planted period across $16,32,64,128$ — Spearman $+1.00$, monotone — and a window shorter than the structure resolves nothing.

*The reads divide by the measured extent, and a mask is not a zero (Def 2.1).* A rank-3 signal at $(T,F)=(200,256)$ with 0–90% of its channels blanked, 20 draws per fraction. Taken through NaN or through a mask over a finite wrong value, $\varphi_F$, $\varphi_T$, $\varphi$, $\mathcal E$ and the Strehl ratio all match the same frame with those channels *deleted* to $2.8\times10^{-16}$ — floating-point equality, on frames whose nominal width is up to ten times their measured extent. Substituting $0$ for the same channels is a different measurement and reads as one: $\varphi_F$ moves by $0.006$, $0.018$, $0.051$ and $0.137$ at 25%, 50%, 75% and 90% blanked, rising with the fraction blanked because that is the factor $L_a/L_{\mathrm{eff}}$ the extent corrects.

*A read is a property of the signal, not of the units it was recorded in (Def 8.1).* A rank-3 signal at $(T,F)=(300,64)$, multiplied through 42 orders of magnitude of recording scale ($10^{12}$ to $10^{-30}$): $\varphi$, $\varphi_T$, $\varphi_F$, $\mathcal E$, the Strehl ratio, $a_\delta$, the focus, $K_{\mathrm{signal}}$ and the coherence are all invariant to $10^{-14}$ relative — $K_{\mathrm{signal}}=3$ and coherence $0.404$ at strain scale ($10^{-21}$) exactly as at unit scale. Replacing that derived floor with a fixed absolute one — the value the derived floor itself takes at unit scale, then held constant — breaks the read at $7$ of the $8$ scales: it resolves all $64$ modes at $10^{3}$ and above, and none at $10^{-6}$ and below, against a true $3$ throughout. It is correct only at the scale it was fixed at. Uniformly quantized over its own range from 16 bits down to 2, the derived floor tracks the quantization noise and stays within a factor of $1.69$ of the unquantized floor at every depth. The resolved count holds at the planted $3$ from 16 bits down to **4**, and departs below it — $4$ at 3 bits, $1$ at 2. The mechanism is the assumption running out rather than the read failing: the quantization step grows from $\sigma_q=0.38$ at 4 bits to $0.81$ and $1.90$, so at 2 bits the record is a four-level staircase and its error is a deterministic, signal-correlated distortion, not the i.i.d. bulk the floor is read against. The floor's own non-monotone response there ($24.0\to14.2\to19.0$) is the visible sign of it.

*Channel scatter separates a decay that is structure from one that is noise (Def 4.8).* An uncorrelated record and an AR(1) record with a planted correlation length $\rho=8$, $T=1500$, each read at $F=4,16,64,256$ channels. On the uncorrelated record the noise and tail shares coincide at every width (ratio $0.88$–$1.05$): all of the power away from zero lag is channel disagreement. With a correlation length planted they separate by up to $227\times$ (ratio $0.237$ falling to $0.0044$), so the same tail is read as structure. The scatter falls monotonically with $F$ in both families ($0.113\to0.0020$ and $0.193\to0.0035$), and the uncorrelated $a_\delta$ closes on its answer of $1$ from below as it does ($0.36\to0.97$) — the read is consistent, and the remedy a wide scatter points at is more channels.

### 14.1 Real data: fast radio bursts

The synthetic checks plant a known truth; a real substrate tests whether the *same untuned instrument* transfers. We apply the plain library (zero tuning) to the publicly released CHIME/FRB Catalog-1 waterfalls [CHIME/FRB Collaboration 2021; data at the CANFAR archive, CISTI.CANFAR/21.0007] at native 16384-channel resolution through the read-side filter (`research/figures/frb_panel.py`), which supplies only observer facts: dead channels are dropped — never zero-filled — and the surviving channels are passed to the aperture front door. "Dead" is the catalogue's own mask together with a zero-variance test, and the surviving width is $9{,}760$–$11{,}696$ of the $16{,}384$ recorded channels across the four events. The read is then taken at whatever width the guard of Definition 2.2 folds to, which for these frames is $5{,}949$–$10{,}972$ — the instrument reading at the resolution it judges the record to carry, which is the untuned behaviour and not a choice made for the figure. The fold changes nothing it resolves: $K_{\mathrm{signal}}$ is identical read folded or at the live width on all four, and the contrast is equal or slightly higher folded. Each width is recorded per burst in the companion table. The catalog waterfalls arrive already dedispersed; the library adds no dedispersion of its own, and any residual sweep shows as diagonal structure the read catches directly. Everything downstream is the library.

For each burst the filter resolves the signal above the derived noise floor ($K_{\mathrm{signal}}\ge 1$) and reconstructs it as the projection onto its resolved modes (Def 8.4): the Gavish–Donoho shrinkage attenuates the noise sea and the $\varphi_F>\varphi_T$ geometry cut removes persistent narrowband interference, leaving the burst morphology intact. The per-burst reads annotated on each panel — the resolved count $K_{\mathrm{signal}}$, the bounded *contrast* $\sigma_1/\Phi$ (the leading singular value over the screen floor), and the coherence $z$ of §5 — report the contrast; a per-mode tail probability would rest on a Tracy–Widom approximation [Chiani 2014] far outside its calibrated range at these deviates. The figure routine writes these per-burst values, with the live-channel and RFI counts, to a companion table (`frb_panel.csv`) regenerated from the CANFAR waterfalls. Figure 2 places the entroptics reconstruction beside CHIME's own *fitburst* forward model and the raw waterfall on four bright events, drawn at random from the release and read with no per-substrate tuning. The three panels are shown side by side for direct comparison and no similarity score is reported, because there is nothing to score against: the source of a fast radio burst is unknown, and the forward model is a parametric fit to the same waterfall, so a correlation against it would measure agreement with a model rather than with the burst. The per-burst reads in `frb_panel.csv` carry the record's shape alongside them — $T=19$–$38$ against $16{,}384$ recorded channels — so the aspect ratio each read was taken at is on the page.

*An independent check on the read.* CHIME's forward model is fitted with the shape of a burst already in it — Gaussians in time against a running power law in frequency — and solves for that shape's parameters. The read is given no model of a burst at all, and keeps whatever modes clear the floor. The two share no assumptions, so agreement between them on the same waterfall is evidence that the read is finding the burst rather than an artifact of its own construction.

Correlation against the forward model, every quantity scored on the live channels: the raw waterfall reads $0.116$–$0.371$ (mean $0.197$), the read $0.520$–$0.612$ (mean $0.564$). The read arrives smoothed along frequency and the model is smooth too, so a box-average of the raw to the same width is scored alongside it — everything the fold does and nothing else. It reads $0.213$. **The fold accounts for $4\%$ of the gain and the read for the rest.** All three columns are in the companion table, per burst.

Neither is ground truth. The source of a fast radio burst is unknown, and the forward model is itself a fit to this same waterfall with no per-channel freedom. The agreement says that two methods assuming nothing in common find the same burst; it does not say either has recovered it correctly.

The panels differ most obviously in how clean they look, and that is the display rather than the data: each is contrast-stretched between its own median and 99.7th percentile, and the model is noiseless and non-negative (median $0$, maximum $\approx0.5$) where the waterfall spans $-0.9$ to $+7.0$.

A further **12 waterfalls drawn uniformly at random** from the same public release, at a fixed seed, are read by the identical path (`research/figures/frb_spotcheck.py`, table in `frb_spotcheck.csv`). All twelve are read without incident, and $K_{\mathrm{signal}}\ge1$ on eight of them. The four that resolve nothing are the four whose leading singular value sits at or below the derived floor (contrast $0.98$–$0.99$, against $1.30$–$4.22$ for the events of Figure 2), the floor refusing a mode it cannot separate from the bulk. Record length does not separate the two groups: both span $T=19$–$95$. What this establishes is that the untuned instrument runs on records it was not selected against, and reports what it finds on them.

![Figure 2. Entroptics on real CHIME/FRB Catalog-1 bursts, read untuned, four panels per event. Left to right: the raw dedispersed waterfall as measured; CHIME's own *fitburst* forward model; the entroptics reconstruction — the shrinkage against the derived floor with the persistent-structure cut, taken on the folded screen and mapped back onto the recorded frequency axis for display; and what the read did not keep, after matching the amplitude scale by least squares, since the read carries morphology rather than the input's scale. Per-burst $K_{\mathrm{signal}}$, contrast $\sigma_1/\Phi$, coherence $z$ and the width read at annotate each row. **Each panel is contrast-stretched independently**, between its own median and 99.7th percentile: the four hold quantities on different scales, and the model in particular is noiseless and non-negative, so the comparison to make by eye is of morphology and not of contrast.](figures/frb_panel.png)

---

## 15. Formal verification

Part of the development is machine-checked in Lean 4 / Mathlib: forty-four theorems that compile with no `sorry`, each depending only on the standard axioms `propext`, `Classical.choice`, and `Quot.sound`. The count is not a measure of coverage — roughly twenty carry substantive content and the rest are one-line specialisations, ring identities, or aliases of Mathlib results. The development states no definitions of its own: each theorem is a claim about explicit sums and matrices, and the identification with the reads above is carried by this paper, not by Lean. What is **not** certified is as important as what is: Definition 5.3's variance — the denominator of every $z$ in the paper — rests on [Cliff & Ord 1981] plus the simulation of §14; so does Definition 2.2's fold guard, the equality cases of Lemmas 3.2 and 3.4, and the strict case of Proposition 10.5. Each certified statement is the finite, discrete fact the paper asserts.

- The entropy bound $\mathrm H(q)\in[0,\log_2 n]$, hence the fill-fraction range $\varphi=2^{\mathrm H(q)}/n\in[1/n,1]$ (Lemma 3.2), and the Strehl bound $\mathcal S\in[1/T,1]$ (Lemma 3.4).
- The congruence and orthogonal-conjugation invariance of a fixed correlation operator's spectrum (its characteristic polynomial a similarity invariant), whose data sub-case is the fill fraction's invariance under variable permutation and per-variable phase (Proposition 3.5).
- The positive semidefiniteness of the biased-autocovariance Toeplitz matrix whose entry depends on the lag alone (Lemma 4.2), with the peak-at-zero-lag corollary from its $2\times2$ minor (Corollary 4.3).
- The discrete scale-invariance of the shape factor through the block-repeat identities $\mathrm H'=\mathrm H+\log_2 s$ and $\xi'=s\,\xi$ (Proposition 4.5).
- The crossing partition and the brightness bound, radiance out at most radiance in with equality under concentration (Propositions 10.4 and 10.5).
- The permutation-null mean via the two-point stabilizer count $(N-2)!$ (Theorem 5.2).
- The algebra behind the coupling's null moments (Theorem 5.6) — the four steps below, with the assembly into $\operatorname{tr}(C_AC_B)/(T-1)$ done on paper above and not in Lean: that summing a function of $\sigma(i)$ over the symmetric group counts each image exactly $(T-1)!$ times whichever index $i$ is; the vanishing first moment $\mathbb E_\pi[S]=0$ for a centred sending side; the same-index second-moment term $\sum_\sigma A_{\sigma(i)d}A_{\sigma(i)e}=(T-1)!\,(C_A)_{de}$; the independence of the distinct-index term from *which* pair of indices is taken; and its value $T(T-1)\sum_\sigma f(\sigma 0)g(\sigma 1)=T!\big((\sum f)(\sum g)-\sum_a f_ag_a\big)$. Both moment structures are established by explicit bijections — the decomposition of a permutation of $\mathrm{Fin}(n{+}1)$ into its value at $0$ and a permutation of the rest, and right multiplication by an explicitly constructed carrier permutation. The coefficient identity $1/T+1/(T(T-1))=1/(T-1)$ and the reduction of the resulting entrywise double sum to $\operatorname{tr}(C_AC_B)$ for symmetric Gram matrices are certified alongside.
- The Weyl perturbation bound for the top eigenvalue composed into the certified attenuation interval (Lemma 6.2).
- The per-row phase invariance of the second moment together with an antipodal cloud that separates focus from resultant (Proposition 7.2).
- The monotonicity of the derived noise floor in its significance quantile and the antitone dependence of the resolved dimension on the floor (§8).
- The propagator identity $P_{yx}=A\,P_{xx}$ with operator recovery $P_{yx}P_{xx}^{-1}=A$ and shared spectrum (Theorem 9.2), and the accumulator additivity behind splicing (Theorem 9.3).

Two certified statements are narrower than their headline. Theorem 9.2 is verified for an invertible accumulator (ordinary inverse $P_{xx}^{-1}$); the rank-deficient pseudoinverse-on-a-subspace form is proved on paper (§9). The noise-floor monotonicity certifies the floor formula in its quantile argument; the Tracy–Widom law it invokes is cited. The classical limit laws (Tracy–Widom, Marchenko–Pastur, Weyl) are cited, not re-proved. The optical identification of §12 — that these entropic and spectral quantities are the aperture's fill fraction, étendue, Strehl ratio, OTF, and carrier — is a definitional dictionary, not a theorem; the certified lemmas bound and relate the quantities without asserting the naming.

---

## 16. Conclusion

Entroptics reads a signal at the resolution the signal itself dictates. From the entropy of the power marginals we obtain a matched scale; from the singular and correlation spectra, the classical optical invariants; from the biased autocorrelation, an OTF whose entropy width is the diffraction limit, with the Mercer ratio as an internal consistency check that doubles as a nonstationarity flag; and from an online linear operator, the decay rates the entropy width approximates. The governing lemmas are proved above, and those §15 names are machine-checked; the whole is deterministic and backend-portable.

What the construction does not do is worth stating as plainly. The derived floor is calibrated against an i.i.d. bulk: noise correlated across channels, or across rows, is counted as signal, and the calibration of §14 speaks to neither. The resolved count over-reads at extreme aspect ratio. The geometry cut of Definition 8.4 discards persistent structure whether or not that structure is interference. The rates of §9 assume a linear one-step map on a stationary record, and the delay depth of the lift is a stated input. On the real substrate of §14.1 the per-mode tail probabilities the floor is defined by fall outside their approximation's calibrated range and are not reported. The signal supplies the aperture, the resolution, and the width of the fold; whether to fold at all is decided at a level the caller sets.

---

## References

- E. Abbe, *Beiträge zur Theorie des Mikroskops und der mikroskopischen Wahrnehmung*, Arch. mikroskop. Anat. **9** (1873) 413–468.
- R. G. Baraniuk, P. Flandrin, A. J. E. M. Janssen, O. J. J. Michel, *Measuring time-frequency information content using the Rényi entropies*, IEEE Trans. Inform. Theory **47** (2001) 1391–1409.
- M. Born, E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press, 1999.
- L. L. Campbell, *Minimum coefficient rate for stationary random processes*, Information and Control **3** (1960) 360–371.
- L. L. Campbell, *Exponential entropy as a measure of extent of a distribution*, Z. Wahrscheinlichkeitstheorie verw. Gebiete **5** (1966) 217–225.
- CHIME/FRB Collaboration, *The First CHIME/FRB Fast Radio Burst Catalog*, Astrophys. J. Suppl. Ser. **257** (2021) 59, arXiv:2106.04352. Data (public): CANFAR archive, CISTI.CANFAR/21.0007, https://www.canfar.net/.
- M. Chiani, *Distribution of the largest eigenvalue for real Wishart and Gaussian random matrices and a simple approximation for the Tracy–Widom distribution*, J. Multivariate Anal. **129** (2014) 69–81.
- A. D. Cliff, J. K. Ord, *Spatial Processes: Models and Applications*, Pion, 1981.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
- L. De Lathauwer, B. De Moor, J. Vandewalle, *A multilinear singular value decomposition*, SIAM J. Matrix Anal. Appl. **21** (2000) 1253–1278.
- D. Gabor, *Theory of communication*, J. IEE **93** (1946) 429–457.
- M. Gavish, D. L. Donoho, *Optimal shrinkage of singular values*, IEEE Trans. Inform. Theory **63** (2017) 2137–2152.
- J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts & Co., 2005.
- N. Halko, P. G. Martinsson, J. A. Tropp, *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions*, SIAM Rev. **53** (2011) 217–288.
- W. Hoeffding, *A combinatorial central limit theorem*, Ann. Math. Statist. **22** (1951) 558-566.
- W. James, C. Stein, *Estimation with quadratic loss*, in *Proc. Fourth Berkeley Symp. Math. Statist. Prob.* **1** (1961) 361–379.
- T. Jiang, *The limiting distributions of eigenvalues of sample correlation matrices*, Sankhyā A **66** (2004) 35–48.
- I. M. Johnstone, *On the distribution of the largest eigenvalue in principal components analysis*, Ann. Statist. **29** (2001) 295–327.
- J. Josse, J. Pagès, F. Husson, *Testing the significance of the RV coefficient*, Comput. Statist. Data Anal. **53** (2008) 82–91.
- F. Kazi-Aoual, S. Hitier, R. Sabatier, J.-D. Lebreton, *Refined approximations to permutation tests for multivariate inference*, Comput. Statist. Data Anal. **20** (1995) 643–656.
- M. H. Kassner, *Screens for the absorption of micro-wave radiation*, M.Sc. thesis, Dept. of Physics, McGill University, 1950. https://doi.org/10.82308/34546
- A. Khinchin, *Korrelationstheorie der stationären stochastischen Prozesse*, Math. Ann. **109** (1934) 604–615.
- B. O. Koopman, *Hamiltonian systems and transformation in Hilbert space*, Proc. Natl. Acad. Sci. USA **17** (1931) 315–318.
- S. Kornblith, M. Norouzi, H. Lee, G. Hinton, *Similarity of neural network representations revisited*, Proc. 36th Int. Conf. Machine Learning (ICML), PMLR **97** (2019), arXiv:1905.00414.
- H. J. Landau, H. O. Pollak, *Prolate spheroidal wave functions, Fourier analysis and uncertainty II*, Bell Syst. Tech. J. **40** (1961) 65–84.
- A. Lesne, *Shannon entropy: a rigorous notion at the crossroads between probability, information theory, dynamical systems and statistical physics*, Math. Struct. Comp. Sci. **24** (2014) e240311.
- W. Lukosz, *Optical systems with resolving powers exceeding the classical limit*, J. Opt. Soc. Am. **56** (1966) 1463–1472.
- V. A. Marchenko, L. A. Pastur, *Distribution of eigenvalues for some sets of random matrices*, Math. USSR-Sb. **1** (1967) 457–483.
- K. V. Mardia, P. E. Jupp, *Directional Statistics*, Wiley, 2000.
- J. Mercer, *Functions of positive and negative type, and their connection with the theory of integral equations*, Phil. Trans. R. Soc. Lond. A **209** (1909) 415–446.
- G. A. Miller, *Note on the bias of information estimates*, in *Information Theory in Psychology* (1955) 95–100.
- P. Robert, Y. Escoufier, *A unifying tool for linear multivariate statistical methods: the RV-coefficient*, J. R. Statist. Soc. C (Applied Statistics) **25** (1976) 257–265.
- O. Roy, M. Vetterli, *The effective rank: a measure of effective dimensionality*, Proc. 15th European Signal Processing Conf. (EUSIPCO) (2007) 606–610.
- C. E. Shannon, *A Mathematical Theory of Communication*, Bell Syst. Tech. J. **27** (1948) 379–423, 623–656.
- D. Slepian, H. O. Pollak, *Prolate spheroidal wave functions, Fourier analysis and uncertainty I*, Bell Syst. Tech. J. **40** (1961) 43–63.
- F. Takens, *Detecting strange attractors in turbulence*, in: Dynamical Systems and Turbulence, Lecture Notes in Mathematics **898**, Springer, 1981, pp. 366–381.
- G. Toraldo di Francia, *Degrees of freedom of an image*, J. Opt. Soc. Am. **59** (1969) 799–804.
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
- J. Ziv, A. Lempel, *Compression of individual sequences via variable-rate coding*, IEEE Trans. Inform. Theory **24** (1978) 530–536.