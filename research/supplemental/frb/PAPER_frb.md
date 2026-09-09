# An untuned rank-selection and reconstruction procedure recovers CHIME/FRB burst morphology

### Applied to public Catalog 1 waterfalls, with no per-substrate tuning and no burst template.

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`  
ORCID [0009-0002-0150-4027](https://orcid.org/0009-0002-0150-4027)

*Pre-print. September 2026.*

---

## Abstract

We apply a parameter-free rank-selection and reconstruction procedure to publicly released CHIME/FRB Catalog 1 waterfalls and compare the result against CHIME's own *fitburst* forward model. The procedure selects its rank against a derived Tracy–Widom noise floor, reconstructs the field on the resolved modes with Gavish–Donoho shrinkage, and removes persistent narrowband structure by a geometric cut on each mode's entropic footprint. No constant is fitted to the data, no template of a burst enters anywhere, and nothing is tuned per event. On four bright events the reconstruction correlates with the forward model at $0.515$–$0.609$ (mean $0.560$), against $0.115$–$0.373$ (mean $0.198$) for the raw waterfall and $0.214$ for a box-average control that applies the same frequency smoothing and nothing else. To make those figures interpretable in the absence of ground truth we report two reference points: the forward model corrupted by the record's own measured noise scores $0.219$ against the model, and the forward model degraded by exactly the resolution the read is taken at scores $0.994$. The read therefore sits well above what the record's noise permits and well below what its resolution permits, and the residual gap is attributable to the forward model's own parametric form, which we cannot quantify without refitting it. A further 12 waterfalls drawn uniformly at random from the same release are read by the identical path; all twelve are read without incident and eight resolve a mode above the floor. Neither the read nor the forward model is ground truth, and the noise floor is calibrated against an i.i.d. bulk which a serially correlated record violates; both limits are stated and quantified.

---

## 1. Introduction

A dedispersed fast radio burst waterfall is a two-dimensional array of intensity against frequency and time, carrying a burst of unknown morphology on a noise background, with a substantial fraction of its channels lost to radio-frequency interference (RFI) or to instrumental masking. Two questions have to be answered before anything can be said about the burst: how many degrees of freedom in the record are signal rather than noise, and which parts of the measured field to keep.

Both are usually answered with a model. A forward model fixes the burst's functional form — CHIME's *fitburst* uses Gaussians in time against a running power law in frequency — and solves for that form's parameters; RFI excision applies thresholds tuned per instrument and often per observation. Both approaches work, and both make the answer depend on choices the analyst supplies.

This paper reports what happens when neither is supplied. We apply the read-side filter of the *entroptics* construction [Sessford 2026] to public CHIME/FRB Catalog 1 waterfalls and compare its output against *fitburst* on the same records.

**What "untuned" means here.** It is a specific and checkable claim, not a rhetorical one:

1. **No constant is fitted to data or calibrated to a substrate.** Every fixed number in the procedure is a derived mathematical quantity — a $\chi^2$ median, an influence-function variance, a universal Tracy–Widom quantile — or a stated criterion. The provenance of each is tabulated in the instrument paper [Sessford 2026, §11.1].
2. **Nothing is set per event.** The same call is made on all sixteen waterfalls read here, with the same arguments. Nothing is inspected and adjusted between events.
3. **No template of a burst enters.** The reconstruction is supported on the measured screen's own singular vectors and synthesises nothing.
4. **What the reader does supply** is an operating point: a false-alarm level $\alpha$, which by the Neyman–Pearson lemma encodes the relative cost of a false alarm against a miss and cannot be derived from the data, together with the null it is taken against. Both are left at their defaults throughout ($\alpha=0.05$, the derived edge). Two facts about the record are declared rather than inferred: which axis is ordered, and that the feature channels are commensurate.

The claim under test is transfer, not accuracy. The construction is validated on synthetic signals with a planted ground truth [Sessford 2026, §12]; a real substrate tests whether the same untuned instrument still works when nobody chose the record.

---

## 2. Method

This section is self-contained enough to follow without the instrument paper, which carries the definitions, the proofs and the provenance of every constant [Sessford 2026].

Write $W\in\mathbb{R}^{T\times F}$ for the dedispersed waterfall with $T$ time samples and $F$ frequency channels, the time axis *ordered* and the frequency axis carrying commensurate channels.

**Observer facts.** Dead channels are *dropped*, never zero-filled. A channel that was measured and read zero is an observation of no power; a channel that was never measured is absent, and absence is not an observation of zero — substituting zero widens the axis the signal is scored against by exactly the channels carrying nothing. "Dead" is the catalogue's own mask together with a zero-variance test. This is the only step that uses knowledge of the instrument, and it supplies no burst information.

**Whiten.** Each surviving channel is rescaled to a common robust median-absolute-deviation noise scale, with each channel's scale shrunk toward the pooled cross-channel scale by a data-derived James–Stein weight. A channel whose MAD falls at or below the frame's pooled MAD times the working arithmetic's machine epsilon is given no scale at all: dividing by a manufactured stand-in would lift round-off to unit amplitude, which the reconstruction would then resolve as signal. The step equalises per-channel scale and performs **no decorrelation** — it is a per-channel divide, not $\Sigma^{-1/2}$ — so noise correlated *across* channels survives it intact.

**Fold.** The frequency axis is coarsened to the resolution the record's own power marginal says it carries, $F_{\mathrm{eff}}=\operatorname{round}(F/\delta_F)$ with $\delta_F=F_{\mathrm{eff}\text{-}\mathrm{measured}}/2^{\mathrm H_F}$ read from the Shannon entropy of the frequency power marginal. The fold is licensed only where two conditions both hold: that the concentration is real against an exact Dirichlet null, and that the frequency axis is *continuous* — that neighbouring channels are more alike than a random relabelling, tested against the closed-form permutation null. The time axis keeps native spacing, which the ordered reads require. For these frames the fold takes $9{,}760$–$11{,}696$ live channels to $5{,}949$–$10{,}972$.

**The derived floor.** The largest squared singular value of an $N\times F_{\mathrm{eff}}$ pure-noise screen concentrates at the Johnstone centre $\mu=(\sqrt{N-1}+\sqrt{F_{\mathrm{eff}}})^2$ with fluctuation scale $\varsigma_J=(\sqrt{N-1}+\sqrt{F_{\mathrm{eff}}})\big(\tfrac1{\sqrt{N-1}}+\tfrac1{\sqrt{F_{\mathrm{eff}}}}\big)^{1/3}$, and $(s_1^2-\mu)/\varsigma_J$ follows the real ($\beta=1$) Tracy–Widom law [Tracy & Widom 1996; Johnstone 2001]. The per-cell noise variance is read robustly from the median row energy, de-biased by the Wilson–Hilferty $\chi^2$ median $c_F=(1-\tfrac{2}{9F})^3$ and by the centring degree of freedom $(N-1)/N$. With the universal Tracy–Widom$_1$ quantile $q_\alpha$ ($q_{0.05}=0.9793$),

$$\Phi=\sqrt{\hat\sigma^2\,(\mu+q_\alpha\varsigma_J)},\qquad K_{\mathrm{signal}}=\#\{k:s_k>\Phi\}.$$

Given $\alpha$, the floor is fixed by the shape $(N,F_{\mathrm{eff}})$ alone: $\mu,\varsigma_J,c_F$ and $(N-1)/N$ depend on it and $q_\alpha$ is a universal constant. Nothing in $\Phi$ is fitted.

**The mode footprint.** Each resolved mode carries a left (time) singular vector $u_k$ and a right (frequency) singular vector $v_k$. Its footprint is the pair of entropic fill fractions

$$\varphi^{(k)}_T=\frac{2^{\mathrm H(|u_k|^2)}}{N},\qquad \varphi^{(k)}_F=\frac{2^{\mathrm H(|v_k|^2)}}{F_{\mathrm{eff}}},$$

which read the *shape* a mode has where the scalar floor sees only its size: a broadband transient has $\varphi^{(k)}_F\to1$ with $\varphi^{(k)}_T$ small, persistent narrowband interference the reverse.

**The geometry cut.** A mode with $\varphi^{(k)}_F\le\varphi^{(k)}_T$ — wider in frequency than in time, the signature of channelised interference — is dropped; a broadband transient $\varphi^{(k)}_F>\varphi^{(k)}_T$ is kept. This is a *shape* test and not a detection: it carries no null and no level, and it drops persistent structure whether or not that structure is interference. It is a stated criterion.

**The read-side filter.** The output is the field reconstructed from its surviving modes, $\widehat S=U\operatorname{diag}(\tilde s)V^{\mathsf H}$, with $\tilde s$ the singular spectrum shrunk against the derived floor after [Gavish & Donoho 2017]: modes at or below the floor map to zero and the survivors are de-biased toward it. Because $U$ and $V$ are read from the data, $\widehat S$ is supported on the measured screen's own modes and introduces no direction the record does not carry. The per-channel whitening is inverted before the output is returned, so the read carries the burst's morphology *and* the waterfall's own amplitude scale; $W-\widehat W$ is therefore a plain difference and is exactly what the filter discarded. A channel's median is removed before the decomposition and is never offered to the cut, so it is carried through untouched.

With shrinkage the map is not a projection — the survivors are de-biased — and the Frobenius optimality of [Gavish & Donoho 2017], an asymptotic statement for i.i.d. noise at a known $\sigma$ cut at the bulk edge, is not claimed for it.

---

## 3. Results

The catalogue waterfalls arrive already dedispersed; the procedure adds no dedispersion of its own, and any residual sweep shows as diagonal structure the read catches directly. Everything downstream of the channel drop is the library.

### 3.1 Four events

Figure 1 places the reconstruction beside *fitburst* and the raw waterfall on four bright events, drawn at random from the release and read with no per-substrate tuning. Table 1 carries the per-burst reads.

![Figure 1. Four CHIME/FRB Catalog 1 bursts read untuned, one row per event. Left to right: the raw dedispersed waterfall, CHIME's *fitburst* forward model, the reconstruction, and what the read did not keep. Each panel is contrast-stretched independently between its own median and 99.7th percentile — the four hold quantities on different scales, and the model in particular is noiseless and non-negative — so the comparison to make by eye is of morphology and not of contrast.](../../figures/frb_panel.png)

| event | DM | $T$ | live | read at | $K_{\mathrm{signal}}$ | contrast | $z$ | kept | dropped |
|---|---|---|---|---|---|---|---|---|---|
| FRB20190425A | 128.2 | 19 | 9,760 | 5,949 | 3 | 4.22 | 3.1 | 3 | 0 |
| FRB20190106B | 316.6 | 38 | 11,632 | 10,972 | 3 | 1.30 | 4.7 | 2 | 1 |
| FRB20190227A | 394.0 | 38 | 11,696 | 10,727 | 5 | 2.00 | 5.8 | 4 | 1 |
| FRB20190323B | 789.6 | 19 | 11,312 | 10,550 | 1 | 1.41 | 3.4 | 1 | 0 |

*Table 1. Per-burst reads, from `tables/events.csv`. "live" is the surviving width of the 16,384 recorded channels; "read at" is the width the fold takes it to. Contrast is $\sigma_1/\Phi$, the leading singular value over the screen floor; $z$ is the ordered-axis coherence. "kept" and "dropped" are the resolved modes on each side of the geometry cut.*

Each event resolves at least one mode above the floor. The reported per-mode quantity is the bounded contrast $\sigma_1/\Phi$ rather than a tail probability: a per-mode $p_k$ would rest on a Tracy–Widom approximation [Chiani 2014] far outside its calibrated range at these deviates, and is therefore not reported.

The record shapes are extreme — $T=19$–$38$ against $16{,}384$ recorded channels — so the aspect ratio each read was taken at is on the page beside it.

**The fold changes nothing it resolves.** $K_{\mathrm{signal}}$ is identical read folded or at the live width on all four events, and the contrast is equal or slightly higher folded. The read is then mapped back onto the recorded frequency axis for display; the mapping invents nothing, since the read is constant across each folded group.

### 3.2 A random spot-check

A further **12 waterfalls drawn uniformly at random** from the same public release, at a fixed seed, are read by the identical path (`tables/spotcheck.csv`). All twelve are read without incident, and $K_{\mathrm{signal}}\ge1$ on eight of them.

The four that resolve nothing are the four whose leading singular value sits at or below the derived floor — contrast $0.98$–$0.99$, against $1.30$–$4.22$ for the events of Figure 1 — the floor declining to certify a mode it cannot separate from the bulk. Record length does not separate the two groups: the four that resolve nothing span $T=19$–$57$ and the eight that resolve span $T=19$–$95$, so the shortest records sit in both.

This establishes that the untuned instrument runs on records it was not selected against, and reports what it finds on them. It is not a performance measurement: nothing here is scored against a reference.

### 3.3 Agreement with the forward model

*fitburst* is fitted with the shape of a burst already in it and solves for that shape's parameters. The read is given no model of a burst at all and keeps whatever modes clear the floor. The two share no assumptions, so agreement between them on the same waterfall is evidence that the read is finding the burst rather than an artifact of its own construction.

Every quantity is scored on the **same cells** (the live channels — the read is undefined on dead ones, and scoring each series wherever it happens to be finite scores them on different pixels) and with its **per-channel baseline removed**. The baseline removal is what makes the series comparable, and it is applied to all of them: `model_wfall` carries no per-channel standing level at all — its per-channel median is numerically zero, at most $3.9\times10^{-6}$ across the four events against model maxima of $0.108$–$0.506$ — because it is a statement about the burst and not about each channel's baseline. A series carrying a baseline would therefore be scored against one that does not, and the baseline would count as disagreement. Centring one series and not the others would hand that one a comparison the references never got.

| series | range | mean |
|---|---|---|
| raw waterfall | $0.115$–$0.373$ | $0.198$ |
| box-average control | $0.118$–$0.427$ | $0.214$ |
| **the read** | $\mathbf{0.515}$–$\mathbf{0.609}$ | $\mathbf{0.560}$ |

*Table 2. Correlation against the fitburst forward model, from `tables/agreement.csv`.*

The read arrives smoothed along frequency, and the model is smooth too, so the box-average control rebins the raw waterfall to the read's own folded width — everything the fold does and nothing the read does. It reads $0.214$. **The fold accounts for $4.6\%$ of the gain and the read for the rest.**

---

## 4. What sets the scale

A correlation of $0.560$ against a series that is not ground truth means nothing on its own. There is no ground truth here to supply a scale — the source of a fast radio burst is unknown, and the forward model is a parametric fit to the same waterfall — so we report two reference points that are computable from the record itself and say what each does and does not bound.

**The noise-limited reference.** Take the forward model itself, corrupt it with this record's own per-channel noise, and score it against the model. The noise scale is read robustly (MAD) from the off-burst columns of the raw waterfall, where the on-burst window is the smallest contiguous run of time samples containing $99\%$ of the model's band-summed power — a stated criterion, and the only free choice in this section. This answers: *what would a method that recovered the forward model exactly score, if it were measured through this waterfall's noise?*

**The fold-resolution ceiling.** Take the forward model, degrade it by exactly the fold the read is taken at, and score it against the model. This answers: *what is the most any read at that frequency resolution could score, from resolution alone?*

| reference | range | mean |
|---|---|---|
| noise-limited (perfect recovery, measured through this noise) | $0.117$–$0.439$ | $0.219$ |
| raw waterfall (measured) | $0.115$–$0.373$ | $0.198$ |
| **the read** | $\mathbf{0.515}$–$\mathbf{0.609}$ | $\mathbf{0.560}$ |
| fold-resolution ceiling (perfect recovery at the read's resolution) | $0.9853$–$0.9987$ | $0.994$ |

*Table 3. The read against its two reference points, from `tables/agreement.csv`.*

Two things follow, and they are the point of the section.

**The raw waterfall's low agreement is noise, not disagreement about the burst.** A series that *is* the forward model, measured through this record's noise, scores $0.219$; the raw waterfall scores $0.198$. The two are not meaningfully different. So the forward model's functional form is not what holds the raw frame down to $0.198$ — the noise is, entirely. That is the floor the read has to beat, and it beats it on every event.

**Resolution is not what caps the read.** A perfect recovery at the read's own folded width scores $0.994$. The fold costs at most $1.5\%$ of the attainable correlation on the worst of the four events and $0.1\%$ on the best. So the gap between $0.559$ and $1$ is not the resolution the instrument chose.

**What we cannot bound.** The remaining gap is attributable to the forward model's own form. *fitburst* is smooth and has no per-channel freedom, while both the waterfall and the read carry per-channel structure that no smooth parametric surface can represent. Quantifying that would require refitting *fitburst* — perturbing its parameters, or refitting it to a resampled record — and we do not have its fitting code. **We therefore do not report a ceiling that accounts for the model's form, and the residual gap between $0.559$ and $0.994$ should not be read as error in the read.** It is a mixture of the read's own residual and a representational limit of the reference, in unknown proportion.

One consequence of scoring on the full frame is worth naming. Off the burst, both the read and the model are close to zero, and that agreement counts toward the score. Restricted to the on-burst window alone, the raw waterfall reads $0.297$ and the read $0.468$ (means; per-event values in `tables/agreement.csv`). The read's margin over the raw survives the restriction — the read is $2.8\times$ the raw on the full frame and $1.6\times$ on the on-burst window — but the absolute figures are lower, and a reader comparing against a full-frame number elsewhere should use the same convention.

---

## 5. Limits

**Neither is ground truth.** The source of a fast radio burst is unknown, and the forward model is itself a fit to this same waterfall with no per-channel freedom. The agreement says that two methods assuming nothing in common find the same burst; it does not say either has recovered it correctly.

**The floor's null is an i.i.d. bulk, and that is a real restriction.** The Tracy–Widom law the floor is derived from is the null of the largest eigenvalue under an i.i.d. Gaussian bulk. Noise correlated *across* channels — common-mode drift, narrowband interference — or heavy-tailed noise concentrates its variance into a few modes and is, on the singular spectrum alone, indistinguishable from signal; the floor counts it as resolved. Correlation *along* the ordered axis moves the bulk itself. Measured on pure AR(1) rows with no planted signal, the derived floor resolves a mode in $100\%$ of draws at every correlation length from $\rho=2$ to $32$, against $0$–$5\%$ on i.i.d. rows [Sessford 2026, §12]. The effect is in the spectrum rather than in any estimator: at shape $(200,200)$ and $\rho=2$ the leading singular value of the raw field averages $37.5$ against a Bai–Yin i.i.d. edge of $28.3$, with $13.3$ values above it, where the i.i.d. control sits on the edge at $28.0$ with $0.1$. No method reading against that edge can be right on such a record, and the standard singular-value and model-order selectors over-read the same records by two to four times as much.

What this bounds is a record whose *noise* is correlated along the ordered axis. A dedispersed waterfall, whose noise is close to white in time, is not affected, and the reads above are not. A record carrying $1/f$ drift or a common-mode gain would be.

**The Tracy–Widom approximation falls outside its calibrated range at these deviates.** The floor is a threshold and is used as one. A per-mode tail probability $p_k$ would require evaluating the Tracy–Widom$_1$ survival function far into a tail where the Chiani Gamma approximation [Chiani 2014], with its stated maximum CDF error of $\approx7\times10^{-3}$, is outside the range it was calibrated over. We therefore report the bounded contrast $\sigma_1/\Phi$ and not a $p$-value. The count $K_{\mathrm{signal}}$ is unaffected, since it depends on the threshold and not on the tail's shape.

**The geometry cut is a criterion, not a test.** It carries no null and no level, and it drops persistent structure whether or not that structure is interference. A genuinely persistent narrowband astrophysical signal would be removed by it.

**The sample is small.** Four events with a figure and twelve more read blind is a demonstration of transfer, not a survey. Nothing here supports a population statement about Catalog 1.

---

## 6. Reproduction

Every number in this paper is written by one script:

```
python research/supplemental/frb/reproduce.py
```

It runs the two committed figure routines (`research/figures/frb_panel.py`, `research/figures/frb_spotcheck.py`) and writes three tables beside itself:

| table | holds | used in |
|---|---|---|
| `tables/events.csv` | per-burst reads for the four events | Table 1, §3.1 |
| `tables/spotcheck.csv` | the 12-waterfall random draw | §3.2 |
| `tables/agreement.csv` | every correlation, including both references | Tables 2 and 3, §§3.3, 4 |

The script imports `entroptics` and calls its public front door; it reimplements nothing. Every seed is fixed and the tables are byte-reproducible across runs.

**The data is a separate download.** The CHIME/FRB Catalog 1 waterfalls are a public release from the CANFAR archive (CISTI.CANFAR/21.0007) and are not part of the software repository. `README.md` beside this paper documents the fetch and the one line of configuration that points the scripts at it. No script here downloads the release: it is far larger than a reproduction script should pull unasked, and a partial download would overwrite good tables with empty ones while still exiting zero. The scripts refuse, with instructions, if they find nothing.

---

## Declaration of generative AI use

The author used Anthropic's Claude Opus (versions 4.8 and 5) in the preparation of this work. Its
contribution was to write code, and to generate and validate content. The ideas, the construction
and the claims are the author's. No other generative AI tool was used. The author reviewed and
edited all output and takes full responsibility for the content of this publication.

---

## References

- M. Chiani, *Distribution of the largest eigenvalue for real Wishart and Gaussian random matrices and a simple approximation for the Tracy–Widom distribution*, J. Multivariate Anal. **129** (2014) 69–81.
- CHIME/FRB Collaboration, *The First CHIME/FRB Fast Radio Burst Catalog*, Astrophys. J. Suppl. Ser. **257** (2021) 59, arXiv:2106.04352. Data (public): CANFAR archive, CISTI.CANFAR/21.0007, https://www.canfar.net/.
- M. Gavish, D. L. Donoho, *Optimal shrinkage of singular values*, IEEE Trans. Inform. Theory **63** (2017) 2137–2152.
- W. James, C. Stein, *Estimation with quadratic loss*, in *Proc. Fourth Berkeley Symp. Math. Statist. Prob.* **1** (1961) 361–379.
- I. M. Johnstone, *On the distribution of the largest eigenvalue in principal components analysis*, Ann. Statist. **29** (2001) 295–327.
- I. J. Sessford, *Entroptics: reading a 2-D signal as a finite optical aperture at its own entropy-matched resolution*, pre-print, 2026. https://github.com/Agience/entroptics
- C. A. Tracy, H. Widom, *On orthogonal and symplectic matrix ensembles*, Comm. Math. Phys. **177** (1996) 727–754.
- E. B. Wilson, M. M. Hilferty, *The distribution of chi-square*, Proc. Natl. Acad. Sci. USA **17** (1931) 684–688.
