# Why the CTR polarization twist scales as 1/γ

**Question.** When the radiator foil is tilted by an angle ψ, the transition-radiation
(TR) polarization is no longer purely radial: a small *azimuthal* (non-radial) field
component $E_\phi$ appears. The simulation (`results_l3/source_polarization.png`) measures
its peak as $|E_\phi/E_r| = 0.49\%$ at ψ = 30°, γ = 117.4, and shows it falls as $1/\gamma$.
This note derives that result in closed form:

$$\boxed{\ \left.\frac{|E_\phi|}{|E_r|}\right|_{\max}
   \;=\; \frac{\sin 2\psi}{\big(1+\beta\cos 2\psi\big)\,\gamma}
   \;\xrightarrow[\beta\to1]{}\; \frac{\tan\psi}{\gamma}\ }$$

The $1/\gamma$ is not because the symmetry-breaking field is weak — it is order $\tan\psi$,
*independent of γ*. It is because the radial TR field it competes against is relativistically
enhanced by a factor of γ. The twist is the ratio of an un-boosted kinematic background to a
boosted resonant lobe.

This is the companion to `LEVEL3_DERIVATION.md` (which sets up the image-charge amplitude)
and explains the $\sim 1/\gamma$ remark in `WRITEUP.md` §3.10.

---

## 1. Setup: image-charge amplitude and the specular frame

Backward TR from a charge crossing a perfectly conducting foil is the radiation of the
charge plus its mirror image. The far-field **vector amplitude** in observation direction
$\hat n$ is the difference of two relativistic velocity-field kernels (see
`LEVEL3_DERIVATION.md`; this is exactly what `thz_source.gf_oblique_field` evaluates):

$$\mathbf A(\hat n) \;=\; \mathbf T(\hat n,\boldsymbol\beta) \;-\; \mathbf T(\hat n,\boldsymbol\beta_i),
\qquad
\mathbf T(\hat n,\mathbf b) \;=\; \frac{\hat n\,(\hat n\cdot\mathbf b)-\mathbf b}{1-\hat n\cdot\mathbf b}
 \;=\; -\,\frac{\hat n\times(\hat n\times\mathbf b)}{1-\hat n\cdot\mathbf b}.$$

The second form shows $\mathbf T$ is **transverse** to $\hat n$ (it is the part of $\mathbf b$
perpendicular to the line of sight, divided by the Doppler denominator) — as a radiation
field must be.

**Geometry.** The electron moves along $+\hat z$ with $\boldsymbol\beta=\beta\hat z$. The foil
normal is tilted by ψ in the $x$–$z$ plane, $\hat n_f=(\sin\psi,0,\cos\psi)$. The image
velocity is the specular reflection of $\boldsymbol\beta$ about the foil,
$\boldsymbol\beta_i=\boldsymbol\beta-2(\boldsymbol\beta\cdot\hat n_f)\hat n_f$:

$$\boldsymbol\beta_i=\beta\big(\hat z-2\cos\psi\,\hat n_f\big)=\beta\,(-\sin 2\psi,\,0,\,-\cos 2\psi),
\qquad |\boldsymbol\beta_i|=\beta .$$

At ψ = 0 this is $-\beta\hat z$ (the head-on mirror charge), and the backward radiation is
centered on $-\hat z$. For general ψ the backward lobe centers on the **image-velocity
direction** $\hat s\equiv\hat\beta_i$, because (see §2) the image kernel's denominator
$1-\hat n\cdot\boldsymbol\beta_i$ is smallest there. Build an orthonormal **specular frame**
on that axis:

$$\hat s=(-\sin 2\psi,0,-\cos 2\psi),\qquad
\hat e_p=\hat s\times\hat y=(\cos 2\psi,0,-\sin 2\psi),\qquad
\hat e_s=\hat y=(0,1,0).$$

$\hat e_p$ is the **in-plane** axis (in the tilt plane), $\hat e_s$ the **out-of-plane** axis.
Parametrize observation directions near $\hat s$ by a small polar angle θ and azimuth χ:

$$\hat n=\frac{\hat s+\theta\,\hat\rho}{|\hat s+\theta\hat\rho|},\qquad
\hat\rho=\cos\chi\,\hat e_p+\sin\chi\,\hat e_s,\qquad
\hat\chi=-\sin\chi\,\hat e_p+\cos\chi\,\hat e_s,$$

with $\hat\rho$ the **radial** and $\hat\chi$ the **azimuthal** transverse unit vectors. The
detector's $(X,Y)$ map to $(\theta\cos\chi,\theta\sin\chi)=(X/z_0,\,Y/z_0)$.

The polarization "twist" is the angle of $\mathbf A$ away from $\hat\rho$; equivalently the
ratio of its azimuthal to radial components, $A_\chi/A_\rho$.

---

## 2. The image term: the radial donut, amplitude ∝ γ

Evaluate $\mathbf T(\hat n,\boldsymbol\beta_i)$ with $\boldsymbol\beta_i=\beta\hat s$. The
denominator is the **resonant** one:

$$1-\hat n\cdot\boldsymbol\beta_i = 1-\beta\cos\theta
\;\approx\; (1-\beta)+\tfrac12\beta\theta^2
\;\approx\; \tfrac12\!\left(\frac1{\gamma^2}+\theta^2\right),$$

using $1-\beta\approx 1/2\gamma^2$. This is small — of order $1/\gamma^2$ — and it is what
concentrates the radiation into a cone of width $\theta\sim1/\gamma$.

For the numerator, expand $\hat n=\hat s+\theta\hat\rho-\tfrac12\theta^2\hat s+O(\theta^3)$.
A short calculation gives the transverse part

$$\big[\hat n(\hat n\cdot\boldsymbol\beta_i)-\boldsymbol\beta_i\big]_\perp
=\beta\,\theta\,\hat\rho+O(\theta^2\hat s),$$

i.e. **purely radial** to leading order (the $O(\theta^2)$ correction lies along $\hat s$ and
does not radiate). Hence

$$\mathbf T(\hat n,\boldsymbol\beta_i)\;\approx\;
\frac{2\beta\theta}{\,1/\gamma^2+\theta^2\,}\;\hat\rho \;\equiv\; R(\theta)\,\hat\rho .$$

This is exactly the Ginzburg–Frank radial donut: zero on axis, **radially polarized**, and
peaked at $\theta=1/\gamma$ with

$$R_{\max}=R(1/\gamma)=\frac{2\beta/\gamma}{2/\gamma^2}=\beta\gamma\;\sim\;\gamma .$$

**Key fact #1:** the radial field on the ring is relativistically enhanced, $E_r\sim\gamma$.
The image term carries *no* azimuthal component at this order — by itself it is perfectly
radial. So the twist must come from the other term.

---

## 3. The electron term: an un-boosted, fixed background, amplitude O(1)

Now $\mathbf T(\hat n,\boldsymbol\beta)$ with $\boldsymbol\beta=\beta\hat z$. We observe near
$\hat n\approx\hat s$, which points *away* from $+\hat z$ (the angle between them is
$\pi-2\psi$, since $\hat s\cdot\hat z=-\cos2\psi$). So the denominator is **non-resonant**:

$$1-\hat n\cdot\boldsymbol\beta = 1-\beta(\hat n\cdot\hat z)
\;\approx\; 1+\beta\cos 2\psi \;=\; O(1).$$

For the numerator, resolve $\hat z$ in the specular frame:

$$\hat z=(\hat z\cdot\hat s)\hat s+(\hat z\cdot\hat e_p)\hat e_p+(\hat z\cdot\hat e_s)\hat e_s
=-\cos 2\psi\,\hat s-\sin 2\psi\,\hat e_p .$$

Then, with $\hat n\approx\hat s$ and $\hat n\cdot\boldsymbol\beta\approx-\beta\cos2\psi$,

$$\big[\hat n(\hat n\cdot\boldsymbol\beta)-\boldsymbol\beta\big]_\perp
=\big[-\beta\cos2\psi\,\hat s-\beta\hat z\big]_\perp
=-\beta\,[\hat z]_\perp
=\beta\sin 2\psi\;\hat e_p .$$

The longitudinal ($\hat s$) pieces cancel and the transverse remainder is a **single fixed
vector along $\hat e_p$**, independent of θ and χ. Therefore

$$\mathbf T(\hat n,\boldsymbol\beta)\;\approx\;
\frac{\beta\sin 2\psi}{1+\beta\cos 2\psi}\;\hat e_p \;\equiv\; C(\psi)\,\hat e_p,
\qquad
C(\psi)=\frac{\beta\sin 2\psi}{1+\beta\cos 2\psi}\;\xrightarrow[\beta\to1]{}\;\tan\psi .$$

**Key fact #2:** the electron term is a γ-independent, O(1) vector that points along the
fixed in-plane axis $\hat e_p$ — it does **not** rotate with the observation azimuth χ. It is
simply the projection of the (un-mirrored) electron velocity onto the backward observation
plane; that projection is $\sin2\psi$ by geometry and vanishes at ψ = 0.

Physically: the real electron is not boosted *toward* the backward detector (it flies forward),
so its contribution to the backward field gets no $1/\gamma$ Doppler enhancement. It stays
order unity while the image (specular) lobe is amplified by γ.

---

## 4. Combine: radial vs azimuthal, and the 1/γ ratio

Add the two transverse amplitudes:

$$\mathbf A_\perp(\theta,\chi)=C(\psi)\,\hat e_p-R(\theta)\,\hat\rho .$$

Project onto the radial and azimuthal unit vectors using
$\hat e_p\cdot\hat\rho=\cos\chi$, $\hat e_p\cdot\hat\chi=-\sin\chi$:

$$A_\rho=C(\psi)\cos\chi-R(\theta),\qquad
A_\chi=-\,C(\psi)\sin\chi .$$

The **non-radial component is entirely from the electron background**, with a $\sin\chi$
azimuthal structure: it vanishes in the tilt plane (χ = 0, π, where the background is itself
radial/in-plane) and is largest out of plane (χ = ±π/2). That is exactly the top/bottom
$E_\phi$ dipole seen in the right panel of `source_polarization.png`.

The polarization twist angle Δ (deviation from radial) is

$$\tan\Delta=\frac{A_\chi}{A_\rho}
=\frac{-C(\psi)\sin\chi}{\,C(\psi)\cos\chi-R(\theta)\,}.$$

On the ring, $R(\theta)\approx R_{\max}=\beta\gamma\gg C(\psi)$, so the denominator is
dominated by $-R$ and

$$\Delta(\chi)\;\approx\;\frac{C(\psi)}{\beta\gamma}\,\sin\chi,
\qquad
\Delta_{\max}=\frac{C(\psi)}{\beta\gamma}
=\frac{\sin 2\psi}{(1+\beta\cos 2\psi)\,\gamma}
\;\xrightarrow[\beta\to1]{}\;\frac{\tan\psi}{\gamma}.$$

Since $|E_\phi/E_r|_{\max}=|A_\chi/A_\rho|_{\max}=\Delta_{\max}$ for small Δ, this is the
boxed result. **The $1/\gamma$ is the ratio of the un-boosted background $C=O(1)$ to the
boosted radial lobe $R_{\max}\sim\gamma$.**

---

## 5. Numerical verification

`gf_oblique_field` evaluated on the source grid, peak $|E_\phi/E_r|$ over the ring annulus
($I>0.25\,I_{\max}$), versus the closed form. (Reproduce: the scan script in the project
notes; uses `default_config` with `gf_tilt_model='level3'`.)

**Dependence on tilt (γ = 117.4 fixed):**

| ψ | measured $|E_\phi/E_r|$ | $\tan\psi/\gamma$ | $\sin2\psi/[(1+\beta\cos2\psi)\gamma]$ |
|---:|---:|---:|---:|
| 10° | 0.150% | 0.150% | 0.150% |
| 20° | 0.309% | 0.310% | 0.310% |
| 30° | 0.489% | 0.492% | 0.492% |
| 40° | 0.710% | 0.715% | 0.715% |
| 45° | 0.845% | 0.852% | 0.852% |

**Dependence on γ (ψ = 30° fixed) — the 1/γ law:**

| γ | measured | $\tan30°/\gamma$ | measured × γ |
|---:|---:|---:|---:|
| 50 | 1.143% | 1.155% | 0.5715 |
| 100 | 0.574% | 0.577% | 0.5742 |
| 117.4 | 0.489% | 0.492% | 0.5747 |
| 200 | 0.288% | 0.289% | 0.5757 |
| 400 | 0.144% | 0.144% | 0.5765 |

`measured × γ` is constant to ~1% over a factor of 8 in γ — a clean $1/\gamma$ scaling — and
equals $\tan30°=0.577$ as predicted (the residual ~0.4% drift is the $(1+\beta\cos2\psi)$
factor plus the finite-θ truncation dropped in the leading-order expansion). At the AWA
operating point this gives 0.49%, matching the standalone figure.

---

## 6. The same 1/γ governs the intensity asymmetry

The crescent (intensity) asymmetry seen in `asymmetry.png` shares the origin. Comparing the
two in-plane lobes (χ = 0 vs χ = π) of $A_\rho=C\cos\chi-R$:

$$|A_\rho(\chi{=}0)|=R-C,\qquad |A_\rho(\chi{=}\pi)|=R+C
\;\Rightarrow\;
\frac{I_{\text{bright}}}{I_{\text{dim}}}=\left(\frac{R+C}{R-C}\right)^2\approx 1+\frac{4C}{R}
\sim 1+\frac{4\tan\psi}{\gamma}.$$

Same $C/R\sim\tan\psi/\gamma$ ratio — so the lobe-intensity imbalance is also $O(1/\gamma)$,
a few percent at γ = 117 (consistent with the measured $L/R\approx0.96$). The *magnitude*
asymmetry (in $A_\rho$) and the *direction* twist (in $A_\chi$) are the two faces of the same
sub-dominant background interfering with the γ-enhanced radial lobe.

---

## 7. Assumptions and validity

- **Point charge, ideal infinite conductor.** This is the image-charge TR amplitude only;
  finite foil size, conductivity, and formation-length effects are not included (they would
  add their own, generally larger, geometric asymmetry — the part Level-1's `gf_tilt_lobe_k`
  absorbs empirically).
- **Leading order in θ ~ 1/γ and in C/R.** The expansion keeps the lowest non-vanishing term
  in each; corrections are $O(1/\gamma^2)$, consistent with the ~1% agreement above.
- **Non-grazing tilt.** $C(\psi)=\tan\psi$ diverges as ψ → 90°; the small-twist linearization
  (and the paraxial backward-lobe picture) breaks down well before grazing incidence.
- **Source plane.** This is the polarization at the foil. The relay transports it unrotated
  because mirrors are modeled as thin lenses (`SRWLOptL`); a real off-axis paraboloid adds a
  separate geometric rotation (`SRWLOptMirPar`) — see `WRITEUP.md` §3.10.

---

## 8. Summary

1. Backward TR from a tilted foil = image (specular) kernel − electron kernel.
2. The **image kernel** is resonant (denominator $\sim1/\gamma^2$): a radially polarized
   donut peaking at θ ~ 1/γ with amplitude $E_r\sim\beta\gamma$.
3. The **electron kernel** is non-resonant: a fixed in-plane vector of size
   $C(\psi)=\beta\sin2\psi/(1+\beta\cos2\psi)\to\tan\psi$, *independent of γ*, with no
   azimuthal rotation — it is the unboosted projection of the forward electron velocity.
4. Their interference makes the polarization deviate from radial by
   $\Delta=\big[C(\psi)/\beta\gamma\big]\sin\chi$, i.e.
   $|E_\phi/E_r|_{\max}=\tan\psi/\gamma$.
5. The $1/\gamma$ is the ratio (un-boosted background)/(γ-boosted radial lobe). For γ = 117
   it is ~0.5% — the simulation reproduces $\tan\psi/\gamma$ across ψ and across γ to ~1%.
