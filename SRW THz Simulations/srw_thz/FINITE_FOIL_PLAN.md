# Finite-Foil Far-Zone Diffraction for the CTR Source — Implementation Plan & Methods

**Goal.** Add **finite radiator (foil) diffraction** to the CTR source. Today all three source
builders (`gf_analytic_field`, `gf_oblique_field`, the Level-1 heuristic) are **infinite-foil,
point-charge** far-field angular patterns; the only finite-foil physics anywhere is the *scalar*
per-frequency energy rolloff `interf_source.foil_amp` (`S_foil`), which changes no spatial
structure. This upgrade replaces that scalar with a **shape-aware, real, far-zone finite-foil
form factor** `H(θ)` that reproduces the low-frequency suppression *and* its angular signature —
ring broadening, partial null-fill, and edge-diffraction fringes — and folds in the **45°
elliptical foreshortening** of the tilted circular foil.

Scope was fixed with the requester: **far-zone only** (no pre-wave-zone / Fresnel phase — `H` is
real) and **include the 45° elliptical projection from the start**. This does not change any
existing behaviour: the infinite-foil source stays the default; finite-foil diffraction is opt-in
via a new orthogonal flag that composes with `analytic` / `level3`.

The companion physics/verification write-up will be `FINITE_FOIL_SOURCE.md` (mirrors
`ANALYTIC_45_SOURCE.md`); this document is the *method*.

---

## 1. The physics — finite foil = truncated Hankel transform of the virtual-photon field

Backward TR from a perfect conductor is the order-1 Hankel transform of the electron's radial
"virtual photon" field reflected in the foil,

$$E_r(r)\ \propto\ \kappa\,K_1(\kappa r),\qquad \kappa=\frac{k}{\beta\gamma}
  \quad\Longrightarrow\quad r_{\rm eff}=\kappa^{-1}=\frac{\beta\gamma}{k}=\frac{\beta\gamma\lambda}{2\pi}.$$

A foil of radius $a$ **truncates** that radiating current at $r=a$:

$$E_a(\theta)\ \propto\ \int_0^{a}\kappa K_1(\kappa r)\,J_1(q r)\,r\,dr,\qquad q=k\sin\theta\simeq k\theta,$$

while the infinite foil is the same integral to $\infty$:

$$E_\infty(\theta)\ \propto\ \int_0^{\infty}\kappa K_1(\kappa r)\,J_1(q r)\,r\,dr
  \;=\;\frac{q}{\kappa^{2}+q^{2}}\ \propto\ \frac{\theta}{\gamma^{-2}+\theta^{2}},$$

which is exactly the Ginzburg–Frank pattern already encoded in `gf_shape` / `gf_analytic_field`.
The **finite-foil form factor** is their ratio,

$$\boxed{\,H(\theta)\equiv\frac{E_a(\theta)}{E_\infty(\theta)}\,}\qquad H\to1\ \text{as}\ a\to\infty .$$

**Why this is significant here (not a small correction).** With $\gamma\simeq117$ and
$a=0.5''=12.7$ mm, $a/r_{\rm eff}\approx0.02$ (10 GHz), $0.23$ (100 GHz), $0.9$ (400 GHz): the
foil is smaller than the natural field extent across essentially the whole 10–400 GHz band, so
$H$ departs strongly from 1 band-wide.

### 1.1 Closed form (Lommel) — no quadrature needed

The truncated integral has an exact closed form. With $u=J_1(qr)$, $v=K_1(\kappa r)$ satisfying
the (modified) Bessel equations, the Lommel construction
$\frac{1}{r}\frac{d}{dr}\!\big[r(vu'-uv')\big]=-(q^2+\kappa^2)uv$ integrates to a pure boundary
term. The $r\!\to\!0$ end contributes $q/\kappa$; the $r=a$ end gives the edge Bessel functions:

$$\int_0^{a} r\,J_1(qr)K_1(\kappa r)\,dr
  =\frac{1}{q^{2}+\kappa^{2}}\Big[\tfrac{q}{\kappa}
     -a\,q\,K_1(\kappa a)J_0(qa)-a\,\kappa\,K_0(\kappa a)J_1(qa)\Big].$$

Dividing by $E_\infty\propto q/[\kappa(q^2+\kappa^2)]$ and writing the two dimensionless groups

$$x\equiv\kappa a=\frac{ka}{\beta\gamma}\ \ (\text{foil size vs. field extent}),\qquad
  y\equiv qa=ka\,\theta\ \ (\text{foil size}\times\text{angle}),$$

gives the **final, exact, real far-zone form factor**

$$\boxed{\;H(\theta)=1-x\,K_1(x)\,J_0(y)-\frac{x^{2}}{y}\,K_0(x)\,J_1(y)\;}
\qquad
H(0)=1-x\,K_1(x)-\tfrac{x^{2}}{2}\,K_0(x).$$

**Verified limits.**
- $a\to\infty$ ($x\to\infty$): $K_0,K_1\to0\Rightarrow H\to1$ (infinite foil recovered).
- $a\to0$ ($x\to0$): $xK_1(x)\to1\Rightarrow H(0)\to0$ (tiny foil radiates nothing coherently).
- $J_0,J_1$ oscillations $\Rightarrow$ **edge-diffraction fringes**; their zeros set fringe radii.
- The base on-axis null survives: $E_\infty(\theta)\!\to\!0$ as $\theta\!\to\!0$ and $H(0)$ is
  finite, so $H\!\cdot\!E_\infty\!\to\!0$ on axis (the null fills only *partially*, via the
  broadened ring, not by breaking the topological null).

Implementation is therefore a handful of `scipy.special` calls (`kn`, `jv`) — **no numerical
integration**, per-pixel cost negligible.

---

## 2. The 45° elliptical projection

Viewed along the specular (collection) axis, the circular foil is foreshortened. The angle
between the foil plane and the plane $\perp$ the specular axis equals the foil tilt $\psi$
(shown below), so the projected aperture is an **ellipse**: semi-axis $a\cos\psi$ along the
in-tilt-plane transverse direction ($\theta_x$, $=\hat e_p$) and $a$ along $\theta_y$
($=\hat e_s$). A smaller aperture along $\theta_x$ ⇒ **broader diffraction along $\theta_x$**.

Because $H$ enters only through $y=ka\theta$, the projection is applied by evaluating $H$ at an
**anisotropic effective angle**:

$$\boxed{\;\theta_{\rm ell}=\sqrt{\cos^{2}\!\psi\,\theta_x^{2}+\theta_y^{2}}\;},\qquad
  y=k\,a\,\theta_{\rm ell}\ \ \text{(and }x\text{ unchanged).}$$

At $\psi=45^\circ$, $\cos^2\psi=\tfrac12$: the tilt-axis aperture is $\approx0.71a$, so the
finite-foil fringes/broadening are $\approx1/\cos45^\circ\approx1.41\times$ **wider along
$\theta_x$ than $\theta_y$** — an elliptical diffraction signature that stacks on the analytic
source's own image-resonance crescent. Limits: $\psi\to0\Rightarrow\theta_{\rm ell}\to\theta$
(circular), $a\to\infty\Rightarrow H\to1$.

*Tilt angle identity.* Beam $\hat z$; foil normal $\hat n_{\rm foil}=(\sin\psi,0,\cos\psi)$;
specular axis $\hat s=(\sin2\psi,0,\cos2\psi)$. Then $\hat n_{\rm foil}\!\cdot\!\hat s=\cos\psi$,
so the foil-plane / $\perp$-specular-plane angle is $\psi$ and the foreshortening along the tilt
axis is $\cos\psi$. (At $\psi=45^\circ$, $\hat s=\hat x$, matching `gf_analytic_field`.)

**Second-order effect deliberately omitted (documented).** The virtual-photon *illumination*
projected onto the tilted foil is itself elliptical (stretched by $1/\cos\psi$ in the foil
plane). We keep the illumination isotropic ($x$ from the circular $\kappa a$) and apply the
ellipse only to the aperture-truncation angle $\theta_{\rm ell}$. The fully rigorous version is
the 2-D Fraunhofer integral over the tilted disk with elliptical illumination; noted in §8 as a
future increment.

---

## 3. Composition — a multiplicative factor on the existing source

$H$ is a scalar **magnitude** form factor; the base builder carries polarization and the
image-resonance crescent. Both $E_a$ and $E_\infty$ are Hankel transforms of the *same* radial
field, so $H=E_a/E_\infty$ is well-defined and, for the symmetric/radial case, **exact**. We
multiply the base field per pixel:

$$E_x^{\rm wfr}(x,y)\leftarrow H(\theta_{\rm ell})\,E_x^{\rm wfr},\qquad
  E_y^{\rm wfr}(x,y)\leftarrow H(\theta_{\rm ell})\,E_y^{\rm wfr}.$$

For the tilted vector field (`analytic`/`level3`) this is a **documented approximation**: the
aperture reshapes amplitude-vs-angle while the tilt sets polarization — the two factorize
cleanly. Everything downstream (interleave into `arEx/arEy`, `Rx=Ry=z0`, `presCA=0`, capped
propagation) is unchanged.

---

## 4. Interface (config + CLI)

Add to `ThzConfig` (`thz_config.py`):

```python
finite_foil_diffraction: bool = False   # apply the far-zone finite-foil form factor H(theta_ell)
foil_radius_m: float = 0.5 * 0.0254     # radiator (Mo foil) radius a; promoted from InterfConfig
```

- `foil_radius_m` is **promoted into the base config** so both `srw_thz` and the interferometer
  share one knob (the interferometer's existing `foil_radius_m` default is identical — remove the
  duplicate from `InterfConfig` and inherit).
- The tilt $\psi$ reuses the existing `screen_tilt_deg` (fixed at 45 for `analytic`; general for
  `level3`); $\psi=0$ gives the circular factor.

CLI (`run_thz.py` **and** `run_interf.py`):

```python
p.add_argument('--finite-foil', action='store_true',
               help='apply far-zone finite-foil diffraction form factor H(theta) to the CTR source')
p.add_argument('--foil-radius', type=float, default=None, help='radiator foil radius a [m]')
```

Mapped in the override builders to `finite_foil_diffraction` / `foil_radius_m`.

**Double-count guard (required).** `H` reproduces the low-f suppression that the scalar `S_foil`
approximates. When `finite_foil_diffraction` is on, force `finite_foil_enabled=False` and emit a
one-line warning; log both flags to `run_meta.txt`. (Do this in `build_cfg`/`__post_init__` so it
holds no matter how the config is constructed.)

---

## 5. Code changes (function level)

**`thz_source.py` — new `finite_foil_formfactor(theta_ell, f_Hz, cfg)`** returning the real
array `H` on the pixel grid:

```python
from scipy.special import kn, jv          # K0,K1 via kn(0,·)/kn(1,·); J0,J1 via jv

def finite_foil_formfactor(theta_ell, f_Hz, cfg):
    """Far-zone finite circular-foil TR form factor H = E_a/E_inf (Sec 1.1). ->1 as a->inf."""
    k    = 2*pi*f_Hz / C_LIGHT
    beta = sqrt(1 - 1/cfg.gamma**2)
    a    = cfg.foil_radius_m
    x    = k*a/(beta*cfg.gamma)                      # kappa*a  (scalar per frequency)
    y    = k*a*theta_ell                             # array over pixels (uses theta_ell, Sec 2)
    K1x, K0x = kn(1, x), kn(0, x)
    H = 1.0 - x*K1x*jv(0, y) - (x*x)*K0x*jv(1, y)/np.where(y == 0.0, 1.0, y)
    H0 = 1.0 - x*K1x - 0.5*x*x*K0x                   # y->0 limit
    return np.where(y == 0.0, H0, H)
```

**`thz_source.src_ctr_wavefront`** — apply after the base `(EX, EY)` are built:

```python
if getattr(cfg, 'finite_foil_diffraction', False):
    psi = radians(cfg.screen_tilt_deg)
    theta_ell = np.sqrt((cos(psi)*thx)**2 + thy**2)     # thx=X/z0, thy=Y/z0 already defined
    H = finite_foil_formfactor(theta_ell, f_Hz, cfg)
    EX = (EX * H).astype(np.float32); EY = (EY * H).astype(np.float32)
```

(Placed to wrap all three `gf_tilt_model` branches, so it composes with `level1/level3/analytic`.)

**`interf_source.py`** — no functional change; `make_source_conv` already calls `make_source`
which routes through `src_ctr_wavefront`. Confirm `foil_amp` is skipped when
`finite_foil_diffraction` is on (guard in §4 sets `finite_foil_enabled=False`).

**`run_thz.py`, `run_interf.py`** — add the two CLI flags, the override mapping, the double-count
guard/warning, and `finite_foil_diffraction` / `foil_radius_m` in the `run_meta.txt` block.

**`compare_finitefoil.py`** (new, mirrors `compare_analytic_45.py`) — the §7 checks + a
`results_l3/compare_finitefoil.png` figure (H vs θ at 10/100/400 GHz; circular vs elliptical
ring; H·E vs E source maps).

---

## 6. Normalization

`H` is a pure dimensionless ratio ($\to1$ for a large foil), so multiplying the already
energy-normalized base field (`_gf_amplitude_norm`) needs **no renormalization**: the
infinite-foil amplitude stays the reference and $H$ applies the physical finite-foil reduction on
top. The angle-integrated $\int|H\,E|^2\,d\Omega < \int|E|^2\,d\Omega$ *is* the low-frequency
energy loss — and it supersedes the scalar `S_foil` (hence the §4 guard). Units stay relative
(`unitElFld=0`).

---

## 7. Verification plan (`compare_finitefoil.py`, source-level, no propagation)

1. **Reduces to infinite foil.** `finite_foil_diffraction=False` *or* $a\to\infty$: `|E|²`
   correlation with the current source $\approx1$; $\max|H-1|\to0$.
2. **On-axis suppression.** $H(0)=1-xK_1(x)-\tfrac{x^2}{2}K_0(x)$ matches a direct numerical
   quadrature of $E_a(0^+)/E_\infty(0^+)$ to machine precision (validates the Lommel closed form).
3. **Energy vs `S_foil`.** Angle-integrated $\int|H E|^2/\int|E|^2$ tracks
   $S_{\rm foil}(f)=[\ln(\min(2\gamma,ka))-1]/[\ln2\gamma-1]$ in trend and in the
   $a\!\gg\!r_{\rm eff}$ (→1) and $a\!\ll\!r_{\rm eff}$ (→0) limits.
4. **Fringes.** Ring intensity minima fall at the $J_0/J_1$-driven zeros of $H(k a\theta)$.
5. **Elliptical signature.** Fringe spacing along $\theta_x$ is $1/\cos45\approx1.41\times$ that
   along $\theta_y$; at $\psi=0$ the pattern is azimuthally symmetric.
6. **Low-f onset.** Suppression turns on near $ka\approx1$ (foil sub-wavelength), consistent with
   the $a/r_{\rm eff}$ table.
7. **End-to-end.** `python run_interf.py --nfreq 9 --tilt-model analytic --screen-tilt 45
   --finite-foil` completes within the capped-propagation budget; compare ring / power-capture /
   recovered spectrum to the infinite-foil run.

---

## 8. Scope and limitations (state in outputs / `FINITE_FOIL_SOURCE.md`)

- **Far-zone only (by request).** `H` is real; no pre-wave-zone/Fresnel phase. Since
  $z_0\sim0.18$–$0.5$ m $\ll\gamma^2\lambda\approx41$ m (100 GHz), the observation is formally in
  the near (pre-wave) zone; the far-field projection $x=z_0\theta$ is retained and this remains a
  **known caveat**, unchanged from the existing source. The pre-wave-zone generalization (Fresnel
  phase in the radial integral) is the natural future increment for full rigor.
- **Elliptical aperture, isotropic illumination.** The 45° projection is applied to the aperture
  truncation ($\theta_{\rm ell}$) but not to the virtual-photon illumination (§2). Full rigor =
  2-D Fraunhofer integral over the tilted disk with elliptical illumination.
- **Scalar-$H$ factorization on the tilted vector field.** Exact for the radial/symmetric case;
  a documented approximation for `analytic`/`level3` (aperture reshapes amplitude; tilt sets
  polarization).
- **Point charge, ideal conductor, circular foil.** Inherits the base-source assumptions; adds
  finite radiator *size* only (not foil conductivity, thickness, or edge-current corrections).

---

## 9. Implementation checklist

1. `thz_config.py`: add `finite_foil_diffraction`; promote `foil_radius_m` to base; remove the
   `InterfConfig` duplicate (inherit).
2. `thz_source.py`: add `finite_foil_formfactor`; apply the `H(θ_ell)` block in
   `src_ctr_wavefront` after the base field build (`scipy.special` import).
3. `run_thz.py` / `run_interf.py`: `--finite-foil`, `--foil-radius`; override mapping;
   double-count guard + warning; `run_meta.txt` logging.
4. `compare_finitefoil.py`: the seven §7 checks + figure.
5. Validate: infinite-foil limit ≡ current source; Lommel closed form vs quadrature; energy vs
   `S_foil`; fringes; elliptical 1.41× check; coarse end-to-end run.
6. Document: `FINITE_FOIL_SOURCE.md` (companion to `ANALYTIC_45_SOURCE.md`); update `README.md`
   flags and `WRITEUP.md`.

---

## 10. Recommended order of work

Start with `finite_foil_formfactor` + checks 1–6 of §7 (source-level, fast, no propagation) to
confirm `H`, its closed-form correctness, the energy tie-in to `S_foil`, and the elliptical
signature. Only then wire the CLI/guard/meta and run a coarse end-to-end interferometer sweep
(check 7). This isolates the new physics (one analytic factor) from the already-validated
propagation/extraction plumbing.
