# Level-3 Oblique-CTR Source with Polarization — Implementation Plan & Methods

**Goal.** Replace the Level-1 tilted-screen *heuristic* (`--screen-tilt`, foreshortening +
`tanh` lobe bias, polarization left radial) with the **exact point-charge oblique
transition-radiation field**, including its true **p/s polarization**, and propagate both
polarization components through the beamline to a polarization-sensitive detector.

The physics is derived in `LEVEL3_DERIVATION.md` (image-charge method); this document is
the *method* — exactly what to compute, where it goes in the code, how polarization is
carried end-to-end, and how it is verified. It does not change any existing behaviour:
Level 1 stays the default; Level 3 is opt-in.

---

## 1. The key simplification — use the vector amplitude directly

The image-charge derivation gives the **exact** far-field radiation amplitude as a single
vector (perfect conductor, point charge, sudden crossing):

$$\vec{\mathcal A}(\hat n)=\frac{\hat n\times(\hat n\times\vec\beta)}{1-\hat n\cdot\vec\beta}
  -\frac{\hat n\times(\hat n\times\vec\beta_i)}{1-\hat n\cdot\vec\beta_i}
  \;\equiv\; \mathbf T(\hat n,\vec\beta)-\mathbf T(\hat n,\vec\beta_i),$$

with the helper

$$\mathbf T(\hat n,\vec b)=\frac{\hat n(\hat n\cdot\vec b)-\vec b}{1-\hat n\cdot\vec b},
  \qquad \vec\beta=\beta\hat z,\quad \vec\beta_i=\beta\big(\hat z-2\cos\psi\,\hat n_s\big),
  \quad \hat n_s=(\sin\psi,0,\cos\psi).$$

$\vec{\mathcal A}$ is automatically transverse to $\hat n$, so **its two transverse
components *are* the two polarization components** — there is no need to evaluate the
$E_p,E_s$ closed forms or do a separate p/s decomposition. We compute $\vec{\mathcal A}$ on
the source grid and project it onto the wavefront's transverse axes. This is exact for any
$\psi$ and reduces to Ginzburg–Frank at $\psi=0$ (verified analytically in the derivation).

> The closed-form $E_p,E_s$ (derivation §5) remain useful as a **cross-check** and for
> intuition, but the implementation uses the vector $\vec{\mathcal A}$.

---

## 2. Coordinate frames and the field-to-wavefront mapping

The beamline propagation axis is the **specular direction** $\hat s=\hat\beta_i$ (the optics
are aligned to collect the specularly-emitted TR). Build the transverse basis on it:

$$\hat e_s=\hat y\ \text{(out of tilt plane)},\qquad
  \hat e_p=\frac{\hat s\times\hat y}{|\hat s\times\hat y|}\ \text{(in tilt plane)}.$$

A source-plane pixel at transverse position $(x,y)$ on the $z_0$ plane corresponds to the
lab-frame observation direction

$$\hat n=\frac{\hat s+(x/z_0)\,\hat e_p+(y/z_0)\,\hat e_s}{\lvert\cdots\rvert}.$$

Evaluate $\vec{\mathcal A}(\hat n)$ (a lab-frame 3-vector) at every pixel, then project onto
the wavefront transverse basis:

$$E_x^{\rm wfr}(x,y)=\vec{\mathcal A}\cdot\hat e_p,\qquad
  E_y^{\rm wfr}(x,y)=\vec{\mathcal A}\cdot\hat e_s.$$

Because $\hat e_p,\hat e_s$ *are* the wavefront $x,y$ axes by construction, this projection
is the polarization assignment: the in-plane (p) field goes to `arEx`, the out-of-plane
(s) field to `arEy`. (For a tilt in an arbitrary azimuth, rotate by the azimuth $\chi$
first; for the default $x$-plane tilt, $\chi=0$.)

---

## 3. Polarization handled end-to-end

This is the heart of the upgrade. Three stages:

**3.1 Source — both components carry true p/s.**
`arEx` $=E_x^{\rm wfr}$ and `arEy` $=E_y^{\rm wfr}$ from §2. Unlike Level 1 (which fills a
single radial amplitude into both via $\cos\phi,\sin\phi$), here $E_x$ and $E_y$ are
*independent* functions with the correct oblique structure: $E_x$ (p) acquires the constant
in-plane background that fills the null and breaks the lobe symmetry; $E_y$ (s) stays
GF-like and antisymmetric in $y$.

**3.2 Propagation — preserved automatically.**
SRW propagates `arEx` and `arEy` as independent scalar fields through drifts and the
thin-lens-equivalent mirrors, which apply the same quadratic phase to both. So linear
polarization is preserved component-by-component with no extra work.
*Limitation:* real off-axis mirrors rotate/mix polarization (s vs p reflectivity and the
reflection geometry); the thin-lens model does **not** capture mirror-induced polarization
rotation. Level 3 therefore models the *source* polarization exactly but assumes
polarization-preserving optics. (A future step could insert Jones-matrix mirror elements.)

**3.3 Detection — project onto the crystal axis.**
The TFLN EO crystal responds to the THz field along its electro-optic axis $\hat c$ (a fixed
lab direction). The measured complex field is

$$E_{\rm meas}(f)=E_x\cos\chi_c+E_y\sin\chi_c\quad\text{at the sampling location},$$

with $\chi_c$ the crystal-axis azimuth (`crystal_pol_axis_deg`). This replaces Level 1's
radial projection: Level 1 assumed the detector sees the radial field; Level 3 measures a
genuine Cartesian polarization, which is what an EO crystal actually does. Sample at the
bright-lobe location (the asymmetric pattern is no longer azimuthally uniform), using the
existing intensity-weighted bright-region average.

**3.4 Polarization diagnostics (new outputs).**
With both components available we can emit, per the broadband map:
- `pol_p_map.png`, `pol_s_map.png` — the in-plane and out-of-plane intensity $|E_x|^2$,
  $|E_y|^2$;
- `pol_angle_map.png` — the local field-orientation angle $\arctan(|E_y|/|E_x|)$, showing
  the radial pattern at $\psi=0$ distorting into the mixed oblique pattern.

---

## 4. Interface (config + CLI)

Add to `ThzConfig`:

```python
gf_tilt_model: str = 'level1'      # 'level1' (heuristic) | 'level3' (exact oblique field)
crystal_pol_axis_deg: float = 0.0  # EO-crystal sensitive axis azimuth in the x-y plane
gf_include_electron_term: bool = True   # keep the non-resonant electron term in A (the
                                        # source of the polarization change); off = image
                                        # term only (radial), for comparison
```

CLI flags in `run_thz.py`: `--tilt-model {level1,level3}`, `--crystal-pol-axis DEG`.
`screen_tilt_deg` is shared; `gf_tilt_lobe_k` is ignored by Level 3.

---

## 5. Code changes (function level)

**`thz_source.py` — new `gf_oblique_field(X, Y, cfg)`** returning `(EX, EY)` arrays:

```python
def gf_oblique_field(X, Y, cfg):
    psi   = radians(cfg.screen_tilt_deg)
    gamma = cfg.gamma; beta = sqrt(1 - 1/gamma**2); z0 = cfg.z_first_oe_m
    zhat  = (0,0,1); ns = (sin psi, 0, cos psi)
    bvec  = beta*zhat
    bivec = beta*(zhat - 2*cos psi * ns)        # image (specular) velocity
    shat  = bivec/|bivec|
    ep    = normalize(cross(shat, (0,1,0)))      # in-plane transverse axis
    es    = (0,1,0)                               # out-of-plane axis
    thx, thy = X/z0, Y/z0
    n = shat + thx[...]*ep + thy[...]*es          # 3-vector per pixel; then normalize
    def T(n, b): return (n*(n·b)[...,None] - b) / (1 - (n·b))[...,None]
    A = T(n, bvec) - T(n, bivec)                  # exact vector amplitude (per pixel)
    if not cfg.gf_include_electron_term:
        A = -T(n, bivec)                          # image-only (radial) comparison
    EX = (A·ep); EY = (A·es)
    return A_norm * EX, A_norm * EY               # A_norm from _gf_amplitude_norm-style scaling
```

**`thz_source.src_ctr_wavefront`** — branch at the field-build step:
```python
if cfg.gf_tilt_model == 'level3':
    EX, EY = gf_oblique_field(X, Y, cfg)          # exact; handles psi=0 (=GF) and psi>0
else:
    ... existing Level-1 build (gf_shape * lobe, radial) ...
```
Everything downstream (interleave into `arEx/arEy`, `Rx=Ry=z0`, `presCA=0`) is unchanged.
The frequency-adaptive mesh and capped propagation apply identically (both components live
on the same grid).

**`thz_extract.py` — polarization-aware sampling.** Add a Cartesian-axis option to the ring
sampler (or a sibling function): project onto $\hat c$ before the intensity-weighted
bright-region average:
```python
def crystal_axis_complex_field(wfr, cfg):
    Ex, Ey = complex Ex, Ey on the native mesh            # both components
    Ec = Ex*cos(chi_c) + Ey*sin(chi_c)                    # crystal-axis projection
    ... intensity-weighted average over the bright region (reuse the §3.6 machinery) ...
```
For `gf_tilt_model='level3'`, the driver uses this; the radial extractor stays for the
symmetric/Level-1 path.

**`run_thz.py`** — thread the flags; choose the extractor by `gf_tilt_model`; write the
polarization diagnostic maps; log `gf_tilt_model`, `crystal_pol_axis_deg` to `run_meta.txt`.

**`thz_verify.py`** — add `oblique_field_check` (see §7).

---

## 6. Normalization

Reuse the single-electron TR energy normalization from `_gf_amplitude_norm`
($dW/d\omega\approx\alpha\hbar(2\ln\gamma-1)$, derivation §3): scale $\vec{\mathcal A}$ so the
angular-integrated $|\vec{\mathcal A}|^2$ equals $dW/d\omega$. Because the vector amplitude
already contains both polarizations, integrate $|E_x|^2+|E_y|^2$. Units stay relative
(`unitElFld=0`); absolute calibration still requires $K_{\rm BPD}$.

---

## 7. Verification plan

Run `oblique_field_check` before any sweep:

1. **Reduces to GF.** At $\psi=0$: $E_x,E_y$ radial ($\propto\theta_x,\theta_y$),
   azimuthally symmetric, on-axis null — identical to `gf_shape`. (Numerically compare to
   the Level-1 symmetric build.)
2. **Null is filled in p only.** At $\psi>0$: along the line $\theta_x=0$, $E_x\to$ the
   constant background $C(\psi)$ (non-zero), while $E_y\to0$ — the polarization-change
   signature.
3. **Lobe asymmetry sign + magnitude.** In-plane lobe ratio matches the
   $\sin2\psi/(1+\beta\cos2\psi)$ cross-term prediction; compare to the $\sim1/\gamma$
   scaling (small for our beam — see §8).
4. **s-component antisymmetry.** $E_y$ is odd in $\theta_y$, even in $\theta_x$.
5. **Energy.** Angular-integrated $|E_x|^2+|E_y|^2$ matches the normalization target.
6. **End-to-end.** Full `--src ctr --tilt-model level3 --screen-tilt 30` run completes
   within the capped-propagation budget; compare its crescent + polarization maps to the
   Level-1 result.

---

## 8. Scope and limitations (state explicitly in outputs)

- **Point charge, ideal conductor, sudden crossing.** No formation-length or finite-foil
  physics; the spectrum is flat and the intrinsic polarization/asymmetry scales as
  $\sim1/\gamma$ (≈1% at $\gamma=117$). The large crescents seen in practice are dominated
  by finite-source/foreshortening/collection geometry, which this point-charge field does
  **not** contain. Consequence: Level 3 is *more correct in polarization* than Level 1 but
  will show a *weaker* intensity asymmetry; for matching measured crescents one still needs
  a geometric/foreshortening factor (Level 1's role) or a finite-source model on top.
- **Polarization-preserving optics.** Mirror-induced polarization rotation is not modelled
  (thin-lens-equivalent mirrors). A Jones-matrix mirror element is the next increment.
- **Far-field-of-source projection** $x=z_0\theta$ (shared with Level 1): valid for
  $z_0\gg\gamma\lambda$.

---

## 9. Implementation checklist

1. `thz_config.py`: add `gf_tilt_model`, `crystal_pol_axis_deg`, `gf_include_electron_term`.
2. `thz_source.py`: add `gf_oblique_field`; branch in `src_ctr_wavefront`; extend the
   amplitude normalization to use $|E_x|^2+|E_y|^2$.
3. `thz_extract.py`: add `crystal_axis_complex_field`; add polarization-map savers.
4. `run_thz.py`: `--tilt-model`, `--crystal-pol-axis`; extractor dispatch; polarization
   diagnostics; meta logging.
5. `thz_verify.py`: `oblique_field_check` (the six checks of §7).
6. Validate: $\psi=0$ ≡ GF; null-fill; asymmetry sign/scaling; full coarse run; compare
   to Level 1.
7. Document: fold results into `WRITEUP.md` §3.9 (add a "Level 3" subsection) and update
   `README.md` flags.

---

## 10. Recommended order of work

Start with `gf_oblique_field` + `oblique_field_check` (source-level, fast, no propagation)
to confirm the field is correct at the source plane (checks 1–5). Only then wire the
extractor, diagnostics, and a coarse end-to-end run. This isolates the new physics from the
propagation/extraction plumbing, which is already validated.
