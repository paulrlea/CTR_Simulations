# The Analytic 45° CTR Source (`--tilt-model analytic`)

This documents the CTR source model built from the supplied closed-form 45° transition-radiation
relation, how it maps onto the existing Level-3 physics, where it is implemented in the code, and
how it was verified. It is the companion to `LEVEL3_DERIVATION.md` (general-ψ image-charge field)
and `LEVEL3_TWIST_DERIVATION.md` (the 1/γ polarization analysis).

---

## 1. The relation

For a 45° radiator foil, the backward-TR far field was supplied in spherical
$(\mathbf e_\theta, \mathbf e_\phi)$ components as

$$
E_{45°}(r,\omega) = \frac{e\,\beta\, e^{-i\omega r/c}}{4\pi^2\epsilon_0 c\,r}
\left[
\frac{\sin\phi}{1-\beta\cos\phi\sin\theta}\,\mathbf e_\phi
+\frac{(\beta-\cos\theta)\cos\phi-\sin\theta}{(1-\beta\cos\theta)(1-\beta\cos\phi\sin\theta)}\,\mathbf e_\theta
\right].
$$

Frame: the electron velocity is $\boldsymbol\beta=\beta\hat z$, $\theta$ is measured from $\hat z$,
and $\phi$ is the azimuth, so the two denominators are the **electron resonance**
$(1-\beta\cos\theta)=(1-\beta\,\hat n\!\cdot\!\hat z)$ and the **image resonance**
$(1-\beta\cos\phi\sin\theta)=(1-\beta\,\hat n\!\cdot\!\hat x)$. The second denominator identifies the
image (specular) velocity as $\boldsymbol\beta_i=\beta\hat x$ — exactly the mirror of $\beta\hat z$
in a 45° foil. The backward-TR lobe is centred on $+\hat x$ ($\theta=90°,\ \phi=0$).

---

## 2. Why this is the Level-3 source (equivalence)

`gf_oblique_field` (Level 3) builds the point-charge image-charge amplitude

$$
\mathbf A(\hat n)=\mathbf T(\hat n,\boldsymbol\beta)-\mathbf T(\hat n,\boldsymbol\beta_i),
\qquad
\mathbf T(\hat n,\mathbf b)=\frac{\hat n(\hat n\cdot\mathbf b)-\mathbf b}{1-\hat n\cdot\mathbf b}.
$$

Decomposing $\mathbf A$ for $\boldsymbol\beta=\beta\hat z,\ \boldsymbol\beta_i=\beta\hat x$ into
spherical components gives, term by term,

$$
\mathbf T(\hat n,\beta\hat z)=\frac{\beta\sin\theta}{1-\beta\cos\theta}\,\mathbf e_\theta,
\qquad
\mathbf T(\hat n,\beta\hat x)=\frac{-\beta\cos\theta\cos\phi}{1-\beta\cos\phi\sin\theta}\,\mathbf e_\theta
+\frac{\beta\sin\phi}{1-\beta\cos\phi\sin\theta}\,\mathbf e_\phi,
$$

so that $\mathbf A = -\beta\cdot[\text{the bracket above}]$ **exactly**, i.e.

$$
\boxed{\;E_{45°} = -\frac{e}{4\pi^2\epsilon_0 c\,r}\,\mathbf A(\hat n)\;}
$$

The supplied relation is therefore an algebraic rewrite of the Level-3 image-charge amplitude for
the 45° case — not an approximation. The `analytic` model implements the relation directly (working
in $(\theta,\phi)$ and $(\mathbf e_\theta,\mathbf e_\phi)$) instead of forming the vector difference
$\mathbf T-\mathbf T$. (Full derivation of the identity, and the resulting $\tan\psi/\gamma$
polarization twist, is in `LEVEL3_TWIST_DERIVATION.md`.)

---

## 3. Implementation

### 3.1 The source field — `thz_source.gf_analytic_field`

Each pixel $(X,Y)$ maps to an observation direction about the $+\hat x$ (image) axis via the
far-field-of-source projection $x=z_0\theta$; the $(\mathbf e_\theta,\mathbf e_\phi)$ field is
converted to Cartesian and projected onto the wavefront transverse basis
$(\mathbf e_p=\hat z,\ \mathbf e_s=\hat y)$ to give the SRW arrays `(EX, EY)`.

```python
def gf_analytic_field(X, Y, cfg):
    """CTR source from the supplied closed-form 45-degree TR relation (an algebraic rewrite
    of the Level-3 image-charge field; the two are identical at psi=45 ...).

        E_45(theta,phi) ~  [ sin(phi) / (1 - beta cos(phi) sin(theta)) ] e_phi
                         + [ ((beta - cos theta) cos phi - sin theta)
                             / ((1 - beta cos theta)(1 - beta cos phi sin theta)) ] e_theta
    ...
    """
    gamma = cfg.gamma
    beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
    z0 = cfg.z_first_oe_m

    # observation directions about the specular (+x) axis, from the far-field projection.
    shat = np.array([1.0, 0.0, 0.0])                        # image/specular axis (45-deg foil)
    ep = np.cross(shat, [0.0, 1.0, 0.0]); ep /= np.linalg.norm(ep)   # in-plane transverse = +z
    es = np.array([0.0, 1.0, 0.0])                          # out-of-plane transverse = +y
    thx, thy = X / z0, Y / z0
    nvec = (shat[None, None, :] + thx[..., None] * ep[None, None, :]
            + thy[..., None] * es[None, None, :])
    nvec /= np.linalg.norm(nvec, axis=2, keepdims=True)
    nx, ny, nz = nvec[..., 0], nvec[..., 1], nvec[..., 2]

    theta = np.arccos(np.clip(nz, -1.0, 1.0))
    phi = np.arctan2(ny, nx)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)

    D1 = 1.0 - beta * ct                                    # electron resonance (never 0, beta<1)
    D2 = 1.0 - beta * cp * st                               # image resonance
    B_th = ((beta - ct) * cp - st) / (D1 * D2)
    B_ph = sp / D2

    # build the Cartesian field from spherical components and project onto (e_p, e_s):
    #   e_theta = (ct cp, ct sp, -st),  e_phi = (-sp, cp, 0)
    Evec = np.stack([B_th * (ct * cp) + B_ph * (-sp),
                     B_th * (ct * sp) + B_ph * (cp),
                     B_th * (-st)], axis=2)
    scale = _gf_amplitude_norm(cfg) * beta / (2.0 * z0)     # matches gf_oblique_field magnitude
    EX = (scale * (Evec @ ep)).astype(np.float32)
    EY = (scale * (Evec @ es)).astype(np.float32)
    return EX, EY
```

Notes:
- **No singularities.** For $\beta<1$, $D_1=1-\beta\cos\theta\ge 1-\beta>0$ and
  $D_2=1-\beta\cos\phi\sin\theta\ge 1-\beta>0$, so the denominators never vanish — cleaner than the
  vector $\mathbf T-\mathbf T$ form.
- **Amplitude match.** `scale` $=A_{\rm norm}\beta/2z_0$ reproduces `gf_oblique_field`'s magnitude,
  because that function applies $-(A_{\rm norm}/z_0)/2$ to $\mathbf A=-\beta\,B$. So the two are
  numerically interchangeable at 45°.

### 3.2 Selecting it — `thz_source.src_ctr_wavefront`

```python
    if cfg.gf_tilt_model == 'level3':
        # exact point-charge oblique-TR field with true p/s polarization (reduces to
        # symmetric GF at screen_tilt_deg=0)
        EX, EY = gf_oblique_field(X, Y, cfg)
    elif cfg.gf_tilt_model == 'analytic':
        # same physics via the supplied closed-form 45-deg TR relation (gf_analytic_field);
        # identical to level3 at psi=45. The foil angle is fixed at 45 deg by the relation.
        EX, EY = gf_analytic_field(X, Y, cfg)
    else:
        # Level-1 heuristic: foreshorten the pattern in x, bias the two x-lobes; radial pol.
        ...
```

### 3.3 Config option — `thz_config.ThzConfig`

```python
    # Tilt model: 'level1' = geometric heuristic ... 'level3' = exact point-charge oblique-TR
    # vector field ...
    # 'analytic' = the supplied closed-form 45-deg TR relation (gf_analytic_field); an algebraic
    # rewrite of level3, identical at psi=45 (verified in compare_analytic_45.py). The foil angle
    # is fixed at 45 deg by the relation, so screen_tilt_deg is only used to trigger the
    # polarization-aware extraction (set it to 45 to match the physics).
    gf_tilt_model: str = 'level1'
```

### 3.4 Driver — `run_thz.py`

CLI choice:

```python
    p.add_argument('--tilt-model', default=None, choices=['level1', 'level3', 'analytic'],
                   help='tilted-screen model: level1 heuristic, level3 exact oblique field, '
                        'or analytic (supplied closed-form 45-deg TR relation = level3 at 45 deg)')
```

The `analytic` model is treated like `level3` everywhere downstream — the polarization-aware
crystal-axis extraction, the diagnostic polarization maps, and the metadata log:

```python
    # polarization-aware on-crystal field (Cartesian pol -> project on crystal axis)
    if cfg.gf_tilt_model in ('level3', 'analytic') and cfg.screen_tilt_deg != 0.0:
        Ering = crystal_axis_complex_field(wfr, cfg) * ch
    else:
        Ering = ring_peak_complex_field(wfr, cfg) * ch
```

```python
    # detector_pol_{p,s,angle}.png at the spectral peak
    if args.src == 'ctr' and cfg.gf_tilt_model in ('level3', 'analytic') and cfg.screen_tilt_deg != 0.0:
        ...
```

```python
    # run_meta.txt
            if cfg.screen_tilt_deg != 0.0:
                if cfg.gf_tilt_model in ('level3', 'analytic'):
                    fh.write(f"crystal_pol_axis_deg={cfg.crystal_pol_axis_deg}\n")
                    fh.write(f"gf_include_electron_term={cfg.gf_include_electron_term}\n")
                else:
                    fh.write(f"gf_tilt_lobe_k={cfg.gf_tilt_lobe_k}\n")
```

---

## 4. Verification

### 4.1 The relation equals the image-charge amplitude (`compare_analytic_45.py`)

Evaluating both closed forms on a $(\theta,\phi)$ grid:

```
A_theta/B_theta : mean = -0.999964   max|dev from -beta| = 4.37e-13
A_phi/B_phi     : mean = -0.999964   max|dev from -beta| = 4.33e-13
transverse check max|A.n| = 1.76e-13   (expect ~0)
```

i.e. $\mathbf A = -\beta\,B$ to machine precision, both components, and $\mathbf A$ is transverse.
Comparing the actual simulation source `gf_oblique_field(ψ=45°)` to the equation's $|E|^2$ over the
source grid gives **correlation r = 1.00000** (after the x-mirror tilt-direction convention). See
`results_l3/compare_analytic_45.png`.

### 4.2 `gf_analytic_field` is a drop-in for `gf_oblique_field` at 45°

```
analytic vs level3 |E|^2 correlation:  x-mirror = 1.00000
analytic: I(center)/I_peak = 7.14e-05        (near-null center; ~1/gamma^2 fill)
analytic: peak at r = 4.18 mm                (expect z0/gamma = 4.26 mm)
amplitude match: max|E| analytic = 6.230e-17,  level3 = 6.230e-17,  ratio = 1.0000
```

### 4.3 End-to-end pipeline

```
python run_thz.py --src ctr --tilt-model analytic --screen-tilt 45 --nfreq 9
# -> 9/9 frequencies OK; detector_pol_*.png written; run_meta: gf_tilt_model=analytic
```

---

## 5. Usage

```bash
python run_thz.py --src ctr --tilt-model analytic --screen-tilt 45 --out results_analytic45
```

Full-resolution run and space-time reconstruction use the same flag:

```bash
python run_thz.py --src ctr --tilt-model analytic --screen-tilt 45 --nfreq 161 --out results_analytic45
python run_thz_spacetime.py --screen-tilt 45 --out results_analytic45_spacetime   # (level3 core; equivalent)
```

---

## 6. Scope and conventions

- **45°-only.** The foil angle (image ⊥ beam, $\boldsymbol\beta_i=\beta\hat x$) is baked into the
  relation's denominators. For any other tilt use `level3` (the general-ψ vector construction).
  `screen_tilt_deg` under `analytic` only triggers the polarization-aware extraction — set it to 45.
- **Tilt-direction convention.** The relation centres the backward lobe on $+\hat x$; `level3` at
  ψ=+45° centres it on $-\hat x$. This is a pure mirror (which way the foil tilts) and is physically
  equivalent — it only flips the crescent side (hence the x-mirror in the r=1.00000 check).
- **Same limitations as Level 3 otherwise:** point-charge, ideal-conductor, infinite-foil,
  far-field-of-source; absolute scale is the GF spectral-energy normalization (relative units),
  not the equation's $e/4\pi^2\epsilon_0 c r$ prefactor.
