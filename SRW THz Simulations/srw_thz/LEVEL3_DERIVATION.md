# Level-3 Oblique-Incidence CTR Source — Math and Derivation

The Level-1 tilted-screen model (`--screen-tilt`, see `WRITEUP.md` §3.9) reproduces the
*qualitative* asymmetry of CTR from a tilted foil with a geometric heuristic
(foreshortening + lobe bias). **Level 3** replaces that heuristic with the *exact*
oblique-incidence transition-radiation (TR) field. This note explains the underlying
physics, the cleanest way to derive it, how it reduces to what is already implemented,
and how it would drop into the code.

---

## 1. The quantity we are after

We want the single-electron backward-TR **vector field amplitude** in the far field, as a
function of the two observation angles $(\theta_x,\theta_y)$ measured from the specular
direction, split into the in-plane ($p$) and out-of-plane ($s$) polarizations:

$$\vec{E}(\theta_x,\theta_y,\omega) = E_p(\theta_x,\theta_y)\,\hat{e}_p
  + E_s(\theta_x,\theta_y)\,\hat{e}_s .$$

Once $E_p$ and $E_s$ are known, the existing pipeline is unchanged: fill `arEx`, `arEy`
from them (rotating $\hat e_p,\hat e_s$ into $x,y$), project to coordinates via
$x=z_0\theta$, set $R_x=R_y=z_0$, and propagate. Everything in `WRITEUP.md` §3 stays; only
the amplitude function changes.

---

## 2. The exact result we are generalizing (normal incidence)

The Ginzburg–Frank formula — what `thz_source.gf_shape` implements — is the $\psi=0$ case.
In the small-angle, ultrarelativistic limit the field amplitude is

$$\vec{E}(\vec\theta) \;\propto\; \frac{\vec\theta}{\gamma^{-2}+\theta^2},
  \qquad \theta^2=\theta_x^2+\theta_y^2 ,$$

which is **radial in angle** (hence radial polarization), **zero on axis**, and peaked at
$\theta\simeq 1/\gamma$. Its modulus squared is the Ginzburg–Frank intensity

$$\frac{d^2W}{d\omega\,d\Omega}
  =\frac{e^2}{4\pi^3\varepsilon_0 c}\,\frac{\beta^2\sin^2\theta}{(1-\beta^2\cos^2\theta)^2}
  \;\xrightarrow{\text{small }\theta}\;
  \frac{e^2}{4\pi^3\varepsilon_0 c}\,\frac{\theta^2}{(\gamma^{-2}+\theta^2)^2}.$$

This is exact and is the anchor: the oblique formula **must** reduce to it as $\psi\to0$.

---

## 3. Cleanest derivation route — equivalent photons + specular reflection

This method makes every feature (the null, radial polarization, the $1/\gamma$ cone, the
specular centring, and the tilt asymmetry) fall out naturally. Four steps.

### Step 1 — the electron's field at the screen (Weizsäcker–Williams)

A relativistic electron's transverse field, Fourier-transformed in time, at transverse
distance $\rho$ from the trajectory:

$$\vec{E}_\perp(\rho,\omega)=\frac{e}{2\pi\varepsilon_0 v}\,\frac{\omega}{\gamma v}\,
  K_1\!\left(\frac{\omega\rho}{\gamma v}\right)\hat\rho ,$$

a radial field ($K_1$ = modified Bessel function), with the longitudinal part suppressed
by $1/\gamma^2$. This is the flux of "virtual photons" incident on the screen.

### Step 2 — its angular spectrum

Take the 2-D spatial Fourier transform over the screen plane. With transverse wavevector
$\vec q_\perp=(\omega/c)\,\vec\theta$,

$$\tilde{\vec E}(\vec\theta,\omega)\;\propto\;\frac{\vec\theta}{\theta^2+\gamma^{-2}} .$$

Steps 1–2 already reproduce the normal-incidence TR amplitude of §2. The on-axis null is
just $\vec\theta\to 0$; the radial polarization is the $\hat\theta$ direction.

### Step 3 — reflect off the conductor

TR is the specular reflection of these virtual photons. Each plane-wave component with
incident wavevector $\vec k_{\rm in}$ reflects to

$$\vec k_{\rm out}=\vec k_{\rm in}-2(\vec k_{\rm in}\!\cdot\hat n)\,\hat n ,
  \qquad \hat n=(\sin\psi,\,0,\,\cos\psi),$$

and the perfect-conductor boundary condition fixes the reflection coefficients
$r_s=-1$, $r_p=+1$ (tangential $\vec E$ vanishes on the surface). The reflected, now-real
photons are the backward TR.

### Step 4 — express in the observation (specular) frame

Rotate into the frame whose axis is the specular direction
$\hat v_{\rm spec}=\hat v-2(\hat v\!\cdot\hat n)\hat n$ (for $\psi=45^\circ$ this is
$90^\circ$ from the beam). Write the observation direction as small angles
$(\theta_x,\theta_y)$ about $\hat v_{\rm spec}$, with $x$ in the plane of incidence. The
reflection of Step 3 maps the **incident** angular spectrum (centred on the beam, radial)
onto the **observed** one (centred on specular), and the Jacobian/projection of the tilted
plane does two things:

- compresses the in-plane angle by $\cos\psi$ (the **foreshortening**), and
- splits the originally-radial field into $p$ and $s$ with *different* angular dependence
  and makes the two in-plane lobes **unequal** (the **asymmetry**).

The Level-1 `θ_eff = √((θx/cosψ)² + θy²)` and the lobe factor are precisely the
**leading-order** content of this step — Level 3 performs Step 4 exactly instead of
parameterizing it.

---

## 4. Structural form of the oblique result

Carrying Step 4 through (or equivalently the image-charge method of §5) gives, to leading
order in $1/\gamma$, two components of the form

$$E_p \propto
  \frac{\theta_x\cos\psi \;-\; \big(\gamma^{-2}+\theta_\perp^2\big)^{1/2}\,g(\psi)}
       {\gamma^{-2}+\theta_x^2\cos^2\psi+\theta_y^2},
  \qquad
  E_s \propto
  \frac{\theta_y}{\gamma^{-2}+\theta_x^2\cos^2\psi+\theta_y^2}.$$

Read off the structure:

- the denominator is the **foreshortened** $\gamma^{-2}+\theta^2$ (in-plane angle scaled
  by $\cos\psi$);
- $E_s$ stays "GF-like" (odd in $\theta_y$, vanishing in the plane of incidence);
- $E_p$ acquires a $\psi$-dependent term **linear in $\theta_x$** that breaks the up/down
  lobe symmetry — the rigorous origin of the Level-1 lobe bias.

> **Honesty flag.** The *structure* above (denominator foreshortening; $s$ unchanged; $p$
> asymmetric, linear-in-$\theta_x$) is reliable. The exact coefficient $g(\psi)$ and the
> overall prefactor should be taken **verbatim from a primary source** (§6), not trusted
> from this sketch.

---

## 5. Rigorous alternative — image charges

Equivalent, and good for cross-checking. Replace the conductor by the **image charge**
$-e$ at the mirror position, moving with the reflected velocity

$$\vec\beta_{\rm img}=\vec\beta-2(\vec\beta\!\cdot\hat n)\,\hat n$$

(so it travels along $\hat v_{\rm spec}$). Backward TR is the radiation from this image
charge being "created/annihilated" at the crossing — i.e. the field of a uniformly moving
charge that abruptly terminates:

$$\frac{d^2W}{d\omega\,d\Omega}=\frac{e^2}{4\pi^3\varepsilon_0 c}
  \left|\frac{\hat k\times\vec\beta_{\rm img}}{1-\hat k\cdot\vec\beta_{\rm img}}\right|^2 .$$

Insert $\vec\beta_{\rm img}$ for the tilted plane, expand $\hat k$ in $(\theta_x,\theta_y)$
about $\hat v_{\rm spec}$, and the same $E_p,E_s$ emerge. At $\psi=0$,
$\vec\beta_{\rm img}=-\beta\hat z$ and this collapses to the Ginzburg–Frank result.

---

## 6. Derivation steps to carry out, and references

1. Fix conventions: beam $\hat z$, screen normal $\hat n=(\sin\psi,0,\cos\psi)$, specular
   axis $\hat v_{\rm spec}$, and the $(\hat e_p,\hat e_s)$ basis about it.
2. Take the incident angular spectrum $\tilde{\vec E}\propto\vec\theta/(\theta^2+\gamma^{-2})$
   (Step 2).
3. Apply the reflection map (Step 3) with $r_p,r_s$ and the wavevector reflection.
4. Change variables to $(\theta_x,\theta_y)$ about specular, including the Jacobian;
   collect $E_p,E_s$.
5. Check the two limits: $\psi\to0$ → GF radial field; $\theta\to0$ → null.
6. Verify the coefficients against a primary source.

**Primary references** (this is where to pin down $g(\psi)$ and the prefactor):

- **V. E. Pafomov** — canonical derivation of oblique-incidence transition radiation; his
  formulas are the standard.
- **M. L. Ter-Mikaelian**, *High-Energy Electromagnetic Processes in Condensed Media* —
  compiles the TR/OTR results including oblique incidence.
- **V. L. Ginzburg & V. N. Tsytovich**, *Transition Radiation and Transition Scattering*.
- OTR-diagnostics literature, where tilted-foil two-component angular distributions are
  written in practical notation (easiest to transcribe into code): **Rule & Fiorito**;
  **M. Castellano & V. A. Verzilov**, *Phys. Rev. ST Accel. Beams* **1**, 062801 (1998);
  **X. Artru et al.**

---

## 7. How it drops into the code

Replace the scalar `gf_shape` with a function returning the two components on the angle
grid:

```python
def gf_oblique(theta_x, theta_y, beta, psi):
    # returns (E_p, E_s) on the angle mesh from the exact oblique-TR formula
    ...
    return E_p, E_s
```

then in `src_ctr_wavefront`, rotate $(\hat e_p,\hat e_s)\to(\hat x,\hat y)$ and fill:

```python
Ex = E_p*np.cos(chi) - E_s*np.sin(chi)     # chi = azimuth of the p-axis in the x-y plane
Ey = E_p*np.sin(chi) + E_s*np.cos(chi)
```

Everything else — the projection $x=z_0\theta$, `Rx=Ry=z0`, capped propagation, and the
intensity-weighted ring extraction — is unchanged, because those already handle an
*arbitrary* asymmetric $\vec E(x,y)$.

### A code-friendly Level-3 route that avoids the closed form

Because the pipeline already fills an arbitrary angular field, you do **not** need the
analytic $E_p,E_s$: you can implement **Step 3/4 numerically**. Build the radial incident
spectrum $\vec\theta/(\theta^2+\gamma^{-2})$ on the angle grid, apply the specular
reflection and the $r_p,r_s$ boundary conditions geometrically, and resample onto the
observation angles about $\hat v_{\rm spec}$. This reproduces the exact oblique pattern
(within the small-angle/ideal-conductor assumptions) without transcribing $g(\psi)$, and
is arguably the cleanest Level-3 implementation for this code.

---

## 8. Summary

| | Level 1 (implemented) | Level 3 |
|---|---|---|
| Foreshortening | heuristic $\theta_x/\cos\psi$ | exact, from the reflection Jacobian |
| Lobe asymmetry | heuristic $\tanh(\theta_x\gamma)$ bias | exact, from the $p$-component |
| Polarization | radial (unchanged) | true $p$/$s$ split |
| Derivation | parameterized | equivalent-photon reflection or image charge |
| Source needed | none | Pafomov / Ter-Mikaelian coefficients, **or** numerical Step 3/4 |

The derivation is standard; only the exact $p$-component coefficient $g(\psi)$ and overall
prefactor must come from a primary source — or be sidestepped entirely by doing the
reflection numerically.
