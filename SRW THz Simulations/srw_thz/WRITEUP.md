# SRW and the AWA CTR-THz Propagation Pipeline

---

## 1. What is SRW?

The Synchrotron Radiation Workshop (SRW) is an open-source physics code developed by
Oleg Chubar (Brookhaven National Laboratory) and Pascal Elleaume (ESRF), designed
primarily to calculate and propagate synchrotron radiation wavefronts through optical
beamlines. The version used here is **srwpy 4.2.1 (core v4.10)**, installed as a Python
package with a compiled C++ extension (`srwlpy`) that performs all heavy numerical work.

### 1.1 Core Physical Model

SRW operates on the concept of a **single-electron electric field wavefront**. For a
given source (a relativistic electron in a magnetic field, or a Gaussian beam), it
computes the full complex electric field $E(x, y)$ — amplitude and phase — on a
transverse observation plane at a specified distance from the source. This is distinct
from intensity-only codes: SRW tracks the coherent field, which means it can model
interference, diffraction, and wavefront distortion correctly.

The two primary source types in SRW are:

- **Synchrotron radiation from magnetic fields** (`srwl.CalcElecFieldSR`): calculates
  the radiation emitted by an electron traversing a user-defined magnetic field
  (bending magnet, undulator, wiggler, arbitrary 3D field). This is the code's
  original and primary application.

- **Gaussian beams** (`srwl.CalcElecFieldGaussian`): propagates a user-defined
  Gaussian-Hermite mode from its waist. Used here as a surrogate THz source (see §2).

A third path — filling the wavefront arrays (`wfr.arEx`, `wfr.arEy`) directly with
analytically prescribed field values — is also possible, and is how the analytic
Ginzburg-Frank CTR source (§3) is implemented.

### 1.2 Wavefront Propagation

Once the source wavefront is computed, SRW propagates it through a sequence of optical
elements using the **Huygens-Fresnel diffraction integral**, implemented as a
fast Fourier transform (FFT) pipeline, via the call `srwl.PropagElecField(wfr, optBL)`,
where `optBL` is an `SRWLOptC` container holding an ordered list of optical elements and
their associated propagation parameters. Each element modifies the wavefront before it is
propagated to the next. Supported element types include:

| SRW class | Physical element |
|---|---|
| `SRWLOptD` | Free-space drift (propagation by FFT) |
| `SRWLOptA` | Aperture or obstacle (rectangle or circle) |
| `SRWLOptL` | Thin lens (quadratic phase multiplication) |
| `SRWLOptT` | Transmission element (tabulated amplitude + phase) |
| `SRWLOptMirPl/El/Par/Tor` | Mirror (plane, elliptical, parabolic, toroidal) |

Each optical element has an associated 12-element propagation parameter array that
controls automatic mesh resizing (whether to expand the transverse window before or
after propagation), resolution scaling, and whether to treat the leading quadratic phase
term semi-analytically — a critical option for long-drift propagation, activated by
setting parameter index 3 to 1.

### 1.3 Partially Coherent Emission

For incoherent synchrotron radiation from a finite-emittance electron beam, SRW
provides `srwl_wfr_emit_prop_multi_e`, which samples over the 6D electron phase space
using a macro-particle approach. A special case with `_char=20` sums the single-electron
fields coherently — this is SRW's built-in coherent synchrotron radiation (CSR) mode.
This path is not used in the present model (see §2.1).

### 1.4 Limitations Relevant Here

- **No transition radiation source.** SRW's emission engine is built around electrons
  in magnetic fields. There is no `CalcElecFieldTransitionRadiation` function; the
  physics of a relativistic electron crossing a material boundary (transition radiation)
  is not implemented. This is the central constraint that shapes the present pipeline.

- **Long-wavelength numerical cost.** SRW's transverse mesh must contain the entire
  beam at every propagation plane. For THz radiation at $\gamma = 117$, the Gaussian
  source radius at 10 GHz is $\sigma_r \approx 1.12$ m, requiring source meshes of
  thousands of points per axis and producing detector meshes in the tens of millions of
  points at some frequencies. This is unavoidable: the physics of low-frequency THz
  diffraction over metre-scale distances demands it.

- **Per-frequency propagation.** SRW is a monochromatic code. Broadband results require
  sweeping over a frequency grid and assembling the results externally — which is what
  the driver script does.

---

## 2. The AWA CTR-THz Propagation Pipeline

### 2.1 Source

The experiment generates THz radiation via **Coherent Transition Radiation (CTR)**:
60 MeV electrons ($\gamma = 117.4$, $\beta \approx 0.999964$) cross a thin radiative
foil. CTR is not a magnetic-field process, so SRW cannot compute it natively. Instead, a
**Gaussian-beam surrogate** is synthesised per frequency.

An `SRWLGsnBm` is initialised with its waist at the foil ($z = 0$) and its RMS beam size
$\sigma_r(f)$ chosen so that the Gaussian far-field divergence $\lambda/(\pi\sigma_r)$
equals the CTR emission angle $1/\gamma$:

$$\sigma_r(f) = \text{div\_match}\times \frac{\lambda(f)\,\gamma}{\pi}$$

At 100 GHz this gives $\sigma_r = 11.2$ cm; at 10 GHz it is 1.12 m — much larger than the
2.3 cm collection aperture radius. This overfilling of the aperture at low frequencies
is the physical mechanism that produces the geometric low-frequency cutoff in the
detected spectrum, and it is reproduced correctly by the surrogate.

The surrogate is a deliberate approximation. It overestimates the on-axis field (the
true CTR angular pattern has a donut shape with a null on axis, from the Ginzburg-Frank
formula) and uses linear horizontal polarisation rather than the true radial
polarisation. An analytic **Ginzburg-Frank source** now exists behind the same interface
(`make_source(f, cfg, src='ctr')`); it captures the on-axis null and radial polarisation
and is documented in full in §3. The Gaussian remains the default for its smoothness and
speed; the GF source is selected with `--src ctr`.

**Coherence and spectral weighting** are handled externally. For a Gaussian electron
bunch of rms length $\sigma_t = 2.0$ ps, the bunch form factor is

$$F(f) = \exp\!\left[-\frac{(2\pi f \sigma_t)^2}{2}\right]$$

At 100 GHz, $F = 0.454$; at 200 GHz, $F = 0.042$. The form factor weights each
single-frequency propagation result at assembly time rather than being baked into the
SRW source field.

### 2.2 Frequency Sweep

Because SRW is monochromatic, the broadband CTR pulse is computed as an ordered sweep
of $N = 161$ monochromatic propagations covering 10–400 GHz ($\approx 2.4$ GHz spacing),
chosen to oversample the sapphire window's 24.0 GHz fringe spacing by roughly $10\times$.

For each frequency $f$:

1. `make_source(f, cfg)` builds and computes the Gaussian source wavefront via
   `srwl.CalcElecFieldGaussian`.
2. `build_beamline(cfg)` assembles the `SRWLOptC` container (identical across
   frequencies except for the window's complex scalar response, which is applied
   post-propagation).
3. `srwl.PropagElecField(wfr, optBL)` propagates the wavefront to the detector plane.
4. The 2D intensity $|E(x,y)|^2$ and the on-axis complex field
   $\mathrm{Re}(E) + i\,\mathrm{Im}(E)$ are extracted and immediately resampled onto a
   fixed $401\times401$ detector grid — a critical memory-management step, since native
   meshes can reach tens of millions of points at the lowest frequencies.
5. The window's per-frequency complex transmission $t_{\rm win}(f)$ (see §2.4) and mirror
   reflectivities are applied as scalar multipliers.

Total wall time for 161 frequencies: $\approx 2.5$ minutes; peak memory $\approx 2$ GB.

### 2.3 Beamline

The optical layout (drawn from `thz_config.py`, all distances best-guess):

```
 foil   ──50cm──  collimator  ──20cm──  sapphire window  ──30cm──  OAP1  ──30cm──  OAP2  ──30cm──  EO crystal
(z=0)             ⌀4.6 cm               2.03mm, ⌀3.5cm            f=1.0m           f=0.3m         (z=1.60m)
```

**Collimator (`SRWLOptA`, circular, $D = 4.6$ cm):** The primary aperture that sets the
geometric low-frequency cutoff. At 10 GHz the source $\sigma_r = 112$ cm; at 100 GHz it
is 11.2 cm. In both cases $\sigma_r \gg D/2 = 2.3$ cm, so the aperture clips the far
wings of the beam and suppresses the low-frequency content. The Fresnel number at the
collimator, $N = D^2/(4\lambda L)$, ranges from 0.035 at 10 GHz to 1.4 at 400 GHz,
confirming the propagation is in the Fraunhofer (far-field) regime for most of the band.

**Sapphire window (`SRWLOptA` + scalar $t_{\rm win}(f)$):** The 2.03 mm flat window is
spatially a no-op (uniform amplitude and phase across the beam), so it is modelled as its
clear aperture ($\varnothing 3.5$ cm) in the spatial beamline, plus an analytic
per-frequency complex transmission applied as a scalar after propagation (see §2.4).

**OAP1 — collimating mirror (`SRWLOptL`, $f = 1.0$ m):** Modelled as a thin lens
equivalent of an off-axis paraboloid. Its focal length is set equal to the foil-to-OAP1
distance ($50+20+30 = 100$ cm) so that the diverging CTR beam is collimated between the
two mirrors.

**OAP2 — focusing mirror (`SRWLOptL`, $f = 0.3$ m):** Focuses the collimated beam onto
the EO crystal. Gold mirrors have reflectivity $\approx 1$ in the THz band; both are
applied as scalar factors in `channel_response(f)`.

**EO sensor (detector plane, $z = 1.60$ m):** The TFLN thin-film lithium niobate sensor
samples the local electric field; the simulation records the on-axis complex field for
waveform reconstruction and the 2D intensity map for spatial characterisation.

### 2.4 Sapphire Window — Fabry-Pérot Etalon

The sapphire window is the dominant optical artefact in the measurement. At THz
frequencies, crystalline sapphire has ordinary-ray refractive index $n \approx 3.07$ and
low bulk absorption (modelled as $\alpha \propto f^2$). The flat parallel surfaces form a
Fabry-Pérot etalon with complex amplitude transmission

$$t_{\rm win}(f) = \frac{(1 - R)\,e^{i\beta}}{1 - R\,e^{2i\beta}}$$

where

$$\beta = \frac{2\pi n d f}{c}\quad\text{(round-trip phase)},\qquad
  R = \left(\frac{n-1}{n+1}\right)^2 = 0.259\quad\text{(reflectance per face)}.$$

Key etalon parameters:

| Quantity | Value |
|---|---|
| Surface reflectance $R = \left(\tfrac{n-1}{n+1}\right)^2$ | 0.259 (25.9%) per face |
| Free spectral range $\mathrm{FSR} = c/2nd$ | 24.03 GHz |
| Round-trip echo delay $2nd/c$ | 41.62 ps |
| Finesse $\pi\sqrt{R}/(1-R)$ | 2.16 |
| Transmittance range | 0.334 – 1.000 (mean 0.570) |
| Fringe visibility | 0.485 |

These produce the regular $\approx 24$ GHz notches visible in the detected spectrum and
the post-pulse echoes in $E(t)$ at 41.3 ps before the main pulse (matching the 41.6 ps
prediction). The echo/peak amplitude ratio measured from $E(t)$ is 0.292, consistent with
the single-face reflectance $R = 0.259$.

Three window models are available via `--window-mode`:
- **`slab`** (default): full Fabry-Pérot, as described above.
- **`wedge`**: etalon suppressed (reflections walk off the aperture). Same Fresnel
  insertion loss ($T \approx 0.55$), no spectral ripple or time-domain echoes.
- **`ar`**: anti-reflection coated, residual $R = 0.01$. Near-unity transmission,
  minimal etalon.

### 2.5 Output Assembly

**Detected power spectrum** (`spectrum.txt` / `spectrum.png`): for each frequency, the
simulated intensity is integrated over the detector active area (5 mm radius, best
guess) on the common $401\times401$ grid, weighted by $|F(f)|^2$ and $Q^2$. The result is
the transfer function of the full source-to-detector channel: geometric cutoff at the
collimator on the low side, bunch form factor rolloff on the high side, sapphire etalon
modulation throughout.

**Broadband spatial intensity map** (`spatial_map.npz/png`): the form-factor-weighted sum
$\sum_f |F(f)|^2\,|E(x,y,f)|^2$ over all frequencies, resampled to the common detector
grid. Shows the THz spot at the EO crystal. Peak broadband intensity
$1.92\times10^{17}$ (arb); FWHM $\approx 17.5$ mm (horizontal).

**Time-domain waveform $E(t)$** (`waveform.txt/png`): the on-axis complex field per
frequency, multiplied by $F(f)$ and charge $Q$, assembled into a waveform by inverse FFT,

$$E(t) = \mathrm{IFFT}_f\!\left[\, E_{\rm det}(f,\, x{=}0,\, y{=}0)\cdot F(f)\cdot Q \,\right].$$

This is the quantity the EO sampling chain measures. The simulation gives it in
arbitrary units; converting to physical field strength requires the $K_{\rm BPD}$
calibration from `proposal.tex`. Peak $|E(t)| = 5.8\times10^9$ arb; waveform span
$\approx 410$ ps.

**Etalon analysis** (`etalon_transmission.png`, `etalon_spectrum.png`): produced
automatically by `thz_etalon.py` each run. The spectrum overlay compares the current
etalon-modulated spectrum to the etalon-removed (wedged window) equivalent, obtained by
rescaling without re-running SRW.

### 2.6 Known Approximations and Limitations

1. **Default Gaussian surrogate vs Ginzburg-Frank**: the default (`--src gaussian`)
   source has a Gaussian angular profile (on-axis maximum, linear polarisation). True CTR
   has an on-axis null and radial polarisation. After focusing by OAP2, the Gaussian
   produces a bright central spot; the Ginzburg-Frank pattern produces a ring with a
   central minimum — which is what the EO crystal would actually see for a perfectly
   aligned beam. The Ginzburg-Frank source (`--src ctr`, §3) reproduces that ring; it
   carries its own approximations (far-field-of-source; low-frequency mesh capping).

2. **Best-guess geometry**: all drift distances, mirror focal lengths, and detector
   active area are assumed, not measured. They are clearly flagged in `thz_config.py`
   and can be updated when the real beamline parameters are available.

3. **Sapphire treated as isotropic**: only the ordinary-ray index ($n = 3.07$) is used.
   The extraordinary index ($n \approx 3.4$) and birefringence are not modelled.

4. **Absolute field scale**: the simulation is in SRW arbitrary units. The physical
   E-field at the crystal requires normalisation to the coherent CTR power, via the
   Ginzburg-Frank spectral energy formula and the $K_{\rm BPD}$ electro-optic calibration.

5. **Mesh artefacts at low frequency**: at 10–30 GHz the collimated beam between OAP1
   and OAP2 can produce SRW meshes of 10–30 million points. These are physically
   required (the beam really does fill this space) but cause a faint square halo in the
   broadband spatial map at low intensity levels. The central spot and spectrum are
   unaffected.

---

## 3. The Ginzburg-Frank CTR Source (`--src ctr`)

The Gaussian surrogate gets the aperture/window/propagation physics right but not the
*shape* of transition radiation. The Ginzburg-Frank (GF) source is the physically
faithful CTR model: it has the correct on-axis null, radial polarisation, and the
$\sim 1/\gamma$ opening angle. It is implemented in `thz_source.src_ctr_wavefront` and
plugs into the identical beamline and assembly path. This section documents how it was
built and the four non-obvious problems that had to be solved to make it correct and fast.

### 3.1 The Ginzburg-Frank pattern

For a relativistic electron ($\gamma = 117.4$, $\beta \approx 0.999964$) crossing a
perfectly conducting foil, the backward-TR far-field angular amplitude is

$$E_{\rm GF}(\theta) = A\,\frac{\beta\sin\theta}{1 - \beta^2\cos^2\theta},$$

with an exact **on-axis null** ($E = 0$ at $\theta = 0$), a peak at
$\theta \approx 1/\gamma \approx 8.5$ mrad, and a heavy $1/\theta$ tail. The radiation is
**radially polarised**: in Cartesian components,

$$E_x = E_{\rm GF}(\theta)\cos\phi,\qquad E_y = E_{\rm GF}(\theta)\sin\phi,\qquad
  \phi = \operatorname{atan2}(\theta_y, \theta_x).$$

In the ideal-conductor limit the TR spectrum is flat (frequency-independent shape), so
the spectral content of the *detected* signal comes from the bunch form factor $F(f)$ and
the beamline, not from the source — the same $F(f)$ weighting as the Gaussian path.

### 3.2 Building the field: coordinate-space far-field projection

The natural place to write the GF pattern is angular space, then FFT to coordinates with
`SetRepresElecField`. **This was tried and abandoned.** The FFT conjugate relation fixes
the coordinate pixel size at $dx = \lambda/(2\theta_{\max})$; for the GF angular extent
($\pm 5/\gamma$) this gives $dx \approx 29$ mm at 120 GHz, while the physical ring sits at
$z_0/\gamma \approx 4.3$ mm — i.e. **sub-pixel at every frequency in the band**
($\text{ring}/dx \approx 0.01\text{–}0.5$). The angular route cannot resolve the
structure it is meant to represent.

Instead the field is built **directly in coordinate space** at the first-element plane
$z_0 = 0.5$ m, using the far-field-of-source projection $x = z_0\theta$:

$$E_x(x,y) = \frac{A}{z_0}\,E_{\rm GF}(r/z_0)\cos\phi,\qquad
  E_y(x,y) = \frac{A}{z_0}\,E_{\rm GF}(r/z_0)\sin\phi,\qquad r = \sqrt{x^2 + y^2}.$$

The diverging spherical curvature is carried as the wavefront radius $R_x = R_y = z_0$
(`wfr.Rx`, `wfr.Ry`; SRW treats the quadratic phase analytically), so the data array
holds the real radial amplitude only; `presCA = 0` (coordinate representation), ready for
the standard propagation path. The mesh is sized to the pattern ($\pm z_0\theta_{\max}$,
$\sim\pm 21$ mm) so the ring is resolved with $\sim 25$ points. This is the far field of a
point-like source and is most valid when $z_0 \gg$ the TR formation length $\gamma\lambda$
(true at high $f$; the same caveat as the Gaussian surrogate).

### 3.3 Physical normalisation

The amplitude $A$ is set so the angular-integrated $|E_{\rm GF}|^2$ equals the
single-electron TR spectral energy (ultra-relativistic limit):

$$\frac{dW}{d\omega} \approx \alpha\hbar\,(2\ln\gamma - 1)
  \qquad\Longrightarrow\qquad
  A = \sqrt{\frac{dW/d\omega}{\sum |E_{\rm GF}|^2\,d\Omega}},$$

where $\alpha$ is the fine-structure constant and $\hbar$ the reduced Planck constant
(`_gf_amplitude_norm`). The value is recorded in `run_meta.txt`
($\texttt{gf\_amplitude\_A} \approx 9.1\times10^{-19}$). The coherent bunch then scales as
$(Q/e)\,F(f)$, applied at assembly. Units remain relative (`unitElFld = 0`); absolute
calibration still needs $K_{\rm BPD}$ from the EO chain.

### 3.4 Problem 1 — frequency-adaptive source mesh

The GF coordinate field has a *frequency-independent* spatial structure (the 4.3 mm
ring), so a fixed fine mesh would resolve it at every frequency — but at low $f$ the ring
is **sub-wavelength** ($\lambda = 30$ mm at 10 GHz), and sampling a 4 mm feature at
0.7 mm pixels is $\sim 40\times$ sub-wavelength oversampling that injects huge
(evanescent) spatial-frequency content and stalls the propagator. The source mesh size is
therefore frequency-adaptive (`_gf_n_ang`):

$$dx = \max\!\left(\frac{\text{ring}}{6},\ \text{gf\_src\_lambda\_frac}\cdot\lambda\right),
  \qquad
  n_{\rm ang} = \operatorname{clip}\!\left(\frac{2\,x_{\rm half}}{dx},\
  \text{gf\_min\_ang},\ \text{gf\_n\_ang}\right) \in [32, 256],$$

so the source resolves the ring where it is super-wavelength (high $f$) and degrades to a
smooth coarse blob where it is sub-wavelength (low $f$) — which is physically what the
radiation looks like there anyway. The source mesh also extends past the collimator
($\text{gf\_src\_aper\_margin}\cdot R_{\rm collimator}$) so the *collimator*, not the mesh
boundary, truncates the heavy $1/\theta$ tail.

### 3.5 Problem 2 — capped propagation (auto-resize runaway)

The GF coordinate source is small ($\sim$ cm). SRW's auto-resize, which works well for
the metre-scale Gaussian source, **runs away** for the GF source at low frequency: trying
to contain the hugely-diffracting beam at the source's sampling it grows the mesh without
bound (observed: $\sim 22$ GB, OOM-killed). The fix is a **capped** propagation mode
(`build_beamline(cfg, capped=True)`, used automatically for `--src ctr`):

- auto-resize **off** on every element;
- each drift expands the transverse range by a **fixed $\times 2$** at constant resolution.

The mesh then stays bounded at $n_{\rm src}\cdot 2^{\,\#\text{drifts}} \lesssim 1600$
points/axis and propagates in 1–2 s at every frequency. Strongly-diffracting low-$f$ beams
are mildly clipped, but those frequencies are deep in the collimator cutoff (negligible
detected signal), so the impact is small. The Gaussian path keeps auto-resize (its large
smooth source needs it; fixed factors would explode its $\pm 3.9$ m mesh).

### 3.6 Problem 3 — radial-projection waveform extraction

After OAP2 the GF beam focuses to a **ring** (central null), so the on-axis field used
for the Gaussian waveform is zero. The waveform must be sampled at the ring. The first
attempt averaged the $E_x$ component over the ring annulus — and produced a quasi-CW
smear, not a transient. The cause is the radial polarisation: $E_x = E_r\cos\phi$, so the
azimuthal average $\langle\cos\phi\rangle = 0$ **cancels the signal**, leaving only
phase-scrambled numerical residue ($\approx 10^4\times$ too small). The fix projects onto
the radial unit vector before averaging (`ring_peak_complex_field`):

$$E_r = E_x\cos\phi + E_y\sin\phi \qquad\text{(azimuthally coherent for a radial field).}$$

This recovers the true field (verified directly: $1.3\times10^4\times$ larger than
$\langle E_x\rangle$, with $\langle|E_r|\rangle \approx |\langle E_r\rangle|$, i.e.
coherent) and a clean spectral phase. The reconstructed $E(t)$ becomes a **localised
bipolar transient** ($\text{peak/RMS} \approx 4.6$, versus $\approx 1.4$ for CW) on a
quiet baseline, with sapphire etalon pre-echoes spaced $\sim 42$ ps — the Gaussian-style
transient, now with the correct CTR ring geometry. The radial average over an annulus is
computed in $O(N)$ with `np.bincount` (an earlier $O(n_r N)$ mask loop was a second
bottleneck at large low-$f$ meshes). An optional `r_fixed` argument samples at a single
fixed crystal radius (consistent phase reference across all frequencies); the
per-frequency ring peak is the default and was marginally sharper.

### 3.7 Verification and results

`thz_verify.gf_pattern_check` confirms, before any sweep: (a) on-axis null
$I(0)/I_{\rm peak} < 1\%$; (b) ring radius at the source plane $= z_0/\gamma$ to within
$\sim 2\%$; (c) `presCA = 0` after build; (d) the ring **expands** under a test drift
(validating the $R_x = R_y = z_0$ diverging curvature). A full 161-frequency run completes
in $\approx 4.5$ min with $\approx 0.7$ GB peak RAM (the capping holds memory flat).
Outputs (`results_gf/`): a clean **ring** with central null in the broadband spatial map,
the same bandpass+etalon spectrum as the Gaussian (both built from intensity), and the
localised $E(t)$ transient described above. Side-by-side, `results/` (Gaussian, filled
spot) vs `results_gf/` (GF, ring) is the clearest illustration of what the analytic source
adds.

### 3.8 GF-specific limitations

- **Far-field of source.** The $x = z_0\theta$ projection assumes $z_0 \gg \gamma\lambda$;
  at low $f$ the formation length $\gamma\lambda$ is comparable to $z_0$, so low-$f$ GF is
  approximate (shared with the Gaussian surrogate).
- **Low-$f$ capping clips the beam.** Bounded-range propagation mildly truncates the
  strongly-diffracting low-$f$ beam; acceptable because those frequencies are in the
  cutoff, but it means the low-$f$ tail of the GF spectrum is less trustworthy.
- **Ideal conductor, flat spectrum.** Finite foil conductivity/size and the true TR
  formation physics are not modelled; the source shape is the idealised GF pattern.

### 3.9 Tilted screen — asymmetric CTR (Level-1 heuristic)

Real CTR stations tilt the radiator foil (commonly 45°) so the backward radiation exits
toward an optical port. A tilt breaks the azimuthal symmetry of the donut: the emission
re-centres on the specular direction, the pattern foreshortens in the tilt plane, and the
two lobes become unequal — the donut turns into a **crescent**. A `--screen-tilt` option
adds this at the *Level-1 (geometric heuristic)* fidelity; it is applied only to the GF
source and $\text{screen\_tilt\_deg} = 0$ is an exact no-op.

The heuristic modifies the source amplitude in two ways (tilt in the $x$-plane, angle
$\psi$). First, the angular pattern is foreshortened in the tilt plane (an elongated
footprint gives a narrower far-field lobe):

$$\theta_{\rm eff} = \sqrt{(\theta_x/\cos\psi)^2 + \theta_y^2}.$$

Second, the two $x$-lobes are biased (bright toward specular, dim away), bounded by a
$\tanh$ and clamped non-negative:

$$\text{lobe}(\theta_x) = \max\!\left(0,\ 1 + \text{gf\_tilt\_lobe\_k}\cdot\tan\psi\cdot
  \tanh(\theta_x\gamma)\right),$$

giving the source field

$$E(x,y) = \frac{A}{z_0}\,E_{\rm GF}(\theta_{\rm eff})\,\text{lobe}(\theta_x)\,
  (\cos\phi,\ \sin\phi).$$

`gf_tilt_lobe_k` (default 1.0) tunes the lobe imbalance: with $k = 1$ a 45° tilt nearly
extinguishes one lobe (a strong crescent), while 30° leaves both lobes visible but clearly
unequal. The radial polarisation is retained (its rigorous p/s modification is a Level-3
upgrade). Verified behaviour: at the source plane, $\text{tilt}=0$ gives centroid $x = 0$
and left/right lobe ratio $= 1.00$ (exact symmetry); $\text{tilt}=45°$ shifts the centroid
$+10.8$ mm and drives the lobe ratio to $\approx 0.03$. Propagated through the relay the
detector shows the expected asymmetric crescent with the central null preserved
(`results_gf_tilt/`, `--screen-tilt 30`). Note the focusing relay inverts the image, so
the bright lobe appears on the opposite side at the detector from the source.

Because the ring is now one-sided, the waveform extraction (`ring_peak_complex_field`)
switches to an **intensity-weighted** complex average over the ring annulus when tilted —
so the bright lobe dominates the sampled on-crystal field. For the symmetric case the
radial field is azimuthally uniform and this equals the plain mean (no change to the
validated symmetric result).

**Level-1 caveats.** This is a geometric heuristic, not the rigorous oblique-incidence
transition-radiation amplitude: the foreshortening and lobe bias are parameterised
($\cos\psi$, $\tan\psi\cdot\tanh$, `gf_tilt_lobe_k`) rather than derived, and the
polarisation is left radial. For quantitative comparison with data the source amplitude
should be upgraded to the full oblique-incidence Ginzburg-Frank / Ter-Mikaelian
expression with separate p- and s-components (Level 3).

### 3.10 Tilted screen — exact oblique-TR vector field (Level-3)

Level-1 parameterises the crescent; **Level-3 derives it** from the point-charge
image-charge construction, and — crucially — produces the *true vector polarisation*
rather than assuming it stays radial. It is selected with `--tilt-model level3` (the
default tilt model is `level1`); at $\text{screen\_tilt\_deg}=0$ it reduces exactly to the
symmetric GF field of §3.1. The full derivation is in `LEVEL3_DERIVATION.md`; this section
documents what the code computes and how it is verified.

**Image-charge amplitude.** Backward transition radiation from a charge crossing a
conducting plane is the field of the charge plus its mirror image. For a foil tilted by
$\psi$ in the $x$-plane, the electron velocity is $\boldsymbol\beta = \beta\hat z$ and the
image velocity is the *specular reflection* of $\boldsymbol\beta$ about the foil normal
$\hat n = (\sin\psi, 0, \cos\psi)$:

$$\boldsymbol\beta_i = \beta\,(\hat z - 2\cos\psi\,\hat n).$$

The far-field vector amplitude in observation direction $\hat n_{\rm obs}$ is the
difference of the two relativistic radiation kernels

$$\mathbf{A}(\hat n_{\rm obs}) = \mathbf{T}(\hat n_{\rm obs}, \boldsymbol\beta)
  - \mathbf{T}(\hat n_{\rm obs}, \boldsymbol\beta_i),\qquad
  \mathbf{T}(\hat n, \mathbf b) = \frac{\hat n\,(\hat n\cdot\mathbf b) - \mathbf b}
  {1 - \hat n\cdot\mathbf b}.$$

`gf_oblique_field(X, Y, cfg)` evaluates this over the source grid: it builds the
observation directions from the specular frame ($\hat s$ along the image velocity,
$\hat e_p = \hat s\times\hat y$ the in-plane axis, $\hat e_s = \hat y$ the out-of-plane
axis), forms $\mathbf A$, and projects onto $(\hat e_p, \hat e_s)$ to get the two Cartesian
field components $(E_x, E_y)$. **The polarisation is now an output, not an assumption** —
it falls directly out of $\mathbf A$. Setting `gf_include_electron_term=False` drops the
$\mathbf T(\hat n,\boldsymbol\beta)$ (electron) term, leaving only the image term; the
difference between the two isolates the p/s split, which is the non-resonant electron-term
contribution.

**Polarisation structure.** At $\psi=0$ the amplitude is purely radial and the field
matches GF. At $\psi>0$ the broken collinearity of $\boldsymbol\beta$ and
$\boldsymbol\beta_i$ does two things, both visible in the diagnostic maps
(`detector_pol_*.png`, saved automatically at the spectral-peak frequency for a tilted
Level-3 run):

- the **p-component** ($E_x$, in the tilt plane) acquires a small on-axis fill (the radial
  null is no longer exact in p), and its two lobes become unequal;
- the **s-component** ($E_y$, out of plane) stays null on axis and remains odd in $y$.

The polarisation-angle map $\arctan(|E_y|/|E_x|)$ shows the canonical radial fingerprint
($\approx 0°$ on the $x$-axis, $\approx 90°$ on the $y$-axis). Both the intensity asymmetry
of the in-plane lobes and the non-radial *polarisation twist* scale as $\sim 1/\gamma$ — the
twist has the closed form $|E_\phi/E_r|_{\max} = \sin2\psi/[(1+\beta\cos2\psi)\gamma] \to
\tan\psi/\gamma$, derived and numerically verified (to ~1% across $\psi$ and $\gamma$) in
`LEVEL3_TWIST_DERIVATION.md`. So for the AWA beam ($\gamma = 117$) the asymmetry is
only a few percent — confirming that for highly relativistic beams the *point-charge*
oblique asymmetry is genuinely small, and that the large crescents seen in practice are
dominated by finite-source/foreshortening geometry (which Level-1's $k$ absorbs
empirically). Level-3 is the polarisation-correct model; Level-1 is the
stronger-asymmetry heuristic.

**Polarisation-aware extraction.** An EO crystal is sensitive to one linear axis, so for a
Level-3 tilted run the on-crystal field is projected onto the crystal axis $\chi=$
`crystal_pol_axis_deg` (`crystal_axis_complex_field`): $E_c = E_x\cos\chi + E_y\sin\chi$,
intensity-weighted ($|E_c|^2$) over the ring annulus. This replaces the radial projection
(§3.6) used for the symmetric/Level-1 paths, which would mix the now-distinct p and s
components. $\chi=0$ (default) reads the in-plane (p) field.

**Verification.** `thz_verify.oblique_field_check` confirms, at the source plane, with no
propagation: (a) $\psi=0$ reproduces the symmetric GF field (relative difference
$4.7\times10^{-5}$); (b) $\psi>0$ fills the on-axis null in p ($|E_x(0)|/E_{\rm peak}
\approx 5\times10^{-3}$) but not in s ($|E_y(0)| = 0$); (c) $E_y$ is odd in $y$ (oddness
residual $0$); (d) the in-plane lobe ratio is asymmetric ($L/R \approx 1.03$ at $30°$,
i.e. the few-percent $\sim 1/\gamma$ effect). A coarse end-to-end run
(`--src ctr --tilt-model level3 --screen-tilt 30`) propagates cleanly and produces the
ring spatial map plus the three polarisation maps.

**Level-3 limitations.** The construction is the *point-charge* (ideal-conductor, infinite
foil) oblique TR field — it does not include finite foil size, conductivity, or the
formation-length physics, so the dominant real-world crescent asymmetry (finite source +
foreshortening) is still not captured at Level-3; that asymmetry lives in Level-1's tunable
$k$. Mirrors are modelled as thin lenses (`SRWLOptL`), which do **not** rotate polarisation,
so the p/s split propagates unrotated to the detector; a real off-axis paraboloid imposes a
geometric polarisation rotation that would require `SRWLOptMirPar` to capture. Level-3 is
therefore the right tool for the *source* polarisation structure, not yet for end-to-end
polarisation transport through the relay.

**Space-time reconstruction at the detector (`run_thz_spacetime.py`).** The standard driver
keeps only the per-frequency intensity plus a single ring phasor. To see the *full*
spatio-temporal pulse the EO crystal samples, `run_thz_spacetime.py` keeps the complex 2D
field at the detector for every frequency, projects it onto the crystal axis $E_c=E_x\cos\chi
+E_y\sin\chi$, weights by the bunch form factor $F(f)$ and charge, and inverse-FFTs **per
pixel** to build $E_c(x,y,t)$. At the *foil* the TR spectrum is flat so the field factorises
into (transverse pattern) × (Gaussian bunch pulse) — the source-plane picture. After the
relay it no longer factorises: the ring radius and focusing are frequency-dependent, so the
Gaussian bunch pulse is reshaped into a localised bipolar transient whose constant-$x$ bands
become **chromatically curved arcs** in the $x$–$t$ plane. The sapphire etalon adds a train
of decaying echoes spaced by the $41.6$ ps round-trip (visible at $-42$ and $-84$ ps in the
$x$–$t$ map; their amplitude ratio tracks the $R=0.259$ per-face reflectance). The
active-area $E(t)$ reproduces the 1-D `waveform.py` transient, cross-validating the per-pixel
reconstruction. Outputs (a data cube, $x$–$t$ map, snapshots, and an animation) land in a
separate folder (`results_l3_spacetime/`); the echo train appears before the main peak as a
reconstruction phase-convention artefact (the $41.6$ ps *spacing* is physical, the sign is
not).

---

## 4. Software and Reproducibility

| Component | Version / detail |
|---|---|
| SRW (srwpy) | 4.2.1, pip wheel (cp313, manylinux) |
| SRW core (srwlpy) | v4.10 |
| Python | 3.13.9, venv at `Suli2026/.venv` |
| numpy / scipy / matplotlib | 2.5.0 / 1.18.0 / 3.11.0 |
| Host | Fedora 41 (ARGO gateway, ANL internal network) |

To reproduce a full run:

```bash
source Suli2026/.venv/bin/activate
cd Suli2026/srw_thz
python run_thz.py                         # Gaussian source -> results/ (~2.5 min, ~2 GB)
python run_thz.py --src ctr --out results_gf   # Ginzburg-Frank CTR -> ring (~4.5 min, ~0.7 GB)
python run_thz.py --src ctr --screen-tilt 30 --out results_gf_tilt   # tilted screen (Level-1) -> crescent
python run_thz.py --src ctr --tilt-model level3 --screen-tilt 30 --out results_l3   # Level-3 oblique vector field + pol maps
python thz_etalon.py                      # etalon metrics + plots (no SRW run)
python thz_layout.py                      # beampath schematic (no SRW run)
python thz_verify.py                      # aperture-diffraction + GF pattern checks
```

Key CLI flags: `--src {gaussian,ctr}`, `--nfreq`, `--fmin/--fmax` (GHz), `--sigma-t`
(ps), `--charge` (nC), `--div-match`, `--window-mode {slab,wedge,ar}`, `--no-mirror2`
(single mirror), `--sampfact`, `--gf-n-ang` (GF source mesh), `--screen-tilt` (deg,
tilted-screen asymmetric CTR), `--tilt-model {level1,level3}` (heuristic vs exact oblique
vector field), `--crystal-pol-axis` (deg, EO crystal sensitive axis for Level-3),
`--out` (output directory).
