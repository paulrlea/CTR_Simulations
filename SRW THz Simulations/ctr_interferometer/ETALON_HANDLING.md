# How the etalon (window Fabry–Pérot) is handled in the SRW model

Short version: **SRW does not model the window etalon.** SRW is used only for the *spatial*
wavefront propagation (diffraction, apertures, focusing). The window's multiple-internal-
reflection Fabry–Pérot response is supplied **analytically, as a per-frequency complex scalar**
applied to the field outside SRW. This is exact for a flat, uniform, normal-incidence slab
because such an element is transversely uniform, so its complex transmission commutes with the
linear (Fourier-optics) propagation and can be factored out.

## Why it is done outside SRW

1. **A flat parallel slab is a spatial no-op.** Over its clear aperture a uniform window applies
   a constant amplitude and a constant (piston) phase to every ray — it does not diffract, focus,
   or steer the beam. Its *only* spatial effect is its clear aperture, which is included in the
   SRW beamline as an ordinary circular aperture (`SRWLOptA`).

2. **SRW's thin elements are single-pass.** SRW's optical elements (thin lens, generic
   transmission `SRWLOptT`, drifts, apertures) each act once on the wavefront. A Fabry–Pérot
   etalon is a *resonant sum of multiple internal reflections* inside the slab — SRW does not
   automatically generate that infinite reflection series. So even if the slab were inserted as
   an `SRWLOptT`, it would give only a single-pass global phase, not the etalon.

3. **A global per-frequency factor commutes with propagation.** Because the etalon transmission
   `t_win(f)` is one complex number per frequency (spatially uniform), and SRW propagation is a
   linear operator `L`, we have `L[t_win(f)·E] = t_win(f)·L[E]`. Applying `t_win(f)` at the
   detector (extraction) is therefore *identical* to applying it at the window — and it is cheap
   (no re-propagation, no spatial mesh cost).

## The analytic model

**In plain terms:** the window is a slab of glass that is slightly reflective on both faces
(~10% each). Most of the THz goes straight through, but a little bounces back and forth between
the two faces before leaking out, and those internal bounces interfere — adding up at some
frequencies (window ~100% transparent) and partly cancelling at others (dips). That periodic
ripple with frequency is the etalon; the formulas below are the exact bookkeeping of the bounces.
- `n = n' + iκ`: one number for what the glass does — `n'` slows/delays the wave, `κ` absorbs it
  (from the absorption coefficient `α`, allowed to grow with frequency).
- `β`: the phase the wave gains crossing the window once (how many wavelengths thick it looks,
  ×2π); its imaginary part is the one-pass absorption.
- `R`: fraction of power reflected at one air–glass face (~10% for fused silica) — this is what
  makes the slab act like a weak two-mirror cavity.
- `t_win`: total transmission = straight-through pass (top) divided by the sum of all internal
  round-trips (bottom, where `e^{2iβ}` is one round-trip's phase). `|t_win|²` is the power that
  gets through, swinging between ~100% (bounces in phase) and a minimum (out of phase) — the
  etalon ripple, whose time-domain partner is the post-pulse echoes / interferogram recurrences.

The window index is complex — the real part slows the wave, the imaginary part absorbs it (from
a power absorption coefficient $\alpha(f)$):

$$
n(f) = n'(f) + i\,\kappa(f), \qquad
\alpha = \frac{4\pi\,\kappa\,f}{c}, \qquad
\alpha(f) = \alpha_{1\text{THz}}\left(\frac{f}{1\,\text{THz}}\right)^{p}.
$$

Built by `interf_beamline.window_n_complex(f, cfg)` (fused-silica interferometer window) and
`thz_beamline.sapphire_n_complex(f, cfg)` (the sapphire equivalent, previous model).

The single-layer Fabry–Pérot amplitude transmission at normal incidence is

$$
\beta(f) = \frac{2\pi\,n(f)\,d\,f}{c}, \qquad
R = |r|^2 = \left(\frac{1-n}{1+n}\right)^{2},
$$

$$
t_{\text{win}}(f) = \frac{(1-R)\,e^{i\beta}}{1 - R\,e^{2i\beta}}.
$$

Here $\beta$ is the (complex) one-way phase thickness — its imaginary part carries the
single-pass absorption — and $R$ is the single air–window intensity reflectance. The numerator
$(1-R)\,e^{i\beta}$ is the single traversal (using $t_{12}\,t_{23} = 1 - r^2 = 1 - R$ exactly);
the denominator $1 - R\,e^{2i\beta}$ is the geometric series over the internal round trips — that
is the etalon. The transmitted power is the familiar Airy function

$$
|t_{\text{win}}(f)|^2 = \frac{1}{\,1 + \dfrac{4R}{(1-R)^2}\,\sin^2\beta\,}.
$$

Implemented in `interf_beamline.window_transmission_fs(f, cfg)` (interferometer, fused silica)
and `thz_beamline.window_transmission(f, cfg)` (previous model, sapphire).

## Where it enters the pipeline

`t_win(f)` is bundled with the other flat per-frequency amplitude factors (beamsplitter split,
mirror reflectivities) into a single **channel response**:

```
channel_response_interf(f, cfg) = t_win(f) · fold_R · (r_BS·t_BS) · retro_R · oap_R
```

and multiplied onto the propagated field at extraction — the field map gets `|ch|²`, the
complex spectral field gets `ch` — in `run_interf.py`, `run_interf_spacetime.py`, and the
`--detector eo` waveform. Since it is applied per frequency across the whole sweep, it correctly
shapes every downstream product: the spectrum, the E(t) waveform, the interferogram, and the 2D
delay–time map.

## Mitigation modes (`window_mode`)

The same function switches the physics via `cfg.window_mode`:

- **`slab`** (default): the full Airy expression above — the real flat window, with the etalon.
- **`wedge`**: `t_win = (1 − R) e^{iβ}` only. A wedged window walks the multiple reflections off
  the aperture so they never recombine coherently → same Fresnel insertion loss, **no etalon**
  (no ripple, no echoes). (A true wedge is a spatially-varying thickness; because the point is
  precisely that the internal reflections *don't* interfere, dropping the denominator captures
  it without needing a spatial map.)
- **`ar`**: anti-reflection coated — surface reflectance replaced by a small residual
  `ar_residual_R`, which nearly removes the etalon and the insertion loss.

## What it produces (fused-silica interferometer window, n′≈1.95, d≈3 mm)

- **Free spectral range** `FSR = c/(2 n' d) ≈ 25.6 GHz` → periodic ripple in the detected
  spectrum.
- **Round-trip echo delay** `2 n' d / c ≈ 39 ps` → post-pulse echoes in the reconstructed E(t),
  and, because the interferogram is the field autocorrelation, **coherence recurrences at
  τ ≈ ±39 ps** flanking the central burst (visible in `interferogram.png` and the 2D map).
- **Surface reflectance** `R ≈ 10 %`, so **transmittance swing ≈ 66–100 %**, low finesse
  (`π√R/(1−R) ≈ 1.1`), echo amplitude ratio ≈ `R ≈ 0.10` per successive echo.

(The previous sapphire window, n′≈3.07, d≈2.03 mm, gave a stronger etalon: R≈26 %, FSR≈24 GHz,
echo delay≈41.6 ps, swing 33–100 %, finesse≈2.2. `thz_etalon.py` quantifies that case and writes
`etalon_transmission.png` / `etalon_spectrum.png`.)

## Assumptions and limitations

- **Normal incidence, single traversal.** The window is crossed once (foil → window → BS); the
  factor is applied once. Oblique-incidence etalon shifts and the s/p split are not modeled.
- **Spatially uniform slab.** Valid because the beam is near-normal and the window is flat; a
  genuinely non-uniform/tilted window would require a spatial optical-path map (`SRWLOptT`),
  which is exactly the regime the `wedge` mode approximates as "etalon removed."
- **Isotropic index.** Sapphire birefringence is ignored (ordinary ray only); fused silica is
  isotropic so this is exact there.
- **The pellicle beamsplitter etalon is neglected.** Its 2 µm membrane gives `FSR = c/2nd` of
  tens of THz — far above the 10–400 GHz band — so it is treated as an ideal thin 30/30 splitter
  (scalar `r`, `t`), no BS etalon.
- **Absorption model** `α(f) ∝ f^p` is a literature/best-guess power law; it damps the etalon
  contrast at higher frequency through `κ(f)`.

## Appendix: step-by-step derivation

Setup: a slab of index $n$, thickness $d$; THz frequency $f$, vacuum wavelength $\lambda=c/f$,
$\omega=2\pi f$. A plane wave inside a medium is $E\propto e^{i(kz-\omega t)}$ with $k=n\omega/c$.

### A.1 Complex index and absorption

Insert $n=n'+i\kappa$ into the wave's spatial factor:

$$e^{ikz}=e^{i n\omega z/c}=\underbrace{e^{\,i n'\omega z/c}}_{\text{phase (oscillation)}}\;\underbrace{e^{-\kappa\omega z/c}}_{\text{amplitude decay}}.$$

So $n'$ sets the phase speed and $\kappa$ the absorption. The intensity is $|E|^2\propto e^{-2\kappa\omega z/c}$; defining the power-absorption coefficient by $I=I_0e^{-\alpha z}$,

$$\alpha=\frac{2\kappa\omega}{c}=\frac{4\pi\kappa f}{c}\quad\Longleftrightarrow\quad \kappa=\frac{\alpha c}{4\pi f}.$$

This is how a measured $\alpha(f)$ becomes $\kappa$; the fit $\alpha(f)=\alpha_{1\text{THz}}(f/1\text{THz})^{p}$ makes absorption rise with frequency ($p\approx2$).

### A.2 One-pass phase $\beta$

Phase accumulated on one crossing = $k\,d$:

$$\beta=k d=\frac{n\omega}{c}\,d=\frac{2\pi n d f}{c}=2\pi\,\frac{nd}{\lambda}$$

— the optical thickness $nd$ in vacuum wavelengths, $\times 2\pi$. Complex $n$ makes $\beta$ complex, with $\mathrm{Im}\,\beta=2\pi\kappa d f/c=\alpha d/2$, so one crossing dims the amplitude by $e^{-\alpha d/2}$ (intensity $e^{-\alpha d}$) — consistent with A.1.

### A.3 Surface reflectance (Fresnel, normal incidence)

Amplitude reflection air ($n_1=1$) → glass ($n_2=n$), and the surface transmissions:

$$r=\frac{1-n}{1+n},\qquad t_{12}=\frac{2}{1+n},\qquad t_{21}=\frac{2n}{1+n}.$$

Stokes relation (used in the numerator below):

$$t_{12}\,t_{21}=\frac{4n}{(1+n)^2}=1-r^2\equiv 1-R,\qquad R\equiv r^2.$$

$R$ is the power reflected per face; $1-R$ is the two-surface single-pass amplitude factor. Fused silica $n=1.95$: $r=-0.322$, $R\approx0.10$. (The internal reflection is $r'=-r$, so each internal bounce contributes $r'^2=r^2=R$.)

### A.4 Summing the internal bounces $\Rightarrow t_{\text{win}}$

The **direct** transmitted beam: in ($t_{12}$), across ($e^{i\beta}$), out ($t_{21}$):

$$t_{12}\,e^{i\beta}\,t_{21}=(1-R)\,e^{i\beta}.$$

Each additional **round trip** inside the slab adds two internal reflections and two crossings:

$$\underbrace{r'}_{\text{face 2}}\underbrace{e^{i\beta}}_{\text{back}}\underbrace{r'}_{\text{face 1}}\underbrace{e^{i\beta}}_{\text{forward}}=r'^2e^{2i\beta}=R\,e^{2i\beta}.$$

The exiting beams form a geometric series with ratio $x=Re^{2i\beta}$ (using $\sum_{m\ge0}x^m=\tfrac{1}{1-x}$):

$$t_{\text{win}}=(1-R)e^{i\beta}\sum_{m=0}^{\infty}\left(Re^{2i\beta}\right)^{m}=\frac{(1-R)\,e^{i\beta}}{1-R\,e^{2i\beta}}.$$

Numerator = single pass; denominator = all internal round trips (the etalon).

### A.5 Transmitted power (Airy function)

Take $|t_{\text{win}}|^2$ (lossless: $R,\beta$ real):

$$|t_{\text{win}}|^2=\frac{(1-R)^2}{|1-Re^{2i\beta}|^2}.$$

Expand the denominator and use $\cos2\beta=1-2\sin^2\beta$:

$$|1-Re^{2i\beta}|^2=1-2R\cos2\beta+R^2=(1-R)^2+4R\sin^2\beta,$$

$$\boxed{\;|t_{\text{win}}|^2=\frac{(1-R)^2}{(1-R)^2+4R\sin^2\beta}=\frac{1}{1+\dfrac{4R}{(1-R)^2}\sin^2\beta}\;}$$

Transmission is $1$ at resonance ($\sin\beta=0\Rightarrow\beta=m\pi$) and dips to $\left(\tfrac{1-R}{1+R}\right)^2$ in between; the bracket $\tfrac{4R}{(1-R)^2}$ (coefficient of finesse) sets the dip depth.

### A.6 From the equations to the observables

Resonances $\beta=m\pi$ give peaks at $f_m=\dfrac{mc}{2n'd}$, evenly spaced by

$$\text{FSR}=\frac{c}{2n'd}\approx 25.6\ \text{GHz}\ \ (\text{fused silica}).$$

The round-trip phase $2\beta$ corresponds to a time $\dfrac{2n'd}{c}\approx 39\ \text{ps}$ — the delay of each internal echo, hence the post-pulse echoes in $E(t)$ and the interferogram recurrences at $\tau\approx\pm39$ ps.
