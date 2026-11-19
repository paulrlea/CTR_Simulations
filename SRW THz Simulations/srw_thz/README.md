# srw_thz — AWA CTR-THz generation + propagation in SRW

Simulates the THz pulse in the AWA 1550-nm EO-sampling experiment: a coherent
THz source at the radiative foil, propagated through the sapphire window,
collection aperture, and a two-mirror relay (OAP1 collimate + OAP2 focus) to the
EO-sensor (TFLN) plane, using **SRW (`srwpy`) as the wavefront-propagation engine**.

Beamline order: foil → collimator → sapphire window → OAP1 (collimating, f1) →
OAP2 (focusing, f2) → detector. Each mirror is a thin-lens-equivalent off-axis
paraboloid; toggle with `--no-mirror` (both) or `--no-mirror2` (single mirror).

## Key modeling choices (read before trusting numbers)

- **The source is Coherent Transition Radiation (CTR), not bending-magnet CSR.**
  SRW has no transition-radiation source, so the THz source is **synthesized**.
- **Two source models** (select with `--src`), both behind `make_source(f, cfg)`:
  - **`gaussian`** (default, `thz_source.src_gaussian`): per-frequency `SRWLGsnBm`
    with divergence tuned to the CTR ~1/γ angle via `cfg.div_match`. Smooth; captures
    the aperture/window/propagation physics but **not** CTR's on-axis null / radial
    polarization.
  - **`ctr`** (`thz_source.src_ctr_wavefront`): analytic **Ginzburg–Frank** TR pattern
    `E(θ)=A·β sinθ/(1−β²cos²θ)` with the correct **on-axis null** and **radial
    polarization**, built in coordinate space via the far-field-of-source projection
    `x=z0·θ`. Amplitude `A` is physical (single-electron TR spectral energy
    `dW/dω≈αħ(2lnγ−1)`). After OAP2 it focuses to a **ring** (not a spot). The E(t)
    waveform is sampled at the ring via the **radial field component**
    `E_r = Ex·cosφ + Ey·sinφ` (`thz_extract.ring_peak_complex_field`) — averaging `Ex`
    alone cancels azimuthally (`⟨cosφ⟩=0`) and scrambles the phase into quasi-CW noise;
    the radial projection is azimuthally coherent and yields a localized transient.
    A **tilted screen** (`--screen-tilt` deg) turns the ring into an asymmetric crescent;
    two fidelities via `--tilt-model`: **`level1`** (default heuristic: foreshortening +
    `tanh` lobe bias, polarization left radial) and **`level3`** (exact point-charge
    oblique-TR vector field from the image-charge construction
    `A = T(n,β) − T(n,β_i)`, which yields the **true p/s polarization** as an output, not
    an assumption). `screen_tilt_deg=0` reduces both to the symmetric GF source.
    For Level-3 the EO crystal axis is set with `--crystal-pol-axis` (deg) and the
    on-crystal field is the projection `E_c = Ex·cosχ + Ey·sinχ`
    (`thz_extract.crystal_axis_complex_field`); `detector_pol_{p,s,angle}.png` maps are
    written automatically. The point-charge oblique asymmetry scales as ~1/γ (a few % at
    γ=117), so Level-3 is polarization-correct but its intensity asymmetry is small — the
    large empirical crescent lives in Level-1's tunable `gf_tilt_lobe_k`.
  - Both weight by the bunch form factor `F(f)=exp[−(2πf σ_t)²/2]`.
- **Engine = per-frequency monochromatic sweep** (10–400 GHz). Outputs:
  1. broadband 2D **spatial intensity map** at the detector, and
  2. on-crystal **time-domain E(t)** waveform (inverse FFT of the on-axis complex
     field × F(f)).
- **Window = sapphire (DN63CF, 2.03 mm).** The datasheet "50 %@250 nm" is a UV
  spec and is **discarded**; THz constants (n≈3.07, weak absorption) come from
  literature (`thz_config`). A flat slab is a spatial no-op, so it is modeled as
  (a) its clear aperture (`SRWLOptA`) plus (b) an analytic complex transmission
  `t_win(f)` (Fresnel + Fabry-Perot etalon + absorption) applied per frequency —
  exact because a global complex factor commutes with linear propagation. This
  also reproduces the etalon ring-down in E(t).
- **Two-mirror relay** (`thz_beamline.build_beamline`): OAP1's focal length defaults
  to the foil→OAP1 distance (collimates the diverging THz), OAP2's to the
  OAP2→detector distance (focuses the collimated beam onto the crystal). Both
  reflectivities (gold ≈1 in THz) enter `channel_response` as scalars.
- **BEST-GUESS knobs** (not pinned by the documents, flagged in `thz_config`):
  drift distances, mirror focal lengths, detector active area.

## Files

| file | role |
|---|---|
| `thz_config.py` | all parameters (`ThzConfig`, `default_config(**overrides)`) |
| `thz_source.py` | `make_source`, Gaussian surrogate, `E_eV`, `form_factor`, `sigma_r` |
| `thz_beamline.py`| `build_beamline` (SRWLOptC), `window_transmission`, `channel_response` |
| `thz_extract.py` | intensity/complex-field extraction (radial + `crystal_axis_complex_field`, `save_polarization_maps`), spatial-map + E(t) assembly, savers |
| `run_thz.py` | driver + CLI |
| `run_thz_spacetime.py` | Level-3 **space-time** detector reconstruction: keeps the full complex 2D field per frequency, projects on the crystal axis, weights by F(f)·Q, irffts **per pixel** → E_c(x,y,t) movie + x–t map (separate output folder) |
| `thz_verify.py` | aperture-diffraction (Airy) sanity check; Butterworth cutoff reference |
| `thz_layout.py` | beampath **schematic** (`draw_beamline`) from config, no SRW run |
| `thz_etalon.py` | quantify the sapphire-window etalon + mitigation plots (no SRW run) |

## Usage

```bash
source ../.venv/bin/activate           # PYTHONPATH-isolated venv
python run_thz.py                                    # full run (Gaussian) -> results/
python run_thz.py --src ctr --out results_gf         # Ginzburg-Frank CTR source (ring)
python run_thz.py --src ctr --screen-tilt 30 --out results_gf_tilt   # tilted screen, Level-1 (crescent)
python run_thz.py --src ctr --tilt-model level3 --screen-tilt 30 --out results_l3   # Level-3 oblique vector field + pol maps
python run_thz_spacetime.py --out results_l3_spacetime   # Level-3 space-time E_c(x,y,t) at detector (cube + x-t map + pulse.gif)
python thz_layout.py                                 # just the beampath schematic
python thz_etalon.py                                 # just the etalon metrics + plots
python run_thz.py --window-mode wedge --out wedge    # mitigation: wedged window
python thz_verify.py                                 # propagation + GF pattern checks
```
Headless plotting is forced (Agg). Outputs: `layout.png`, `spatial_map.{png,npz}`,
`waveform.{png,txt}`, `spectrum.{png,txt}`, `etalon_transmission.png`,
`etalon_spectrum.png`, `run_meta.txt`.

Useful flags: `--src {gaussian,ctr}`, `--nfreq`, `--fmin/--fmax` (GHz), `--sigma-t` (ps),
`--charge` (nC), `--div-match` (cutoff knob), `--sampfact`, `--det-npix`,
`--max-src-pts` (memory guard), `--no-window`, `--window-mode {slab,wedge,ar}`,
`--no-mirror`, `--no-mirror2`, `--screen-tilt` (deg; GF tilted-screen asymmetric CTR),
`--tilt-model {level1,level3}` (heuristic vs exact oblique vector field),
`--crystal-pol-axis` (deg; EO crystal sensitive axis, Level-3 polarization extraction).

## Sapphire-window etalon

The periodic ~24 GHz ripple in the spectrum and the post-pulse echoes in E(t) come
entirely from the sapphire window (the only multi-surface element): each air–sapphire
face reflects R = ((n−1)/(n+1))² ≈ 26%, forming a Fabry–Pérot cavity. `thz_etalon.py`
quantifies it — FSR = c/2nd ≈ 24 GHz, finesse ≈ 2.2, transmittance swing 33–100 %,
round-trip echo delay 2nd/c ≈ 41.6 ps, echo amplitude ratio ≈ 0.26 — and writes
`etalon_transmission.png` (|t_win(f)|² for slab/wedge/AR) and `etalon_spectrum.png`
(the current spectrum vs the same spectrum with the etalon removed). Mitigations are first-class via
`--window-mode`: `slab` (real flat window, etalon), `wedge` (etalon removed, ~55 % flat
transmission), `ar` (anti-reflection, ~96 %). These run through the full pipeline, and
the overlay plot rescales an existing spectrum without re-running SRW.

## Resolution & performance

Defaults: `n_freq=161` (≈2.4 GHz spacing, resolves the ~24 GHz sapphire etalon),
`sampFactNxNyForProp=1.0`, `det_npix=401`. A full run is ~6–8 min and peaks a few GB
(the two-mirror relay roughly doubled runtime vs a single mirror).
Each frequency is resampled to the fixed `det_npix` grid immediately and only that is
kept, so memory does **not** grow with the sweep length — important because at some low
frequencies SRW's auto-resize produces very large native meshes (tens of millions of
points). Those arise because long-wavelength THz diffracts over ~metres past the 2.3 cm
collimator and the two-mirror relay carries large collimated beams between the OAPs;
this is the physics that produces the low-frequency cutoff and cannot be cropped away
(cropping inflates the central power). The `max_src_pts` guard throttles source sampling
there (logged each time); lowering it can paradoxically *raise* the downstream mesh, so
4000 is a good default. On a small-RAM machine, raise `--fmin` (drops the most expensive
points) or lower `--sampfact`. `sampFact=2.0` gives slightly sharper high-frequency maps
but is heavier with the two-mirror relay.

## Verification & calibration

- `thz_verify.py` confirms SRW reproduces the circular-aperture far-field Airy
  null: ratio → 1 as the Fresnel number → 0 (0.73 at N=0.2, 0.94–0.96 at N≈0.01).
- **Cutoff calibration:** tune `cfg.div_match` so the simulated low-frequency
  rolloff (spectrum) matches the measured ~45 GHz high-pass
  (`thz_verify.butterworth_highpass` is the reference curve).

## Known limitations / next steps

- The broadband spatial map can show a faint square halo at low intensity — a
  mesh-truncation footprint from the very large low-frequency meshes (frequencies
  deep in the cutoff that carry negligible power). It does not affect the central
  spot or the spectrum.
- Sapphire treated as isotropic (ordinary-ray index); birefringence ignored.
- Absolute scaling is arbitrary (relative units); charge enters as a global ∝Q².
- **GF (`--src ctr`) low-frequency caveat:** below ~35 GHz the ring (z0/γ≈4.3 mm) is
  sub-wavelength and the source aperture is ~λ, so SRW auto-resize would run away. The
  GF path therefore uses a *capped* propagation (`build_beamline(cfg, capped=True)`:
  no auto-resize, fixed range ×2 per drift) which bounds the mesh and runs fast at all
  frequencies but mildly clips the hugely-diffracting low-f beam. Those frequencies are
  deep in the collimator cutoff (negligible signal), so the impact is small. The source
  mesh is also frequency-adaptive (coarse blob at low f, ring-resolving at high f).
- GF is the far field of a point-like source: most valid when z0 ≫ formation length γλ
  (high f); shares the low-f approximation with the Gaussian surrogate.
- **Level-3 polarization transport:** the exact oblique field gives the correct **source**
  p/s polarization, but mirrors are thin lenses (`SRWLOptL`), which do not rotate
  polarization — a real off-axis paraboloid imposes a geometric rotation that would need
  `SRWLOptMirPar`. So Level-3 is faithful for source polarization, not yet for end-to-end
  polarization transport through the relay. It also remains a point-charge (ideal-conductor,
  infinite-foil) model, so the dominant real crescent asymmetry still lives in Level-1's `k`.
