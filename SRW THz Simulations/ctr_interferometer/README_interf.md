# CTR beam diagnostic — propagation through the interferometer path (`run_interf.py`)

A **CTR-based beam diagnostic**: propagates the AWA CTR-THz beam — **the same source
profile as the previous experiments** (`srw_thz/`) — through the **new** Michelson-type
interferometer optical path drawn in `draw_interferometer.py` / `beampath.png`, using SRW
as the wavefront-propagation engine. Scanning the delay stage measures the CTR field
**autocorrelation**, whose spectrum carries the bunch form factor (this is *not* FTIR
spectroscopy of a sample).

**Detector: Golay-cell power detector (default).** Unlike the previous EO field meter,
a Golay cell measures time-integrated, polarization-blind, phase-blind POWER over its
input aperture, so the measurement is a **delay-stage scan → interferogram
(autocorrelation) → recovered CTR spectrum**. The SRW propagation is identical to the EO
case; only the detector-plane *reduction* differs. Use `--detector eo` for the previous
field-meter model (E(t) waveform, single delay, no scan) — see the "Detector models"
section.

## Optical path (new)

```
Mo target → fused-silica window → 4" flat fold mirror ("L3 monitor")
          → pellicle beamsplitter (30/30) → 75 mm flat retro mirror (delay arm)
          → back to beamsplitter → gold OAP (focus) → detector
```

We model the physical path the **delay-arm** light takes to the detector: reflect into
the arm at the BS, retro-reflect off the 75 mm flat, transmit back out toward the OAP,
which focuses onto the detector. The delay is fixed at **0** (no scan). The reference
arm would only add a scalar coherent factor at τ=0 and is out of scope.

## Source: Ginzburg–Frank CTR ⊛ Gaussian

Reuses `srw_thz/thz_source.src_ctr_wavefront` (analytic TR ring, on-axis null, radial
polarization; physical amplitude from the single-electron TR spectral energy), then
**convolves the source field with a Gaussian** of RMS `gf_conv_sigma_m` — the finite
transverse electron-beam size that rounds the idealized point-charge pattern.

- `--conv-domain field` (default, coherent): blurs the complex Ex/Ey. Preserves the
  radial-polarization on-axis null (topological) and smooths/broadens the ring.
- `--conv-domain intensity` (incoherent): blurs |E|², filling the source on-axis null.
- `--conv-sigma 0`: exact pure-GF pass-through (before/after comparison).

At the **detector** the pattern is a CTR **ring** in all cases — set by the robust ~1/γ
angular structure and the OAP imaging — so the convolution is a modest perturbation
there (the null-fill is a source-plane effect that coherent radial imaging largely
restores). See the `results_smoke_*` folders for the quantified comparison.

## Detector models (`--detector`)

The SRW propagation to the detector plane is identical for both detectors; only the
reduction of the detector-plane field into an observable differs.

This is a **CTR-based beam diagnostic**, not FTIR spectroscopy of a sample: the Michelson +
Golay measures the first-order **field autocorrelation** of the CTR pulse, whose spectrum
carries the bunch form factor |F(f)|² (longitudinal-bunch information).

**`golay` (default) — Golay-cell power detector.** Measures time-integrated,
polarization-blind, phase-blind power over its aperture, so it gives one number per delay
setting; the delay stage is scanned to build the autocorrelation:
- **Observable** = total intensity (`|Ex|²+|Ey|²`, both pols) integrated over the Golay
  aperture (`--golay-aperture`, = `det_active_radius_m`). No E(t) waveform.
- **`power_capture.png`** — captured power fraction vs aperture radius. Because the beam
  focuses to a CTR **ring** with an on-axis null, a small on-axis aperture reads near
  zero: at the 5 mm nominal aperture only **~5%** of the power is captured (ring peak
  ~6.8 mm); you need ~15–20 mm (or a Winston cone / repositioned focus) for 60–80%.
- **`interferogram.png`** — the Golay reading vs delay τ (with a stage-travel axis),
  `P(τ)=Σ_f S(f)[1+cos(2πfτ)]` for a balanced (symmetric) Michelson: the CTR field
  autocorrelation. Shows the central burst at zero path difference and coherence
  recurrences at the fused-silica window round trip 2nd/c ≈ 39 ps.
- **`recovered_spectrum.png`** — CTR power spectrum recovered by cosine-transforming the
  interferogram (autocorrelation), overlaid on the directly-integrated spectrum
  (Wiener–Khinchin self-consistency check).
- Delay↔stage: `τ = 2·x_stage/c`. Nyquist step for 400 GHz ≈ 1.25 ps (0.19 mm stage);
  ~2.4 GHz resolution needs τ_max ≈ 205 ps (~31 mm travel).

**`eo` — electric-field meter (the previous setup).** Coherent, phase- and
polarization-resolved: samples the ring complex field and inverse-FFTs to the on-crystal
**E(t) waveform** (`waveform.png`); propagation-only, no delay scan. This is the model
`srw_thz/` used.

## Files (this folder)

| file | role |
|---|---|
| `interf_config.py` | `InterfConfig(ThzConfig)`: geometry, apertures, BS 30/30, fused-silica window constants, convolution knob. All distances/focal length **best-guess**. |
| `interf_source.py` | `make_source_conv(f, cfg)`: GF wavefront (reused) ⊛ Gaussian. |
| `interf_beamline.py` | `build_interf_beamline` (SRWLOptC), `window_transmission_fs` (fused silica), `channel_response_interf` (window × mirrors × BS r·t), `detector_z_m`. |
| `interf_golay.py` | Golay observables: power-capture curve, aperture-resolved spectrum, CTR autocorrelation interferogram + recovered spectrum, and their savers/plots. |
| `run_interf.py` | driver + CLI; reuses `srw_thz/thz_extract` (EO) and `interf_golay` (Golay). |
| `animate_interferogram.py` | animates the Golay reading as the delay stage scans → `interferogram.gif` (reads `interferogram.txt`, no re-propagation). |
| `run_interf_spacetime.py` | **absolute** space-time reconstruction: per-pixel complex field → `P_inst(t;τ)=Σ_ap|E(t)+E(t−τ)|²` [W] as a 2D map over delay & time, calibrated to J/W via the coherent CTR energy (N²|F|², N=Q/e). Re-propagates. |
| `draw_interferometer.py` | the beampath schematic (`beampath.png`). |

Notes: `DELAY_PHASE.md` (how the delay stage is modelled) and `ETALON_HANDLING.md` (how the
window Fabry–Pérot etalon is handled analytically outside SRW).

Everything in `srw_thz/thz_source.py`, `thz_extract.py`, `thz_config.py`,
`thz_beamline.py` (the capped `PP_CAP_THIN` params) is **reused unchanged**; the new
modules add the path in `sys.path` so they run from any cwd.

## Usage

```bash
source ../.venv/bin/activate            # PYTHONPATH-isolated venv
python run_interf.py --nfreq 9          # quick smoke test (Golay)
python run_interf.py                    # full run (n_freq=161, Golay) → results_interf/
python run_interf.py --detector eo      # previous EO field meter → E(t) waveform
python run_interf.py --golay-aperture 0.003         # 3 mm Golay aperture
python run_interf.py --conv-domain intensity        # incoherent source smear
python run_interf.py --oap-f 0.20 --conv-sigma 3e-3 # tune OAP focus / beam size
python animate_interferogram.py                     # -> results_interf/interferogram.gif (delay-scan movie)
python run_interf_spacetime.py                      # -> results_interf_spacetime/ (absolute 2D delay-time map)
```

### Absolute space-time map (`run_interf_spacetime.py`)

Produces what the Golay sees in **absolute units** as a function of **both** delay τ and fast
time t: `P_inst(t;τ) = Σ_aperture |E(x,y,t) + E(x,y,t−τ)|²` [W] — the reference pulse at t=0,
the delayed pulse riding the diagonal t=τ, interfering at ZPD (τ=0). Its time-integral is the
interferogram `P(τ) = ∫P_inst dt` [J/pulse]. Needs the complex field, so it re-propagates.
Absolute scale from the coherent CTR bunch energy `dW/dω = N²|F(f)|²·(2α/π)ℏ(ln2γ−1)` (N=Q/e),
with the **finite-foil low-frequency rolloff** applied at the source (see below), integrated
over the aperture-collected spectrum — still **order-of-magnitude** (rests on the TR model +
coherent scaling + the assumed charge/bunch length); the *shapes* are robust. Outputs:
`map_power_delay_time.png` (2D map, W), `interferogram_absolute.png` (nJ/pulse vs τ),
`pulse_power_time.png` (single-arm pulse power vs t), `spacetime.npz`, `run_meta.txt`. Defaults
give W_arm≈6 µJ/arm through the 5 mm aperture, peak ≈2.6 MW at ZPD (4× the single-arm peak).
(Before the TR-prefactor + finite-foil corrections this read ~35 µJ / 16 MW — a ~6× overestimate.)

Outputs (`results_interf/`), Golay: `spatial_map.{png,npz}` (detector ring + aperture
overlay), `power_capture.{png,txt}`, `interferogram.{png,txt}`, `interferogram.gif` (delay-scan
movie), `recovered_spectrum.{png,txt}`,
`spectrum.{png,txt}`, `layout.png`, `run_meta.txt`. EO (`--detector eo`): `waveform.{png,txt}`
instead of the interferogram/power-capture.

Useful flags: `--detector --golay-aperture --tau-max-ps --n-tau --nfreq --fmin --fmax
--sigma-t --charge --conv-sigma --conv-domain --oap-f --div-match --gf-n-ang
--cap-range-factor --det-npix --no-window --window-mode`.

## Key modeling assumptions (all flagged in `interf_config.py`)

- **Detector = Golay power cell** (default): observable is aperture-integrated total power;
  the delay stage is scanned to build the CTR field autocorrelation. `--detector eo`
  restores the field-meter E(t).
- **Interferogram** `P(τ)=Σ_f S(f)[1+cos2πfτ]`, τ = relative optical delay (ZPD at τ=0).
  The reference arm is the measured **1.854"** (`L_bs_to_reference_m`); it is longer than the
  delay arm's nominal 1.342" by ΔL=0.512", so **ZPD occurs at a 13.0 mm (0.512") delay-stage
  displacement**, not at the nominal position — reflected only in the interferogram's
  stage-travel axis (`stage_zero_offset_m`), not the shape/spectrum. The two arms are assumed
  to deliver the same detector field (exact at ZPD, ≤2.8% path difference elsewhere), so no
  re-propagation is needed for the arm length. For EO the delay is fixed at 0 (single arm).
- **GF ⊛ Gaussian** with a best-guess `gf_conv_sigma_m` (2 mm default) — tune to the beam.
- **Absolute-energy corrections** (make the magnitude realistic; shapes mostly via the low-f
  one): single-electron TR energy uses the standard `(2α/π)ℏ(ln2γ−1)` (not the earlier
  ~3×-too-big `αℏ(2lnγ−1)`); a **finite-foil low-frequency rolloff** `S(f)=[ln(min(2γ,ka))−1]/
  [ln2γ−1]` (a=`foil_radius_m`=0.5″) suppresses the low-f coherent energy the ideal infinite
  foil overestimates; a **transverse form factor** `exp[−(kσ_rθ)²/2]` (`beam_sigma_r_m`) is
  included but ≈1 in-band for mm beams. Absolute magnitude remains order-of-magnitude and ∝Q²
  — calibrate to a measured Golay signal for anything quoted as absolute.
- **Fused-silica window**: n≈1.95, best-guess absorption (rises ~f²); milder etalon than
  the old sapphire (R≈10% vs 26%). The 2 µm pellicle BS etalon FSR is far above band →
  treated as an ideal thin 30/30 splitter (scalar r·t, no BS etalon).
- **Best-guess distances & OAP focal length** (auto-set to image the source onto the
  detector); the drawing is schematic.
- **Capped propagation** (GF source needs it): mesh range grows `cap_range_factor` (1.6)
  per drift at constant resolution — milder than the old model's 2.0 because this path
  has 6 drifts; still contains the beam at the OAP while bounding wavefronts to a few M pts.
