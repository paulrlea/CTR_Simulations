# SRW CTR-THz Model vs. AWA Experimental EO Data

*Cross-analysis of `srw_thz/` simulation against the July-31 2025 EO fine/coarse
scans (`SULI_THZ_SUM26/Data_raw/July31_`). Experimental waveform built as
V_sig(τ) = peak5.max − peak4.max (the notebook's validated drift-corrected EO
signature); coarse scans reduced from raw `BPDwf`. Figures in this folder.*

---

## Headline results

| Quantity | SRW model | Experiment | Verdict |
|---|---|---|---|
| Sapphire etalon echo **delay** | 41.6 ps (=2nd/c) | 37–43 ps (median ≈40 ps, 8 scans) | **✓ match (~5%)** |
| Etalon echo **amplitude ratio** | 0.26–0.29 (lossless Fresnel) | ~0.20 (autocorr side-peak) | ✓ right order; data lower (absorption) |
| Spectral **envelope peak** | ~115 GHz | ~51 GHz (centroid ~157) | **✗ SRW ~2× high — source model** |
| Recovered bunch **σ_t** | 2.0 ps (assumed) | 1.5–1.6 ps (companion nb) | exp bunch *shorter*, not the cause |
| Band vs EO walk-off **T_RF** | — | all content T_RF ≥ 0.98 | **✓ EO maps field faithfully** |

---

## 1. Sapphire Fabry–Pérot echo — CONFIRMED

SRW predicts a window etalon post-pulse at Δt = 2nd/c = **41.6 ps**, field ratio
~0.29, from the only multi-surface element (the ~2 mm sapphire window).

This is a **parameter-free** prediction (depends only on n≈3.07, d≈2.0 mm).
Autocorrelation of V_sig(τ) shows a side-peak at:
- **Fine scans** (5 charges): 37.6–42.7 ps, height 0.19–0.22 (median 39.3 ps / 0.205)
- **Coarse scans** (3): 39.3–41.0 ps, height 0.10–0.20

Consistent across all 8 independent scans → the post-pulse structure in the EO
waveforms is the **sapphire-window etalon (instrumental), not bunch physics**.
The measured echo ratio (~0.2) sits just below the lossless Fresnel estimate
(0.26–0.29), as expected once sapphire absorption + non-normal incidence are
included. Back-solving the lag gives window thickness d ≈ **1.9 mm** (vs 2.03 mm
assumed) — a refinable handle on the actual window.

> **Delay-axis note:** the echo appears at *smaller* τ than the main pulse
> (V_sig main pulse sits at τ≈106–115 ps in the fine scans). The EO-sampling
> delay axis is sign-reversed (later THz arrival ↔ smaller τ). This is why a
> naïve "look 41 ps *after* the peak" search fails on the fine scans — the echo
> is 41 ps *before* it, and the fine scans only have ~25 ps of room after.

## 2. Spectrum — SRW peak is ~2× too high (source, not propagation)

- SRW envelope peaks at **~115 GHz** (the raw 119.7 GHz is one etalon fringe;
  the 24.4 GHz ripple is confirmed = the predicted 24.0 GHz FSR).
- Experiment peaks at **~51 GHz** (companion notebook; my FFT: 48–81 GHz),
  centroid ~157 GHz, with a sharp foil low-frequency cutoff at **44.8 GHz**.

The measured bunch is actually **shorter** (σ_t≈1.5 ps) than SRW's assumed
2.0 ps — a shorter bunch radiates *higher* frequencies, so the form factor is
**not** the cause of the mismatch. The discrepancy is in the **low-frequency
cutoff**: SRW's Gaussian-surrogate source places its geometric/collimator cutoff
near ~100 GHz, whereas the real CTR **radiative-foil cutoff is ~45 GHz**. The
experimental peak is essentially pinned just above that foil cutoff.

**Conclusion:** SRW captures the *propagation* physics (etalon, spot, window
transmission) but its *source* spectrum is wrong — exactly the limitation flagged
in `FINDINGS.md` ("Gaussian surrogate of the CTR"). The data now calibrates the
fix: set the source low-f cutoff ≈ 45 GHz and σ_t ≈ 1.5 ps.

## 3. EO detector fidelity in this band — field maps faithfully

Velocity-mismatch transfer function T_RF(f) = sinc(πf(n_RF−n_opt)l/c):

| f | 50 | 120 | 250 | 310 | 640 GHz |
|---|---|---|---|---|---|
| T_RF | 0.997 | 0.981 | 0.919 | 0.877 | 0.536 |

All detected content (SRW envelope ≤150 GHz, experiment ≤~100 GHz) sits where
**T_RF ≥ 0.98** — i.e. far below the ~640 GHz walk-off (half-power) cutoff. So in
*this* experiment the velocity-mismatch reshaping is **<2%**: the EO detector
records the THz field essentially **shape-preserving (1-to-1)**. The departures
from 1-to-1 here are (a) the cosine/saturation amplitude nonlinearity at high
charge and (b) the detector's *low-frequency* aperture rolloff (<250 GHz, Wilke),
which suppresses absolute amplitude — not the velocity-mismatch high-f cutoff.

## 4. Spatial

SRW THz spot FWHM at the chip = **17.5 mm**. The actual TFLN waveguide active
region is ~10 µm × 600 µm — effectively a point at beam centre — so the detector
samples the **on-axis peak field**, which justifies SRW's on-axis E(t) extraction.
No spatial-averaging correction needed; the relevant concern is pointing/overlap
of the 17.5 mm spot onto the sub-mm chip.

---

## Actionable recommendations
1. **Retune SRW source:** low-f cutoff ≈45 GHz, σ_t≈1.5 ps → should bring the
   envelope peak from 115→~55 GHz, matching data.
2. **Window:** echo lag implies d≈1.9 mm sapphire; confirm against the real part
   spec. A wedged/AR window would remove the 0.2-ratio post-pulse.
3. **Echo-targeted scan:** to image the etalon train directly, run a delay scan
   that puts the main pulse *early* with ≥120 ps of room on the echo side.
4. Absolute E-field calibration still needs the single lumped κ (=πK_BPD/E_π);
   the low-f aperture rolloff must be folded into κ since the band is <250 GHz.
