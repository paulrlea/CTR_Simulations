# AWA CTR-THz Simulation Findings
*Generated from `srw_thz/` SRW propagation model. All E-field values are in
arbitrary (simulation) units — see scaling note at the bottom.*

---

## Electron Beam (AWA)
- Beam energy: **60 MeV** → Lorentz γ = **117.4**
- Bunch charge: **10 nC** (coherent power ∝ Q²)
- RMS bunch length σ_t: **2.0 ps**
- CTR opening angle ~1/γ: **8.52 mrad**

## Beamline Layout
```
foil → 50 cm → collimator (⌀4.6 cm) → 20 cm → sapphire window
     → 30 cm → OAP1 (collimate, f=1.0 m)
     → 30 cm → OAP2 (focus, f=0.3 m)
     → 30 cm → EO crystal (TFLN)
```
Total foil-to-detector: **1.60 m**

## Detected THz Spectrum (at EO crystal)
- Spectral peak: **119.7 GHz** (λ = 2.50 mm)
- −3 dB bandwidth: **112–127 GHz** (~15 GHz)
- Bunch form factor at peak: **F(120 GHz) = 0.323** (Gaussian: e^{−(2π f σ_t)²/2})
- Window transmittance at peak: **|t_win|² = 0.984**
- Band shaped by: geometric collimator cutoff (low-f) + form factor rolloff (high-f)

## Electric Field at EO Crystal — E(t) (on-axis, from inverse FFT)
- **Peak |E(t)| = 5.805 × 10⁹** (arb, see scaling below)
- Peak arrival time: ~387.7 ps (propagation delay; absolute t₀ arbitrary)
- Pulse structure: single-cycle bipolar transient
- **Prior etalon echo: 1.694 × 10⁹ arb** at t = 346.5 ps
  → **echo/peak amplitude ratio = 0.292**, echo delay = **41.3 ps**

## Spatial Profile at EO Crystal (z = 1.60 m)
- Peak broadband intensity: 1.920 × 10¹⁷ arb
- THz spot FWHM: **17.5 mm** (horizontal slice through centre)
- Beam enclosed by the ±50 mm detector view window

## Sapphire Window — Fabry–Pérot Etalon
The ~24 GHz spectral ripple and post-pulse echoes are caused **entirely by the
sapphire window** (the only multi-surface element). Each face reflects:
- Surface reflectance per face: **R = ((n−1)/(n+1))² = 0.259** (25.9%)
- Fringe spacing (FSR): **24.03 GHz** = c/(2nd)
- Echo round-trip delay: **41.62 ps** = 2nd/c ≈ confirmed by E(t) (41.3 ps measured)
- Finesse: **2.16** (low → broad, near-sinusoidal fringes)
- Transmittance range: **T_min = 0.347, T_max ≈ 1.000** (mean 0.57)
- Fringe visibility: **0.485**
- Echo amplitude per round trip: **~0.26** (field)

## Source Parameters at Spectral Peak (120 GHz)
- Photon energy: 495 µeV
- Gaussian source σ_r (waist, div_match=1): **93.6 mm** (overfills collimator → cutoff)
- Fresnel number at collimator (D=4.6 cm, z=0.5 m): ~0.07 (Fraunhofer regime)

## E-Field Scaling Note
The model computes `srwl.CalcElecFieldGaussian` (a Gaussian surrogate of the CTR),
propagates with `srwl.PropagElecField`, and extracts on-axis Re(E)+Im(E) per frequency.
The source field amplitude is set to `pulseEn=1 J` (nominal SRW unit) with an arbitrary
`repRate=1 Hz`. The coherent CTR field scales as **E ∝ Q × F(f) × E_single_electron**.
To recover physical units, the simulation field needs to be normalised to the
**total coherent CTR energy**, which for a Gaussian bunch is:
```
  U_CTR(f) ∝ Q² × F(f)² × (e²γ²)/(4π² ε₀ c) × [angular integral clipped by aperture]
```
The charge Q = 10 nC is the natural scale. The form factor F(f) is dimensionless and
already embedded in the waveform. The single-electron TR spectral energy and the
geometric beamline acceptance (aperture, mirror efficiency) set the absolute scale —
these are the unknowns that need to be fitted or independently measured.

## Key Unknowns / Best-Guess Parameters (flagged in `thz_config.py`)
- Drift distances between elements (assumed, not measured)
- Mirror focal lengths (set to f₁=1.0 m, f₂=0.3 m by the collimate+focus heuristic)
- EO crystal active area (assumed 5 mm radius)
- Absolute E-field calibration (requires K_BPD from `proposal.tex`)
