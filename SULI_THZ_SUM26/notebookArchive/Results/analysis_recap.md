# EO Sampling Data Analysis — Recap
**AWA THz / TFLN MZI Experiment, SULI 2026**

---

## 1. Overview

The goal was to extract the electro-optic (EO) modulated signal from BPD (balanced photodetector) waveform data taken at the AWA facility. Each scan captures BPD waveforms across 101–200 phase steps, where advancing the RF phase shifts the time delay between the electron bunch THz pulse and the laser probe pulse. The raw waveform at each step is dominated by the laser envelope — a large |sin(2πft)| shaped background at f = 81.25 MHz/2 = 40.625 MHz — sitting on top of which is the small EO-modulated THz signal we care about. The challenge is separating these two.

---

## 2. Baseline Removal — Methods Attempted

### 2.1 Mean over phase steps (`mean_steps`) ✓ Best general method

**Method:** Compute the mean waveform across all phase steps and subtract it from each trace.

**Why it works:** The THz pulse shifts in time by ~342 ps (coarse) or ~136 ps (fine) across the scan. This shift is small relative to the 100 ns waveform window, but sufficient to wash out the THz contribution when averaging over 101–160 steps. The laser envelope, being static, survives the mean and is cleanly removed.

**Limitation:** Per-step laser amplitude drift (shot-to-shot power variation) produces horizontal banding in the residual heatmaps — each step's laser power deviates slightly from the mean, leaving a residual proportional to the baseline shape.

| Metric | Value |
|---|---|
| Residual RMS (10 nC coarse) | 10.64 mV |
| Banding (row-mean std) | 3.21 mV |
| THz peak amplitude | 136.5 mV |

---

### 2.2 Rolling average (`rolling`) ✗ Not recommended

**Method:** Replace the global mean with a per-trace rolling boxcar average with window = one laser period (T = 1/81.25 MHz ≈ 12.3 ns, or ~615 samples). Averages out the local mean independently for each trace.

**Result:** Eliminates banding completely (row-mean std → 0) but leaves the sinusoidal AC ripple of the |sin| envelope in the residual, raising the overall RMS from 10.6 to 35.5 mV. Banding is reduced but at the cost of much noisier residuals.

---

### 2.3 Scaled mean (`scaled_mean`) ✓ Good, slightly better than mean_steps

**Method:** For each step, find the least-squares scale factor that best fits the mean baseline template to that trace, then subtract. Scale factor: a_i = (trace_i · mean) / ||mean||².

**Result:** Accounts for per-step laser power drift, reducing banding from 3.21 mV to 0.08 mV while keeping the same low overall RMS (10.1 mV). Mathematically equivalent to SVD with N_SVD = 1.

---

### 2.4 SVD subtraction (`svd`) ⚠️ Only safe at N_SVD = 1

**Method:** Decompose the (steps × samples) trace matrix via SVD. The leading components capture the dominant structured background; subtracting them leaves the THz signal.

**Finding:** The first SVD component accounts for 99.47% of variance — this is the laser envelope. The second component (0.41%) captures the THz signal itself, not additional baseline. This is a fundamental geometric limitation: the THz pulse shifts by only 342 ps in a 100 ns window (0.34% of the waveform), so it appears nearly stationary to SVD and is incorrectly identified as a structured background component.

| N_SVD | THz Peak | Residual RMS |
|---|---|---|
| 1 | 136.5 mV ✓ | 10.1 mV |
| 2 | **21.1 mV** ✗ | 4.8 mV |

**Conclusion:** SVD with N_SVD ≥ 2 over-subtracts and destroys the THz signal. N_SVD = 1 is safe but equivalent to `scaled_mean`. `mean_steps` is preferred because it is more physically transparent and qualitatively cleaner.

---

### 2.5 DipoleON reference subtraction ✓ Most accurate, limited availability

**Method:** Use a dedicated dipole-ON scan (beam deflected away from the EO crystal → no THz contribution) as the baseline template. Average all 5 channels and 200 steps of the dipoleON scan to obtain the cleanest possible laser envelope estimate, then apply per-step optimal scaling before subtracting.

**Physics:** When the dipole magnet is ON, the electron beam is deflected away from the EO crystal target. The BPD then measures only the laser envelope with zero THz modulation — the ideal physical background.

**Result:**

| Metric | mean_steps | dipoleON reference |
|---|---|---|
| THz peak amplitude | 136.5 mV | **155.1 mV** |
| Residual RMS | 10.64 mV | 18.9 mV |
| Banding (row-mean std) | 3.21 mV | **0.08 mV** |

The 14% higher peak with dipoleON is physically significant: `mean_steps` contaminates its baseline estimate with THz signal contributions from the ~20 steps near the optimal delay, partially suppressing the measured amplitude. The dipoleON result is the more accurate measurement. The higher overall RMS comes from the genuinely larger THz signal contribution, not worse noise.

**Limitation:** A paired dipoleON scan exists only for the 10 nC coarse scan in July 31. The full charge series has no paired backgrounds, so `mean_steps` remains the practical method for multi-scan comparisons.

---

## 3. Charge Series and Sensitivity

### 3.1 Data

Fine scans (July 31, `finescaneQ*`) covering 0.25–10 nC were processed with `mean_steps`. The 10 fine-scan channels were identified as repeated shots per phase step (not separate detector channels) and averaged before stripping.

### 3.2 Linear regression results

| Group | Free-fit sensitivity | Origin-forced sensitivity | R² (free) |
|---|---|---|---|
| Coarse scans | 10.5 ± 1.4 mV/nC | 14.9 ± 1.9 mV/nC | 0.95 |
| Fine scans | 13.1 ± 0.8 mV/nC | 15.6 ± 1.0 mV/nC | 0.98 |
| **dipoleON corrected (10 nC)** | **15.5 mV/nC** | — | — |

The free-fit intercepts are positive and non-negligible (coarse: +32.8 mV; fine: +17.6 mV), suggesting `mean_steps` systematically underestimates the signal — consistent with the 14% suppression seen in the dipoleON comparison. The origin-forced fits (physically motivated: no signal at zero charge) are more consistent across the two series and are preferred for quoting the sensor sensitivity.

**Best estimate of sensitivity: ~15 mV/nC** (dipoleON corrected, consistent with origin-forced fine scan regression).

---

## 4. Comparison to Wilke et al., *Scientific Reports* 14:4822 (2024)

The paper describes the same TFLN MZI sensor used in the AWA measurement. Its parameters serve as the calibration reference.

### 4.1 Sensor parameters (from paper)

| Parameter | Value |
|---|---|
| Probe wavelength λ | 1550 nm |
| Extraordinary index n_e | 2.15 |
| Pockels coefficient r₃₃ | 30.9 pm/V |
| MZI arm length l | 600 μm |
| Half-wave field E_π | 4.2 MV/m |
| Min detectable field (lock-in) | 1.3 kV/m |

### 4.2 Converting our BPD signal to E_THz

Using Eq. (1) and (3) from the paper:

**Δφ = V_signal / V_max = 155 mV / 424 mV = 0.37 rad**

**E_THz = (Δφ / π) × E_π = 0.49 MV/m at 10 nC**

This sits comfortably within the paper's demonstrated dynamic range (1.3 kV/m → 4.6 MV/m). The transduction coefficient is **0.32 μV per V/m**.

### 4.3 Beamline geometry factor

From E_THz at 10 nC, the electric field the AWA bunch produces at the crystal per unit charge is:

**49 kV/m per nC**

This is a property of the AWA beamline geometry (bunch-to-crystal distance and orientation), not the sensor.

### 4.4 Theoretical vs practical minimum detectable charge

| | Minimum detectable field | Minimum detectable charge |
|---|---|---|
| Paper (lock-in detection) | 1.3 kV/m | ~27 pC |
| Our measurement (phase scanning, no lock-in) | ~9 mV noise floor → 590 V/m | ~580 pC |
| **Gap** | | **~22×** |

### 4.5 Interpretation of the gap

The paper explicitly identifies laser pulse-to-pulse amplitude fluctuations as the dominant noise source (*"The SNR is determined by the laser pulse to laser pulse energy fluctuations"*). This is exactly what produced the banding in our residual heatmaps and what the dipoleON scaled subtraction reduced — but could not eliminate, since the laser noise is intrinsic to the measurement.

The paper achieves its 27 pC theoretical limit by using lock-in detection (beam chopping + synchronous demodulation at ~100 kHz), which makes the measurement insensitive to slow laser drift. Our phase-scanning approach accumulates this noise directly on the residual.

**Implication:** Implementing beam chopping synchronised to the oscilloscope acquisition would, in principle, reduce the minimum detectable charge from ~580 pC toward the ~27 pC theoretical limit of the sensor — a potential 20× improvement in sensitivity.

---

## 5. Summary

| Finding | Result |
|---|---|
| Best practical baseline method | `mean_steps` (simple, qualitatively cleanest) |
| Most accurate single-scan method | DipoleON reference subtraction |
| `mean_steps` amplitude bias | ~14% underestimate vs dipoleON reference |
| Sensor sensitivity | ~15 mV/nC |
| E_THz at 10 nC | ~0.49 MV/m |
| Beamline field factor | 49 kV/m per nC |
| Practical Q_min | ~580 pC |
| Theoretical Q_min (lock-in) | ~27 pC |
| Dominant noise source | Laser pulse-to-pulse amplitude fluctuations |
| Path to improvement | Lock-in / beam-chopped detection |
