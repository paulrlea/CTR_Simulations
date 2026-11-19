# BPD Signal Processing: Period-Fill Baseline Removal and Super-Gaussian Low-Pass Filter

*Refers to `THz_Sensor_Characterization.ipynb`, Cell 5 (`_period_fill_indices`, `fft_lowpass`, `reduce_bpd`).*

---

## 1. Period-Fill Baseline Removal

### Background

Each BPD scope trace captures a 100 ns window at 50 GS/s (20 ps/sample, 5000 samples per shot). The AWA photoinjector laser runs at 81.25 MHz, so the scope window contains roughly 8 laser pulses spaced by

$$T_\text{laser} = \frac{1}{81.25 \text{ MHz}} \approx 12.308 \text{ ns} \approx 615 \text{ samples}$$

The probe pulse — the laser shot that temporally overlaps the electron bunch — lands somewhere within the window. The EO modulation we want to measure is a small perturbation riding on top of the periodic laser-train carrier. Straightforward background subtraction (e.g., fitting a polynomial or using a pre-shot region) fails here because the carrier is not smooth — it has sharp peaks every 12.3 ns — and the probe pulse sits inside that carrier, not outside it.

The solution is **period-filling**: rather than modeling the baseline analytically, we *look up* what the same carrier phase looks like at points in time that are not contaminated by the EO signal, and use those as the local baseline estimate.

### Signal Window

The EO-modulated probe pulse is known to arrive between 45 ns and 75 ns into the scope window (constants `SIGNAL_START_NS = 45.0`, `SIGNAL_END_NS = 75.0`). In sample-index space:

$$i_0 = \left\lfloor\frac{45.0\,\text{ns}}{0.02\,\text{ns/sample}}\right\rceil = 2250, \qquad i_1 = \left\lfloor\frac{75.0\,\text{ns}}{0.02\,\text{ns/sample}}\right\rceil = 3750$$

This 30 ns window spans **2.438 laser periods** — a non-integer number of periods, which is why the extrapolation logic needs to handle asymmetric hop counts on the left and right.

### Reference Index Calculation (`_period_fill_indices`)

```python
def _period_fill_indices(t_ns, s0, s1):
    dt = (t_ns[-1] - t_ns[0]) / (len(t_ns) - 1)
    T  = T_LASER_NS / dt
    i0, i1 = int(round(s0/dt)), int(round(s1/dt))
    idx = np.arange(i0, i1)
    nl  = np.ceil((idx + 1 - i0) / T).astype(int)
    nr  = np.ceil((i1 - idx)       / T).astype(int)
    return i0, i1, np.round(idx - nl*T).astype(int), np.round(idx + nr*T).astype(int)
```

For each sample index `idx` in `[i0, i1)`, the function computes two reference indices — one to the left of the window (`sl`) and one to the right (`sr`) — at the same laser carrier phase as `idx`, but outside the contaminated region.

**Left hop count `nl`:**
$$n_L = \left\lceil \frac{\text{idx} + 1 - i_0}{T} \right\rceil$$

This is the smallest integer such that `idx − n_L · T < i0`. In other words, go back exactly enough periods so the reference lands before the signal window begins. Adding 1 in the numerator ensures that a sample exactly at the window edge (where `idx = i0`) requires `nl = 1`, not `nl = 0`.

**Right hop count `nr`:**
$$n_R = \left\lceil \frac{i_1 - \text{idx}}{T} \right\rceil$$

The smallest integer such that `idx + n_R · T ≥ i1`. Go forward enough periods so the reference lands at or after the window end.

**Reference indices:**
$$s_L = \text{round}(\text{idx} - n_L \cdot T), \qquad s_R = \text{round}(\text{idx} + n_R \cdot T)$$

The `round()` is necessary because `T = 615.38` is non-integer; without it, accumulated fractional errors would put the reference at a slightly wrong carrier phase.

### The Extrapolation: Why the Hop Counts Are Asymmetric

The window is 2.438 periods wide, which means `nl` and `nr` are generally not equal, and both can exceed 1. Three representative examples:

| Location in window | `idx` | scope time | `nl` | `nr` | `sl` (ref left) | `sr` (ref right) |
|---|---|---|---|---|---|---|
| Near left edge | 2255 | 45.10 ns | 1 | 3 | 1640 (32.80 ns) | 4101 (82.02 ns) |
| Center | 3000 | 60.00 ns | 2 | 2 | 1769 (35.38 ns) | 4231 (84.62 ns) |
| Near right edge | 3745 | 74.90 ns | 3 | 1 | 1899 (37.98 ns) | 4360 (87.20 ns) |

A sample near the left edge of the window is only ~0.008 periods from the left boundary, so one hop to the left clears it easily (`nl = 1`). But it is ~2.43 periods from the right boundary, so it needs three hops to the right (`nr = 3`). The situation reverses for samples near the right edge. Center-of-window samples need two hops on each side.

The key point is that both references are **always outside the signal window**, regardless of where in the window `idx` sits. This guarantees that the reference values are uncontaminated by the EO modulation.

### Baseline Construction and Subtraction

```python
bl = traces.copy()
bl[:, i0:i1] = 0.5 * (traces[:, sl] + traces[:, sr])
res = fft_lowpass(traces - bl, dt)
```

The baseline inside the window is set to the **mean of the two period-coherent references**. Outside the window, the baseline equals the raw trace (so the subtraction `traces - bl` is zero outside `[i0, i1]`, and the signal outside the window is zeroed out).

The average of `sl` and `sr` is not arbitrary. The two references bracket the contaminated window at the same carrier phase. Their mean provides a linear-interpolation estimate of what the carrier would look like at `idx` in the absence of the electron bunch. First-order secular drift in the laser power or DC offset cancels in the average, because `sl` is displaced by roughly the same time offset to the left as `sr` is to the right (for center-window samples, exactly so; for edge samples, the offset is asymmetric but both references sample the same stationary carrier phase).

### Physical Validity

The method rests on two assumptions:

1. **Phase stationarity**: the laser carrier has the same amplitude and phase at each repetition. Shot-to-shot laser amplitude noise (jitter) contributes directly to the baseline uncertainty. The 10-shot average performed before baseline subtraction reduces this noise by $1/\sqrt{10}$.

2. **No signal contamination outside the window**: the references `sl` and `sr` are clean. This holds as long as the electron bunch interacts with the probe only within the declared window, which is satisfied by construction of `SIGNAL_START_NS` and `SIGNAL_END_NS`.

---

## 2. Super-Gaussian FFT Low-Pass Filter

After baseline subtraction, the residual still contains high-frequency scope noise (quantization, thermal noise, and residual laser noise not removed by period-fill). A low-pass filter is applied in the frequency domain.

### Filter Definition

```python
def fft_lowpass(arr, dt_ns):
    n = arr.shape[1]
    freqs = np.fft.rfftfreq(n, d=dt_ns)
    H = np.exp(-0.5 * (freqs / FFT_SIGMA_GHZ) ** (2 * FFT_ORDER))
    return np.fft.irfft(np.fft.rfft(arr, axis=1) * H, n=n, axis=1)
```

The filter transfer function is:

$$H(f) = \exp\!\left[-\frac{1}{2}\left(\frac{f}{\sigma}\right)^{2n}\right]$$

with $\sigma = 0.35\,\text{GHz}$ (`FFT_SIGMA_GHZ`) and $n = 6$ (`FFT_ORDER`). The exponent is $2n = 12$.

This is called a **super-Gaussian** or **flat-top Gaussian** because as $n$ increases, the filter shape transitions from a gentle bell curve ($n=1$, ordinary Gaussian) toward a rectangular brick-wall filter ($n \to \infty$). With $n = 6$, the passband below $\sigma$ is nearly flat and the stopband above $\sigma$ has a very steep transition.

### Transfer Function Shape

The ordinary Gaussian ($n = 1$) would be:

$$H_1(f) = \exp\!\left[-\frac{f^2}{2\sigma^2}\right]$$

which attenuates already at $f \ll \sigma$ and rolls off smoothly over many octaves. A super-Gaussian with $n = 6$ evaluates as:

| Frequency | $H_{n=1}$ | $H_{n=6}$ |
|---|---|---|
| 0.10 GHz | 0.960 | $\approx 1.000$ |
| 0.25 GHz | 0.734 | 0.990 |
| 0.35 GHz ($= \sigma$) | 0.607 | 0.607 |
| 0.42 GHz | 0.477 | 0.010 |
| 0.60 GHz | 0.204 | $< 10^{-9}$ |

The passband ($f < 0.3$ GHz) is essentially unattenuated for the super-Gaussian, while a standard Gaussian would have already removed 27% of the amplitude at 0.25 GHz.

### Bandwidth Specifications

The $-3\,\text{dB}$ (half-amplitude) point:

$$H(f_{-3\,\text{dB}}) = \frac{1}{2} \implies f_{-3\,\text{dB}} = \sigma\,(2\ln 2)^{1/(2n)} = 0.35 \times (2\ln 2)^{1/12} \approx 0.360\,\text{GHz}$$

The $1\%$ amplitude point (effective stopband onset):

$$H(f_{1\%}) = 0.01 \implies f_{1\%} = \sigma\,(2\ln 100)^{1/12} \approx 0.421\,\text{GHz}$$

The transition from $-3\,\text{dB}$ to $-40\,\text{dB}$ (amplitude) spans only $0.421 - 0.360 = 0.061\,\text{GHz}$ — a transition band less than 20% wide relative to the cutoff frequency. A standard Gaussian ($n=1$) requires orders of magnitude more bandwidth to achieve the same stopband rejection.

### Why Super-Gaussian for This Application

Three properties make this filter appropriate:

1. **Signal preservation in the passband.** The EO-modulated probe pulse, after baseline subtraction and probe-time detection, contributes signal energy in the DC–0.3 GHz band (the probe pulse envelope is broadened to a few ns by the detector/scope response). The flat passband ensures the probe amplitude and shape are not distorted by filter rolloff.

2. **Sharp rejection above the cutoff.** The laser repetition frequency is 81.25 MHz and its harmonics fall at 81.25, 162.5, 243.75, 325, 406.25 MHz, ... The period-fill removes these from the signal window, but shot noise and digitizer noise above $\sim 0.4\,\text{GHz}$ remain. The $n = 6$ filter has $< 1\%$ amplitude transmission by 421 MHz, so noise in that region is suppressed by more than $40\,\text{dB}$ in amplitude.

3. **No time-domain ringing from the probe pulse region.** Unlike a rectangular (brick-wall) filter, the super-Gaussian has no Gibbs phenomenon and no significant sidelobes in the impulse response. Any sharp feature in the baseline-subtracted residual — such as the edge of the period-fill window — is not turned into an oscillatory artifact by the filter.

### Implementation Notes

The filter is applied via the real FFT (`np.fft.rfft`), which computes only the non-negative-frequency half of the spectrum and assumes the input is real-valued. The multiplication `np.fft.rfft(arr, axis=1) * H` broadcasts over the trace/shot batch axis. `np.fft.irfft(..., n=n)` reconstructs the full $n$-sample real-valued trace, ensuring the output length matches the input exactly even when `n` is odd.

The filter is computed over the entire 100 ns trace including the period-filled region, not just the signal window. This is correct because the baseline subtraction has already made the trace zero outside `[i0, i1]` (the subtracted region), so the LP filter is effectively acting on a windowed signal. No pre-windowing or zero-padding is applied beyond the natural FFT length.
