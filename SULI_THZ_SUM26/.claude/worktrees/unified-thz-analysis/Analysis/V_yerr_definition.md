# Vertical errorbars `amp_yerr`, `p2p_yerr`, `integ_yerr` — definition

Source code (cells 2 and 5a of `THz_Sensor_Characterization.ipynb`):

```python
# cell 2 — inside reduce_bpd(rec):
amp_peak_ps    = np.abs(signed_amp_ps).max(axis=0)                  # (n_shots,) V
p2p_global_ps  = signed_amp_ps.max(axis=0) - signed_amp_ps.min(axis=0)  # (n_shots,) V
integ_abs_ps   = np.trapezoid(np.abs(signed_amp_ps), dly_ps, axis=0)    # (n_shots,) V·ps
amp_peak_err   = float(amp_peak_ps  .std(ddof=1) / np.sqrt(n_sh))
p2p_global_err = float(p2p_global_ps.std(ddof=1) / np.sqrt(n_sh))
integ_abs_err  = float(integ_abs_ps .std(ddof=1) / np.sqrt(n_sh))

# cell 5a — assembling per-scan arrays:
amp_yerr   = np.array([rec['amp_peak_err']   for rec in scans])
p2p_yerr   = np.array([rec['p2p_global_err'] for rec in scans])
integ_yerr = np.array([rec['integ_abs_err']  for rec in scans])
```

## Setup — per-shot signed amplitude

For scan $k$ with $N_s^{(k)}$ delay steps (indexed by $i$) and $n=10$ shots
per step (indexed by $j$), let $V_{ij}^{(k)}(t)$ be the raw BPD waveform of
shot $(i,j)$. The per-shot reduction pipeline (baseline-fill + FFT low-pass +
average over a probe-time window of half-width `PROBE_HALF_WIN`) gives the
per-shot signed amplitude

$$
S_{ij}^{(k)} \;=\;
\frac{1}{|\mathcal{W}|}\sum_{t\,\in\,\mathcal{W}} \widetilde{V}_{ij}^{(k)}(t)
\qquad (\text{V}),
$$

stored as `signed_amp_ps[k]` of shape $(N_s, n)$. Here $\mathcal{W}$ is the
sample window around the probe-time index $t_p$, and $\widetilde{V}$ is the
baseline-corrected, low-pass-filtered waveform.

## Per-shot metrics and their per-scan SEM

For each shot $j$ in scan $k$, three scalar metrics are computed across the
delay-step axis $i$:

$$
\begin{aligned}
M^{\text{pk}}_{kj}   &\;=\; \max_i \bigl|S_{ij}^{(k)}\bigr|
   && \text{(peak |amplitude|, V)} \\[2pt]
M^{\text{p2p}}_{kj}  &\;=\; \max_i S_{ij}^{(k)} \;-\; \min_i S_{ij}^{(k)}
   && \text{(global P2P, V)} \\[2pt]
M^{\text{int}}_{kj}  &\;=\; \int_\tau \bigl|S_{ij}^{(k)}(\tau)\bigr|\,d\tau
   \;\approx\; \mathrm{trapz}_i\!\bigl(|S_{ij}^{(k)}|,\,\tau_i\bigr)
   && \text{(integrated |signal|, V·ps)}
\end{aligned}
$$

For each metric $X \in \{\text{pk},\text{p2p},\text{int}\}$, the per-scan
mean and **standard error of the mean** across the $n=10$ shots are

$$
\bar{M}^X_k \;=\; \frac{1}{n}\sum_{j=1}^{n} M^X_{kj},
\qquad
\widehat{\sigma}^X_k
\;=\; \sqrt{\,\frac{1}{n-1}\sum_{j=1}^{n}\bigl(M^X_{kj} - \bar{M}^X_k\bigr)^2\,},
$$

and the plotted vertical errorbar is the SEM ($\mathrm{ddof}=1$, hence $n-1$):

$$
\boxed{\;
\sigma^X_{V,k} \;=\; \frac{\widehat{\sigma}^X_k}{\sqrt{n}}
\;=\; \sqrt{\,\frac{1}{n(n-1)}\sum_{j=1}^{n}\bigl(M^X_{kj} - \bar{M}^X_k\bigr)^2\,}
\;}
$$

which is `amp_yerr[k]`, `p2p_yerr[k]`, or `integ_yerr[k]` depending on $X$
(units: V or V·ps).

## What `σ_V` captures

By construction $\sigma^X_{V,k}$ is an **empirical** scatter — no error
propagation, no analytic noise model. It folds in every fluctuation that
makes the metric differ from shot to shot:

| Source                                | Folded in? |
| ------------------------------------- | :--------: |
| BPD detector electronics noise        | yes |
| Probe-laser intensity / timing jitter | yes |
| ICT shot-to-shot charge variation     | yes |
| Real bunch-to-bunch envelope changes  | yes |
| Drift over the duration of the scan   | yes (between-step is absorbed into the per-shot scalar before std is taken) |

The cell 5e "background-only" alternative ($\sigma_{\text{bg}}/\sqrt{n}$ with
$\sqrt{2}$ and $|d\tau|\sqrt{N_s}$ propagation factors) only contains the
first row and is flat with $Q$ — `σ_V` here grows with $Q$ as the real
bunch-to-bunch scatter does.

## Why SEM (not population $\sigma$) — opposite convention from `Q_xerr`

The vertical bar uses $\mathrm{std}/\sqrt{n}$ (SEM); the horizontal bar
[`Q_xerr`](Q_xerr_definition.md) uses population $\mathrm{std}$. They answer
different questions on purpose:

| Axis | Bar                                          | Meaning |
| ---- | -------------------------------------------- | ------- |
| $y$  | $\widehat{\sigma}^X_k / \sqrt{n}$            | "how well do we know the mean metric for this scan" |
| $x$  | $\sigma_{Q,k}$ (population $\sigma$)         | "how wide was the cloud of individual shot charges" |

The y-axis bar is the right quantity for fitting: a weighted log-log fit
should use the uncertainty on the *mean* metric, which is the SEM.

## Subtle: marker position vs. SEM definition

The y-position of the marker in cell 5d uses the **metric of the
10-shot-averaged trace** (`signed_amp`), while $\widehat{\sigma}^X_k$ is the
std of the **metric of each per-shot trace** ($M^X_{kj}$ from
`signed_amp_ps`):

| Quantity         | Code                                            | Formula |
| ---------------- | ----------------------------------------------- | ------- |
| Marker $y$ (peak)| `np.abs(rec['signed_amp']).max()`               | $\max_i \bigl|\tfrac{1}{n}\sum_j S_{ij}\bigr|$ |
| SEM-input (peak) | `amp_peak_ps[j] = np.abs(signed_amp_ps).max(0)` | $\max_i \bigl|S_{ij}\bigr|$, then $\mathrm{std}_j/\sqrt{n}$ |

For the **integral** these commute (it is linear in $S$): the metric of the
mean equals the mean of the metric. For the **peak** and **P2P** they do not
— $\max\circ\mathrm{mean} \le \mathrm{mean}\circ\max$ (Jensen-style), so the
marker can sit a few % below the per-shot mean metric, and the SEM is the
scatter *around the per-shot mean*, not around the marker. In practice the
gap is small (< 1 σ_V for these scans) but worth knowing when the bar is
read at face value.

## Dropout asymmetry vs. `Q_xerr`

`reduce_bpd` operates on `raw = rec['BPDwf']` *without* masking the ICT
dropout shots. So if a step contained a gun mis-fire, that shot's
`signed_amp_ps[:,j]` is essentially baseline noise and the per-shot metric
$M^X_{kj}$ for that shot is small — it inflates $\widehat{\sigma}^X_k$ and
biases $\bar{M}^X$ slightly downward. The x-axis bar **does** mask dropouts
([`Q_xerr_definition.md`](Q_xerr_definition.md)), so the two bars summarise
slightly different populations of shots. Dropout fractions are at the
~0.1–0.4 % level in this dataset (see cell 4 table), so the effect is small,
but it is the one place where x and y are not fully symmetric.

## Related alternative estimators (available in `rec`)

| Quantity                         | Code                                     | Meaning |
| -------------------------------- | ---------------------------------------- | ------- |
| `amp_yerr[k]` (plotted)          | `rec['amp_peak_err']`                    | SEM of peak |amp| across 10 shots |
| `p2p_yerr[k]` (plotted)          | `rec['p2p_global_err']`                  | SEM of global P2P across 10 shots |
| `integ_yerr[k]` (plotted)        | `rec['integ_abs_err']`                   | SEM of ∫|V|dτ across 10 shots |
| Population $\sigma$ (no /√n)     | `rec['amp_peak_ps'].std(ddof=1)`         | per-shot spread, not SEM |
| Background-noise propagation     | cell 5e (`amp_yerr_bg`, etc.)            | analytic σ_bg/√n with √2 / |dτ|√N factors |
