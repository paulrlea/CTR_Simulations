# Horizontal errorbar `Q_xerr` — definition

Source code (cell 5a of `THz_Sensor_Characterization.ipynb`):

```python
Q_xerr = np.array([
    np.nanstd(rec['ict']['per_shot_kept'].ravel()) for rec in scans
])
```

## Definition

For scan $k$ with delay steps indexed by $i = 1, \dots, N_s^{(k)}$ and shots
per step indexed by $j = 1, \dots, 10$, let $Q_{ij}^{(k)}$ be the per-shot
ICT charge (nC) and let

$$
\mathcal{K}_k \;=\; \bigl\{(i,j)\,:\,Q_{ij}^{(k)} > 0\bigr\},
\qquad
N_k \;=\; |\mathcal{K}_k|
$$

be the set (and count) of kept shots — i.e. shots not flagged as photo-injector
dropouts. The marker x-position is the pooled mean

$$
\bar{Q}_k \;=\; \frac{1}{N_k}\sum_{(i,j)\,\in\,\mathcal{K}_k} Q_{ij}^{(k)}.
$$

The horizontal errorbar is the **population standard deviation** of the kept
shots (`np.nanstd` default, $\mathrm{ddof}=0$):

$$
\boxed{\;\sigma_{Q,k} \;=\; \sqrt{\,\frac{1}{N_k}\sum_{(i,j)\,\in\,\mathcal{K}_k}\bigl(Q_{ij}^{(k)} - \bar{Q}_k\bigr)^2\,}\;}
\qquad (= \texttt{Q\_xerr[k]},\ \text{nC}).
$$

## Equivalent "shot + drift" decomposition

When every step has the same kept-shot count $n$ ($\approx 10$, modulo
dropouts), the within-step / across-step variance split is exact:

$$
\sigma_{Q,k}^2 \;=\;
\underbrace{\frac{1}{N_s}\sum_{i=1}^{N_s}\sigma_{i,k}^2}_{\sigma_{\text{shot},k}^2\ \text{(within-step)}}
\;+\;
\underbrace{\frac{1}{N_s}\sum_{i=1}^{N_s}\bigl(\mu_{i,k}-\bar{Q}_k\bigr)^2}_{\sigma_{\text{drift},k}^2\ \text{(across-step)}}
$$

with the per-step mean and population variance

$$
\mu_{i,k} \;=\; \frac{1}{n}\sum_{j} Q_{ij}^{(k)},
\qquad
\sigma_{i,k}^2 \;=\; \frac{1}{n}\sum_{j}\bigl(Q_{ij}^{(k)} - \mu_{i,k}\bigr)^2.
$$

The two pieces correspond to the columns reported in the cell 4 charge-summary
table:

| Column        | Symbol                                  | Meaning                       |
| ------------- | --------------------------------------- | ----------------------------- |
| `shot CV %`   | $\sigma_{\text{shot},k}\,/\,\bar{Q}_k$  | within-step shot-to-shot RMS  |
| `step CV %`   | $\sigma_{\text{drift},k}\,/\,\bar{Q}_k$ | across-step slow drift in $Q$ |

and the plotted bar satisfies
$\sigma_{Q,k} = \sqrt{\sigma_{\text{shot},k}^2 + \sigma_{\text{drift},k}^2}$.

## Why population $\sigma$, not standard error of the mean

`Q_xerr` is intentionally **not** a SEM. Using
$\sigma_{Q,k}/\sqrt{N_k}\;\approx\;6\%/\sqrt{1600}\;\approx\;0.15\%$
would make the bar invisible. Reporting the population $\sigma$ instead
makes the bar mean

> "any individual shot contributing to this scan landed at $Q$ within
> $\pm\sigma_{Q,k}$ of the marker,"

which is the relevant quantity for interpreting the per-shot scatter of the
BPD signal that the vertical bar represents (see
[`amp_yerr`, `p2p_yerr`, `integ_yerr`](#) in cell 5a).

## Dropout handling is consistent

`Q_mean` and `Q_xerr` both use `np.nan{mean,std}` over `per_shot_kept`, so
they summarise the **same** population of kept shots. The x-position and the
x-bar are therefore symmetric in their treatment of bad shots.

## Related alternative summaries (available in `rec['ict']`)

| Quantity                            | Code                                | Meaning                                                 |
| ----------------------------------- | ----------------------------------- | ------------------------------------------------------- |
| `Q_xerr[k]` (plotted)               | `np.nanstd(per_shot_kept.ravel())`  | $\sigma_{Q,k}$ — pooled population σ over kept shots    |
| Pure shot σ                         | `rec['ict']['shot_std_mean']`       | $\sigma_{\text{shot},k}$ — drift removed                |
| Drift σ                             | `rec['ict']['per_step'].std()`      | $\sigma_{\text{drift},k}$ — across-step only (cell 4d)  |
| SEM on mean charge                  | `Q_xerr[k] / np.sqrt(N_k)`          | uncertainty on $\bar{Q}_k$ itself                       |
