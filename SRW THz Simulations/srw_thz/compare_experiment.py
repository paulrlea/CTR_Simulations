#!/usr/bin/env python
"""
Compare the simulated on-crystal THz waveform (results_l3_spacetime, the field behind
waveform_active.png) against the measured EO-sampled waveform from the AWA TFLN detector.

Simulation  : E_c(t) = active-area mean of the reconstructed detector cube (spacetime.npz).
Experiment  : signed_amp(delay) from the balanced-photodetector EO scan, exported by
              THz_Sensor_Characterization.ipynb to Results/combined_csvs/<Q>.csv.

Both are RELATIVE amplitudes (sim = arb SRW units; exp = mV before EO calibration), so the
comparison is of SHAPE: each is peak-normalised and time-aligned to its main transient.
The experimental sign and delay-origin are arbitrary, so the sign/flip that best matches
the simulation is chosen for display and reported honestly.

Caveats (printed on the figure):
  * The srw_thz model uses a SAPPHIRE window (etalon FSR 24 GHz, 42 ps echo); the measured
    setup ALSO uses a SAPPHIRE window, so material/dispersion match and the etalon fine
    structure is comparable. Any residual echo-spacing difference reflects window THICKNESS.
  * Sim is the Level-3 (30 deg) CTR source through best-guess optics; not a calibrated match.

Usage:  python compare_experiment.py [--charge 10nC]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIM_NPZ = '/home/paulrleaa/Documents/1_Research/Suli2026/srw_thz/results_l3_spacetime/spacetime.npz'
EXP_DIR = ('/home/paulrleaa/Documents/1_Research/Suli2026/SULI_THZ_SUM26/.claude/worktrees/'
           'unified-thz-analysis/Results/combined_csvs')
ACTIVE_R_M = 0.005          # EO active-area radius used in the sim (waveform_active)


def load_sim():
    d = np.load(SIM_NPZ)
    Ec, x, y, t = d['Ec'], d['x'], d['y'], d['t_ps']
    XX, YY = np.meshgrid(x, y)
    mask = (XX**2 + YY**2) <= ACTIVE_R_M**2
    wf = np.array([fr[mask].mean() for fr in Ec])      # active-area mean vs time
    return t.astype(float), wf.astype(float)


def load_exp(charge):
    path = os.path.join(EXP_DIR, f'{charge}.csv')
    d = np.genfromtxt(path, delimiter=',', names=True)
    return d['delay_ps'].astype(float), d[d.dtype.names[1]].astype(float)   # delay_ps, signed_amp_mV


def center_norm(t, y):
    """Peak-|.|-align to t=0 and normalise to unit peak."""
    y = y - np.median(y)                    # remove any DC/baseline offset
    i = int(np.argmax(np.abs(y)))
    return t - t[i], y / np.abs(y[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charge', default='10nC', help='experimental scan CSV stem (e.g. 10nC)')
    ap.add_argument('--out', default=SIM_NPZ.replace('spacetime.npz', 'sim_vs_experiment.png'))
    a = ap.parse_args()

    ts, ys = center_norm(*load_sim())
    te, ye = center_norm(*load_exp(a.charge))

    # choose experimental sign to best match the sim over the common window (arbitrary EO sign)
    lo, hi = max(ts.min(), te.min()), min(ts.max(), te.max())
    grid = np.linspace(lo, hi, 400)
    si = np.interp(grid, ts, ys)
    ei = np.interp(grid, te, ye)
    r_pos = np.corrcoef(si, ei)[0, 1]
    r_neg = np.corrcoef(si, -ei)[0, 1]
    flip = r_neg > r_pos
    if flip:
        ye = -ye; ei = -ei
    r = max(r_pos, r_neg)

    # robust main-transient width: time between the max and min excursions near the peak
    def swing(t, y, win=20.0):
        m = np.abs(t) <= win
        tt, yy = t[m], y[m]
        return abs(tt[np.argmax(yy)] - tt[np.argmin(yy)])
    fw_s, fw_e = swing(ts, ys), swing(te, ye)
    span_e = te.max() - te.min()

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (a) overlay on the experimental delay span (the honest apples-to-apples window)
    for A in (ax[0], ax[1]):
        A.axhline(0, color='0.8', lw=0.8); A.axvline(0, color='0.8', lw=0.8)
        A.plot(te, ye, 'o-', ms=3, lw=1.3, color='#1f6fd1',
               label=f'experiment (EO, {a.charge})')
        A.plot(ts, ys, '-', lw=2.0, color='#d12f2f', label='simulation (Level-3, active-area)')
        A.set_xlabel('t  [ps]  (relative to main peak)')
        A.set_ylabel('E (peak-normalised)'); A.grid(alpha=0.3); A.legend(fontsize=9)
    ax[0].set_xlim(-span_e/2*1.1, span_e/2*1.1)
    ax[0].set_title(f'Zoom to experimental span (±{span_e/2:.0f} ps)')
    ax[1].set_xlim(ts.min(), ts.max())
    ax[1].set_title('Full simulation window (etalon echoes visible)')

    fig.suptitle(f'Simulated vs measured on-crystal THz waveform  —  '
                 f'shape corr r = {r:.2f}{"  (exp sign flipped)" if flip else ""}',
                 fontsize=12, y=0.99)
    fig.text(0.5, -0.02,
             'Both peak-normalised & peak-aligned (relative units). Caveat: sim AND measured both '
             'use a SAPPHIRE window (material matches); residual echo-spacing differs with thickness.',
             ha='center', fontsize=8.3, color='#666')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(a.out, dpi=140, bbox_inches='tight')

    print(f'wrote {a.out}')
    print(f'experiment: {a.charge}, delay span {span_e:.1f} ps, main bipolar-swing width = {fw_e:.1f} ps')
    print(f'simulation: window ±{ts.max():.0f} ps,           main bipolar-swing width = {fw_s:.1f} ps')
    print(f'shape correlation over common window = {r:.3f}'
          f'{"  (experimental sign flipped to match)" if flip else ""}')


if __name__ == '__main__':
    main()
