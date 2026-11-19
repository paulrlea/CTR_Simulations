#!/usr/bin/env python
"""
Compare the simulated on-crystal THz waveform against the EXTREMAS-method reconstruction
of the measured EO waveform (companion to compare_experiment.py, which uses period fill).

Simulation : E_c(t) = active-area mean of the reconstructed detector cube (spacetime.npz).
Experiment : reconstructed from the balanced-photodetector EO scan exported to
             Results/combined_csvs/<Q>.csv. Three experimental extractions are compared:
   * period fill            : signed_amp_mV column (fixed-window mean; the current pipeline)
   * extremas (max-only)     : peak5_max - peak4_max  (classic extremas.py / Method A)
   * extremas (largest fluct): signed peak-5-vs-peak-4 deviation of largest magnitude,
                               = dmax if |dmax|>=|dmin| else dmin, with
                               dmax = peak5_max-peak4_max, dmin = peak5_min-peak4_min.
                               This is the extremas analogue of "capture the largest
                               fluctuation" (see THz_Sensor_Characterization / Baseline
                               method comparison, section 2e).

All are RELATIVE amplitudes; the comparison is of SHAPE — each is peak-normalised, baseline-
removed, and peak-aligned. The experimental sign/flip that best matches the sim is chosen.

Caveats (same as compare_experiment.py):
  * both sim and measured use a SAPPHIRE window (same material / dispersion / reflectivity),
    so the etalon fine structure is comparable; any residual difference in echo spacing
    reflects window THICKNESS (sim FSR 24 GHz, 42 ps echo), not material.
  * sim = Level-3 (30 deg) CTR source through best-guess optics; not a calibrated match.

Usage:  python compare_experiment_extremas.py [--charge 10nC]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# resolve paths relative to this script so it works regardless of CWD / where srw_thz lives
_HERE = os.path.dirname(os.path.abspath(__file__))
SIM_NPZ = os.path.join(_HERE, 'results_l3_spacetime', 'spacetime.npz')
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
    """Return delay_ps and a dict of the three experimental extractions (mV)."""
    d = np.genfromtxt(os.path.join(EXP_DIR, f'{charge}.csv'), delimiter=',', names=True)
    n = d.dtype.names
    delay = d['delay_ps'].astype(float)
    signed = d[n[1]].astype(float)                                  # period fill
    p5max, p5min, p4max, p4min = (d[n[2]], d[n[3]], d[n[4]], d[n[5]])
    dmax = (p5max - p4max).astype(float)                            # extremas max-only
    dmin = (p5min - p4min).astype(float)
    largest = np.where(np.abs(dmax) >= np.abs(dmin), dmax, dmin)    # extremas largest fluctuation
    return delay, {'period fill': signed,
                   'extremas (max-only)': dmax,
                   'extremas (largest fluctuation)': largest}


def center_norm(t, y):
    """Baseline-remove, peak-|.|-align to t=0, normalise to unit peak."""
    y = y - np.median(y)
    i = int(np.argmax(np.abs(y)))
    return t - t[i], y / np.abs(y[i])


def shape_corr_and_flip(ts, ys, te, ye):
    lo, hi = max(ts.min(), te.min()), min(ts.max(), te.max())
    g = np.linspace(lo, hi, 400)
    si, ei = np.interp(g, ts, ys), np.interp(g, te, ye)
    r_pos, r_neg = np.corrcoef(si, ei)[0, 1], np.corrcoef(si, -ei)[0, 1]
    return (max(r_pos, r_neg), r_neg > r_pos)


def swing(t, y, win=20.0):
    m = np.abs(t) <= win
    tt, yy = t[m], y[m]
    return abs(tt[np.argmax(yy)] - tt[np.argmin(yy)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charge', default='10nC', help='experimental scan CSV stem (e.g. 10nC)')
    ap.add_argument('--out', default=os.path.join(_HERE, 'results_l3_spacetime',
                                                  'sim_vs_experiment_extremas.png'))
    a = ap.parse_args()

    ts, ys = center_norm(*load_sim())
    delay, exp = load_exp(a.charge)

    styles = {'period fill':                    dict(color='0.6',     lw=1.2, ls='-',  marker=''),
              'extremas (max-only)':            dict(color='#e08a1e', lw=1.4, ls='--', marker=''),
              'extremas (largest fluctuation)': dict(color='#1f6fd1', lw=1.6, ls='-',  marker='o', ms=3)}

    prepared, results = {}, {}
    for name, w in exp.items():
        te, ye = center_norm(delay, w)
        r, flip = shape_corr_and_flip(ts, ys, te, ye)
        if flip:
            ye = -ye
        prepared[name] = (te, ye)
        results[name] = (r, flip, swing(te, ye), w.min(), w.max())

    span_e = delay.max() - delay.min()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    for A in ax:
        A.axhline(0, color='0.85', lw=0.8); A.axvline(0, color='0.85', lw=0.8)
        for name, (te, ye) in prepared.items():
            A.plot(te, ye, label=f'{name}  (r={results[name][0]:.2f})', **styles[name])
        A.plot(ts, ys, '-', lw=2.4, color='#d12f2f', label='simulation (Level-3, active-area)', zorder=5)
        A.set_xlabel('t  [ps]  (relative to main peak)')
        A.set_ylabel('E (peak-normalised)'); A.grid(alpha=0.3); A.legend(fontsize=8.5)
    ax[0].set_xlim(-span_e/2*1.1, span_e/2*1.1)
    ax[0].set_title(f'Zoom to experimental span (±{span_e/2:.0f} ps)')
    ax[1].set_xlim(ts.min(), ts.max())
    ax[1].set_title('Full simulation window (etalon echoes visible)')

    fig.suptitle('Simulated vs measured on-crystal THz waveform — experiment via EXTREMAS method',
                 fontsize=12, y=0.99)
    fig.text(0.5, -0.02,
             'All peak-normalised & aligned (relative units). extremas (max-only)=peak5_max−peak4_max; '
             'largest fluctuation uses the peak-5-vs-peak-4 deviation of greater magnitude (max or min). '
             'Caveat: both sim and measured use a SAPPHIRE window (material matches); residual '
             'echo-spacing differences reflect window thickness. Compare main transient/width.',
             ha='center', fontsize=8, color='#666')
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(a.out, dpi=140, bbox_inches='tight')

    print(f'wrote {a.out}\n')
    print(f'charge {a.charge}: experimental delay span {span_e:.1f} ps')
    print(f'simulation main bipolar-swing width = {swing(ts, ys):.1f} ps\n')
    print(f'{"experiment extraction":34s} {"shape r":>8} {"flip":>5} {"swing ps":>9} {"range mV":>16}')
    for name, (r, flip, sw, lo, hi) in results.items():
        print(f'{name:34s} {r:>8.3f} {str(flip):>5} {sw:>9.1f}   [{lo:7.1f},{hi:7.1f}]')


if __name__ == '__main__':
    main()
