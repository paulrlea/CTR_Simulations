#!/usr/bin/env python
"""
Golay aperture transmission vs frequency, for several aperture radii.

The Golay cell reads TIME-INTEGRATED power over its circular input aperture (radius R).
For a given frequency f, the fraction of the detector-plane power that makes it through
the aperture is the purely GEOMETRIC aperture transmission

    T(f, R) = [ I(f) integrated over r <= R ] / [ I(f) integrated over the whole plane ].

The per-frequency channel response |t_chan(f)|^2 and bunch form factor |F(f)|^2 Q^2 are
UNIFORM scalars over the detector plane, so they cancel in this ratio -- T(f,R) depends
only on the beam geometry (CTR ring size / divergence) and the aperture, not on the bunch
or the window/BS losses.  Because the CTR ring radius grows toward low frequency (larger
diffraction/divergence), a fixed aperture passes less of the low-f power: the aperture acts
as a high-pass-like spatial filter.  This script sweeps the beamline once and plots
T(f,R) for R = 5, 7.5, 10 mm.

Run inside the SRW venv (has srwpy; repo-root .venv, python3.13):
    source ../../.venv/bin/activate     # Suli2026/.venv  (from ctr_interferometer/)
    python aperture_freq_response.py                 # dense<60GHz + log to 400 (default)
    python aperture_freq_response.py --df-dense 0.5  # even finer around the 35 GHz notch

The sub-40 GHz band needs care: (1) the GF source mesh floor is raised (gf_min_ang=96) so
the low-f detector plane is properly sampled -- the old default (32) left the aperture only
~1 pixel in radius and gave noisy/unconverged low-f values; (2) the frequency grid is dense
(1 GHz) below f_dense_max so the REAL, narrow on-axis diffraction notch near 35 GHz (an
Arago/Poisson-type null from the hard circular-aperture stack + lens) is rendered smoothly
instead of as a jagged single-point spike.
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SRW_THZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "srw_thz")
if _SRW_THZ not in sys.path:
    sys.path.insert(0, _SRW_THZ)
from thz_extract import FreqResult, intensity_2d, resample_to_grid   # noqa: E402
from srwpy.srwlib import srwl                                        # noqa: E402

from interf_config import default_interf_config                      # noqa: E402
from interf_source import make_source_conv                           # noqa: E402
from interf_beamline import (build_interf_beamline,                  # noqa: E402
                             channel_response_interf, detector_z_m)
import interf_golay as golay                                         # noqa: E402

# aperture RADII to compare [m]
RADII_M = [0.005, 0.0075, 0.010]
COLORS = ['#2c6fbb', '#d9a521', '#d12f2f']


def build_freqs_GHz(args):
    """Hybrid frequency grid: FINE below f_dense_max (to resolve the ~35 GHz diffraction
    notch from the hard-aperture stack) then LOG-spaced up to fmax (smooth high-f roll-on).
    A single coarse grid renders the narrow low-f notch as a jagged single-point spike."""
    fmin, fdense, fmax = args.fmin, args.f_dense_max, args.fmax
    lo = np.arange(fmin, fdense, args.df_dense)
    hi = np.logspace(np.log10(fdense), np.log10(fmax),
                     max(2, args.n_hi))
    return np.unique(np.round(np.concatenate([lo, hi]), 6))


def parse_args():
    p = argparse.ArgumentParser(description="Golay aperture transmission vs frequency")
    p.add_argument('--fmin', type=float, default=10.0, help='min frequency [GHz]')
    p.add_argument('--fmax', type=float, default=400.0, help='max frequency [GHz]')
    p.add_argument('--f-dense-max', type=float, default=60.0,
                   help='upper edge of the fine (1 GHz) low-f band [GHz]')
    p.add_argument('--df-dense', type=float, default=1.0,
                   help='step in the fine low-f band [GHz] (resolves the ~35 GHz notch)')
    p.add_argument('--n-hi', type=int, default=60,
                   help='number of log-spaced points from f_dense_max to fmax')
    p.add_argument('--gf-min-ang', type=int, default=96,
                   help='source-mesh floor (32 default under-samples low f; 96 is converged)')
    p.add_argument('--out', default='results_aperture_freq', help='output directory')
    return p.parse_args()


def main():
    args = parse_args()
    # gf_min_ang raises ONLY the low-f source resolution (high f already exceeds it), so the
    # >~60 GHz curve is unchanged while the sub-40 GHz values become converged (not noisy).
    cfg = default_interf_config(gf_min_ang=args.gf_min_ang,
                                f_min_GHz=args.fmin, f_max_GHz=args.fmax)
    os.makedirs(args.out, exist_ok=True)

    freqs = build_freqs_GHz(args) * 1e9
    print(f"Aperture freq-response sweep: {len(freqs)} freqs "
          f"{freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz; detector z={detector_z_m(cfg):.2f} m; "
          f"radii={[r*1e3 for r in RADII_M]} mm")

    # --- propagate the beamline once, keeping the per-frequency detector-plane maps ---
    records = []
    t0 = time.time()
    for i, f in enumerate(freqs):
        try:
            wfr = make_source_conv(f, cfg)
            optBL, _ = build_interf_beamline(cfg, capped=True)
            srwl.PropagElecField(wfr, optBL)
            ch = channel_response_interf(f, cfg)
            I2d, x, y = intensity_2d(wfr)
            Ig, _, _ = resample_to_grid(I2d, x, y, cfg)
            # channel weighting is a uniform scalar -> cancels in T(f,R); keep it for
            # consistency with run_interf's records (harmless to the ratio).
            records.append(FreqResult(f, Ig * abs(ch) ** 2, 0j, 0j, src_mode="ctr"))
            del wfr, I2d, x, y, Ig
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  |t_chan|={abs(ch):.3f}")
        except Exception as e:
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  FAILED: {e}")

    if not records:
        print("No frequencies succeeded; aborting."); sys.exit(1)
    print(f"Propagation done ({time.time()-t0:.1f}s, {len(records)}/{len(freqs)} ok).")

    # --- geometric aperture transmission T(f,R) per frequency ---
    _, _, RR, dcell = golay._grid(cfg)
    a = dcell * dcell
    f_GHz = np.array([r.f_Hz / 1e9 for r in records])
    P_total = np.array([float(r.I_grid.sum()) * a for r in records])
    P_total = np.where(P_total > 0, P_total, np.nan)

    T = {}
    for R in RADII_M:
        Pap = golay.per_freq_aperture_power(records, cfg, R)   # sum over r<=R * a
        T[R] = Pap / P_total

    # --- save data ---
    cols = [f_GHz] + [T[R] for R in RADII_M]
    hdr = 'f[GHz]  ' + '  '.join(f'T_r={R*1e3:g}mm' for R in RADII_M)
    np.savetxt(os.path.join(args.out, 'aperture_freq_response.txt'),
               np.column_stack(cols), header=hdr)
    np.savez(os.path.join(args.out, 'aperture_freq_response.npz'),
             f_GHz=f_GHz, radii_m=np.array(RADII_M), T=np.array([T[R] for R in RADII_M]))

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for R, c in zip(RADII_M, COLORS):
        ax.plot(f_GHz, 100.0 * T[R], lw=1.9, color=c,
                label=f'r = {R*1e3:g} mm  (⌀ {2*R*1e3:g} mm)')
    ax.set_xlabel('frequency  f  [GHz]')
    ax.set_ylabel('aperture transmission  T(f)  [%]')
    ax.set_title('Golay aperture transmission vs frequency (CTR beam)')
    ax.set_ylim(0, 102)
    ax.set_xlim(f_GHz.min(), f_GHz.max())
    ax.grid(alpha=0.3)
    # mark the on-axis diffraction notch (hard-aperture stack + lens) -- a real model
    # feature: a LOCAL dip inside the low-f roll-on (NOT the fmin floor where T->0), found as
    # the largest-aperture curve's minimum in a mid band bracketed by its rising edges.
    band = (f_GHz >= 22.0) & (f_GHz <= 55.0)
    if band.any():
        i_notch = np.nanargmin(np.where(band, T[RADII_M[-1]], np.nan))
        f_notch = f_GHz[i_notch]
        ax.axvspan(f_notch - 3, f_notch + 3, color='0.6', alpha=0.12, zorder=0)
        ax.annotate(f'on-axis diffraction\nnotch ~{f_notch:.0f} GHz\n(hard apertures)',
                    xy=(f_notch, 6), xytext=(f_notch + 28, 34),
                    fontsize=7.5, color='0.35', ha='left',
                    arrowprops=dict(arrowstyle='->', color='0.5', lw=0.9))
    ax.legend(title='aperture radius', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'aperture_freq_response.png'), dpi=140)
    plt.close(fig)

    print(f"Wrote outputs to {os.path.abspath(args.out)}/")
    for R in RADII_M:
        tv = T[R]
        print(f"  r={R*1e3:4g} mm: T range {np.nanmin(tv)*100:.1f}-{np.nanmax(tv)*100:.1f}% "
              f"(@ {f_GHz[np.nanargmin(tv)]:.0f} / {f_GHz[np.nanargmax(tv)]:.0f} GHz)")


if __name__ == '__main__':
    main()
