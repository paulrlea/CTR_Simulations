#!/usr/bin/env python
"""
CTR detector-plane pattern with NO apertures in the beamline.

Propagates the Ginzburg-Frank CTR source (same as run_interf) through the interferometer
path with EVERY clear-aperture stop removed (window / fold / BS / retro / OAP) -- only the
drifts and the focusing OAP lens remain (build_interf_beamline(..., apertures=False)).  The
beam is therefore never trimmed, so the detector plane (where the Golay cell sits, z ~ 0.94 m)
carries the FULL, unclipped CTR pattern.

This is the reference pattern for the aperture study: the Golay aperture-transmission curve
(aperture_freq_response.py) is the fraction of THIS pattern that a finite aperture passes.
The circular on-axis null + radial-polarization ring of coherent transition radiation is the
key feature; low frequencies form larger rings (wider wings), high frequencies a tight core.

Outputs (results_pattern_no_aperture/):
    pattern_2d.png        broadband detector intensity (linear + log), Golay radii overlaid
    pattern_freqs.png     representative single-frequency rings (shrink with frequency)
    pattern_radial.png    azimuthal-average profile + encircled-power fraction vs radius
    pattern.npz           broadband map, axes, radial profile, encircled fraction

Run inside the SRW venv (repo-root .venv, python3.13):
    source ../../.venv/bin/activate     # Suli2026/.venv  (from ctr_interferometer/)
    python ctr_pattern_no_aperture.py
    python ctr_pattern_no_aperture.py --nfreq 121 --fmax 400
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_SRW_THZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "srw_thz")
if _SRW_THZ not in sys.path:
    sys.path.insert(0, _SRW_THZ)
from thz_source import form_factor                              # noqa: E402
from thz_extract import FreqResult, intensity_2d, resample_to_grid  # noqa: E402
from srwpy.srwlib import srwl                                   # noqa: E402

from interf_config import default_interf_config                 # noqa: E402
from interf_source import make_source_conv                      # noqa: E402
from interf_beamline import (build_interf_beamline,             # noqa: E402
                             channel_response_interf, detector_z_m)
import interf_golay as golay                                    # noqa: E402

# Golay aperture radii to overlay for reference [m] (matches aperture_freq_response.py)
APER_RADII_M = [0.005, 0.0075, 0.010]
FREQ_PANELS_GHz = [50.0, 150.0, 300.0]


def parse_args():
    p = argparse.ArgumentParser(description="Unclipped CTR detector-plane pattern (no apertures)")
    p.add_argument('--nfreq', type=int, default=97, help='number of log-spaced sweep frequencies')
    p.add_argument('--fmin', type=float, default=10.0, help='min frequency [GHz]')
    p.add_argument('--fmax', type=float, default=400.0, help='max frequency [GHz]')
    p.add_argument('--gf-min-ang', type=int, default=96,
                   help='source-mesh floor (32 under-samples low f; 96 converged)')
    p.add_argument('--view-mm', type=float, default=20.0,
                   help='half-extent of the plotted map view [mm] (data saved to full extent)')
    p.add_argument('--out', default='results_pattern_no_aperture', help='output directory')
    return p.parse_args()


def azimuthal_average(Itot, RR, nb):
    """Azimuthally-averaged radial profile and its bin-edge radii [m]."""
    rmax = RR.max()
    dr = rmax / nb
    idx = np.minimum((RR.ravel() / dr).astype(int), nb - 1)
    num = np.bincount(idx, Itot.ravel(), nb)
    cnt = np.maximum(np.bincount(idx, None, nb), 1)
    r_mid = (np.arange(nb) + 0.5) * dr
    return r_mid, num / cnt


def main():
    args = parse_args()
    cfg = default_interf_config(gf_min_ang=args.gf_min_ang, n_freq=args.nfreq,
                                f_min_GHz=args.fmin, f_max_GHz=args.fmax)
    os.makedirs(args.out, exist_ok=True)

    freqs = cfg.freq_grid_Hz()
    print(f"CTR pattern, NO apertures: {len(freqs)} freqs {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz; "
          f"detector z={detector_z_m(cfg):.2f} m")

    # ---- propagate the aperture-free beamline, keep per-frequency detector maps ----
    records = []
    t0 = time.time()
    for i, f in enumerate(freqs):
        try:
            wfr = make_source_conv(f, cfg)
            optBL, _ = build_interf_beamline(cfg, capped=True, apertures=False)
            srwl.PropagElecField(wfr, optBL)
            ch = channel_response_interf(f, cfg)
            I2d, x, y = intensity_2d(wfr)
            Ig, _, _ = resample_to_grid(I2d, x, y, cfg)
            records.append(FreqResult(f, Ig * abs(ch) ** 2, 0j, 0j, src_mode="ctr"))
            del wfr, I2d, x, y, Ig
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  |t_chan|={abs(ch):.3f}")
        except Exception as e:
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  FAILED: {e}")
    if not records:
        print("No frequencies succeeded; aborting."); sys.exit(1)
    print(f"Propagation done ({time.time()-t0:.1f}s, {len(records)}/{len(freqs)} ok).")

    # ---- broadband (|F|^2 Q^2-weighted) detector-plane pattern ----
    xc, yc, RR, dcell = golay._grid(cfg)
    Itot = golay.broadband_map(records, cfg)
    r_ring = golay.ring_radius_m(records, cfg)
    a = dcell * dcell

    # radial profile + encircled-power fraction
    nb = max(64, cfg.det_npix // 2)
    r_mid, prof = azimuthal_average(Itot, RR, nb)
    order = np.argsort(RR.ravel())
    cum = np.cumsum(Itot.ravel()[order]) * a
    enc_r = RR.ravel()[order]
    enc_frac = cum / cum[-1] if cum[-1] > 0 else cum
    frac_at = {R: float(np.interp(R, enc_r, enc_frac)) for R in APER_RADII_M}

    np.savez(os.path.join(args.out, 'pattern.npz'),
             I=Itot, x=xc, y=yc, r_mid=r_mid, prof=prof,
             enc_r=enc_r[::37], enc_frac=enc_frac[::37],   # decimate the CDF for storage
             ring_radius_m=r_ring, aper_radii_m=np.array(APER_RADII_M))

    view = args.view_mm if args.view_mm else cfg.det_half_extent_m * 1e3
    ext = [xc[0] * 1e3, xc[-1] * 1e3, yc[0] * 1e3, yc[-1] * 1e3]

    def overlay_circles(ax):
        th = np.linspace(0, 2 * np.pi, 200)
        for R, c in zip(APER_RADII_M, ['#3fd0ff', '#ffd23f', '#ff5c5c']):
            ax.plot(R * 1e3 * np.cos(th), R * 1e3 * np.sin(th), color=c, lw=1.3,
                    label=f'r={R*1e3:g} mm ({frac_at[R]*100:.0f}%)')
        ax.plot(r_ring * 1e3 * np.cos(th), r_ring * 1e3 * np.sin(th),
                color='w', lw=0.9, ls='--', alpha=0.7, label=f'ring r={r_ring*1e3:.1f} mm')

    # ---- Figure A: broadband 2D map, linear + log ----
    figA, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    Inorm = Itot / (Itot.max() if Itot.max() > 0 else 1)
    im1 = a1.imshow(Inorm, origin='lower', extent=ext, aspect='equal', cmap='inferno')
    a1.set_title('broadband CTR pattern — linear')
    figA.colorbar(im1, ax=a1, label='intensity [norm]')
    floor = max(Inorm[Inorm > 0].min(), 1e-4) if np.any(Inorm > 0) else 1e-4
    im2 = a2.imshow(np.clip(Inorm, floor, 1), origin='lower', extent=ext, aspect='equal',
                    cmap='inferno', norm=LogNorm(vmin=floor, vmax=1))
    a2.set_title('broadband CTR pattern — log (wings)')
    figA.colorbar(im2, ax=a2, label='intensity [norm, log]')
    a2.text(0.5, -0.16, 'axis-aligned cross / faint grid = square computational-mesh edge '
            'diffraction\n(numerical, no circular stops); physical CTR is the central ring',
            transform=a2.transAxes, fontsize=6.8, color='0.4', ha='center', va='top')
    for ax in (a1, a2):
        overlay_circles(ax)
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        ax.set_xlim(-view, view); ax.set_ylim(-view, view)
    a1.legend(fontsize=6.5, loc='upper right', framealpha=0.5,
              title='Golay aperture (encircled %)', title_fontsize=6.5)
    figA.suptitle('CTR intensity at the Golay detector plane — NO apertures in beamline', y=1.0)
    figA.tight_layout(); figA.savefig(os.path.join(args.out, 'pattern_2d.png'), dpi=140)
    plt.close(figA)

    # ---- Figure B: representative single-frequency rings ----
    fset = [min(records, key=lambda r, ft=ft: abs(r.f_Hz - ft * 1e9)) for ft in FREQ_PANELS_GHz]
    figB, axes = plt.subplots(1, len(fset), figsize=(4.2 * len(fset), 4.2))
    if len(fset) == 1:
        axes = [axes]
    for ax, rec in zip(axes, fset):
        In = rec.I_grid / (rec.I_grid.max() if rec.I_grid.max() > 0 else 1)
        im = ax.imshow(In, origin='lower', extent=ext, aspect='equal', cmap='inferno')
        th = np.linspace(0, 2 * np.pi, 200)
        for R, c in zip(APER_RADII_M, ['#3fd0ff', '#ffd23f', '#ff5c5c']):
            ax.plot(R * 1e3 * np.cos(th), R * 1e3 * np.sin(th), color=c, lw=1.1)
        ax.set_title(f'{rec.f_Hz/1e9:.0f} GHz')
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        ax.set_xlim(-view, view); ax.set_ylim(-view, view)
        figB.colorbar(im, ax=ax, fraction=0.046, label='I [norm]')
    figB.suptitle('CTR single-frequency rings at the detector plane (no apertures) — '
                  'rings shrink with frequency; circles = Golay r = 5 / 7.5 / 10 mm', y=1.0)
    figB.tight_layout(); figB.savefig(os.path.join(args.out, 'pattern_freqs.png'), dpi=140)
    plt.close(figB)

    # ---- Figure C: radial profile + encircled-power fraction ----
    figC, axL = plt.subplots(figsize=(7.2, 4.2))
    axL.plot(r_mid * 1e3, prof / (prof.max() if prof.max() > 0 else 1),
             color='#2c6fbb', lw=1.8, label='azimuthal-avg intensity [norm]')
    axL.set_xlabel('radius  r  [mm]'); axL.set_ylabel('azimuthal-avg intensity [norm]',
                                                       color='#2c6fbb')
    axL.tick_params(axis='y', labelcolor='#2c6fbb')
    r_view = max(view, 25.0)
    axL.set_xlim(0, r_view); axL.set_ylim(0, 1.05); axL.grid(alpha=0.3)
    axR = axL.twinx()
    axR.plot(enc_r * 1e3, enc_frac * 100, color='#d12f2f', lw=1.8,
             label='encircled power [%]')
    axR.set_ylabel('encircled power [%]', color='#d12f2f')
    axR.tick_params(axis='y', labelcolor='#d12f2f'); axR.set_ylim(0, 102)
    for k, (R, c) in enumerate(zip(APER_RADII_M, ['#3fb0ff', '#c99a10', '#d12f2f'])):
        axL.axvline(R * 1e3, color=c, ls=':', lw=1.2)
        axL.text(R * 1e3, 0.90 - 0.10 * k, f' {frac_at[R]*100:.0f}% @ r={R*1e3:g}mm',
                 fontsize=7.5, color=c, ha='left', va='center')
    axL.axvline(r_ring * 1e3, color='0.4', ls='--', lw=1.0)
    figC.suptitle('CTR radial profile + encircled power at the detector plane (no apertures)')
    figC.tight_layout(); figC.savefig(os.path.join(args.out, 'pattern_radial.png'), dpi=140)
    plt.close(figC)

    print(f"Wrote outputs to {os.path.abspath(args.out)}/")
    print(f"  broadband ring radius = {r_ring*1e3:.2f} mm")
    for R in APER_RADII_M:
        print(f"  encircled power within r={R*1e3:4g} mm : {frac_at[R]*100:5.1f}%")


if __name__ == '__main__':
    main()
