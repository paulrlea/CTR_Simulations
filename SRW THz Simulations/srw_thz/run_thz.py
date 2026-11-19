#!/usr/bin/env python
"""
Driver: sweep frequency, synthesize the CTR-THz source, propagate through the
beamline with SRW, and assemble the broadband spatial map + on-crystal E(t).

Run inside the project venv (PYTHONPATH-isolated):
    source ../.venv/bin/activate
    python run_thz.py --nfreq 9 --charge 10
Headless plotting is forced (Agg), so no display is needed.

Outputs land in <out_dir>/:
    spatial_map.png/.npz, waveform.png/.txt, spectrum.png/.txt, run_meta.txt
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')

from thz_config import default_config
from thz_source import make_source, form_factor
from thz_beamline import build_beamline, channel_response, detector_z_m
from thz_layout import draw_beamline
from thz_etalon import write_analysis
from thz_extract import (FreqResult, intensity_2d, onaxis_complex_field,
                         ring_peak_complex_field, crystal_axis_complex_field,
                         save_polarization_maps, resample_to_grid,
                         assemble_spatial_map, assemble_waveform,
                         detected_power_spectrum, save_spatial_map,
                         save_waveform, save_spectrum)

from srwpy.srwlib import srwl


def parse_args():
    p = argparse.ArgumentParser(description="AWA CTR-THz SRW propagation")
    p.add_argument('--src', default='gaussian', choices=['gaussian', 'ctr'])
    p.add_argument('--nfreq', type=int, default=None, help='number of sweep frequencies')
    p.add_argument('--fmin', type=float, default=None, help='min frequency [GHz]')
    p.add_argument('--fmax', type=float, default=None, help='max frequency [GHz]')
    p.add_argument('--charge', type=float, default=None, help='bunch charge [nC]')
    p.add_argument('--sigma-t', type=float, default=None, help='rms bunch length [ps]')
    p.add_argument('--div-match', type=float, default=None, help='divergence-match knob')
    p.add_argument('--sampfact', type=float, default=None, help='SRW sampling factor')
    p.add_argument('--det-npix', type=int, default=None, help='detector grid points per axis')
    p.add_argument('--max-src-pts', type=int, default=None, help='source mesh cap (memory guard)')
    p.add_argument('--gf-n-ang', type=int, default=None, help='GF source angular mesh points per axis')
    p.add_argument('--screen-tilt', type=float, default=None,
                   help='CTR screen tilt [deg] in the x-plane (GF source only; 0 = symmetric)')
    p.add_argument('--tilt-model', default=None, choices=['level1', 'level3', 'analytic'],
                   help='tilted-screen model: level1 heuristic, level3 exact oblique field, '
                        'or analytic (supplied closed-form 45-deg TR relation = level3 at 45 deg)')
    p.add_argument('--crystal-pol-axis', type=float, default=None,
                   help='EO-crystal sensitive-axis azimuth [deg] (level3 polarization extraction)')
    p.add_argument('--finite-foil', action='store_true',
                   help='apply the far-zone finite-foil diffraction form factor H(theta) to the '
                        'CTR source (ring broadening, null-fill, edge fringes). See FINITE_FOIL_PLAN.md.')
    p.add_argument('--foil-radius', type=float, default=None,
                   help='radiator (Mo foil) radius a [m] for --finite-foil (default 0.5in = 0.0127)')
    p.add_argument('--no-window', action='store_true')
    p.add_argument('--window-mode', default=None, choices=['slab', 'wedge', 'ar'],
                   help='sapphire window model: slab=etalon, wedge=no etalon, ar=anti-reflection')
    p.add_argument('--no-mirror', action='store_true', help='disable OAP1 (and OAP2)')
    p.add_argument('--no-mirror2', action='store_true', help='disable only OAP2 (single mirror)')
    p.add_argument('--out', default=None, help='output directory')
    return p.parse_args()


def build_cfg(args):
    ov = {}
    if args.nfreq is not None: ov['n_freq'] = args.nfreq
    if args.fmin is not None: ov['f_min_GHz'] = args.fmin
    if args.fmax is not None: ov['f_max_GHz'] = args.fmax
    if args.charge is not None: ov['bunch_charge_nC'] = args.charge
    if args.sigma_t is not None: ov['sigma_t_ps'] = args.sigma_t
    if args.div_match is not None: ov['div_match'] = args.div_match
    if args.sampfact is not None: ov['sampFactNxNyForProp'] = args.sampfact
    if args.det_npix is not None: ov['det_npix'] = args.det_npix
    if args.max_src_pts is not None: ov['max_src_pts'] = args.max_src_pts
    if args.gf_n_ang is not None: ov['gf_n_ang'] = args.gf_n_ang
    if args.screen_tilt is not None: ov['screen_tilt_deg'] = args.screen_tilt
    if args.tilt_model is not None: ov['gf_tilt_model'] = args.tilt_model
    if args.crystal_pol_axis is not None: ov['crystal_pol_axis_deg'] = args.crystal_pol_axis
    if args.finite_foil: ov['finite_foil_diffraction'] = True
    if args.foil_radius is not None: ov['foil_radius_m'] = args.foil_radius
    if args.no_window: ov['window_enabled'] = False
    if args.window_mode is not None: ov['window_mode'] = args.window_mode
    if args.no_mirror: ov['mirror_enabled'] = False
    if args.no_mirror2: ov['mirror2_enabled'] = False
    if args.out is not None: ov['out_dir'] = args.out
    return default_config(**ov)


def main():
    args = parse_args()
    cfg = build_cfg(args)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # beampath schematic (cheap, from config only)
    draw_beamline(cfg, os.path.join(cfg.out_dir, 'layout.png'))

    freqs = cfg.freq_grid_Hz()
    print(f"SRW CTR-THz run: src={args.src}  gamma={cfg.gamma:.1f}  "
          f"Q={cfg.bunch_charge_nC} nC  sigma_t={cfg.sigma_t_ps} ps")
    print(f"  {len(freqs)} frequencies {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz; "
          f"detector at z={detector_z_m(cfg):.2f} m")

    records = []
    t0 = time.time()
    for i, f in enumerate(freqs):
        try:
            wfr = make_source(f, cfg, args.src)
            optBL, _ = build_beamline(cfg, capped=(args.src == 'ctr'))
            srwl.PropagElecField(wfr, optBL)
            ch = channel_response(f, cfg)
            nx, ny = wfr.mesh.nx, wfr.mesh.ny
            I2d, x, y = intensity_2d(wfr)
            # Gaussian -> on-axis field; GF/CTR -> ring field (on-axis null). For the
            # Level-3 oblique source the EO crystal sees a Cartesian polarization, so
            # project onto the crystal axis; Level-1/radial uses the radial projection.
            if args.src == 'ctr':
                Eax = 0j
                if cfg.gf_tilt_model in ('level3', 'analytic') and cfg.screen_tilt_deg != 0.0:
                    Ering = crystal_axis_complex_field(wfr, cfg) * ch
                else:
                    Ering = ring_peak_complex_field(wfr, cfg) * ch
            else:
                Eax = onaxis_complex_field(wfr) * ch
                Ering = 0j
            # resample to the fixed common grid NOW and keep only that, so memory
            # does not scale with the (sometimes huge) native mesh across the sweep
            Ig, _, _ = resample_to_grid(I2d, x, y, cfg)
            records.append(FreqResult(f, Ig * abs(ch) ** 2, Eax, Ering, src_mode=args.src))
            del wfr, I2d, x, y, Ig
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  "
                  f"mesh={nx}x{ny}  F={form_factor(f, cfg.sigma_t_s):.2e}  "
                  f"|t_win|={abs(ch):.3f}")
        except Exception as e:                       # keep the sweep alive
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  FAILED: {e}")

    if not records:
        print("No frequencies succeeded; aborting."); sys.exit(1)

    print(f"Propagation done ({time.time()-t0:.1f}s, {len(records)}/{len(freqs)} ok). Assembling...")

    Ic, xc, yc = assemble_spatial_map(records, cfg)
    save_spatial_map(Ic, xc, yc, cfg,
                     os.path.join(cfg.out_dir, 'spatial_map.png'),
                     os.path.join(cfg.out_dir, 'spatial_map.npz'))

    t_ps, E_t = assemble_waveform(records, cfg)
    save_waveform(t_ps, E_t, cfg,
                  os.path.join(cfg.out_dir, 'waveform.png'),
                  os.path.join(cfg.out_dir, 'waveform.txt'))

    f_GHz, P = detected_power_spectrum(records, cfg)
    save_spectrum(f_GHz, P,
                  os.path.join(cfg.out_dir, 'spectrum.png'),
                  os.path.join(cfg.out_dir, 'spectrum.txt'))

    # quantify the sapphire-window etalon + mitigation overlay (cheap, no re-run)
    write_analysis(cfg, cfg.out_dir)

    # Level-3 polarization diagnostic maps: re-propagate the spectral-peak frequency once
    # and save |Ex|^2 (p), |Ey|^2 (s), and the polarization-angle map.
    if args.src == 'ctr' and cfg.gf_tilt_model in ('level3', 'analytic') and cfg.screen_tilt_deg != 0.0:
        f_pk = float(f_GHz[int(np.argmax(P))] * 1e9)
        try:
            wfp = make_source(f_pk, cfg, 'ctr')
            srwl.PropagElecField(wfp, build_beamline(cfg, capped=True)[0])
            save_polarization_maps(wfp, cfg, os.path.join(cfg.out_dir, 'detector'))
            print(f"  polarization maps at {f_pk/1e9:.0f} GHz -> detector_pol_*.png")
            del wfp
        except Exception as e:
            print(f"  polarization maps skipped: {e}")

    with open(os.path.join(cfg.out_dir, 'run_meta.txt'), 'w') as fh:
        fh.write(f"src={args.src}\n")
        for k in ('beam_energy_MeV', 'bunch_charge_nC', 'sigma_t_ps', 'div_match',
                  'sampFactNxNyForProp', 'window_enabled', 'mirror_enabled',
                  'mirror2_enabled', 'collimator_radius_m', 'window_clear_ap_radius_m',
                  'mirror_focal_length_m', 'mirror2_focal_length_m',
                  'L_mirror1_to_mirror2_m'):
            fh.write(f"{k}={getattr(cfg, k)}\n")
        if args.src == 'ctr':
            from thz_source import _gf_amplitude_norm
            fh.write(f"gf_n_ang={cfg.gf_n_ang}\n")
            fh.write(f"gf_n_sigma={cfg.gf_n_sigma}\n")
            fh.write(f"gf_amplitude_A={_gf_amplitude_norm(cfg):.6e}\n")
            fh.write(f"screen_tilt_deg={cfg.screen_tilt_deg}\n")
            fh.write(f"gf_tilt_model={cfg.gf_tilt_model}\n")
            if cfg.screen_tilt_deg != 0.0:
                if cfg.gf_tilt_model in ('level3', 'analytic'):
                    fh.write(f"crystal_pol_axis_deg={cfg.crystal_pol_axis_deg}\n")
                    fh.write(f"gf_include_electron_term={cfg.gf_include_electron_term}\n")
                else:
                    fh.write(f"gf_tilt_lobe_k={cfg.gf_tilt_lobe_k}\n")
            fh.write(f"finite_foil_diffraction={cfg.finite_foil_diffraction}\n")
            if cfg.finite_foil_diffraction:
                fh.write(f"foil_radius_m={cfg.foil_radius_m}\n")
        fh.write(f"detector_z_m={detector_z_m(cfg):.4f}\n")
        fh.write(f"n_ok={len(records)}/{len(freqs)}\n")

    print(f"Wrote outputs to {os.path.abspath(cfg.out_dir)}/")
    peak_i = max(range(len(P)), key=lambda k: P[k])
    print(f"  spectral peak ~ {f_GHz[peak_i]:.0f} GHz; waveform span {t_ps[-1]:.0f} ps")


if __name__ == '__main__':
    main()
