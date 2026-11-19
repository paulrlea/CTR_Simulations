#!/usr/bin/env python
"""
Driver: propagate the CTR-THz beam (SAME profile as the previous experiments -- the
Ginzburg-Frank CTR source, here convolved with a Gaussian) through the NEW interferometer
optical path, and assemble the detector deliverables.

This is a CTR-based BEAM DIAGNOSTIC (not FTIR spectroscopy).  Detector: a GOLAY CELL power
detector (default) -- it measures time-integrated, polarization-blind, phase-blind POWER
over its input aperture, so the measurement is a DELAY-STAGE SCAN yielding the CTR field
autocorrelation (interferogram).  Outputs:
    spatial_map.{png,npz}     detector intensity + Golay aperture overlay
    power_capture.{png,txt}    captured power vs aperture radius (CTR ring/null tradeoff)
    interferogram.{png,txt}    Golay P(tau) vs delay (CTR autocorrelation; + stage-travel axis)
    recovered_spectrum.{png,txt}  CTR power spectrum from the autocorrelation vs direct
    spectrum.{png,txt}, run_meta.txt, layout.png (beampath.png copy)
Use --detector eo for the previous ELECTRIC-FIELD-METER model instead (E(t) waveform;
single delay, no scan).  The SRW propagation is identical either way -- only the
detector-plane reduction differs.

Run inside the project venv (PYTHONPATH-isolated):
    source ../.venv/bin/activate
    python run_interf.py --nfreq 9                 # quick smoke test (Golay)
    python run_interf.py                           # full run (n_freq=161) -> results_interf/
    python run_interf.py --detector eo             # previous EO field-meter (E(t) waveform)
    python run_interf.py --golay-aperture 0.003    # 3 mm Golay aperture
    python run_interf.py --conv-domain intensity   # incoherent source smear (fills the null)
Headless plotting is forced (Agg).
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')

# reused, unchanged, from the previous experiment code
_SRW_THZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "srw_thz")
if _SRW_THZ not in sys.path:
    sys.path.insert(0, _SRW_THZ)
from thz_source import form_factor                                            # noqa: E402
from thz_config import C_LIGHT                                                # noqa: E402
from thz_extract import (FreqResult, intensity_2d, ring_peak_complex_field,   # noqa: E402
                         resample_to_grid, assemble_spatial_map,
                         assemble_waveform, detected_power_spectrum,
                         save_spatial_map, save_waveform, save_spectrum)
from srwpy.srwlib import srwl                                                 # noqa: E402

# new interferometer-path modules (this folder)
from interf_config import default_interf_config
from interf_source import make_source_conv
from interf_beamline import build_interf_beamline, channel_response_interf, detector_z_m
import interf_golay as golay


def parse_args():
    p = argparse.ArgumentParser(description="CTR-THz propagation through the NEW interferometer path")
    p.add_argument('--nfreq', type=int, default=None, help='number of sweep frequencies')
    p.add_argument('--fmin', type=float, default=None, help='min frequency [GHz]')
    p.add_argument('--fmax', type=float, default=None, help='max frequency [GHz]')
    p.add_argument('--charge', type=float, default=None, help='bunch charge [nC]')
    p.add_argument('--sigma-t', type=float, default=None, help='rms bunch length [ps]')
    p.add_argument('--conv-sigma', type=float, default=None,
                   help='GF Gaussian-convolution RMS at the source plane [m] (0 = pure GF)')
    p.add_argument('--conv-domain', default=None, choices=['field', 'intensity'],
                   help="convolution domain: 'field' (coherent, keeps radial-pol null) or "
                        "'intensity' (incoherent smear, fills the null)")
    p.add_argument('--tilt-model', default=None, choices=['level1', 'level3', 'analytic'],
                   help='GF source model: level1 heuristic (default), level3 exact oblique '
                        'field, or analytic (supplied closed-form 45-deg TR relation = level3 '
                        'at 45 deg). Use with --screen-tilt 45 for the analytic 45-deg source.')
    p.add_argument('--screen-tilt', type=float, default=None,
                   help='CTR screen tilt [deg] (GF source; 0 = symmetric). Set to 45 with '
                        '--tilt-model analytic to match that relation\'s baked-in 45-deg foil.')
    p.add_argument('--finite-foil', action='store_true',
                   help='apply the far-zone finite-foil diffraction form factor H(theta) to the '
                        'CTR source (ring broadening, null-fill, edge fringes; supersedes the '
                        'scalar S_foil low-f rolloff). See FINITE_FOIL_PLAN.md.')
    p.add_argument('--foil-radius', type=float, default=None,
                   help='radiator (Mo foil) radius a [m] for --finite-foil (default 0.5in = 0.0127)')
    p.add_argument('--detector', default='golay', choices=['golay', 'eo'],
                   help="detector model: 'golay' power detector (delay-scan CTR interferogram, "
                        "default) or 'eo' field meter (E(t) waveform, the previous setup)")
    p.add_argument('--golay-aperture', type=float, default=None,
                   help='Golay input aperture radius [m] (nominal; interferogram integrates this)')
    p.add_argument('--tau-max-ps', type=float, default=None,
                   help='max interferometer delay [ps] (default: resolve the frequency spacing)')
    p.add_argument('--n-tau', type=int, default=None, help='number of delay points')
    p.add_argument('--oap-f', type=float, default=None, help='OAP focal length [m] (0/omit = auto image)')
    p.add_argument('--div-match', type=float, default=None, help='CTR divergence-match knob')
    p.add_argument('--gf-n-ang', type=int, default=None, help='GF source angular mesh points/axis')
    p.add_argument('--cap-range-factor', type=float, default=None, help='per-drift mesh range growth')
    p.add_argument('--det-npix', type=int, default=None, help='detector grid points per axis')
    p.add_argument('--no-window', action='store_true')
    p.add_argument('--window-mode', default=None, choices=['slab', 'wedge', 'ar'])
    p.add_argument('--out', default=None, help='output directory')
    return p.parse_args()


def build_cfg(args):
    ov = {}
    if args.nfreq is not None: ov['n_freq'] = args.nfreq
    if args.fmin is not None: ov['f_min_GHz'] = args.fmin
    if args.fmax is not None: ov['f_max_GHz'] = args.fmax
    if args.charge is not None: ov['bunch_charge_nC'] = args.charge
    if args.sigma_t is not None: ov['sigma_t_ps'] = args.sigma_t
    if args.conv_sigma is not None: ov['gf_conv_sigma_m'] = args.conv_sigma
    if args.conv_domain is not None: ov['gf_conv_domain'] = args.conv_domain
    if args.tilt_model is not None: ov['gf_tilt_model'] = args.tilt_model
    if args.screen_tilt is not None: ov['screen_tilt_deg'] = args.screen_tilt
    if args.finite_foil: ov['finite_foil_diffraction'] = True
    if args.foil_radius is not None: ov['foil_radius_m'] = args.foil_radius
    if args.golay_aperture is not None: ov['det_active_radius_m'] = args.golay_aperture
    if args.oap_f is not None: ov['oap_focal_length_m'] = args.oap_f
    if args.div_match is not None: ov['div_match'] = args.div_match
    if args.gf_n_ang is not None: ov['gf_n_ang'] = args.gf_n_ang
    if args.cap_range_factor is not None: ov['cap_range_factor'] = args.cap_range_factor
    if args.det_npix is not None: ov['det_npix'] = args.det_npix
    if args.no_window: ov['window_enabled'] = False
    if args.window_mode is not None: ov['window_mode'] = args.window_mode
    if args.out is not None: ov['out_dir'] = args.out
    return default_interf_config(**ov)


def main():
    args = parse_args()
    cfg = build_cfg(args)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # layout: copy the beampath schematic (drawn by draw_interferometer.py) if present
    here = os.path.dirname(os.path.abspath(__file__))
    src_png = os.path.join(here, 'beampath.png')
    if os.path.exists(src_png):
        shutil.copyfile(src_png, os.path.join(cfg.out_dir, 'layout.png'))

    freqs = cfg.freq_grid_Hz()
    print(f"CTR-THz -> NEW interferometer path:  detector={args.detector}  "
          f"src=GF(*)Gaussian(sigma={cfg.gf_conv_sigma_m*1e3:.2f}mm, {cfg.gf_conv_domain})  "
          f"tilt={cfg.gf_tilt_model}@{cfg.screen_tilt_deg:g}deg  "
          f"finite_foil={cfg.finite_foil_diffraction}"
          f"{f'(a={cfg.foil_radius_m*1e3:.1f}mm)' if cfg.finite_foil_diffraction else ''}  "
          f"gamma={cfg.gamma:.1f}  Q={cfg.bunch_charge_nC} nC  sigma_t={cfg.sigma_t_ps} ps")
    print(f"  {len(freqs)} frequencies {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz;  "
          f"OAP f={cfg.oap_focal_length_m:.3f} m;  detector at z={detector_z_m(cfg):.2f} m")

    records = []
    t0 = time.time()
    for i, f in enumerate(freqs):
        try:
            wfr = make_source_conv(f, cfg)
            optBL, _ = build_interf_beamline(cfg, capped=True)
            srwl.PropagElecField(wfr, optBL)
            ch = channel_response_interf(f, cfg)
            nx, ny = wfr.mesh.nx, wfr.mesh.ny
            I2d, x, y = intensity_2d(wfr)
            # Golay measures total power (both pols) integrated over its aperture -> only the
            # intensity map is needed.  EO field meter also needs the ring complex field.
            Ering = (ring_peak_complex_field(wfr, cfg) * ch) if args.detector == 'eo' else 0j
            Ig, _, _ = resample_to_grid(I2d, x, y, cfg)
            records.append(FreqResult(f, Ig * abs(ch) ** 2, 0j, Ering, src_mode="ctr"))
            del wfr, I2d, x, y, Ig
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  mesh={nx}x{ny}  "
                  f"F={form_factor(f, cfg.sigma_t_s):.2e}  |t_chan|={abs(ch):.3f}")
        except Exception as e:                                     # keep the sweep alive
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  FAILED: {e}")

    if not records:
        print("No frequencies succeeded; aborting."); sys.exit(1)
    print(f"Propagation done ({time.time()-t0:.1f}s, {len(records)}/{len(freqs)} ok). Assembling...")

    od = cfg.out_dir
    if args.detector == 'golay':
        # -------- Golay power detector: CTR autocorrelation interferogram + aperture study --------
        R_nom = cfg.det_active_radius_m
        r_ring = golay.ring_radius_m(records, cfg)

        # detector intensity map with the Golay aperture overlaid (what it integrates)
        golay.save_spatial_map_golay(records, cfg, R_nom, r_ring,
                                     os.path.join(od, 'spatial_map.png'),
                                     os.path.join(od, 'spatial_map.npz'))

        # power captured vs aperture radius (the CTR ring/null tradeoff)
        radii = np.linspace(0.0, 0.020, 81)
        radii, frac, total = golay.power_capture_curve(records, cfg, radii)
        golay.save_power_capture(radii, frac, cfg, r_ring, R_nom,
                                 os.path.join(od, 'power_capture.png'),
                                 os.path.join(od, 'power_capture.txt'))
        cap_nom = float(np.interp(R_nom, radii, frac))

        # detected spectrum at the nominal aperture (direct)
        f_GHz, S = golay.detected_spectrum(records, cfg, R_nom)
        save_spectrum(f_GHz, S, os.path.join(od, 'spectrum.png'),
                      os.path.join(od, 'spectrum.txt'))

        # delay-scanned interferogram P(tau) = CTR field autocorrelation, and the CTR
        # power spectrum recovered from it (cosine transform; Wiener-Khinchin)
        taus = golay.delay_grid(cfg, freqs, args.tau_max_ps, args.n_tau)
        P_tau = golay.interferogram(records, cfg, R_nom, taus)
        tau_et_ps = 2.0 * cfg.window_n * cfg.window_thickness_m / C_LIGHT * 1e12  # window recurrence
        # arm-length asymmetry: ZPD is reached when the delay stage is displaced by
        # dL = L_ref - L_bs_to_delay (reference arm longer than the delay arm's nominal).
        zpd_stage_offset_m = cfg.L_bs_to_reference_m - cfg.L_bs_to_delay_m
        golay.save_interferogram(taus, P_tau, cfg,
                                 os.path.join(od, 'interferogram.png'),
                                 os.path.join(od, 'interferogram.txt'),
                                 etalon_tau_ps=tau_et_ps,
                                 stage_zero_offset_m=zpd_stage_offset_m)
        S_rec = golay.recover_spectrum(taus, P_tau, f_GHz * 1e9)
        golay.save_recovered_spectrum(f_GHz, S, S_rec,
                                      os.path.join(od, 'recovered_spectrum.png'),
                                      os.path.join(od, 'recovered_spectrum.txt'))
        P = S  # for the summary/peak print below
    else:
        # -------- EO field meter (previous setup): E(t) waveform --------
        Ic, xc, yc = assemble_spatial_map(records, cfg)
        save_spatial_map(Ic, xc, yc, cfg,
                         os.path.join(od, 'spatial_map.png'),
                         os.path.join(od, 'spatial_map.npz'))
        t_ps, E_t = assemble_waveform(records, cfg)
        save_waveform(t_ps, E_t, cfg,
                      os.path.join(od, 'waveform.png'),
                      os.path.join(od, 'waveform.txt'))
        f_GHz, P = detected_power_spectrum(records, cfg)
        save_spectrum(f_GHz, P, os.path.join(od, 'spectrum.png'),
                      os.path.join(od, 'spectrum.txt'))

    with open(os.path.join(cfg.out_dir, 'run_meta.txt'), 'w') as fh:
        if args.detector == 'golay':
            fh.write("path=ctr_interferometer (NEW), detector=golay (power), "
                     "scope=CTR interferometric autocorrelation (beam diagnostic)\n")
        else:
            fh.write("path=ctr_interferometer (NEW), detector=eo (field), scope=propagation-only, delay=0\n")
        fh.write("source=ginzburg-frank_convolved_gaussian\n")
        fh.write(f"gf_tilt_model={cfg.gf_tilt_model}\n")
        fh.write(f"screen_tilt_deg={cfg.screen_tilt_deg}\n")
        if cfg.gf_tilt_model in ('level3', 'analytic') and cfg.screen_tilt_deg != 0.0:
            fh.write(f"crystal_pol_axis_deg={cfg.crystal_pol_axis_deg}\n")
            fh.write(f"gf_include_electron_term={cfg.gf_include_electron_term}\n")
        fh.write(f"finite_foil_diffraction={cfg.finite_foil_diffraction}\n")
        if cfg.finite_foil_diffraction:
            fh.write(f"foil_radius_m={cfg.foil_radius_m}\n")
            fh.write(f"finite_foil_enabled={cfg.finite_foil_enabled}  (scalar S_foil; auto-off under diffraction)\n")
        for k in ('beam_energy_MeV', 'bunch_charge_nC', 'sigma_t_ps', 'div_match',
                  'gf_conv_sigma_m', 'gf_conv_domain', 'cap_range_factor',
                  'window_enabled', 'window_mode', 'window_n', 'window_thickness_m',
                  'window_alpha_per_cm_at_1THz',
                  'L_foil_to_window_m', 'L_window_to_fold_m', 'L_fold_to_bs_m',
                  'L_bs_to_delay_m', 'L_bs_to_oap_m', 'L_oap_to_detector_m',
                  'oap_focal_length_m', 'bs_R', 'bs_T',
                  'fold_mirror_dia_m', 'bs_dia_m', 'retro_mirror_dia_m', 'oap_dia_m'):
            fh.write(f"{k}={getattr(cfg, k)}\n")
        fh.write(f"channel_amp_at_150GHz={abs(channel_response_interf(150e9, cfg)):.4f}\n")
        fh.write(f"detector_z_m={detector_z_m(cfg):.4f}\n")
        if args.detector == 'golay':
            fh.write(f"golay_aperture_radius_m={cfg.det_active_radius_m}\n")
            fh.write(f"detector_ring_radius_m={r_ring:.4f}\n")
            fh.write(f"power_captured_frac_at_nominal={cap_nom:.3f}\n")
            fh.write(f"window_etalon_recurrence_ps={tau_et_ps:.2f}\n")
            fh.write(f"L_bs_to_reference_m={cfg.L_bs_to_reference_m:.6f}\n")
            fh.write(f"zpd_stage_travel_m={zpd_stage_offset_m:.6f} (={zpd_stage_offset_m/0.0254:.3f}in one-way)\n")
            fh.write(f"zpd_roundtrip_opd_m={2*zpd_stage_offset_m:.6f} "
                     f"(=2x stage; ={2*zpd_stage_offset_m/0.0254:.3f}in; tau0={2*zpd_stage_offset_m/C_LIGHT*1e12:.2f}ps)\n")
            fh.write(f"delay_max_ps={taus.max()*1e12:.1f}  n_tau={len(taus)}  "
                     f"stage_travel_max_mm={C_LIGHT*taus.max()/2*1e3:.2f}\n")
        fh.write(f"n_ok={len(records)}/{len(freqs)}\n")

    print(f"Wrote outputs to {os.path.abspath(cfg.out_dir)}/")
    peak_i = max(range(len(P)), key=lambda k: P[k])
    if args.detector == 'golay':
        print(f"  detector=golay; spectral peak ~ {f_GHz[peak_i]:.0f} GHz; ring r={r_ring*1e3:.1f} mm; "
              f"power captured @ r={cfg.det_active_radius_m*1e3:.1f}mm = {cap_nom*100:.0f}%; "
              f"interferogram {len(taus)} delays to {taus.max()*1e12:.0f} ps")
    else:
        print(f"  detector=eo; spectral peak ~ {f_GHz[peak_i]:.0f} GHz; waveform span {t_ps[-1]:.0f} ps; "
              f"|t_chan|(150GHz)={abs(channel_response_interf(150e9, cfg)):.3f}")


if __name__ == '__main__':
    main()
