#!/usr/bin/env python
"""
Space-time detector reconstruction for the Level-3 oblique CTR source.

Unlike run_thz.py (which keeps only the per-frequency *intensity* + a single ring
phasor), this driver keeps the full COMPLEX 2D field at the detector for every
frequency, projects it onto the EO-crystal axis, weights by the Gaussian bunch
form factor F(f) and charge, and inverse-FFTs PER PIXEL to reconstruct the true

    E_c(x, y, t)   at the detector plane.

This is the non-separable counterpart of the source-plane figure
results_l3/spacetime_pulse.png: at the source the TR spectrum is flat so the
field factorises into (transverse pattern) x (Gaussian pulse); after the beamline
the ring radius and focusing are frequency-dependent, so the Gaussian bunch pulse
is reshaped by the relay's diffraction + the sapphire etalon into a bipolar
transient with echoes -- captured here as a full space-time movie.

Outputs land in <out_dir>/ (default results_l3_spacetime/):
    xt_map.png           E_c(x, y=0, t) space-time map (tilt-plane cut)
    snapshots.png        E_c(x,y) at several times across the pulse
    pulse.gif            animation of E_c(x,y,t) at the detector
    waveform_active.png  active-area on-crystal E(t) (reconstruction cross-check)
    spacetime.npz        the reconstructed cube (E_c, x, y, t) + axes
    run_meta.txt

Run (venv active, PYTHONPATH-isolated):
    python run_thz_spacetime.py --nfreq 161 --cube-npix 201
"""
import argparse, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from thz_config import default_config
from thz_source import make_source, form_factor
from thz_beamline import build_beamline, channel_response, detector_z_m
from thz_extract import _complex_ExEy, resample_to_grid, common_grid
from srwpy.srwlib import srwl


def parse_args():
    p = argparse.ArgumentParser(description="Level-3 CTR space-time detector reconstruction")
    p.add_argument('--nfreq', type=int, default=161)
    p.add_argument('--fmin', type=float, default=10.0, help='min frequency [GHz]')
    p.add_argument('--fmax', type=float, default=400.0, help='max frequency [GHz]')
    p.add_argument('--screen-tilt', type=float, default=30.0)
    p.add_argument('--crystal-pol-axis', type=float, default=0.0)
    p.add_argument('--charge', type=float, default=10.0, help='bunch charge [nC]')
    p.add_argument('--sigma-t', type=float, default=2.0, help='rms bunch length [ps]')
    p.add_argument('--cube-npix', type=int, default=201, help='detector grid for the cube')
    p.add_argument('--half-extent', type=float, default=0.035, help='detector half-extent [m]')
    p.add_argument('--win-ps', type=float, default=90.0, help='time window half-width around peak [ps]')
    p.add_argument('--out', default='results_l3_spacetime')
    return p.parse_args()


def main():
    a = parse_args()
    cfg = default_config(n_freq=a.nfreq, f_min_GHz=a.fmin, f_max_GHz=a.fmax,
                         bunch_charge_nC=a.charge, sigma_t_ps=a.sigma_t,
                         det_npix=a.cube_npix, det_half_extent_m=a.half_extent,
                         gf_tilt_model='level3', screen_tilt_deg=a.screen_tilt,
                         crystal_pol_axis_deg=a.crystal_pol_axis)
    os.makedirs(a.out, exist_ok=True)
    chi = np.radians(a.crystal_pol_axis)
    Q = cfg.bunch_charge_nC

    freqs = cfg.freq_grid_Hz()
    xc, yc = common_grid(cfg)
    npix = cfg.det_npix
    print(f"Space-time L3 run: tilt={a.screen_tilt} deg  crystal_axis={a.crystal_pol_axis} deg  "
          f"gamma={cfg.gamma:.1f}")
    print(f"  {len(freqs)} freq {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz; "
          f"cube {npix}x{npix} over +/-{a.half_extent*1e3:.0f} mm; det z={detector_z_m(cfg):.2f} m")

    # --- per-frequency: propagate, project onto crystal axis, resample complex field ---
    Ec_stack, f_ok = [], []
    t0 = time.time()
    for i, f in enumerate(freqs):
        try:
            wfr = make_source(f, cfg, 'ctr')
            srwl.PropagElecField(wfr, build_beamline(cfg, capped=True)[0])
            Ex, Ey, x, y = _complex_ExEy(wfr)
            Ec = Ex * np.cos(chi) + Ey * np.sin(chi)        # crystal-axis complex field
            Reg, _, _ = resample_to_grid(Ec.real, x, y, cfg)
            Img, _, _ = resample_to_grid(Ec.imag, x, y, cfg)
            ch = channel_response(f, cfg)                   # window etalon + mirrors (complex)
            Ec_stack.append(((Reg + 1j * Img) * ch).astype(np.complex64))
            f_ok.append(f)
            del wfr, Ex, Ey, Ec
            if (i + 1) % 20 == 0 or i == len(freqs) - 1:
                print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  FAILED: {e}")
    if not Ec_stack:
        print("No frequencies succeeded; aborting."); sys.exit(1)
    f_ok = np.asarray(f_ok)
    Ec_stack = np.asarray(Ec_stack)                         # (nfreq, npix, npix) complex64
    print(f"Propagation done ({time.time()-t0:.0f}s, {len(f_ok)}/{len(freqs)} ok). Reconstructing E(x,y,t)...")

    # --- per-pixel inverse FFT (same uniform-grid Hermitian convention as assemble_waveform) ---
    F = np.exp(-0.5 * (2 * np.pi * f_ok * cfg.sigma_t_s) ** 2)   # bunch form factor (vectorized)
    df = np.min(np.diff(f_ok)) if len(f_ok) > 1 else f_ok[0]
    idx = np.round(f_ok / df).astype(int)
    n_pos = int(idx.max()) + 1
    spec = np.zeros((n_pos, npix, npix), dtype=np.complex64)
    spec[idx] = Ec_stack * (F * Q)[:, None, None]           # zero below f_min (foil cutoff)
    n_time = 2 * (n_pos - 1)
    Et = np.fft.irfft(spec, n=n_time, axis=0).astype(np.float32) * n_time   # (n_time, npix, npix)
    dt = 1.0 / (n_time * df)
    t_ps = np.arange(n_time) * dt * 1e12
    del spec, Ec_stack

    # center the time window on the pulse. The irfft output is periodic, so roll the
    # peak to the middle (absolute origin is an arbitrary phase ref) -> symmetric window
    # showing both sides of the transient without wrap-around truncation.
    energy_t = np.sum(Et.reshape(n_time, -1) ** 2, axis=1)
    ipk = int(np.argmax(energy_t))
    Et = np.roll(Et, n_time // 2 - ipk, axis=0)
    t_rel = (np.arange(n_time) - n_time // 2) * dt * 1e12   # ps, relative to peak
    keep = np.abs(t_rel) <= a.win_ps
    tt = t_rel[keep]
    cube = Et[keep]                                         # (nt, npix, npix)
    vmax = float(np.percentile(np.abs(cube), 99.8))
    ext = [xc[0] * 1e3, xc[-1] * 1e3, yc[0] * 1e3, yc[-1] * 1e3]
    print(f"  reconstructed {n_time} frames, dt={dt*1e12:.2f} ps; pulse peak at t={t_ps[ipk]:.0f} ps "
          f"-> windowed +/-{a.win_ps:.0f} ps ({cube.shape[0]} frames)")

    # --- save cube ---
    np.savez_compressed(os.path.join(a.out, 'spacetime.npz'),
                        Ec=cube, x=xc, y=yc, t_ps=tt, f_peak_ps=t_ps[ipk])

    # --- (1) x-t space-time map at y=0 (tilt-plane cut) ---
    iy0 = npix // 2
    xt = cube[:, iy0, :].T                                  # (npix_x, nt)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    m = ax.imshow(xt, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  extent=[tt[0], tt[-1], xc[0] * 1e3, xc[-1] * 1e3])
    ax.set_xlabel('t [ps]  (relative to pulse peak)'); ax.set_ylabel('x [mm]')
    ax.set_title('Detector space-time field  $E_c(x,\\,y{=}0,\\,t)$  (Level-3, %g° tilt)\n'
                 'Gaussian bunch pulse reshaped by relay diffraction + sapphire etalon' % a.screen_tilt)
    fig.colorbar(m, ax=ax, fraction=0.046, label='$E_c$ arb.')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'xt_map.png'), dpi=130); plt.close(fig)

    # --- (2) transverse snapshots across the pulse ---
    tsnap = [-84, -42, 0, 42, 84]
    fig, axs = plt.subplots(1, len(tsnap), figsize=(3 * len(tsnap), 3.2))
    for ax, tc in zip(axs, tsnap):
        j = int(np.argmin(np.abs(tt - tc)))
        im = ax.imshow(cube[j], origin='lower', extent=ext, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_title(f't = {tt[j]:+.0f} ps', fontsize=10)
        ax.set_xlabel('x [mm]'); ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30])
    axs[0].set_ylabel('y [mm]')
    fig.suptitle('Detector E_c(x,y) snapshots across the CTR pulse (Level-3, %g° tilt)' % a.screen_tilt,
                 y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'snapshots.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    # --- (3) active-area on-crystal waveform (cross-check vs run_thz waveform shape) ---
    XX, YY = np.meshgrid(xc, yc)
    amask = (XX ** 2 + YY ** 2) <= cfg.det_active_radius_m ** 2
    wf = np.array([np.mean(fr[amask]) for fr in cube])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(tt, wf, lw=1.3)
    ax.set_xlabel('t [ps]  (relative to peak)'); ax.set_ylabel('E_c(t) [arb]')
    ax.set_title('On-crystal E(t): active-area mean (%.0f mm radius)' % (cfg.det_active_radius_m * 1e3))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'waveform_active.png'), dpi=130); plt.close(fig)

    # --- (4) animation ---
    nframe = min(cube.shape[0], 80)
    sub = np.linspace(0, cube.shape[0] - 1, nframe).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cube[sub[0]], origin='lower', extent=ext, cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ttl = ax.set_title('')
    fig.colorbar(im, ax=ax, fraction=0.046, label='$E_c$ arb.')

    def upd(k):
        im.set_data(cube[sub[k]])
        ttl.set_text(f'Detector $E_c(x,y,t)$   t = {tt[sub[k]]:+.1f} ps   (Level-3, {a.screen_tilt:g}° tilt)')
        return im, ttl

    anim = FuncAnimation(fig, upd, frames=nframe, blit=False)
    anim.save(os.path.join(a.out, 'pulse.gif'), writer=PillowWriter(fps=15))
    plt.close(fig)

    with open(os.path.join(a.out, 'run_meta.txt'), 'w') as fh:
        fh.write("src=ctr  model=level3 (space-time)\n")
        for k in ('beam_energy_MeV', 'bunch_charge_nC', 'sigma_t_ps', 'screen_tilt_deg',
                  'crystal_pol_axis_deg', 'window_mode', 'mirror_enabled', 'mirror2_enabled'):
            fh.write(f"{k}={getattr(cfg, k)}\n")
        fh.write(f"n_freq={len(f_ok)}/{len(freqs)}  f={a.fmin}-{a.fmax} GHz\n")
        fh.write(f"cube_npix={npix}  half_extent_m={a.half_extent}\n")
        fh.write(f"dt_ps={dt*1e12:.3f}  n_time={n_time}  pulse_peak_ps={t_ps[ipk]:.1f}  win_ps={a.win_ps}\n")
        fh.write(f"detector_z_m={detector_z_m(cfg):.4f}\n")

    print(f"Wrote outputs to {os.path.abspath(a.out)}/")


if __name__ == '__main__':
    main()
