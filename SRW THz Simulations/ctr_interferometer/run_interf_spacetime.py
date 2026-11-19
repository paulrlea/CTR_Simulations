#!/usr/bin/env python
"""
Absolute space-time reconstruction for the CTR interferometer + Golay detector.

Produces what the Golay would see as a function of BOTH the interferometer delay tau and
the fast time t, in ABSOLUTE units:

    P_inst(t; tau) = sum_aperture | E(x,y,t) + E(x,y,t-tau) |^2      [Watts]

At each aperture pixel the CTR pulse E(t) (reference arm) is added to its delayed replica
E(t-tau) (delay arm), squared, and summed over the Golay aperture and both polarizations.
The reference pulse sits at t=0, the delayed pulse rides the diagonal t=tau, and they
interfere near tau=0 (zero path difference). The Golay's actual reading is the time-integral
of this map:  P_golay(tau) = INT P_inst dt   [Joules/pulse]  ->  the interferogram.

Absolute scale (order-of-magnitude): the CTR pulse energy through the aperture is fixed from
the coherent bunch spectral energy
    dW/domega = N^2 |F(f)|^2 * alpha*hbar*(2 ln gamma - 1)        (N = Q/e electrons)
integrated over the propagated, aperture-collected, channel-weighted spectrum S(f); the
reconstructed cube is rescaled so its total energy matches. Rests on the single-electron
TR-energy model + coherent N^2|F|^2 scaling -> treat the magnitude as approximate; the SHAPES
of the map and interferogram are robust.

Needs the COMPLEX field, so it re-propagates (the Golay run is phase-blind, saves only |.|^2).

    source ../.venv/bin/activate
    python run_interf_spacetime.py --nfreq 9      # quick smoke
    python run_interf_spacetime.py                # full run -> results_interf_spacetime/
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
from thz_extract import _complex_ExEy, resample_to_grid, common_grid   # noqa: E402
from thz_config import C_LIGHT                                          # noqa: E402
from srwpy.srwlib import srwl                                          # noqa: E402

from interf_config import default_interf_config
from interf_source import make_source_conv
from interf_beamline import build_interf_beamline, channel_response_interf, detector_z_m

HBAR_J_S = 1.054571817e-34     # reduced Planck [J s]
ALPHA_FS = 7.2973525693e-3     # fine-structure constant
E_CHARGE = 1.602176634e-19     # electron charge [C]


def parse_args():
    p = argparse.ArgumentParser(description="Absolute (delay, time) map for the CTR interferometer + Golay")
    p.add_argument('--nfreq', type=int, default=161)
    p.add_argument('--fmin', type=float, default=10.0, help='min frequency [GHz]')
    p.add_argument('--fmax', type=float, default=400.0, help='max frequency [GHz]')
    p.add_argument('--charge', type=float, default=10.0, help='bunch charge [nC]')
    p.add_argument('--sigma-t', type=float, default=2.0, help='rms bunch length [ps]')
    p.add_argument('--conv-sigma', type=float, default=2.0e-3, help='GF Gaussian conv RMS [m]')
    p.add_argument('--golay-aperture', type=float, default=0.005, help='Golay aperture radius [m]')
    p.add_argument('--cube-npix', type=int, default=201, help='detector grid points per axis')
    p.add_argument('--half-extent', type=float, default=0.012, help='detector half-extent [m]')
    p.add_argument('--tau-max-ps', type=float, default=150.0, help='max |delay| for the 2D map [ps]')
    p.add_argument('--t-win-ps', type=float, default=185.0, help='fast-time window half-width [ps]')
    p.add_argument('--rep-rate', type=float, default=None, help='bunch rep rate [Hz] (adds avg power)')
    p.add_argument('--out', default='results_interf_spacetime')
    return p.parse_args()


def main():
    a = parse_args()
    cfg = default_interf_config(n_freq=a.nfreq, f_min_GHz=a.fmin, f_max_GHz=a.fmax,
                                bunch_charge_nC=a.charge, sigma_t_ps=a.sigma_t,
                                gf_conv_sigma_m=a.conv_sigma,
                                det_active_radius_m=a.golay_aperture,
                                det_npix=a.cube_npix, det_half_extent_m=a.half_extent)
    os.makedirs(a.out, exist_ok=True)
    Q = cfg.bunch_charge_nC
    N_e = (Q * 1e-9) / E_CHARGE                       # electrons in the bunch
    dW_domega_single = (2.0 * ALPHA_FS / np.pi) * HBAR_J_S * (np.log(2.0 * cfg.gamma) - 1.0)  # [J s]

    freqs = cfg.freq_grid_Hz()
    xc, yc = common_grid(cfg)
    npix = cfg.det_npix
    XX, YY = np.meshgrid(xc, yc)
    amask = (XX ** 2 + YY ** 2) <= cfg.det_active_radius_m ** 2

    print(f"Absolute (delay,time) map:  gamma={cfg.gamma:.1f}  Q={Q} nC (N={N_e:.2e} e-)  "
          f"sigma_t={cfg.sigma_t_ps} ps")
    print(f"  {len(freqs)} freq {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz; cube {npix}x{npix} "
          f"+/-{a.half_extent*1e3:.0f} mm; Golay r={cfg.det_active_radius_m*1e3:.1f} mm; "
          f"{int(amask.sum())} aperture px")

    # ---- per frequency: propagate, keep COMPLEX aperture field (both pols) x channel ----
    Ex_ap, Ey_ap, f_ok = [], [], []
    t0 = time.time()
    for i, f in enumerate(freqs):
        try:
            wfr = make_source_conv(f, cfg)
            srwl.PropagElecField(wfr, build_interf_beamline(cfg, capped=True)[0])
            Ex, Ey, x, y = _complex_ExEy(wfr)
            ch = channel_response_interf(f, cfg)
            exr, _, _ = resample_to_grid(Ex.real, x, y, cfg)
            exi, _, _ = resample_to_grid(Ex.imag, x, y, cfg)
            eyr, _, _ = resample_to_grid(Ey.real, x, y, cfg)
            eyi, _, _ = resample_to_grid(Ey.imag, x, y, cfg)
            Ex_ap.append((((exr + 1j * exi) * ch)[amask]).astype(np.complex128))
            Ey_ap.append((((eyr + 1j * eyi) * ch)[amask]).astype(np.complex128))
            f_ok.append(f)
            del wfr, Ex, Ey
            if (i + 1) % 20 == 0 or i == len(freqs) - 1:
                print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [{i+1:>3}/{len(freqs)}] f={f/1e9:6.1f} GHz  FAILED: {e}")
    if not f_ok:
        print("No frequencies succeeded; aborting."); sys.exit(1)
    f_ok = np.asarray(f_ok)
    Ex_ap = np.asarray(Ex_ap)                          # (nfreq, npix_ap) complex64
    Ey_ap = np.asarray(Ey_ap)
    print(f"Propagation done ({time.time()-t0:.0f}s, {len(f_ok)}/{len(freqs)} ok). Reconstructing...")

    # ---- absolute single-arm energy through the aperture (coherent CTR) ----
    # detector pixel area: INT|E|^2 dA = sum_pix |E|^2 * dcell^2 (needed for absolute energy)
    dcell = (2.0 * cfg.det_half_extent_m) / (cfg.det_npix - 1)         # [m]
    F = np.exp(-0.5 * (2 * np.pi * f_ok * cfg.sigma_t_s) ** 2)        # bunch form factor
    ap_spec = (np.abs(Ex_ap) ** 2 + np.abs(Ey_ap) ** 2).sum(axis=1)   # sum_ap |E_srw*ch|^2 per freq
    # The GF source amplitude A already carries dW_domega_single (INT|E_src|^2 dA = dW/domega),
    # so INT_ap|E_srw*ch|^2 dA = ap_spec*dcell^2 = throughput*|ch|^2*dW_domega_single. Coherent
    # bunch -> x N^2 |F|^2:
    dW_domega_arm = (N_e ** 2) * (F ** 2) * ap_spec * dcell ** 2       # [J s] per freq point
    dfreq = float(np.min(np.diff(np.sort(f_ok))))                     # grid spacing [Hz]
    W_arm_abs = 2.0 * np.pi * float(np.sum(dW_domega_arm)) * dfreq     # [J]  (INT ... domega)

    # ---- analytic-signal per-pixel time reconstruction on a uniform grid ----
    idx = np.round(f_ok / dfreq).astype(int)
    n_pos = int(idx.max()) + 1
    n_time = 2 * (n_pos - 1)
    dt = 1.0 / (n_time * dfreq)                                       # [s]
    w = (F * Q).astype(np.complex128)                                # arb per-freq field weight

    def to_time(E_ap):
        spec = np.zeros((n_time, E_ap.shape[1]), dtype=np.complex128)
        spec[idx] = E_ap * w[:, None]                                # positive-freq analytic spectrum
        return np.fft.ifft(spec, axis=0)                             # (n_time, npix_ap) analytic

    ex = to_time(Ex_ap); ey = to_time(Ey_ap)
    del Ex_ap, Ey_ap

    I_arm_arb = (np.abs(ex) ** 2 + np.abs(ey) ** 2).sum(axis=1)       # (n_time,) arb power
    E_arm_arb = float(I_arm_arb.sum() * dt)                           # arb * s
    scale = W_arm_abs / E_arm_arb if E_arm_arb > 0 else 0.0           # [W per arb-power]

    # center the reference pulse at t=0
    ipk = int(np.argmax(I_arm_arb))
    ex = np.roll(ex, n_time // 2 - ipk, axis=0)
    ey = np.roll(ey, n_time // 2 - ipk, axis=0)
    t_rel = (np.arange(n_time) - n_time // 2) * dt                    # [s]

    # ---- 2D map P_inst(t; tau) [W] and the interferogram P_golay(tau) [J] ----
    mmax = int(round(a.tau_max_ps * 1e-12 / dt))
    shifts = np.arange(-mmax, mmax + 1)
    tau_ps = shifts * dt * 1e12
    keep = np.abs(t_rel) <= a.t_win_ps * 1e-12
    tt_ps = t_rel[keep] * 1e12
    M = np.empty((len(shifts), int(keep.sum())), dtype=np.float64)    # [W]
    P_golay_J = np.empty(len(shifts))                                 # [J] per pulse
    for j, m in enumerate(shifts):
        exd = np.roll(ex, m, axis=0); eyd = np.roll(ey, m, axis=0)
        pinst = scale * (np.abs(ex + exd) ** 2 + np.abs(ey + eyd) ** 2).sum(axis=1)  # (n_time,) [W]
        M[j] = pinst[keep]
        P_golay_J[j] = float(pinst.sum() * dt)                        # full-time integral [J]

    # single-arm pulse power [W], rolled to match the centered t_rel axis (as ex/ey were)
    I_arm_W = scale * np.roll(I_arm_arb, n_time // 2 - ipk)
    etalon_ps = 2.0 * cfg.window_n * cfg.window_thickness_m / C_LIGHT * 1e12
    zpd_stage_mm = (cfg.L_bs_to_reference_m - cfg.L_bs_to_delay_m) * 1e3

    print(f"  n_time={n_time} dt={dt*1e12:.2f} ps; single-arm energy W_arm={W_arm_abs:.3e} J; "
          f"peak inst. power={M.max():.3e} W")
    print(f"  Golay energy/pulse: baseline~{2*W_arm_abs:.3e} J, ZPD~{P_golay_J[np.argmin(np.abs(tau_ps))]:.3e} J")

    # ---- save cube ----
    np.savez_compressed(os.path.join(a.out, 'spacetime.npz'),
                        M_W=M, t_ps=tt_ps, tau_ps=tau_ps, P_golay_J=P_golay_J,
                        I_arm_W=I_arm_W, t_full_ps=t_rel * 1e12,
                        W_arm_J=W_arm_abs, scale=scale, dt_s=dt)

    # ---- (1) 2D map: instantaneous power vs (time, delay) ----
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    vmax = float(np.percentile(M, 99.6))
    im = ax.imshow(M, origin='lower', aspect='auto', cmap='inferno', vmin=0, vmax=vmax,
                   extent=[tt_ps[0], tt_ps[-1], tau_ps[0], tau_ps[-1]])
    ax.plot([tt_ps[0], tt_ps[-1]], [tt_ps[0], tt_ps[-1]], color='w', lw=0.8, ls='--',
            alpha=0.6)                                               # delayed pulse rides t = tau
    ax.axhline(0, color='w', lw=0.8, ls=':', alpha=0.6)              # reference pulse at t=0
    for s in (-etalon_ps, etalon_ps):
        ax.axhline(s, color='#66ffcc', lw=0.7, ls=':', alpha=0.7)
    ax.set_xlabel('fast time  t [ps]'); ax.set_ylabel('interferometer delay  tau [ps]')
    ax.set_title('Golay detector: instantaneous power  P(t, tau)  [W]  (absolute)\n'
                 'reference pulse at t=0, delayed pulse at t=tau; interfere at ZPD (tau=0)')
    fig.colorbar(im, ax=ax, label='instantaneous power [W]')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'map_power_delay_time.png'), dpi=140)
    plt.close(fig)

    # ---- (2) absolute interferogram = time-integral of the map ----
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(tau_ps, P_golay_J * 1e9, lw=1.1)                          # [nJ]
    ax.axvline(0, color='#b8470f', ls=':', lw=0.9)
    for s in (-etalon_ps, etalon_ps):
        ax.axvline(s, color='#2a8', ls=':', lw=0.9)
    ax.set_xlabel('interferometer delay  tau [ps]  (ZPD at 0; stage +%.1f mm)' % zpd_stage_mm)
    ax.set_ylabel('Golay energy / pulse [nJ]')
    ax.set_title('Absolute Golay interferogram  P(tau) = INT P(t,tau) dt')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'interferogram_absolute.png'), dpi=140)
    plt.close(fig)

    # ---- (3) single-arm pulse: absolute instantaneous power vs time ----
    tf = t_rel * 1e12
    wsel = np.abs(tf) <= a.t_win_ps
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(tf[wsel], I_arm_W[wsel], lw=1.2, color='#d12f2f')
    ax.set_xlabel('fast time  t [ps]'); ax.set_ylabel('instantaneous power [W]')
    ax.set_title('CTR pulse at the Golay (one arm): absolute power vs time')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'pulse_power_time.png'), dpi=140)
    plt.close(fig)

    with open(os.path.join(a.out, 'run_meta.txt'), 'w') as fh:
        fh.write("ctr_interferometer space-time reconstruction (absolute units)\n")
        fh.write(f"n_freq={len(f_ok)}/{len(freqs)}  f={a.fmin}-{a.fmax} GHz  dfreq={dfreq/1e9:.3f} GHz\n")
        fh.write(f"charge_nC={Q}  N_electrons={N_e:.4e}  gamma={cfg.gamma:.2f}\n")
        fh.write(f"dW_domega_single={dW_domega_single:.4e} J*s  sigma_t_ps={cfg.sigma_t_ps}\n")
        fh.write(f"golay_aperture_m={cfg.det_active_radius_m}  aperture_px={int(amask.sum())}\n")
        fh.write(f"W_arm_J={W_arm_abs:.6e}  (single-arm pulse energy through aperture)\n")
        fh.write(f"baseline_energy_J={2*W_arm_abs:.6e}  ZPD_energy_J={P_golay_J[np.argmin(np.abs(tau_ps))]:.6e}\n")
        fh.write(f"peak_inst_power_W={M.max():.6e}  single_arm_peak_power_W={I_arm_W.max():.6e}\n")
        fh.write(f"dt_ps={dt*1e12:.3f}  n_time={n_time}  tau_max_ps={a.tau_max_ps}  t_win_ps={a.t_win_ps}\n")
        fh.write(f"zpd_stage_offset_mm={zpd_stage_mm:.3f}  etalon_recurrence_ps={etalon_ps:.2f}\n")
        if a.rep_rate:
            fh.write(f"rep_rate_Hz={a.rep_rate}  avg_power_at_ZPD_W="
                     f"{P_golay_J[np.argmin(np.abs(tau_ps))]*a.rep_rate:.4e}\n")
        fh.write(f"detector_z_m={detector_z_m(cfg):.4f}\n")
        fh.write("NOTE: absolute magnitude is order-of-magnitude (single-electron TR model +\n"
                 "coherent N^2|F|^2 scaling); shapes are robust.\n")

    print(f"Wrote outputs to {os.path.abspath(a.out)}/")


if __name__ == '__main__':
    main()
