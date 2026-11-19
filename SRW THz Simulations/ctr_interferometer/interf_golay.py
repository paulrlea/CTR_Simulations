"""
Golay-cell detector model + interferometric-autocorrelation analysis for the CTR
beam diagnostic.

This is a CTR-based BEAM DIAGNOSTIC, not FTIR spectroscopy of a sample: the Michelson +
Golay measures the first-order FIELD AUTOCORRELATION of the coherent transition-radiation
pulse.  A Golay cell reads TIME-INTEGRATED, POLARIZATION-BLIND, phase-blind POWER over its
input aperture -- one scalar per delay.  Scanning the delay tau records the interferogram
P(tau) (the autocorrelation); its cosine transform gives the CTR power spectrum, which
carries the bunch form factor |F(f)|^2 -- the longitudinal-bunch information the diagnostic
exists to measure.

Everything here is a reduction of the per-frequency detector-plane intensity that
run_interf already stores in records[*].I_grid (resampled to the common grid, channel-
weighted).  No re-propagation is needed.

Physics (balanced Michelson, symmetric arms, 30/30 BS):
  both arms deliver the SAME detector-plane field E_det(x,y,f); the delay arm carries an
  extra phase exp(i 2 pi f tau).  The Golay reads, integrated over aperture radius R:
     P(tau) = sum_f w(f) * P_det(f, R) * |1 + exp(i 2 pi f tau)|^2
            = 2 * sum_f S(f,R) * [1 + cos(2 pi f tau)],   S(f,R) = w(f) P_det(f,R)
  with w(f) = |F(f)|^2 Q^2 (bunch form factor + charge).  The tau-modulated part is the
  cosine transform of the detected spectrum S(f) (Wiener-Khinchin); inverting it recovers
  S(f).  Delay <-> stage: tau = 2 * x_stage / c  (retro round trip).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SRW_THZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "srw_thz")
if _SRW_THZ not in sys.path:
    sys.path.insert(0, _SRW_THZ)
from thz_source import form_factor          # noqa: E402
from thz_extract import common_grid         # noqa: E402
from thz_config import C_LIGHT              # noqa: E402


# --------------------------------------------------------------- grid helpers
def _grid(cfg):
    xc, yc = common_grid(cfg)
    XX, YY = np.meshgrid(xc, yc)
    RR = np.sqrt(XX ** 2 + YY ** 2)
    dcell = (2.0 * cfg.det_half_extent_m) / (cfg.det_npix - 1)
    return xc, yc, RR, dcell


def _weights(records, cfg):
    """Per-frequency intensity weight w(f) = |F(f)|^2 * Q^2."""
    Q = cfg.bunch_charge_nC
    return np.array([form_factor(r.f_Hz, cfg.sigma_t_s) ** 2 * Q ** 2 for r in records])


def broadband_map(records, cfg):
    """Form-factor/charge-weighted broadband detector intensity map (arb)."""
    w = _weights(records, cfg)
    Itot = np.zeros((cfg.det_npix, cfg.det_npix))
    for wi, r in zip(w, records):
        Itot += wi * r.I_grid
    return Itot


def ring_radius_m(records, cfg):
    """Azimuthal-peak radius of the broadband detector ring [m]."""
    xc, yc, RR, _ = _grid(cfg)
    Itot = broadband_map(records, cfg)
    rmax = xc.max()
    nb = max(64, cfg.det_npix // 4)
    dr = rmax / nb
    idx = np.minimum((RR.ravel() / dr).astype(int), nb - 1)
    prof = np.bincount(idx, Itot.ravel(), nb) / np.maximum(np.bincount(idx, None, nb), 1)
    return float((np.argmax(prof) + 0.5) * dr)


# ---------------------------------------------------------- power / spectrum
def per_freq_aperture_power(records, cfg, R_ap):
    """P_det(f) = intensity integrated over the Golay aperture (r <= R_ap), per frequency."""
    _, _, RR, dcell = _grid(cfg)
    mask = RR <= R_ap
    a = dcell * dcell
    return np.array([float(r.I_grid[mask].sum()) * a for r in records])


def detected_spectrum(records, cfg, R_ap):
    """S(f, R_ap) = w(f) * P_det(f, R_ap).  Returns (f_GHz, S)."""
    w = _weights(records, cfg)
    Pf = per_freq_aperture_power(records, cfg, R_ap)
    f_GHz = np.array([r.f_Hz / 1e9 for r in records])
    return f_GHz, w * Pf


def power_capture_curve(records, cfg, radii_m):
    """Fraction of total broadband power captured vs aperture radius.
    Returns (radii_m, fraction, total_power).  The CTR ring + on-axis null make this
    rise slowly for small on-axis apertures (a small aperture sits in the null)."""
    _, _, RR, dcell = _grid(cfg)
    Itot = broadband_map(records, cfg)
    a = dcell * dcell
    total = float(Itot.sum()) * a
    cap = np.array([float(Itot[RR <= R].sum()) * a for R in radii_m])
    return np.asarray(radii_m), cap / (total if total > 0 else 1.0), total


# --------------------------------------------------------------- interferogram
def delay_grid(cfg, freqs_Hz, tau_max_ps=None, n_tau=None):
    """Symmetric delay grid tau [s].  Step = 0.8/(2 f_max) (finer than Nyquist for f_max);
    range = tau_max_ps if given, else 1/(2*df) to resolve the frequency-grid spacing."""
    f_max = float(np.max(freqs_Hz))
    dtau = 0.8 / (2.0 * f_max)
    if tau_max_ps is not None:
        tau_max = tau_max_ps * 1e-12
    else:
        df = float(np.min(np.diff(np.sort(freqs_Hz)))) if len(freqs_Hz) > 1 else f_max
        tau_max = 1.0 / (2.0 * df)
    if n_tau is not None:
        return np.linspace(-tau_max, tau_max, int(n_tau))
    n = int(2 * tau_max / dtau) | 1            # odd -> includes tau=0
    return np.linspace(-tau_max, tau_max, n)


def interferogram(records, cfg, R_ap, taus_s):
    """Golay interferogram P(tau) = sum_f S(f)[1 + cos(2 pi f tau)] over the aperture R_ap."""
    f_Hz = np.array([r.f_Hz for r in records])
    _, S = detected_spectrum(records, cfg, R_ap)
    ang = 2.0 * np.pi * np.outer(taus_s, f_Hz)          # [n_tau, n_f]
    return S.sum() + (S[None, :] * np.cos(ang)).sum(axis=1)


def recover_spectrum(taus_s, P, f_Hz_grid):
    """Recover S(f) from the interferogram by discrete cosine transform of its AC part
    (Wiener-Khinchin).  Returns S_rec on f_Hz_grid (arb units)."""
    ac = P - P.mean()
    dtau = float(taus_s[1] - taus_s[0])
    ang = 2.0 * np.pi * np.outer(np.asarray(f_Hz_grid), taus_s)   # [n_f, n_tau]
    return (ac[None, :] * np.cos(ang)).sum(axis=1) * dtau


# --------------------------------------------------------------------- savers
def save_power_capture(radii_m, frac, cfg, r_ring_m, R_nom_m, path_png, path_txt):
    np.savetxt(path_txt, np.column_stack([radii_m * 1e3, frac]),
               header='aperture_radius[mm]  captured_power_fraction')
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(radii_m * 1e3, frac, lw=1.8)
    ax.axvline(r_ring_m * 1e3, color='#d9a521', ls='--', lw=1.2,
               label=f'ring peak r={r_ring_m*1e3:.1f} mm')
    ax.axvline(R_nom_m * 1e3, color='#d12f2f', ls=':', lw=1.4,
               label=f'nominal Golay r={R_nom_m*1e3:.1f} mm')
    ax.set_xlabel('Golay aperture radius [mm]')
    ax.set_ylabel('captured power fraction')
    ax.set_title('Golay power capture vs aperture (CTR ring)')
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)


def save_interferogram(taus_s, P, cfg, path_png, path_txt, etalon_tau_ps=None,
                       stage_zero_offset_m=0.0):
    """taus_s is the RELATIVE optical delay between the two arms (ZPD at tau=0), so the
    interferogram shape and the recovered spectrum are independent of the arm-length
    asymmetry.  stage_zero_offset_m = L_ref - L_bs_to_delay maps relative delay to the
    physical delay-stage position: x_stage = stage_zero_offset + c*tau/2, i.e. ZPD is
    reached at x_stage = stage_zero_offset (not at the stage's nominal position)."""
    x_stage_mm = (stage_zero_offset_m + C_LIGHT * taus_s / 2.0) * 1e3
    np.savetxt(path_txt, np.column_stack([taus_s * 1e12, x_stage_mm, P]),
               header='rel_delay[ps]  stage_travel[mm]  P_golay[arb]  '
                      f'(ZPD at stage={stage_zero_offset_m*1e3:.2f}mm)')
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(taus_s * 1e12, P, lw=1.0)
    ax.set_xlabel('relative delay  tau [ps]  (ZPD at tau=0)')
    ax.set_ylabel('Golay power P(tau) [arb]')
    ax.set_title('CTR interferogram — field autocorrelation vs delay')
    if etalon_tau_ps is not None:
        for s in (-1, 1):
            ax.axvline(s * etalon_tau_ps, color='#2a8', ls=':', lw=1.0)
        ax.text(etalon_tau_ps, ax.get_ylim()[1] * 0.92,
                f'  window etalon\n  recurrence {etalon_tau_ps:.0f} ps',
                color='#2a8', fontsize=7.5, va='top')
    if stage_zero_offset_m:
        opd_mm = 2.0 * stage_zero_offset_m * 1e3          # round-trip OPD = 2 x stage move
        tau0_ps = 2.0 * stage_zero_offset_m / C_LIGHT * 1e12
        ax.text(0.02, 0.97,
                f'ZPD (tau=0): stage +{stage_zero_offset_m*1e3:.1f} mm ({stage_zero_offset_m/0.0254:.3f}")\n'
                f'round-trip OPD = 2x stage = {opd_mm:.1f} mm ({2*stage_zero_offset_m/0.0254:.3f}", tau0 {tau0_ps:.0f} ps)',
                transform=ax.transAxes, color='#b8470f', fontsize=7.0, va='top', ha='left')
    off_mm = stage_zero_offset_m * 1e3
    secax = ax.secondary_xaxis(
        'top', functions=(lambda t: off_mm + C_LIGHT * (t * 1e-12) / 2 * 1e3,
                          lambda x: 2 * ((x - off_mm) * 1e-3) / C_LIGHT * 1e12))
    secax.set_xlabel('delay-stage travel [mm]')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)


def save_recovered_spectrum(f_GHz, S_direct, S_recovered, path_png, path_txt):
    """Overlay the directly-computed detected spectrum and the CTR power spectrum recovered
    from the interferogram autocorrelation (both normalized) -- the Wiener-Khinchin
    self-consistency check.  The recovered spectrum carries the bunch form factor |F(f)|^2."""
    sd = S_direct / (np.max(S_direct) if np.max(S_direct) > 0 else 1)
    sr = S_recovered / (np.max(S_recovered) if np.max(S_recovered) > 0 else 1)
    np.savetxt(path_txt, np.column_stack([f_GHz, sd, sr]),
               header='f[GHz]  S_direct(norm)  S_recovered_from_autocorrelation(norm)')
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(f_GHz, sd, lw=1.8, label='direct (aperture power)')
    ax.plot(f_GHz, sr, lw=1.2, ls='--', label='recovered from interferogram')
    ax.set_xlabel('f [GHz]'); ax.set_ylabel('spectral power [norm]')
    ax.set_title('CTR power spectrum — recovered from autocorrelation vs direct')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)


def save_spatial_map_golay(records, cfg, R_nom_m, r_ring_m, path_png, path_npz):
    """Broadband detector intensity with the Golay aperture + ring overlaid (what the
    Golay integrates)."""
    xc, yc, _, _ = _grid(cfg)
    Itot = broadband_map(records, cfg)
    np.savez(path_npz, I=Itot, x=xc, y=yc, golay_aperture_m=R_nom_m, ring_radius_m=r_ring_m)
    ext = [xc[0] * 1e3, xc[-1] * 1e3, yc[0] * 1e3, yc[-1] * 1e3]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(Itot, origin='lower', extent=ext, aspect='equal')
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R_nom_m * 1e3 * np.cos(th), R_nom_m * 1e3 * np.sin(th),
            color='#d12f2f', lw=1.6, label=f'Golay aperture r={R_nom_m*1e3:.1f} mm')
    ax.plot(r_ring_m * 1e3 * np.cos(th), r_ring_m * 1e3 * np.sin(th),
            color='w', lw=1.0, ls='--', alpha=0.8, label=f'ring r={r_ring_m*1e3:.1f} mm')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title('Detector intensity + Golay aperture')
    ax.legend(fontsize=7, loc='upper right')
    fig.colorbar(im, ax=ax, label='arb. (|F|^2 Q^2)')
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)
