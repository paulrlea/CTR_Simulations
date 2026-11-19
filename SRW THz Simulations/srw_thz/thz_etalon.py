#!/usr/bin/env python
"""
Quantify the sapphire-window Fabry-Perot (etalon) effect and compare mitigations.

The etalon is caused ENTIRELY by the sapphire window (the only multi-surface
element): each air-sapphire face reflects R = ((n-1)/(n+1))^2 of the THz, and the
two faces form a cavity.  Everything here derives from `window_transmission`.

Provides:
  - etalon_metrics(cfg)            -> dict of numbers (FSR, finesse, contrast, ...)
  - plot_window_transmission(...)  -> |t_win(f)|^2 for slab vs wedge vs AR
  - plot_spectrum_etalon(...)      -> the measured spectrum re-weighted to each
                                      window mode (shows the ripple being removed),
                                      computed by rescaling, no SRW re-run needed.

Standalone:  python thz_etalon.py        # prints metrics, writes plots into results/
"""
import os
from dataclasses import replace

import numpy as np

from thz_config import default_config, C_LIGHT
from thz_beamline import window_transmission


def _Tcurve(cfg, f_Hz):
    """|t_win(f)|^2 (power transmittance) for the given config/mode over array f."""
    return np.array([abs(window_transmission(fi, cfg)) ** 2 for fi in f_Hz])


def etalon_metrics(cfg, f_lo_GHz=5.0, f_hi_GHz=400.0, n=40000):
    """Return a dict of etalon descriptors, mixing closed-form (n, d) values with
    measurements off the actual transmittance curve."""
    n_re = cfg.sapphire_n
    d = cfg.window_thickness_m
    R = cfg.ar_residual_R if cfg.window_mode == 'ar' else ((n_re - 1) / (n_re + 1)) ** 2

    FSR = C_LIGHT / (2 * n_re * d)                 # fringe spacing [Hz]
    rt_delay = 2 * n_re * d / C_LIGHT              # echo spacing [s] (=1/FSR)
    finesse = np.pi * np.sqrt(R) / (1 - R) if R < 1 else float('inf')

    f = np.linspace(f_lo_GHz * 1e9, f_hi_GHz * 1e9, n)
    T = _Tcurve(cfg, f)
    Tmax, Tmin, Tmean = float(T.max()), float(T.min()), float(T.mean())
    contrast = (Tmax - Tmin) / (Tmax + Tmin)      # visibility

    # FSR measured from the curve autocorrelation (robust to the absorption envelope)
    g = T - T.mean()
    ac = np.correlate(g, g, 'full')[len(g) - 1:]
    df = f[1] - f[0]
    loc = np.where((ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]))[0]
    FSR_meas = (loc[0] + 1) * df if len(loc) else float('nan')

    return {
        'window_mode': cfg.window_mode,
        'n': n_re, 'thickness_mm': d * 1e3,
        'R_surface': R,
        'FSR_GHz': FSR / 1e9,
        'FSR_measured_GHz': FSR_meas / 1e9,
        'finesse': finesse,
        'roundtrip_delay_ps': rt_delay * 1e12,
        'echo_amp_ratio': R,                       # field amplitude of each successive echo
        'T_min': Tmin, 'T_max': Tmax, 'T_mean': Tmean,
        'peak_to_valley': Tmax - Tmin,
        'visibility': contrast,
    }


def print_metrics(cfg):
    m = etalon_metrics(cfg)
    print(f"Etalon metrics (window_mode={m['window_mode']}, sapphire n={m['n']}, "
          f"d={m['thickness_mm']:.3f} mm):")
    print(f"  surface reflectance R     = {m['R_surface']:.3f}")
    print(f"  FSR (fringe spacing)      = {m['FSR_GHz']:.2f} GHz "
          f"(measured {m['FSR_measured_GHz']:.2f})")
    print(f"  finesse                   = {m['finesse']:.2f}")
    print(f"  round-trip echo delay     = {m['roundtrip_delay_ps']:.2f} ps")
    print(f"  echo amplitude ratio      = {m['echo_amp_ratio']:.3f} (field, per round trip)")
    print(f"  transmittance T_min..T_max= {m['T_min']:.3f} .. {m['T_max']:.3f} "
          f"(mean {m['T_mean']:.3f})")
    print(f"  peak-to-valley / visibility = {m['peak_to_valley']:.3f} / {m['visibility']:.3f}")
    return m


# ----------------------------------------------------------------- plots
_MODES = [('slab', 'C0', 'slab (etalon)'),
          ('wedge', 'C1', 'wedged (no etalon)'),
          ('ar', 'C2', 'AR-coated')]


def plot_window_transmission(cfg, path_png, f_lo_GHz=10.0, f_hi_GHz=400.0, n=2000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = np.linspace(f_lo_GHz * 1e9, f_hi_GHz * 1e9, n)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for mode, c, lab in _MODES:
        T = _Tcurve(replace(cfg, window_mode=mode), f)
        ax.plot(f / 1e9, T, color=c, lw=1.3, label=lab)
    m = etalon_metrics(cfg)
    ax.set_xlabel('f [GHz]'); ax.set_ylabel(r'window power transmittance $|t_{win}|^2$')
    ax.set_title(f"Sapphire window transmission "
                 f"(FSR {m['FSR_GHz']:.1f} GHz, T={m['T_min']:.2f}-{m['T_max']:.2f})")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)


def plot_spectrum_etalon(cfg, spectrum_txt, path_png):
    """Overlay the current spectrum (with the sapphire etalon) against the same
    spectrum with the etalon removed (a wedged window: identical Fresnel insertion
    loss, no ripple). The window enters detected power as a per-frequency scalar
    |t_win(f)|^2, so the etalon-removed curve = P * |t_wedge|^2 / |t_slab|^2
    (exact; no SRW re-run)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.loadtxt(spectrum_txt)
    f_GHz, P = data[:, 0], data[:, 1]
    f = f_GHz * 1e9
    Tslab = _Tcurve(replace(cfg, window_mode='slab'), f)
    Twedge = _Tcurve(replace(cfg, window_mode='wedge'), f)
    P_noetalon = P * Twedge / np.where(Tslab > 0, Tslab, np.nan)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(f_GHz, P, color='C0', lw=1.2, marker='.', ms=2.5, label='current (with etalon)')
    ax.plot(f_GHz, P_noetalon, color='C1', lw=1.6, label='etalon removed (wedged)')
    ax.set_xlabel('f [GHz]'); ax.set_ylabel('detected power [arb]')
    ax.set_title('THz spectrum at detector: etalon vs etalon-removed')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)


def write_analysis(cfg, out_dir):
    """Print metrics and write both plots into out_dir (spectrum overlay only if
    a spectrum.txt is present)."""
    print_metrics(cfg)
    plot_window_transmission(cfg, os.path.join(out_dir, 'etalon_transmission.png'))
    spec = os.path.join(out_dir, 'spectrum.txt')
    if os.path.exists(spec):
        plot_spectrum_etalon(cfg, spec, os.path.join(out_dir, 'etalon_spectrum.png'))


if __name__ == '__main__':
    cfg = default_config()
    write_analysis(cfg, cfg.out_dir)
    print(f"wrote etalon_transmission.png"
          f"{' + etalon_spectrum.png' if os.path.exists(os.path.join(cfg.out_dir,'spectrum.txt')) else ''}"
          f" to {cfg.out_dir}/")
