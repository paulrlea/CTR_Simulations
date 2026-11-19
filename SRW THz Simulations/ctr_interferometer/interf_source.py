"""
Source for the interferometer path: the Ginzburg-Frank CTR source (SAME profile as
the previous experiments) CONVOLVED with a Gaussian.

The reused `thz_source.src_ctr_wavefront` builds the analytic TR field in coordinate
space at the source plane (z0 = foil->window distance) with radial polarization
(Ex = E_r cos phi, Ey = E_r sin phi) and an exact on-axis null.  That is the field of
an idealized POINT charge.  A real electron bunch has a finite transverse size, which
smears the emission and rounds off the (unphysically sharp) on-axis null and 1/theta
tail.  We model that as a Gaussian convolution of the source field:

    E(x,y)  ->  E(x,y) (*) G_sigma(x,y)          [gf_conv_domain = 'field', default]

applied to Ex and Ey SEPARATELY so the radial polarization structure is preserved.
sigma = cfg.gf_conv_sigma_m (RMS of the kernel, in metres, at the source plane).

`gf_conv_sigma_m = 0` -> exact pass-through (unless the finite-foil / transverse form-factor
corrections below are enabled).

Two source-generation spectral corrections are also applied here (to make the absolute CTR
energy realistic; see interf_config):
  - finite-foil low-frequency rolloff: a per-frequency amplitude factor sqrt(S_foil(f)) with
    S_foil = [ln(min(2 gamma, k a)) - 1]/[ln(2 gamma) - 1] (a = foil radius) -- the dominant
    correction, suppressing the low-f coherent energy the ideal infinite foil overestimates;
  - transverse bunch form factor exp[-(k sigma_r theta)^2/2] (theta = r/z0) -- ~1 across the
    10-400 GHz band for mm beams, included for completeness.
"""
import math
import os
import sys
from array import array

import numpy as np
from scipy.ndimage import gaussian_filter

_SRW_THZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "srw_thz")
if _SRW_THZ not in sys.path:
    sys.path.insert(0, _SRW_THZ)

from thz_source import make_source  # noqa: E402
from thz_config import C_LIGHT      # noqa: E402


def foil_amp(f_Hz, cfg):
    """Finite-foil low-frequency suppression AMPLITUDE factor (energy ratio S -> sqrt(S)).
    S_foil(f) = [ln(min(2 gamma, k a)) - 1] / [ln(2 gamma) - 1],  a = cfg.foil_radius_m.
    -> 1 at high f (foil large vs gamma*lambda/2pi), rolls off at low f. 1.0 if disabled."""
    if not getattr(cfg, 'finite_foil_enabled', False):
        return 1.0
    gamma = cfg.gamma
    k = 2.0 * math.pi * f_Hz / C_LIGHT
    u = min(2.0 * gamma, k * cfg.foil_radius_m)
    denom = math.log(2.0 * gamma) - 1.0
    S = (math.log(u) - 1.0) / denom if (u > math.e and denom > 0) else 0.0
    # floor at a tiny nonzero value: a fully-zeroed field (deep-cutoff low f) collapses the
    # SRW mesh and breaks resampling; ~1e-8 energy is physically negligible but keeps it alive.
    return math.sqrt(max(1e-8, min(1.0, S)))


def _read_complex(wfr):
    """Complex Ex, Ey [ny, nx] from the interleaved SRW buffers."""
    nx, ny = wfr.mesh.nx, wfr.mesh.ny
    Ex = np.asarray(wfr.arEx, dtype=np.float64).reshape(ny, nx, 2)
    Ey = np.asarray(wfr.arEy, dtype=np.float64).reshape(ny, nx, 2)
    return Ex[..., 0] + 1j * Ex[..., 1], Ey[..., 0] + 1j * Ey[..., 1]


def _write_complex(wfr, Ex, Ey):
    """Write complex Ex, Ey [ny, nx] back into the interleaved SRW buffers."""
    nx, ny = wfr.mesh.nx, wfr.mesh.ny
    for arr, dat in ((wfr.arEx, Ex), (wfr.arEy, Ey)):
        buf = np.zeros(2 * nx * ny, dtype=np.float32)
        buf[0::2] = np.real(dat).astype(np.float32).ravel()
        buf[1::2] = np.imag(dat).astype(np.float32).ravel()
        a = array('f'); a.frombytes(buf.tobytes())
        arr[:] = a


def _gauss_blur_complex(Z, sig_y, sig_x):
    """Isotropic-in-physical-units Gaussian blur of a complex array (blur Re & Im)."""
    re = gaussian_filter(np.real(Z), sigma=(sig_y, sig_x), mode='constant', cval=0.0)
    im = gaussian_filter(np.imag(Z), sigma=(sig_y, sig_x), mode='constant', cval=0.0)
    return re + 1j * im


def make_source_conv(f_Hz, cfg):
    """GF CTR wavefront (thz_source), Gaussian-convolved (RMS cfg.gf_conv_sigma_m), then with
    the finite-foil low-frequency rolloff and transverse bunch form factor applied.

    Returns an SRWLWfr ready for the standard propagation path.
    """
    wfr = make_source(f_Hz, cfg, 'ctr')
    sigma = float(getattr(cfg, 'gf_conv_sigma_m', 0.0))
    af = foil_amp(f_Hz, cfg)                                  # finite-foil amplitude (per-freq)
    do_tff = getattr(cfg, 'transverse_ff_enabled', False) and cfg.beam_sigma_r_m > 0

    nx, ny = wfr.mesh.nx, wfr.mesh.ny
    dx = (wfr.mesh.xFin - wfr.mesh.xStart) / max(nx - 1, 1)
    dy = (wfr.mesh.yFin - wfr.mesh.yStart) / max(ny - 1, 1)
    sig_x = sigma / dx if (sigma > 0 and dx > 0) else 0.0
    sig_y = sigma / dy if (sigma > 0 and dy > 0) else 0.0
    do_conv = sig_x >= 1e-3 or sig_y >= 1e-3                  # kernel >= ~1 px

    if not (do_conv or do_tff or af != 1.0):                 # nothing to apply -> unmodified GF
        return wfr

    Ex, Ey = _read_complex(wfr)

    if do_conv:
        domain = getattr(cfg, 'gf_conv_domain', 'field')
        if domain == 'intensity':
            # incoherent source-size smearing: blur |E|^2 per component, keep the phase
            ax = np.sqrt(np.maximum(gaussian_filter(np.abs(Ex) ** 2, (sig_y, sig_x),
                                                    mode='constant', cval=0.0), 0.0))
            ay = np.sqrt(np.maximum(gaussian_filter(np.abs(Ey) ** 2, (sig_y, sig_x),
                                                    mode='constant', cval=0.0), 0.0))
            Ex = ax * np.exp(1j * np.angle(Ex))
            Ey = ay * np.exp(1j * np.angle(Ey))
        else:                                                # 'field' (coherent), default
            Ex = _gauss_blur_complex(Ex, sig_y, sig_x)
            Ey = _gauss_blur_complex(Ey, sig_y, sig_x)

    if do_tff:
        # transverse bunch form factor exp[-(k sigma_r theta)^2/2], theta = r/z0
        x = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, nx)
        y = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, ny)
        X, Y = np.meshgrid(x, y)
        k = 2.0 * math.pi * f_Hz / C_LIGHT
        z0 = cfg.z_first_oe_m
        Ftr = np.exp(-0.5 * (k * cfg.beam_sigma_r_m / z0) ** 2 * (X * X + Y * Y))
        Ex = Ex * Ftr; Ey = Ey * Ftr

    if af != 1.0:                                             # finite-foil global amplitude
        Ex = Ex * af; Ey = Ey * af

    _write_complex(wfr, Ex, Ey)
    return wfr
