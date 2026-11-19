"""
Beamline construction for the NEW interferometer optical path, and the fused-silica
window frequency response.

Spatial propagation (apertures, focusing OAP, drifts) is an SRWLOptC container.  The
flat fused-silica window and the flat mirrors are spatial no-ops (constant amplitude +
global phase over their clear aperture), so each contributes only (a) its clear
aperture in the beamline and (b) a per-frequency complex scalar collected in
`channel_response_interf` (window transmission x mirror reflectivities x beamsplitter
split).  A global complex factor commutes with all linear propagation, so applying it
at extraction is exact (the same trick the sapphire model uses in thz_beamline).

Scope: PROPAGATION ONLY.  We model the physical path the DELAY-ARM light takes to the
detector (reflect into the arm at the BS, retro-reflect, transmit out toward the OAP);
delay = 0, no reference arm, no interferogram.

Layout (z increases downstream; foil waist at z=0; source computed at the window plane
z0 = L_foil_to_window):
  window --Lwf--> fold(4") --Lfb--> BS --Lbd--> retro(75mm) --Lbd--> BS --Lbo--> OAP --Lod--> det
"""
import cmath
import math
import os
import sys

from srwpy.srwlib import SRWLOptA, SRWLOptD, SRWLOptL, SRWLOptC

_SRW_THZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "srw_thz")
if _SRW_THZ not in sys.path:
    sys.path.insert(0, _SRW_THZ)

from thz_config import C_LIGHT            # noqa: E402
from thz_beamline import PP_CAP_THIN   # noqa: E402  (capped 'thin': GF source needs it)


# ------------------------------------------------- fused-silica window response
def window_n_complex(f_Hz, cfg):
    """Complex THz refractive index of the (fused-silica) window: n = n' + i*kappa,
    kappa from a power absorption coefficient alpha(f).  Reads the generic
    window_n / window_alpha_* fields (so the sapphire model in thz_config is untouched)."""
    alpha_1THz_per_m = cfg.window_alpha_per_cm_at_1THz * 100.0        # 1/cm -> 1/m
    alpha = alpha_1THz_per_m * (f_Hz / 1e12) ** cfg.window_alpha_freq_exp
    kappa = alpha * C_LIGHT / (4.0 * math.pi * f_Hz)                  # alpha = 4 pi kappa f / c
    return complex(cfg.window_n, kappa)


def window_transmission_fs(f_Hz, cfg):
    """Complex amplitude transmission of the fused-silica window at normal incidence,
    beta = 2 pi n d f / c (n complex -> absorption via Im(n)).  Same Fresnel + Fabry-Perot
    (+ AR / wedge modes) form as thz_beamline.window_transmission, but for the generic
    fused-silica constants.  Returns 1.0 if the window is disabled."""
    if not cfg.window_enabled:
        return 1.0 + 0j
    n = window_n_complex(f_Hz, cfg)
    beta = 2.0 * math.pi * n * cfg.window_thickness_m * f_Hz / C_LIGHT
    mode = getattr(cfg, 'window_mode', 'slab')

    if mode == 'ar':
        R = cfg.ar_residual_R
    else:
        r12 = (1.0 - n) / (1.0 + n)
        R = r12 * r12

    single_pass = (1.0 - R) * cmath.exp(1j * beta)          # Fresnel + absorption, one pass
    if mode == 'wedge':
        return single_pass                                  # etalon walked off
    return single_pass / (1.0 - R * cmath.exp(2j * beta))   # 'slab'/'ar': Fabry-Perot


def channel_response_interf(f_Hz, cfg):
    """Per-frequency complex scalar applied to the propagated field:
        window transmission x fold reflectivity x (BS reflect-in * transmit-out)
        x retro reflectivity x OAP reflectivity.
    Delay-arm BS bookkeeping: the light reflects INTO the arm (amplitude r=sqrt(R))
    then transmits OUT toward the OAP (amplitude t=sqrt(T)), so the BS factor is r*t."""
    t = window_transmission_fs(f_Hz, cfg)
    t = t * cfg.fold_reflectivity
    t = t * math.sqrt(cfg.bs_R) * math.sqrt(cfg.bs_T)       # reflect in, transmit out
    t = t * cfg.retro_reflectivity
    t = t * cfg.oap_reflectivity
    return t


# ------------------------------------------------------------------- beamline
def build_interf_beamline(cfg, capped=True, apertures=True):
    """Return (SRWLOptC, names) for the SPATIAL interferometer beamline starting at the
    window plane (z0 = L_foil_to_window).  Apply channel_response_interf(f) at extraction.

    capped=True (default): bounded fixed-range growth, no auto-resize -- required for the
    GF/CTR source, whose auto-resize otherwise runs away/hangs at low f (matches the old
    --src ctr path).  The per-drift range growth is cfg.cap_range_factor (milder than the
    old 2.0 because this path has more drifts).  capped=False available for smooth sources.

    apertures=True (default): include every physical clear-aperture stop (window, fold, BS,
    retro, OAP).  apertures=False removes ALL stops (drifts + OAP lens only) -- the beam is
    never trimmed, so you see the full, unclipped CTR pattern the detector plane would carry
    in the absence of finite optics (a reference for how much the stops vignette)."""
    rf = getattr(cfg, 'cap_range_factor', 1.6)
    # capped drift: no auto-resize, grow H+V range by rf at constant resolution
    pp_cap_drift = [0, 0, 1.0, 1, 0, rf, 1.0, rf, 1.0, 0, 0, 0]
    pp_thin = PP_CAP_THIN if capped else [1, 1, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
    pp_drift = pp_cap_drift if capped else [1, 1, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
    pp_final = pp_drift
    elems, pp, names = [], [], []

    def add(elem, params, name):
        elems.append(elem); pp.append(params); names.append(name)

    def add_aperture(elem, params, name):
        if apertures:
            add(elem, params, name)

    # fused-silica window clear aperture (source is computed at this plane)
    if cfg.window_enabled:
        add_aperture(SRWLOptA('c', 'a', 2.0 * cfg.window_clear_ap_radius_m), pp_thin, "window_aperture")

    # -> 4" first-surface fold mirror ("L3 monitor")
    add(SRWLOptD(cfg.L_window_to_fold_m), pp_drift, "drift_window_to_fold")
    add_aperture(SRWLOptA('c', 'a', cfg.fold_mirror_dia_m), pp_thin, "fold_mirror_aperture")

    # -> pellicle beamsplitter (first pass: reflect into the delay arm)
    add(SRWLOptD(cfg.L_fold_to_bs_m), pp_drift, "drift_fold_to_bs")
    add_aperture(SRWLOptA('c', 'a', cfg.bs_dia_m), pp_thin, "beamsplitter_aperture_in")

    # -> 75 mm flat retro mirror (delay arm) and back to the BS
    add(SRWLOptD(cfg.L_bs_to_delay_m), pp_drift, "drift_bs_to_retro")
    add_aperture(SRWLOptA('c', 'a', cfg.retro_mirror_dia_m), pp_thin, "retro_mirror_aperture")
    add(SRWLOptD(cfg.L_bs_to_delay_m), pp_drift, "drift_retro_to_bs")

    # -> beamsplitter (second pass: transmit out toward the OAP)
    add_aperture(SRWLOptA('c', 'a', cfg.bs_dia_m), pp_thin, "beamsplitter_aperture_out")

    # -> gold OAP (single focusing element) -> detector
    add(SRWLOptD(cfg.L_bs_to_oap_m), pp_drift, "drift_bs_to_oap")
    add(SRWLOptL(cfg.oap_focal_length_m, cfg.oap_focal_length_m), pp_thin, "oap_focus")
    add_aperture(SRWLOptA('c', 'a', cfg.oap_dia_m), pp_thin, "oap_aperture")
    add(SRWLOptD(cfg.L_oap_to_detector_m), pp_final, "drift_oap_to_detector")

    return SRWLOptC(elems, pp), names


def detector_z_m(cfg):
    """Absolute longitudinal position of the detector plane [m] along the folded path."""
    return (cfg.L_foil_to_window_m + cfg.L_window_to_fold_m + cfg.L_fold_to_bs_m
            + 2.0 * cfg.L_bs_to_delay_m + cfg.L_bs_to_oap_m + cfg.L_oap_to_detector_m)
