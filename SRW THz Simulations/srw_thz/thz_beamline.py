"""
Beamline construction and the sapphire-window frequency response.

Spatial propagation (apertures, focusing mirror, drifts) is handled by an
SRWLOptC container.  The flat sapphire window is a spatial no-op for an
on-axis uniform slab (constant amplitude + global phase), so its clear aperture
is modelled as an SRWLOptA in the beamline, while its FREQUENCY response
(Fresnel reflection + Fabry-Perot etalon + bulk absorption) is returned as an
analytic complex scalar t_win(f).  Because a global complex factor commutes with
all linear propagation, applying t_win(f) at the detector is exact and also
captures the etalon ring-down in the reconstructed E(t).

Layout (z increases downstream; foil waist at z=0):
  foil --L1--> [collimator ap] --L2--> [window ap] --L3--> [mirror/lens] --L4--> detector
The source wavefront is computed at the collimator plane (z = L1), so the
beamline starts at the collimator and the first drift is L2.
"""
import cmath
import math

from srwpy.srwlib import SRWLOptA, SRWLOptD, SRWLOptL, SRWLOptC

from thz_config import C_LIGHT


# ----------------------------------------------------------- propagation params
# [0] resize before, [1] resize after, [2] prec, [3] semi-analytic phase,
# [4] FFT-side resize, [5] H range factor, [6] H resolution factor,
# [7] V range factor, [8] V resolution factor, [9..11] shift (unused).
# Auto-resize before+after is enabled so SRW keeps the (large, strongly varying
# with lambda) THz beam contained and sampled; semi-analytic quadratic phase [3]=1
# on drifts. Manual factors are kept mild to bound mesh growth.
PP_APER_RESIZE = [1, 1, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]   # auto-resize at a hard clip
PP_DRIFT       = [1, 1, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]   # semi-analytic phase ON
PP_THIN        = [1, 1, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]   # lens/aperture
PP_DRIFT_FINAL = [1, 1, 1.0, 1, 0, 1.0, 1.3, 1.0, 1.3, 0, 0, 0]   # mild detector refine

# CAPPED set for the GF/CTR source. Its coordinate field is small (~cm), so SRW
# auto-resize runs away (and hangs) at low f trying to contain the hugely diffracting
# beam. Instead disable auto-resize and grow the range a FIXED 2x per drift at constant
# resolution: the mesh stays bounded (n_src * 2^(#drifts), n_src<=~100) and runs fast at
# every frequency. Low-f beams get mildly clipped (acceptable -- deep in the cutoff).
PP_CAP_THIN  = [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]   # aperture/lens: just mask
PP_CAP_DRIFT = [0, 0, 1.0, 1, 0, 2.0, 1.0, 2.0, 1.0, 0, 0, 0]   # drift: range x2, res x1


def sapphire_n_complex(f_Hz: float, cfg) -> complex:
    """Complex THz refractive index of sapphire at frequency f.
    n = n' + i*kappa, with kappa from a power absorption coefficient alpha(f).
    """
    alpha_1THz_per_m = cfg.sapphire_alpha_per_cm_at_1THz * 100.0   # 1/cm -> 1/m
    alpha = alpha_1THz_per_m * (f_Hz / 1e12) ** cfg.sapphire_alpha_freq_exp
    kappa = alpha * C_LIGHT / (4.0 * math.pi * f_Hz)              # alpha = 4 pi kappa f / c
    return complex(cfg.sapphire_n, kappa)


def window_transmission(f_Hz: float, cfg) -> complex:
    """Complex amplitude transmission of the sapphire window at normal incidence,
    beta = 2 pi n d f / c (n complex, so absorption is included via Im(n)).

    Depending on cfg.window_mode:
      'slab'  : full single-layer Fabry-Perot (etalon)
                t = (1-R) e^{i b} / (1 - R e^{2 i b}),  R = |r12|^2 = ((1-n)/(1+n))^2
      'wedge' : single pass only (multiple reflections walk off the aperture) ->
                same Fresnel insertion loss, NO etalon ripple/echoes:  t = (1-R) e^{i b}
      'ar'    : anti-reflection coated, surface reflectance -> cfg.ar_residual_R
    Note t12*t23 = 1 - r12^2 exactly, so (1-R) is the single-pass amplitude factor.
    Returns 1.0 if the window is disabled.
    """
    if not cfg.window_enabled:
        return 1.0 + 0j
    n = sapphire_n_complex(f_Hz, cfg)
    d = cfg.window_thickness_m
    beta = 2.0 * math.pi * n * d * f_Hz / C_LIGHT
    mode = getattr(cfg, 'window_mode', 'slab')

    if mode == 'ar':
        R = cfg.ar_residual_R                          # small residual surface reflectance
    else:
        r12 = (1.0 - n) / (1.0 + n)                    # air<->sapphire amplitude reflection
        R = r12 * r12                                  # = ((1-n)/(1+n))^2

    single_pass = (1.0 - R) * cmath.exp(1j * beta)     # Fresnel transmission + absorption
    if mode == 'wedge':
        return single_pass                             # etalon removed
    return single_pass / (1.0 - R * cmath.exp(2j * beta))  # 'slab' / 'ar': Fabry-Perot


def channel_response(f_Hz: float, cfg) -> complex:
    """Per-frequency complex scalar applied to the propagated field: window
    transmission times each mirror's reflectivity (gold ~1 in THz)."""
    t = window_transmission(f_Hz, cfg)
    if cfg.mirror_enabled:
        t = t * cfg.mirror_reflectivity
    if cfg.mirror_enabled and cfg.mirror2_enabled:
        t = t * cfg.mirror_reflectivity
    return t


def build_beamline(cfg, capped=False):
    """Return (SRWLOptC, names) for the SPATIAL beamline starting at the
    collimator plane (z = L1).  The window's complex frequency response is NOT
    here; apply channel_response(f) at extraction.

    capped=False: SRW auto-resize (good for the large, smooth Gaussian source).
    capped=True : bounded fixed range growth, no auto-resize (for the small GF
    source, whose auto-resize otherwise runs away/hangs at low f)."""
    pp_aper = PP_CAP_THIN if capped else PP_APER_RESIZE
    pp_drift = PP_CAP_DRIFT if capped else PP_DRIFT
    pp_thin = PP_CAP_THIN if capped else PP_THIN
    pp_final = PP_CAP_DRIFT if capped else PP_DRIFT_FINAL
    elems, pp, names = [], [], []

    # collimator aperture (sets the geometric low-frequency cutoff)
    elems.append(SRWLOptA('c', 'a', 2.0 * cfg.collimator_radius_m))
    pp.append(pp_aper)
    names.append("collimator_aperture")

    if cfg.window_enabled:
        elems.append(SRWLOptD(cfg.L_collimator_to_window_m)); pp.append(pp_drift)
        names.append("drift_collim_to_window")
        elems.append(SRWLOptA('c', 'a', 2.0 * cfg.window_clear_ap_radius_m))
        pp.append(pp_aper)
        names.append("window_aperture")

    if cfg.mirror_enabled:
        elems.append(SRWLOptD(cfg.L_window_to_mirror_m)); pp.append(pp_drift)
        names.append("drift_window_to_mirror1")
        elems.append(SRWLOptL(cfg.mirror_focal_length_m, cfg.mirror_focal_length_m))
        pp.append(pp_thin)
        names.append("mirror1_collimate")
        if cfg.mirror2_enabled:
            elems.append(SRWLOptD(cfg.L_mirror1_to_mirror2_m)); pp.append(pp_drift)
            names.append("drift_mirror1_to_mirror2")
            elems.append(SRWLOptL(cfg.mirror2_focal_length_m, cfg.mirror2_focal_length_m))
            pp.append(pp_thin)
            names.append("mirror2_focus")

    elems.append(SRWLOptD(cfg.L_mirror_to_detector_m)); pp.append(pp_final)
    names.append("drift_to_detector")

    return SRWLOptC(elems, pp), names


def detector_z_m(cfg) -> float:
    """Absolute longitudinal position of the detector plane [m] (for reference)."""
    z = cfg.z_first_oe_m
    if cfg.window_enabled:
        z += cfg.L_collimator_to_window_m
    if cfg.mirror_enabled:
        z += cfg.L_window_to_mirror_m
        if cfg.mirror2_enabled:
            z += cfg.L_mirror1_to_mirror2_m
    z += cfg.L_mirror_to_detector_m
    return z
