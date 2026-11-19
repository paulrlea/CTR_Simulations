"""
Configuration for propagating the CTR-THz beam through the NEW interferometer
optical path (see draw_interferometer.py / beampath.png):

    Mo target -> fused-silica window -> 4" flat fold mirror -> pellicle BS (30/30)
              -> 75 mm flat retro mirror (delay arm) -> back to BS -> gold OAP -> detector

This reuses the SAME source profile as the previous experiments (`srw_thz/`): the
Ginzburg-Frank CTR source (analytic TR ring, on-axis null, radial polarization),
here additionally CONVOLVED with a Gaussian (finite transverse source-size smearing;
see interf_source.py).  SRW is again only the wavefront-propagation engine.

`InterfConfig` subclasses `srw_thz/thz_config.ThzConfig` so every beam / frequency /
GF-source field the reused source code needs is inherited unchanged; we only add the
new geometry, apertures, beamsplitter split, fused-silica window constants, and the
convolution knob.

Scope (per request): PROPAGATION ONLY -- no delay scan / interferogram.  The delay
arm is modelled as the physical path the light takes to the detector; delay = 0.

All distances / focal lengths / window constants marked "BEST GUESS" are not pinned
by the (schematic) drawing and are meant to be tuned.  SI units (metres) throughout.
"""
import os
import sys
from dataclasses import dataclass

# make the reused srw_thz modules importable regardless of cwd
_SRW_THZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "srw_thz")
if _SRW_THZ not in sys.path:
    sys.path.insert(0, _SRW_THZ)

from thz_config import ThzConfig  # noqa: E402  (path set above)


@dataclass
class InterfConfig(ThzConfig):
    # ------------------------------------------------------------------ source
    # Use the Ginzburg-Frank CTR source (same as previous --src ctr), then convolve
    # with a Gaussian.  gf_conv_sigma_m is the RMS of the transverse Gaussian kernel
    # applied to the source FIELD at the source plane (z0 = foil->window distance),
    # representing the finite electron-beam transverse size that rounds the idealized
    # point-charge on-axis null.  0.0 -> exact GF pass-through (no convolution).
    gf_conv_sigma_m: float = 2.0e-3          # BEST GUESS (few-mm beam spot)
    gf_conv_domain: str = "field"            # 'field' (coherent, default) or 'intensity'

    # ---- source-generation spectral corrections (make the absolute energy realistic) ----
    # Finite target (Mo foil) low-frequency rolloff: the infinite-foil TR log integral runs to
    # impact parameter ~gamma*lambda/2pi; a finite disk of radius a caps it at a when
    # a < gamma*lambda/2pi -> progressive low-f suppression. This is the DOMINANT correction to
    # the (over)estimated absolute CTR energy and it reshapes the low-f spectrum.
    finite_foil_enabled: bool = True
    # foil_radius_m is inherited from ThzConfig (0.5" radius); shared with finite_foil_diffraction.
    # Transverse bunch form factor exp[-(k sigma_r theta)^2/2]: coherent suppression from finite
    # transverse beam size. NOTE: ~1 across 10-400 GHz for mm-scale beams (only bites above ~few
    # THz), and it overlaps conceptually with gf_conv_sigma_m (both are "finite transverse size")
    # -- don't crank both to represent one large size.
    transverse_ff_enabled: bool = True
    beam_sigma_r_m: float = 1.0e-3           # BEST GUESS transverse rms beam size on the foil

    # --------------------------------------------------------------- geometry
    # Longitudinal layout of the new path (foil waist at z=0).  The GF source is
    # computed at the FIRST optical element = the fused-silica window (z0 below);
    # the beamline then starts at the window.  Distances below are the AS-BUILT values
    # (user-provided drift lengths between elements), not best-guess.
    #   foil --Lfw--> window --Lwf--> fold(4") --Lfb--> BS(30/30)
    #        --Lbd--> retro(75mm) --Lbd--> BS  --Lbo--> OAP --Lod--> detector
    L_foil_to_window_m: float = 0.18415         # foil -> fused-silica window  (= z0)
    L_window_to_fold_m: float = 0.18415         # window -> 4" fold mirror
    L_fold_to_bs_m: float = 0.3433826         # fold  -> pellicle beamsplitter
    L_bs_to_delay_m: float = 0.0340868         # BS -> 75 mm retro mirror (delay arm, NOMINAL; one way, x2 round trip)
    L_bs_to_reference_m: float = 1.854 * 0.0254  # BS -> fixed flat mirror (reference arm; 1.854")
    L_bs_to_oap_m: float = 0.0823976              # BS -> gold OAP
    L_oap_to_detector_m: float = 0.080391        # OAP -> detector
    # The reference arm (1.854") is longer than the delay arm's nominal length (1.342") by
    # dL = L_ref - L_bs_to_delay.  ZERO PATH DIFFERENCE therefore occurs when the delay stage
    # is displaced by dL (~13.0 mm), i.e. at relative-delay offset tau0 = 2*dL/c (~86.8 ps),
    # NOT at the stage's nominal position.  This shifts only the interferogram's stage-travel
    # axis / ZPD marker -- the beam transport (2.8% path difference) and the recovered
    # spectrum are unchanged, so no re-propagation is needed.

    # Single focusing OAP.  Default focal length images the source plane (total folded
    # foil->OAP path) onto the detector for the default distances:
    #   1/f = 1/s_o + 1/s_i,  s_o = Lfw+Lwf+Lfb+2*Lbd+Lbo,  s_i = Lod.
    # (Recomputed in __post_init__ unless the user overrides oap_focal_length_m.)
    oap_focal_length_m: float = 0.0          # 0 -> auto (image condition); else fixed [m]
    oap_reflectivity: float = 1.0            # gold ~1 in THz

    # ----------------------------------------------------------- apertures [m]
    window_clear_ap_radius_m: float = 0.020  # BEST GUESS fused-silica clear aperture radius
    fold_mirror_dia_m: float = 4.0 * 0.0254  # 4"  first-surface fold mirror
    bs_dia_m: float = 3.0 * 0.0254           # 3"  pellicle beamsplitter
    retro_mirror_dia_m: float = 0.075        # 75 mm flat delay/retro mirror
    oap_dia_m: float = 3.0 * 0.0254          # BEST GUESS OAP clear aperture (~3")

    # ------------------------------------------------------- beamsplitter 30/30
    # Pellicle "30/30": 30% reflected, 30% transmitted (rest lost).  Amplitudes
    # r = sqrt(R), t = sqrt(T).  Delay-arm light reflects INTO the arm then transmits
    # OUT toward the OAP, so its BS amplitude factor is r*t (see interf_beamline).
    bs_R: float = 0.30                       # intensity reflectance
    bs_T: float = 0.30                       # intensity transmittance
    # 2 um pellicle: etalon FSR (c/2nd ~ tens of THz) is far above the 10-400 GHz band,
    # so the BS is treated as an ideal thin splitter (no BS etalon) -- scalar split only.

    # ---------------------------------------------- mirror reflectivities (amp)
    fold_reflectivity: float = 1.0           # Si-overcoat first-surface mirror (~1 in THz)
    retro_reflectivity: float = 1.0          # protected Al 75 mm flat (~1 in THz)

    # ------------------------------------------------- propagation mesh control
    # The GF source needs CAPPED propagation (no SRW auto-resize; it runs away at low f).
    # Instead the mesh RANGE grows a fixed factor per drift at constant resolution, so the
    # point count is n_src * cap_range_factor^(#drifts).  This path has 6 drifts, so the
    # old model's 2.0 would balloon the mesh (2^6=64x); 1.6 (1.6^6~17x) still contains the
    # beam at its widest plane (the OAP) while keeping wavefronts a few M points.  Low-f
    # beams (deep in the collimation cutoff) clip mildly -- acceptable.
    cap_range_factor: float = 1.6

    # --------------------------------------------- window (fused silica / quartz)
    # Generic (material-agnostic) window fields so the previous sapphire model in
    # thz_config is left untouched.  Fused-silica THz constants are literature/BEST
    # GUESS: n' ~ 1.95, weak-but-nonzero absorption rising ~ f^2.  Surface reflectance
    # R = ((n-1)/(n+1))^2 ~ 10% (vs sapphire ~26%), so a milder etalon.
    window_n: float = 1.95                            # real refractive index (THz)
    window_alpha_per_cm_at_1THz: float = 3.0          # BEST GUESS power abs. coeff [1/cm] @1 THz
    window_alpha_freq_exp: float = 2.0                # alpha(f) ~ alpha_1THz*(f/1THz)^exp
    # window_thickness_m / window_enabled / window_mode / ar_residual_R are inherited
    # from ThzConfig; override thickness for the fused-silica viewport (BEST GUESS).
    window_thickness_m: float = 0.003                 # ~3 mm fused-silica window

    # ---------------------------------------------------------------- output
    out_dir: str = "results_interf_dist_true"

    # ----------------------------------------------------- derived / overrides
    def __post_init__(self):
        # Double-count guard: the finite-foil DIFFRACTION form factor H(theta) already
        # reproduces the low-f energy suppression that the scalar S_foil (finite_foil_enabled)
        # approximates. Never apply both -- disable the scalar rolloff when diffraction is on.
        if self.finite_foil_diffraction and self.finite_foil_enabled:
            print("    [finite-foil] diffraction form factor ON -> disabling scalar S_foil "
                  "rolloff (finite_foil_enabled) to avoid double counting.")
            self.finite_foil_enabled = False

        # Auto focal length for the single OAP (image source plane onto detector),
        # unless the user pinned oap_focal_length_m > 0.
        if self.oap_focal_length_m <= 0.0:
            s_o = (self.L_foil_to_window_m + self.L_window_to_fold_m + self.L_fold_to_bs_m
                   + 2.0 * self.L_bs_to_delay_m + self.L_bs_to_oap_m)
            s_i = self.L_oap_to_detector_m
            self.oap_focal_length_m = (s_o * s_i) / (s_o + s_i)

    @property
    def z_first_oe_m(self) -> float:
        """First optical element = the fused-silica window; the GF source is built
        here (z0 sets the ring radius z0/gamma) and the beamline starts here."""
        return self.L_foil_to_window_m


def default_interf_config(**overrides) -> InterfConfig:
    """InterfConfig with field overrides, e.g. default_interf_config(n_freq=9)."""
    cfg = InterfConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"InterfConfig has no field '{k}'")
        setattr(cfg, k, v)
    # re-run the auto-focal-length derivation if geometry was overridden and f not pinned
    if "oap_focal_length_m" not in overrides:
        cfg.oap_focal_length_m = 0.0
        cfg.__post_init__()
    return cfg
