"""
Central configuration for the AWA CTR-THz SRW propagation model.

Physics context (see ../.claude/plans/...simulate...md):
  - Source: Coherent Transition Radiation (CTR) from a radiative foil at the AWA,
    ~60 MeV beam (gamma ~ 120), ps bunches (0.25-10 nC). THz band ~10-400 GHz.
  - SRW cannot generate transition radiation, so the source is SYNTHESIZED as a
    per-frequency Gaussian beam (SRWLGsnBm) whose divergence is tuned to the CTR
    ~1/gamma opening angle, weighted by the bunch form factor F(f).
  - SRW is used as the wavefront-PROPAGATION engine through the beamline.

Values marked "BEST GUESS" are not pinned by the source documents and are meant
to be tuned. Everything is SI (metres, seconds, Hz) except photon energy (eV),
which SRW uses internally.
"""
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
C_LIGHT = 299_792_458.0          # m/s
H_PLANCK_EV = 4.135667696e-15    # eV*s   (E[eV] = H_PLANCK_EV * f[Hz])
ME_C2_GEV = 0.51099895000e-3     # electron rest energy [GeV]


@dataclass
class ThzConfig:
    # ------------------------------------------------------------------ beam
    beam_energy_MeV: float = 60.0          # AWA nominal
    bunch_charge_nC: float = 10.0          # CTR power ~ Q^2 (global scale only)
    sigma_t_ps: float = 2.0                # rms bunch length (sets form factor & spectrum)

    # ------------------------------------------------------- frequency grid
    f_min_GHz: float = 10.0
    f_max_GHz: float = 400.0
    n_freq: int = 161                      # sweep points (coarse=5..9 for smoke tests)
    freq_log: bool = False                 # linear grid by default

    # ----------------------------------------------- source (Gaussian surrogate)
    # sigma_x(f) = div_match * lambda(f) * gamma / pi  -> divergence ~ 1/gamma.
    # div_match is the knob tuned in verification so the low-f rolloff matches
    # the measured ~45 GHz cutoff.
    div_match: float = 1.0
    sampFactNxNyForProp: float = 1.0       # SRW auto-sampling factor (sweep 0.3-3)
    src_nx: int = 120                      # initial mesh points (auto-resized later)
    src_ny: int = 120
    # The source window must contain the (large, at low f) THz beam. Its half-size
    # scales with the source rms size sigma_r(f) ~ lambda*gamma/pi, with a floor.
    src_half_ap_sigmas: float = 3.5        # half-window = this many sigma_r
    src_half_ap_floor_m: float = 0.05      # minimum half-window [m] (high-f)
    # Memory guard: cap the source transverse mesh (points per axis). The mesh grows
    # ~ sampFact*half^2/(lambda*z), which explodes at low f; the guard throttles the
    # effective sampFact there (where fine sampling is unneeded) and logs it.
    max_src_pts: int = 4000

    # ------------------------------------- source (B): Ginzburg-Frank CTR (--src ctr)
    # Analytic transition-radiation angular pattern E(theta) ~ beta sin/(1-beta^2 cos^2),
    # filled in angular space then FFT'd to coordinates. On-axis null + radial pol.
    gf_n_sigma: float = 5.0        # angular coverage +/- gf_n_sigma/gamma [rad]
    gf_n_ang: int = 256            # MAX source mesh points/axis (used at high f)
    gf_min_ang: int = 32           # MIN source mesh points/axis (low f, coarse blob)
    gf_max_ang_pts: int = 512      # safety cap on gf_n_ang
    # Source dx = max(ring/6, gf_src_lambda_frac*lambda). The lambda floor prevents
    # sub-wavelength oversampling at low f (where the ring is smaller than lambda and
    # physically unresolvable) -- that oversampling makes SRW propagation explode.
    gf_src_lambda_frac: float = 0.25
    gf_ring_dr_frac: float = 0.20  # waveform sampled over annulus dr = frac * r_peak
    # The GF pattern has a heavy 1/theta tail, so the source mesh must extend PAST the
    # collimator aperture (the real, physical truncator) -- otherwise the mesh boundary
    # chops the pattern inside the collimator and creates a spurious diffracting edge.
    gf_src_aper_margin: float = 1.5  # source half-extent >= this * collimator_radius
    # Tilted-screen asymmetric CTR (Level-1 heuristic; GF source only). Tilt is in the
    # x (horizontal) plane. screen_tilt_deg=0 -> symmetric GF (exact no-op). Nonzero:
    #   - foreshortens the angular pattern in x by cos(tilt) (elongated source footprint
    #     -> narrower far-field lobe in the tilt plane);
    #   - biases the two x-lobes by 1 + gf_tilt_lobe_k*tan(tilt)*tanh(theta_x*gamma)
    #     (one lobe brightened toward the specular direction, the other dimmed).
    screen_tilt_deg: float = 0.0
    gf_tilt_lobe_k: float = 1.0      # lobe-asymmetry strength (0 = foreshortening only)
    # Tilt model: 'level1' = geometric heuristic above (foreshorten + tanh lobe, radial pol).
    # 'level3' = exact point-charge oblique-TR vector field with true p/s polarization
    # (image-charge result; see LEVEL3_DERIVATION.md / LEVEL3_PLAN.md). level3 reduces to
    # symmetric GF at screen_tilt_deg=0 and ignores gf_tilt_lobe_k.
    # 'analytic' = the supplied closed-form 45-deg TR relation (gf_analytic_field); an algebraic
    # rewrite of level3, identical at psi=45 (verified in compare_analytic_45.py). The foil angle
    # is fixed at 45 deg by the relation, so screen_tilt_deg is only used to trigger the
    # polarization-aware extraction (set it to 45 to match the physics).
    gf_tilt_model: str = 'level1'
    gf_include_electron_term: bool = True   # keep the non-resonant electron term in the
                                            # oblique amplitude (the source of the p/s split);
                                            # False = image term only (radial), for comparison
    # EO-crystal sensitive (electro-optic) axis azimuth in the x-y plane [deg]; the detected
    # field is projected onto this direction (level3 polarization-aware extraction).
    crystal_pol_axis_deg: float = 0.0

    # ---- finite-foil (radiator) diffraction (far-zone; see FINITE_FOIL_PLAN.md) ----
    # When True, multiply the CTR source field by the far-zone finite-foil form factor
    #   H(theta) = 1 - x K1(x) J0(y) - (x^2/y) K0(x) J1(y),   x = k a /(beta gamma), y = k a theta_ell
    # (an exact Lommel closed form for the truncated Hankel transform of the virtual-photon
    # field), with the 45-deg elliptical foreshortening carried through
    #   theta_ell = sqrt(cos^2(psi) theta_x^2 + theta_y^2),  psi = screen_tilt_deg.
    # H -> 1 as a -> inf (recovers the infinite-foil source). This adds ring broadening,
    # partial null-fill, and edge-diffraction fringes AND supersedes the scalar S_foil
    # low-f rolloff (interf_source.finite_foil_enabled must be OFF when this is ON -- the
    # drivers enforce this to avoid double counting).
    finite_foil_diffraction: bool = False
    foil_radius_m: float = 0.5 * 0.0254     # radiator (Mo foil) radius a; 1" target -> 0.5" radius

    # ------------------------------------------------------------- geometry
    # Longitudinal layout from the foil (waist at z=0), 2-mirror relay:
    #   foil -L1-> collimator -L2-> window -L3-> OAP1 -L34-> OAP2 -L4-> detector
    # All distances BEST GUESS (not given in the documents).
    L_foil_to_collimator_m: float = 0.50   # L1
    L_collimator_to_window_m: float = 0.20 # L2
    L_window_to_mirror_m: float = 0.30     # L3 (window -> OAP1)
    L_mirror1_to_mirror2_m: float = 0.30   # L34 (OAP1 -> OAP2; collimated space)
    L_mirror_to_detector_m: float = 0.30   # L4 (last mirror -> detector)

    # collection aperture (clips the radiation; sets geometric cutoff) -- from analysis
    collimator_radius_m: float = 0.023     # ~2.3 cm radius

    # ------------------------------------------------------- window (sapphire)
    # DN63CF UV-sapphire viewport (user datasheet). The datasheet "50%@250nm" is a
    # UV spec and is NOT used here; THz constants below come from literature.
    window_thickness_m: float = 0.080 * 0.0254     # 0.080 in -> 2.032 mm
    window_clear_ap_radius_m: float = 0.0175       # ~35 mm clear aperture (DN63CF) -> r~17.5 mm
    window_enabled: bool = True
    # Crystalline sapphire (Al2O3) ordinary-ray THz optical constants.
    # Below ~0.5 THz sapphire is nearly transparent; n ~ 3.07, weak absorption.
    # (Birefringence ignored in first pass; extraordinary n ~ 3.4.)
    sapphire_n: float = 3.07               # ordinary-ray refractive index (real)
    sapphire_alpha_per_cm_at_1THz: float = 2.0  # power absorption coeff [1/cm] ref @1 THz
    # crude absorption scaling alpha(f) ~ alpha_1THz * (f/1THz)^2 (far-IR-like)
    sapphire_alpha_freq_exp: float = 2.0
    # Window optical model (etalon mitigation study):
    #   'slab'  - flat parallel slab -> full Fabry-Perot etalon (the real DN63CF window)
    #   'wedge' - wedged window: same Fresnel insertion loss, etalon walked off (no ripple/echoes)
    #   'ar'    - anti-reflection coated: surface reflectance -> ar_residual_R (~no etalon)
    window_mode: str = 'slab'
    ar_residual_R: float = 0.01            # residual surface intensity reflectance for 'ar'

    # ----------------------------------------------------------- mirrors/focus
    # Two-mirror relay, each modeled as a thin-lens-equivalent off-axis paraboloid.
    # OAP1 collimates the diverging THz: f1 ~ source(foil)->OAP1 distance
    #   (= L1+L2+L3 = 0.50+0.20+0.30 = 1.00 m by default).
    # OAP2 focuses the collimated beam onto the crystal: f2 ~ OAP2->detector (= L4).
    # Focal lengths BEST GUESS; reflectivity ~1 (gold in THz), applied per mirror.
    mirror_enabled: bool = True            # OAP1 (collimating)
    mirror_focal_length_m: float = 1.00    # f1
    mirror2_enabled: bool = True           # OAP2 (focusing)
    mirror2_focal_length_m: float = 0.30   # f2
    mirror_reflectivity: float = 1.0       # per-mirror amplitude reflectivity

    # --------------------------------------------------------------- detector
    # Common output grid for the broadband spatial map (per-freq meshes differ,
    # so we resample onto this fixed grid before form-factor-weighted summation).
    det_half_extent_m: float = 0.05        # +/- 5 cm view at the detector plane
    det_npix: int = 401
    det_active_radius_m: float = 0.005     # EO-crystal active area (BEST GUESS) for power integ.

    # ---------------------------------------------------------------- output
    out_dir: str = "results"

    # ----------------------------------------------------- derived quantities
    @property
    def gamma(self) -> float:
        return (self.beam_energy_MeV * 1e-3) / ME_C2_GEV

    @property
    def sigma_t_s(self) -> float:
        return self.sigma_t_ps * 1e-12

    @property
    def z_first_oe_m(self) -> float:
        """Longitudinal position of the first optical element (collimator)."""
        return self.L_foil_to_collimator_m

    def freq_grid_Hz(self):
        """Return the frequency sweep grid in Hz."""
        import numpy as np
        if self.freq_log:
            g = np.logspace(np.log10(self.f_min_GHz), np.log10(self.f_max_GHz), self.n_freq)
        else:
            g = np.linspace(self.f_min_GHz, self.f_max_GHz, self.n_freq)
        return g * 1e9


def default_config(**overrides) -> ThzConfig:
    """Convenience: ThzConfig with field overrides, e.g. default_config(n_freq=5)."""
    cfg = ThzConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"ThzConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg
