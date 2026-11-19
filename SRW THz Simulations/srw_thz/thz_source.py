"""
THz source representation for the SRW propagation model.

Default: Gaussian-beam surrogate (Option A).  A per-frequency SRWLGsnBm is built
with its divergence tuned to the CTR ~1/gamma opening angle.  The bunch form
factor F(f) is applied later (as a per-frequency weight in assembly), not baked
into the field here, so the same propagated field can be re-weighted for a
different bunch length without re-propagating.

Stage-2 upgrade (Option B), stubbed below: an analytic Ginzburg-Frank CTR field
with radial polarization and an on-axis null, loaded into a custom SRWLWfr.  It
plugs in behind the same make_source(f, cfg) signature.
"""
from array import array
import math

import numpy as np
from scipy.special import kn, jv          # K0=kn(0,.), K1=kn(1,.);  J0=jv(0,.), J1=jv(1,.)

from srwpy.srwlib import SRWLGsnBm, SRWLWfr, srwl

from thz_config import C_LIGHT, H_PLANCK_EV

ALPHA_FS = 7.2973525693e-3       # fine-structure constant
HBAR_J_S = 1.054571817e-34       # reduced Planck constant [J s]


# ---------------------------------------------------------------- conversions
def E_eV(f_Hz: float) -> float:
    """Photon energy [eV] for frequency f [Hz]. 100 GHz -> 4.136e-4 eV."""
    return H_PLANCK_EV * f_Hz


def wavelength_m(f_Hz: float) -> float:
    return C_LIGHT / f_Hz


def form_factor(f_Hz: float, sigma_t_s: float) -> float:
    """Gaussian bunch longitudinal form factor F(f) = exp[-(2 pi f sigma_t)^2 / 2].

    This is the field-level coherent weight; coherent power ~ |F(f)|^2.
    """
    a = 2.0 * math.pi * f_Hz * sigma_t_s
    return math.exp(-0.5 * a * a)


def sigma_r(f_Hz: float, cfg) -> float:
    """RMS transverse source size [m] chosen so the Gaussian far-field divergence
    approximates the CTR opening angle ~1/gamma:  sigma_x ~ div_match * lambda * gamma / pi.
    Tuned via cfg.div_match in verification to match the measured low-f cutoff.
    """
    lam = wavelength_m(f_Hz)
    return cfg.div_match * lam * cfg.gamma / math.pi


# ---------------------------------------------------------------- source (A)
def src_gaussian(f_Hz: float, cfg) -> SRWLWfr:
    """Build and compute the initial Gaussian wavefront at the first optical
    element plane (z = cfg.z_first_oe_m), for a single frequency f_Hz.
    Returns an SRWLWfr with the electric field populated.
    """
    gsn = SRWLGsnBm()
    gsn.x = gsn.y = 0.0
    gsn.z = 0.0                       # waist at the foil
    gsn.xp = gsn.yp = 0.0
    gsn.avgPhotEn = E_eV(f_Hz)
    gsn.pulseEn = 1.0                 # arbitrary; absolute scaling applied later
    gsn.repRate = 1.0
    gsn.polar = 1                     # linear horizontal (radial pol. is the Option-B upgrade)
    gsn.sigX = sigma_r(f_Hz, cfg)
    gsn.sigY = gsn.sigX
    gsn.sigT = cfg.sigma_t_s          # informational (monochromatic calc)
    gsn.mx = gsn.my = 0

    wfr = SRWLWfr()
    wfr.allocate(1, cfg.src_nx, cfg.src_ny)
    wfr.mesh.zStart = cfg.z_first_oe_m
    wfr.mesh.eStart = gsn.avgPhotEn
    wfr.mesh.eFin = gsn.avgPhotEn
    # window scales with the source size so the Gaussian is always well contained
    half = max(cfg.src_half_ap_floor_m, cfg.src_half_ap_sigmas * gsn.sigX)
    wfr.mesh.xStart, wfr.mesh.xFin = -half, half
    wfr.mesh.yStart, wfr.mesh.yFin = -half, half

    # carry source info into the wavefront's particle-beam slot (as the examples do)
    wfr.partBeam.partStatMom1.x = gsn.x
    wfr.partBeam.partStatMom1.y = gsn.y
    wfr.partBeam.partStatMom1.z = gsn.z
    wfr.partBeam.partStatMom1.xp = gsn.xp
    wfr.partBeam.partStatMom1.yp = gsn.yp

    samp = _guarded_sampfact(f_Hz, half, cfg)
    srwl.CalcElecFieldGaussian(wfr, gsn, [samp])
    return wfr


def _guarded_sampfact(f_Hz: float, half_m: float, cfg) -> float:
    """Throttle the SRW sampling factor at low frequency so the source mesh stays
    under cfg.max_src_pts (per axis), preventing multi-GB wavefronts. The mesh size
    scales as ~ K * sampFact * half^2 / (lambda * z); K~4.8 was calibrated from the
    observed 10/100/400 GHz meshes. Throttling is logged, never silent."""
    K = 4.8
    lam = wavelength_m(f_Hz)
    z = max(cfg.z_first_oe_m, 1e-6)
    npts_pred = K * cfg.sampFactNxNyForProp * half_m ** 2 / (lam * z)
    if npts_pred <= cfg.max_src_pts:
        return cfg.sampFactNxNyForProp
    eff = cfg.sampFactNxNyForProp * cfg.max_src_pts / npts_pred
    eff = max(eff, 0.15)                                   # accuracy floor
    print(f"    [guard] f={f_Hz/1e9:6.1f} GHz: sampFact "
          f"{cfg.sampFactNxNyForProp:.2f}->{eff:.2f} "
          f"(predicted src ~{int(npts_pred)} pts > cap {cfg.max_src_pts})")
    return eff


# ---------------------------------------------------------------- source (B)
def gf_shape(theta, beta):
    """Ginzburg-Frank backward-TR angular amplitude shape (single electron, A=1):
        E(theta) = beta * sin(theta) / (1 - beta^2 cos^2(theta)).
    Vectorized; exact on-axis null at theta=0."""
    th2 = theta * theta
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - th2))   # exact cos(theta), small-angle accurate
    sin_t = theta                                  # sin(theta) ~ theta (theta << 1 here)
    denom = 1.0 - beta * beta * cos_t * cos_t
    return beta * sin_t / denom


def _gf_x_half(cfg) -> float:
    """Source coordinate half-extent at z0: covers the main lobe AND extends past the
    collimator so the collimator (not the mesh edge) truncates the 1/theta GF tail."""
    z0 = cfg.z_first_oe_m
    return max(z0 * cfg.gf_n_sigma / cfg.gamma,
               cfg.gf_src_aper_margin * cfg.collimator_radius_m)


def _gf_n_ang(f_Hz: float, cfg) -> int:
    """Frequency-aware GF source mesh size. Coarse at low f (where the post-collimator
    beam diffracts over metres and a fine mesh would blow up the propagated grid via
    auto-resize), fine at high f (Nyquist dx <= gf_src_lambda_frac*lambda). Always
    resolves the ring (z0/gamma); bounded to [gf_min_ang, gf_n_ang]."""
    lam = wavelength_m(f_Hz)
    r_ring = cfg.z_first_oe_m / cfg.gamma
    # resolve the ring (~6 pts) when it is super-wavelength; floor dx at a fraction of
    # lambda so low-f sources stay coarse blobs (the ring is sub-wavelength there and
    # resolving it would oversample massively and stall SRW propagation)
    dx_target = max(r_ring / 6.0, cfg.gf_src_lambda_frac * lam)
    n = int(round(2.0 * _gf_x_half(cfg) / dx_target))
    return max(cfg.gf_min_ang, min(n, cfg.gf_n_ang))


def _gf_amplitude_norm(cfg) -> float:
    """Physical field amplitude A for the GF pattern, set so the angular-integrated
    |E_GF|^2 equals the single-electron backward-TR spectral energy (single perfect-conductor
    surface, angle-integrated, ultra-relativistic):
        dW/domega = (2 alpha hbar / pi) * [ln(2 gamma) - 1]   [J s].
    (Corrected 2026-07-06: the earlier alpha*hbar*(2 ln gamma - 1) omitted the 1/pi and used
    ln gamma instead of ln 2gamma -> ~3x too large an absolute energy. This only rescales the
    absolute amplitude A; all shapes/relative results are unchanged.)
    Frequency-independent in the ideal-conductor, flat-TR-spectrum limit, so the spectral shape
    of the detected signal comes from the bunch form factor F(f), the finite-foil rolloff, and
    the beamline -- not from A. Returns A (physical-but-approximate units)."""
    gamma = cfg.gamma
    beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
    n = cfg.gf_n_ang
    theta_max = cfg.gf_n_sigma / gamma
    th = np.linspace(-theta_max, theta_max, n)
    TX, TY = np.meshgrid(th, th)
    TH = np.sqrt(TX * TX + TY * TY)
    shape2 = gf_shape(TH, beta) ** 2                       # |E_GF|^2 with A=1
    dOmega = (2.0 * theta_max / (n - 1)) ** 2              # small-angle solid-angle element
    integral = float(np.sum(shape2) * dOmega)
    dW = (2.0 * ALPHA_FS / math.pi) * HBAR_J_S * (math.log(2.0 * gamma) - 1.0)
    return math.sqrt(dW / integral) if integral > 0 else 1.0


def gf_oblique_field(X, Y, cfg):
    """Exact point-charge oblique transition-radiation field (Level 3, image-charge result;
    LEVEL3_DERIVATION.md). Returns (EX, EY) on the coordinate grid (x=X, y=Y at z0), with
    TRUE p/s polarization -- not the radial heuristic of Level 1.

    Method: evaluate the exact vector amplitude
        A(n) = T(n,beta) - T(n,beta_i),   T(n,b) = (n (n.b) - b)/(1 - n.b),
    where beta = beta z_hat (electron) and beta_i is the specular (image) velocity, at the
    observation direction n of each pixel (about the specular axis = beamline axis), then
    project A onto the wavefront transverse basis (e_p in-plane, e_s = y out-of-plane). The
    polarization falls out automatically because A is transverse to n. Scaled by 1/2 so
    that psi=0 reproduces the scalar gf_shape normalization exactly."""
    gamma = cfg.gamma
    beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
    z0 = cfg.z_first_oe_m
    psi = math.radians(cfg.screen_tilt_deg)

    zhat = np.array([0.0, 0.0, 1.0])
    ns = np.array([math.sin(psi), 0.0, math.cos(psi)])      # screen normal
    bvec = beta * zhat                                       # electron velocity
    bivec = beta * (zhat - 2.0 * math.cos(psi) * ns)        # image (specular) velocity
    shat = bivec / np.linalg.norm(bivec)                    # specular axis (=beamline axis)
    yhat = np.array([0.0, 1.0, 0.0])
    ep = np.cross(shat, yhat); ep /= np.linalg.norm(ep)     # in-plane transverse axis
    es = yhat                                                # out-of-plane transverse axis

    thx, thy = X / z0, Y / z0
    # observation direction per pixel about the specular axis, normalized
    nvec = (shat[None, None, :] + thx[..., None] * ep[None, None, :]
            + thy[..., None] * es[None, None, :])
    nvec /= np.linalg.norm(nvec, axis=2, keepdims=True)

    def Tterm(nn, b):
        ndb = nn @ b                                        # (ny, nx)
        num = nn * ndb[..., None] - b[None, None, :]
        return num / (1.0 - ndb)[..., None]

    with np.errstate(divide='ignore', invalid='ignore'):
        Avec = Tterm(nvec, bvec) - Tterm(nvec, bivec)
        if not cfg.gf_include_electron_term:
            Avec = -Tterm(nvec, bivec)                      # image-only (radial) comparison
    # /2 so |A|=2*gf_shape at psi=0 matches the scalar norm; sign chosen so psi=0 gives the
    # same radial-outward convention as the Level-1 / symmetric build (overall sign is a
    # physical convention and does not affect intensity or polarization structure).
    scale = -(_gf_amplitude_norm(cfg) / z0) / 2.0
    EX = (scale * (Avec @ ep)).astype(np.float32)
    EY = (scale * (Avec @ es)).astype(np.float32)
    return EX, EY


def gf_analytic_field(X, Y, cfg):
    """CTR source from the supplied closed-form 45-degree TR relation (an algebraic rewrite
    of the Level-3 image-charge field; the two are identical at psi=45, see
    LEVEL3_TWIST_DERIVATION.md and compare_analytic_45.py):

        E_45(theta,phi) ~  [ sin(phi) / (1 - beta cos(phi) sin(theta)) ] e_phi
                         + [ ((beta - cos theta) cos phi - sin theta)
                             / ((1 - beta cos theta)(1 - beta cos phi sin theta)) ] e_theta

    Frame: electron velocity beta = beta z_hat; image (specular) velocity beta_i = beta x_hat
    -- i.e. a 45-degree foil, image perpendicular to the beam. theta is measured from z_hat and
    phi is the azimuth, so the two denominators are the electron resonance (1 - beta n.z_hat)
    and the image resonance (1 - beta n.x_hat). The backward-TR lobe is centred on +x_hat
    (theta=90 deg, phi=0). Each pixel (X,Y) maps to an observation direction about that +x_hat
    axis via the far-field projection x = z0*theta; the (e_theta, e_phi) field is projected onto
    the wavefront transverse basis (e_p in-plane = z_hat, e_s = y_hat) to give (EX, EY).

    This is the psi=45 special case written in clean closed form; for a general tilt use
    gf_oblique_field (Level 3). Amplitude is normalized identically to gf_oblique_field, so the
    two are drop-in interchangeable at 45 deg (the only difference is a tilt-direction/mirror
    convention: this relation centres the lobe on +x, gf_oblique_field on -x for psi=+45)."""
    gamma = cfg.gamma
    beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
    z0 = cfg.z_first_oe_m

    # observation directions about the specular (+x) axis, from the far-field projection.
    shat = np.array([1.0, 0.0, 0.0])                        # image/specular axis (45-deg foil)
    ep = np.cross(shat, [0.0, 1.0, 0.0]); ep /= np.linalg.norm(ep)   # in-plane transverse = +z
    es = np.array([0.0, 1.0, 0.0])                          # out-of-plane transverse = +y
    thx, thy = X / z0, Y / z0
    nvec = (shat[None, None, :] + thx[..., None] * ep[None, None, :]
            + thy[..., None] * es[None, None, :])
    nvec /= np.linalg.norm(nvec, axis=2, keepdims=True)
    nx, ny, nz = nvec[..., 0], nvec[..., 1], nvec[..., 2]

    theta = np.arccos(np.clip(nz, -1.0, 1.0))
    phi = np.arctan2(ny, nx)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)

    D1 = 1.0 - beta * ct                                    # electron resonance (never 0, beta<1)
    D2 = 1.0 - beta * cp * st                               # image resonance
    B_th = ((beta - ct) * cp - st) / (D1 * D2)
    B_ph = sp / D2

    # build the Cartesian field from spherical components and project onto (e_p, e_s):
    #   e_theta = (ct cp, ct sp, -st),  e_phi = (-sp, cp, 0)
    Evec = np.stack([B_th * (ct * cp) + B_ph * (-sp),
                     B_th * (ct * sp) + B_ph * (cp),
                     B_th * (-st)], axis=2)
    scale = _gf_amplitude_norm(cfg) * beta / (2.0 * z0)     # matches gf_oblique_field magnitude
    EX = (scale * (Evec @ ep)).astype(np.float32)
    EY = (scale * (Evec @ es)).astype(np.float32)
    return EX, EY


def finite_foil_formfactor(theta_ell, f_Hz, cfg):
    """Far-zone finite circular-foil TR form factor H(theta) = E_a(theta)/E_inf(theta), the
    ratio of the truncated-at-radius-a to the infinite Hankel transform of the electron's
    virtual-photon field kappa*K1(kappa r) (see FINITE_FOIL_PLAN.md Sec 1). Exact Lommel
    closed form:

        H = 1 - x K1(x) J0(y) - (x^2 / y) K0(x) J1(y),
            x = kappa a = k a /(beta gamma)   (foil size vs field extent; scalar per freq)
            y = k a theta_ell                 (foil size * angle; per pixel)
        H(y->0) = 1 - x K1(x) - (x^2/2) K0(x)     (on-axis suppression)

    H -> 1 as a -> infinity (infinite-foil source recovered); H(0) -> 0 as a -> 0 (a tiny
    foil radiates no coherent low-f energy). J0/J1 oscillations are the edge-diffraction
    fringes. `theta_ell` already carries the 45-deg elliptical foreshortening (see caller).
    Returns a real float array shaped like theta_ell.
    """
    k = 2.0 * math.pi * f_Hz / C_LIGHT
    beta = math.sqrt(1.0 - 1.0 / cfg.gamma ** 2)
    a = cfg.foil_radius_m
    x = k * a / (beta * cfg.gamma)                       # kappa*a (scalar)
    y = k * a * np.asarray(theta_ell, dtype=np.float64)  # per pixel
    K1x, K0x = float(kn(1, x)), float(kn(0, x))
    y_safe = np.where(y == 0.0, 1.0, y)                  # avoid 0/0; on-axis handled below
    H = 1.0 - x * K1x * jv(0, y) - (x * x) * K0x * jv(1, y) / y_safe
    H0 = 1.0 - x * K1x - 0.5 * x * x * K0x               # y -> 0 limit
    return np.where(y == 0.0, H0, H)


def src_ctr_wavefront(f_Hz: float, cfg) -> SRWLWfr:
    """Analytic Ginzburg-Frank CTR source, built directly in COORDINATE space at the
    first optical element plane (z = z0 = cfg.z_first_oe_m) via the far-field-of-source
    projection x = z0 * theta:

        E(x,y) = (A / z0) * gf_shape(r/z0) * (cos phi, sin phi),  r = sqrt(x^2+y^2)

    with radial polarization (Ex, Ey) and an exact on-axis null. The diverging
    spherical curvature is carried as wfr.Rx = wfr.Ry = z0 (SRW treats the quadratic
    phase analytically), so the data array holds the real radial amplitude only.

    NOTE (approximation): this is the far field of a point-like TR source. It is most
    valid when z0 >> the TR formation length gamma*lambda; at low frequencies
    gamma*lambda can exceed z0 (same caveat as the Gaussian surrogate). The GF source's
    value over the Gaussian is the correct on-axis null and radial polarization.

    The amplitude A is physical (single-electron TR spectral energy); the coherent
    bunch scaling (Q/e)*F(f) is applied later in assembly. Returns a coordinate-space
    SRWLWfr (presCA=0) ready for the standard propagation path.

    If cfg.screen_tilt_deg != 0, a Level-1 heuristic tilted-screen asymmetry is applied
    (foreshortening + x-lobe bias; see thz_config). tilt=0 reproduces the symmetric GF
    pattern exactly.
    """
    if cfg.gf_n_ang > cfg.gf_max_ang_pts:
        raise ValueError(
            f"gf_n_ang={cfg.gf_n_ang} exceeds gf_max_ang_pts={cfg.gf_max_ang_pts}")

    gamma = cfg.gamma
    beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
    z0 = cfg.z_first_oe_m
    n = _gf_n_ang(f_Hz, cfg)       # frequency-aware (coarse at low f to bound memory)
    x_half = _gf_x_half(cfg)       # covers main lobe + extends past collimator
    A = _gf_amplitude_norm(cfg)

    # coordinate grid: element [iy, ix] -> (x=xs[ix], y=xs[iy])
    xs = np.linspace(-x_half, x_half, n)
    X, Y = np.meshgrid(xs, xs)                     # 'xy': X[iy,ix]=xs[ix], Y[iy,ix]=xs[iy]
    R = np.sqrt(X * X + Y * Y)
    thx, thy = X / z0, Y / z0                       # emission angles (small-angle)

    if cfg.gf_tilt_model == 'level3':
        # exact point-charge oblique-TR field with true p/s polarization (reduces to
        # symmetric GF at screen_tilt_deg=0)
        EX, EY = gf_oblique_field(X, Y, cfg)
    elif cfg.gf_tilt_model == 'analytic':
        # same physics via the supplied closed-form 45-deg TR relation (gf_analytic_field);
        # identical to level3 at psi=45. The foil angle is fixed at 45 deg by the relation.
        EX, EY = gf_analytic_field(X, Y, cfg)
    else:
        # Level-1 heuristic: foreshorten the pattern in x, bias the two x-lobes; radial pol.
        psi = math.radians(cfg.screen_tilt_deg)
        cpsi = max(math.cos(psi), 1e-3)
        th_eff = np.sqrt((thx / cpsi) ** 2 + thy ** 2)              # ellipse: narrower in x
        lobe = 1.0 + cfg.gf_tilt_lobe_k * math.tan(psi) * np.tanh(thx * gamma)
        lobe = np.maximum(lobe, 0.0)                                # bounded, non-negative
        with np.errstate(divide='ignore', invalid='ignore'):
            Eamp = (A / z0) * gf_shape(th_eff, beta) * lobe
            PHI = np.arctan2(Y, X)
            EX = np.where(R > 0.0, Eamp * np.cos(PHI), 0.0).astype(np.float32)
            EY = np.where(R > 0.0, Eamp * np.sin(PHI), 0.0).astype(np.float32)

    # finite-foil (radiator) diffraction: multiply by the far-zone form factor H(theta_ell),
    # with the 45-deg elliptical foreshortening theta_ell = sqrt(cos^2(psi) thx^2 + thy^2).
    # H -> 1 as foil_radius -> inf, so this is an exact no-op for a large foil. Composes with
    # all three gf_tilt_model branches (level1/level3/analytic). See FINITE_FOIL_PLAN.md.
    if getattr(cfg, 'finite_foil_diffraction', False):
        cpsi = math.cos(math.radians(cfg.screen_tilt_deg))
        theta_ell = np.sqrt((cpsi * thx) ** 2 + thy ** 2)
        H = finite_foil_formfactor(theta_ell, f_Hz, cfg).astype(np.float32)
        EX = (EX * H).astype(np.float32)
        EY = (EY * H).astype(np.float32)

    # interleave [Re, Im] with Im=0; C order over (iy, ix) -> arEx[2*(ix + iy*n)]
    bufx = np.zeros(2 * n * n, dtype=np.float32); bufx[0::2] = EX.ravel()
    bufy = np.zeros(2 * n * n, dtype=np.float32); bufy[0::2] = EY.ravel()

    wfr = SRWLWfr()
    wfr.allocate(1, n, n)
    wfr.mesh.zStart = z0
    wfr.mesh.eStart = wfr.mesh.eFin = E_eV(f_Hz)
    wfr.mesh.xStart, wfr.mesh.xFin = -x_half, x_half
    wfr.mesh.yStart, wfr.mesh.yFin = -x_half, x_half

    ax = array('f'); ax.frombytes(bufx.tobytes()); wfr.arEx[:] = ax
    ay = array('f'); ay.frombytes(bufy.tobytes()); wfr.arEy[:] = ay

    wfr.presCA = 0                 # coordinate representation
    wfr.presFT = 0                 # frequency domain
    wfr.unitElFld = 0             # arbitrary-but-consistent (physical scale via A)
    wfr.avgPhotEn = E_eV(f_Hz)
    wfr.Rx = wfr.Ry = z0           # diverging spherical wave from the foil at z=0
    wfr.dRx = wfr.dRy = 0.0
    wfr.xc = wfr.yc = 0.0
    wfr.partBeam.partStatMom1.x = 0.0
    wfr.partBeam.partStatMom1.y = 0.0
    wfr.partBeam.partStatMom1.z = 0.0
    wfr.partBeam.partStatMom1.xp = 0.0
    wfr.partBeam.partStatMom1.yp = 0.0
    return wfr


# ---------------------------------------------------------------- factory
def make_source(f_Hz: float, cfg, src: str = "gaussian") -> SRWLWfr:
    """Source factory. src in {'gaussian', 'ctr'}."""
    if src == "gaussian":
        return src_gaussian(f_Hz, cfg)
    if src == "ctr":
        return src_ctr_wavefront(f_Hz, cfg)
    raise ValueError(f"unknown source type {src!r}")
