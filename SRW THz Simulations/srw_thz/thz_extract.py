"""
Result extraction and assembly.

Per frequency we pull from the propagated wavefront:
  - the 2D single-electron intensity |E(x,y)|^2 at the detector plane, and
  - the on-axis complex field E(0,0) (Re and Im), for the time-domain waveform.

These are combined across the frequency sweep with the bunch form factor
F(f) = exp[-(2 pi f sigma_t)^2/2] and a charge scale (coherent field ~ Q, power
~ Q^2) to produce the two requested deliverables:
  (1) broadband 2D spatial intensity map at the detector, and
  (2) the time-domain E(t) waveform at the crystal (inverse FFT over frequency).
"""
from array import array
from dataclasses import dataclass
import math

import numpy as np

from srwpy.srwlib import srwl

from thz_source import form_factor


@dataclass
class FreqResult:
    f_Hz: float
    I_grid: np.ndarray     # [npix, npix] intensity on the COMMON detector grid (channel-weighted)
    E_onaxis: complex      # on-axis complex field (channel-weighted); used for Gaussian source
    E_ring: complex = 0j   # ring-peak complex field (channel-weighted); used for GF/CTR source
    src_mode: str = "gaussian"   # "gaussian" -> waveform from E_onaxis; "ctr" -> from E_ring
    # Note: I_grid is resampled to the fixed common grid at capture time so memory
    # does not scale with the (sometimes huge) per-frequency native SRW mesh.


# --------------------------------------------------------------- raw extraction
def intensity_2d(wfr):
    """Total single-electron intensity over (x, y). Returns (I[ny,nx], x, y)."""
    nx, ny = wfr.mesh.nx, wfr.mesh.ny
    arI = array('f', [0] * nx * ny)
    srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
    I = np.asarray(arI, dtype=float).reshape(ny, nx)   # outer loop y, inner x
    x = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, nx)
    y = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, ny)
    return I, x, y


def onaxis_complex_field(wfr) -> complex:
    """On-axis (x=y=0) complex field of the linear-horizontal component."""
    reE = array('f', [0])
    imE = array('f', [0])
    srwl.CalcIntFromElecField(reE, wfr, 0, 5, 0, wfr.mesh.eStart, 0.0, 0.0)  # Re(E), Lin-Hor
    srwl.CalcIntFromElecField(imE, wfr, 0, 6, 0, wfr.mesh.eStart, 0.0, 0.0)  # Im(E), Lin-Hor
    return complex(reE[0], imE[0])


def ring_peak_complex_field(wfr, cfg, r_fixed=None) -> complex:
    """Complex RADIAL field at the ring of a propagated GF wavefront.

    The Ginzburg-Frank pattern has an on-axis null and RADIAL polarization
    (Ex = E_r cos(phi), Ey = E_r sin(phi)). Averaging Ex alone over the ring cancels
    azimuthally (<cos phi> = 0) -- so we project onto the radial unit vector,
    E_r = Ex*cos(phi) + Ey*sin(phi), which is azimuthally coherent, and average that
    complex value over the annulus. This preserves both amplitude and a clean spectral
    phase (-> a localized transient rather than phase-scrambled quasi-CW).

    Sampling radius: r_fixed if given (a single crystal location, consistent phase
    reference across all frequencies -> sharpest transient), else the per-frequency
    azimuthal-intensity peak r_peak."""
    nx, ny = wfr.mesh.nx, wfr.mesh.ny
    x = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, nx)
    y = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, ny)
    XX, YY = np.meshgrid(x, y)
    R = np.sqrt(XX * XX + YY * YY)
    PHI = np.arctan2(YY, XX)

    # complex Ex, Ey on the native mesh: arE[2*(ix+iy*nx)] = Re, +1 = Im
    Ex = np.asarray(wfr.arEx, dtype=float).reshape(ny, nx, 2)
    Ey = np.asarray(wfr.arEy, dtype=float).reshape(ny, nx, 2)
    Ex = Ex[..., 0] + 1j * Ex[..., 1]
    Ey = Ey[..., 0] + 1j * Ey[..., 1]
    Er = Ex * np.cos(PHI) + Ey * np.sin(PHI)       # radial component (azimuthally coherent)

    if r_fixed is not None:
        r_peak = float(r_fixed)
    else:
        # azimuthal average of |E_r|^2 -> ring-peak radius (vectorized, O(N))
        r_max = min(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1]))
        n_r = max(64, nx // 2)
        dr_bin = r_max / n_r
        ridx = np.minimum((R.ravel() / dr_bin).astype(int), n_r - 1)
        I_az = (np.bincount(ridx, weights=(np.abs(Er) ** 2).ravel(), minlength=n_r)
                / np.maximum(np.bincount(ridx, minlength=n_r), 1))
        r_peak = float((np.argmax(I_az) + 0.5) * dr_bin)

    dr = cfg.gf_ring_dr_frac * r_peak
    ann = (R >= r_peak - dr) & (R <= r_peak + dr)
    if not ann.any():
        return 0j
    Er_ann = Er[ann]
    # For a tilted (asymmetric) screen the ring is brighter on one side; weight the
    # complex radial field by intensity so the bright lobe dominates the sampled field.
    # For the symmetric GF case E_r is azimuthally uniform, so this equals the plain mean.
    if getattr(cfg, 'screen_tilt_deg', 0.0) != 0.0:
        w = np.abs(Er_ann) ** 2
        wsum = float(w.sum())
        return complex((Er_ann * w).sum() / wsum) if wsum > 0 else 0j
    return complex(Er_ann.mean())


def _complex_ExEy(wfr):
    """Return complex Ex, Ey arrays [ny,nx] and the axes (x,y) from a wavefront."""
    nx, ny = wfr.mesh.nx, wfr.mesh.ny
    x = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, nx)
    y = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, ny)
    Ex = np.asarray(wfr.arEx, dtype=float).reshape(ny, nx, 2)
    Ey = np.asarray(wfr.arEy, dtype=float).reshape(ny, nx, 2)
    return (Ex[..., 0] + 1j * Ex[..., 1]), (Ey[..., 0] + 1j * Ey[..., 1]), x, y


def crystal_axis_complex_field(wfr, cfg) -> complex:
    """Polarization-aware sampling for the Level-3 source: the EO crystal responds to the
    field along its electro-optic axis chi_c (cfg.crystal_pol_axis_deg). Project the complex
    field onto that axis, E_c = Ex cos(chi) + Ey sin(chi), find the bright region from the
    total intensity, and return the intensity-weighted complex E_c there (so the bright lobe
    of an asymmetric crescent dominates)."""
    Ex, Ey, x, y = _complex_ExEy(wfr)
    ny, nx = Ex.shape
    XX, YY = np.meshgrid(x, y)
    R = np.sqrt(XX * XX + YY * YY)
    chi = math.radians(getattr(cfg, 'crystal_pol_axis_deg', 0.0))
    Ec = Ex * math.cos(chi) + Ey * math.sin(chi)            # crystal-axis component
    Itot = np.abs(Ex) ** 2 + np.abs(Ey) ** 2

    # ring-peak radius from the total-intensity azimuthal average (vectorized)
    r_max = min(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1]))
    n_r = max(64, nx // 2); dr_bin = r_max / n_r
    ridx = np.minimum((R.ravel() / dr_bin).astype(int), n_r - 1)
    I_az = (np.bincount(ridx, weights=Itot.ravel(), minlength=n_r)
            / np.maximum(np.bincount(ridx, minlength=n_r), 1))
    r_peak = float((np.argmax(I_az) + 0.5) * dr_bin)

    dr = cfg.gf_ring_dr_frac * r_peak
    ann = (R >= r_peak - dr) & (R <= r_peak + dr)
    if not ann.any():
        return 0j
    Ec_ann = Ec[ann]
    w = np.abs(Ec_ann) ** 2                                 # weight toward the bright lobe
    wsum = float(w.sum())
    return complex((Ec_ann * w).sum() / wsum) if wsum > 0 else complex(Ec_ann.mean())


def save_polarization_maps(wfr, cfg, prefix):
    """Diagnostic: from a propagated Level-3 wavefront, save |Ex|^2 (p, in-plane),
    |Ey|^2 (s, out-of-plane), and the local polarization-angle map atan(|Ey|/|Ex|),
    each resampled to the common detector grid. prefix -> <prefix>_pol_{p,s,angle}.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Ex, Ey, x, y = _complex_ExEy(wfr)
    Ip, _, _ = resample_to_grid(np.abs(Ex) ** 2, x, y, cfg)
    Is, xc, yc = resample_to_grid(np.abs(Ey) ** 2, x, y, cfg)
    ang = np.degrees(np.arctan2(np.sqrt(np.maximum(Is, 0)), np.sqrt(np.maximum(Ip, 0))))
    ext = [xc[0] * 1e3, xc[-1] * 1e3, yc[0] * 1e3, yc[-1] * 1e3]
    for data, tag, lab, cmap in (
            (Ip, 'p', r'$|E_x|^2$  (p, in-plane)', 'inferno'),
            (Is, 's', r'$|E_y|^2$  (s, out-of-plane)', 'inferno'),
            (ang, 'angle', r'pol. angle $\arctan(|E_y|/|E_x|)$ [deg]', 'twilight')):
        fig, a = plt.subplots(figsize=(5, 4))
        im = a.imshow(data, origin='lower', extent=ext, aspect='equal', cmap=cmap)
        a.set_xlabel('x [mm]'); a.set_ylabel('y [mm]'); a.set_title(lab)
        fig.colorbar(im, ax=a)
        fig.tight_layout(); fig.savefig(f"{prefix}_pol_{tag}.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------ resampling
def common_grid(cfg):
    """Fixed detector grid axes (xc, yc) shared across all frequencies."""
    xc = np.linspace(-cfg.det_half_extent_m, cfg.det_half_extent_m, cfg.det_npix)
    yc = np.linspace(-cfg.det_half_extent_m, cfg.det_half_extent_m, cfg.det_npix)
    return xc, yc


def resample_to_grid(I, x, y, cfg):
    """Bilinearly resample I(y,x) onto the common detector grid; 0 outside."""
    from scipy.interpolate import RegularGridInterpolator
    xc, yc = common_grid(cfg)
    interp = RegularGridInterpolator((y, x), I, bounds_error=False, fill_value=0.0)
    YY, XX = np.meshgrid(yc, xc, indexing='ij')
    pts = np.stack([YY.ravel(), XX.ravel()], axis=-1)
    Ig = interp(pts).reshape(cfg.det_npix, cfg.det_npix)
    return Ig, xc, yc


# ------------------------------------------------------------------- assembly
def assemble_spatial_map(records, cfg):
    """Form-factor- and charge-weighted broadband intensity on the common grid.
    Returns (I[npix,npix], xc, yc). I_grid is already on the common grid."""
    Q = cfg.bunch_charge_nC
    xc, yc = common_grid(cfg)
    Ic = np.zeros((cfg.det_npix, cfg.det_npix))
    for r in records:
        w = (form_factor(r.f_Hz, cfg.sigma_t_s) ** 2) * (Q ** 2)
        Ic += r.I_grid * w
    return Ic, xc, yc


def detected_power_spectrum(records, cfg):
    """Power within the detector active area vs frequency, form-factor & charge
    weighted. Returns (f_GHz[N], P[N]). Integrates on the common detector grid."""
    Q = cfg.bunch_charge_nC
    xc, yc = common_grid(cfg)
    XX, YY = np.meshgrid(xc, yc)
    mask = (XX ** 2 + YY ** 2) <= cfg.det_active_radius_m ** 2
    dcell = (2.0 * cfg.det_half_extent_m) / (cfg.det_npix - 1)
    fs, Ps = [], []
    for r in records:
        P_raw = float(np.sum(r.I_grid[mask]) * dcell * dcell)
        w = (form_factor(r.f_Hz, cfg.sigma_t_s) ** 2) * (Q ** 2)
        fs.append(r.f_Hz / 1e9)
        Ps.append(P_raw * w)
    return np.asarray(fs), np.asarray(Ps)


def assemble_waveform(records, cfg):
    """On-crystal time-domain field E(t) via inverse FFT of the on-axis complex
    field, weighted by F(f) and charge.  Assumes a (near-)uniform frequency grid;
    bins the sweep onto a 0..f_max uniform axis and irfft's a Hermitian spectrum.
    Returns (t_ps[M], E_t[M])."""
    Q = cfg.bunch_charge_nC
    f = np.array([r.f_Hz for r in records])
    order = np.argsort(f)
    f = f[order]
    # GF/CTR source has an on-axis null -> use the ring-peak field; Gaussian uses on-axis
    use_ring = all(records[i].src_mode == "ctr" for i in order)
    E = np.array([(records[i].E_ring if use_ring else records[i].E_onaxis) for i in order])
    F = np.array([form_factor(fi, cfg.sigma_t_s) for fi in f])
    spec_vals = E * F * Q                       # complex field phasors, weighted

    # uniform bin spacing from the (sorted) grid
    df = np.min(np.diff(f)) if len(f) > 1 else f[0]
    idx = np.round(f / df).astype(int)
    n_pos = int(idx.max()) + 1
    spec = np.zeros(n_pos, dtype=complex)
    spec[idx] = spec_vals                       # zero below f_min (foil cutoff region)

    n_time = 2 * (n_pos - 1)
    E_t = np.fft.irfft(spec, n=n_time) * n_time # irfft normalisation undo
    dt = 1.0 / (n_time * df)                     # s
    t = np.arange(n_time) * dt
    return t * 1e12, E_t                         # t in ps


# --------------------------------------------------------------------- savers
def save_spatial_map(Ic, xc, yc, cfg, path_png, path_npz):
    np.savez(path_npz, I=Ic, x=xc, y=yc)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(Ic, origin='lower', aspect='equal',
                   extent=[xc[0] * 1e3, xc[-1] * 1e3, yc[0] * 1e3, yc[-1] * 1e3])
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title('Broadband THz intensity at detector')
    fig.colorbar(im, ax=ax, label='arb. (|F|^2 Q^2 weighted)')
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)


def save_waveform(t_ps, E_t, cfg, path_png, path_txt):
    np.savetxt(path_txt, np.column_stack([t_ps, E_t]),
               header='t[ps]  E(t)[arb]')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t_ps, E_t, lw=1.2)
    ax.set_xlabel('t [ps]'); ax.set_ylabel('E(t) [arb]')
    ax.set_title('Reconstructed on-crystal THz waveform')
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)


def save_spectrum(f_GHz, P, path_png, path_txt):
    np.savetxt(path_txt, np.column_stack([f_GHz, P]), header='f[GHz]  P[arb]')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(f_GHz, P, marker='o', ms=3, lw=1.2)
    ax.set_xlabel('f [GHz]'); ax.set_ylabel('detected power [arb]')
    ax.set_title('THz spectrum at detector (window+aperture shaped)')
    fig.tight_layout(); fig.savefig(path_png, dpi=130); plt.close(fig)
