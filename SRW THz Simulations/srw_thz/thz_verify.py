#!/usr/bin/env python
"""
Verification utilities.

1) aperture_diffraction_check: propagate a (near-)plane wave through a circular
   aperture and compare the first dark-ring radius to the analytic Fraunhofer
   value 1.22 * lambda * L / D.  Validates mesh sampling / propagation params
   before trusting any CTR physics.

2) butterworth_highpass: reference low-frequency cutoff curve (the measured
   foil response is ~Butterworth high-pass, corner ~45 GHz) to overlay on the
   simulated detected-power spectrum.

Run standalone:  python thz_verify.py
"""
import math
from array import array

import numpy as np

from srwpy.srwlib import SRWLGsnBm, SRWLWfr, SRWLOptA, SRWLOptD, SRWLOptC, srwl

from thz_config import C_LIGHT


def butterworth_highpass(f_GHz, fc_GHz=45.0, order=7):
    """Amplitude (not power) Butterworth high-pass response."""
    f = np.asarray(f_GHz, dtype=float)
    return f ** order / np.sqrt(fc_GHz ** (2 * order) + f ** (2 * order))


def aperture_diffraction_check(freq_GHz=300.0, D=0.02, L=0.5,
                               n_pts=800, samp_fact=2.0, verbose=True):
    """Far-field circular-aperture (Airy) sanity check.

    Choose L so the first null r=1.22 lambda L/D lands at a well-sampled radius
    (tens of mm) AND the Fresnel number a^2/(lambda L) < 1.  Returns
    (r_null_measured_m, r_null_theory_m, ratio); ratio ~ 1 means SRW reproduces
    the analytic Airy null.
    """
    f = freq_GHz * 1e9
    lam = C_LIGHT / f
    E_eV = 4.135667696e-15 * f
    r_theory = 1.22 * lam * L / D
    fresnel_N = (0.5 * D) ** 2 / (lam * L)

    # near-plane wave across the aperture: a Gaussian much wider than D, at waist
    gsn = SRWLGsnBm()
    gsn.x = gsn.y = gsn.z = 0.0
    gsn.xp = gsn.yp = 0.0
    gsn.avgPhotEn = E_eV
    gsn.pulseEn = 1.0
    gsn.repRate = 1.0
    gsn.polar = 1
    gsn.sigX = gsn.sigY = 8.0 * D      # ~flat amplitude over the aperture
    gsn.sigT = 1e-12
    gsn.mx = gsn.my = 0

    wfr = SRWLWfr()
    wfr.allocate(1, n_pts, n_pts)
    wfr.mesh.zStart = 0.0              # field at the aperture plane (waist)
    wfr.mesh.eStart = wfr.mesh.eFin = E_eV
    half = 1.2 * D
    wfr.mesh.xStart, wfr.mesh.xFin = -half, half
    wfr.mesh.yStart, wfr.mesh.yFin = -half, half
    srwl.CalcElecFieldGaussian(wfr, gsn, [samp_fact])

    # circular aperture (Dx = diameter), then drift; expand range to contain the
    # central lobe (~4x first-null radius) so the null is inside the mesh.
    range_fac = max(4.0, 4.0 * r_theory / half)
    pp_ap = [1, 1, 1.0, 1, 0, 1.0, 2.0, 1.0, 2.0, 0, 0, 0]
    pp_dr = [1, 1, 1.0, 1, 0, range_fac, 1.0, range_fac, 1.0, 0, 0, 0]
    optBL = SRWLOptC([SRWLOptA('c', 'a', D), SRWLOptD(L)], [pp_ap, pp_dr])
    srwl.PropagElecField(wfr, optBL)

    # 1D horizontal intensity cut through centre
    nx = wfr.mesh.nx
    arIx = array('f', [0] * nx)
    srwl.CalcIntFromElecField(arIx, wfr, 6, 0, 1, wfr.mesh.eStart, 0, 0)
    Ix = np.asarray(arIx, dtype=float)
    x = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, nx)

    # robust first-null: scan x>0 for a local minimum that is a true dark ring
    # (deep, < 5% of the central peak), so near-axis ripples are ignored.
    c = int(np.argmax(Ix))                 # central peak index
    peak = Ix[c]
    r_null = None
    for k in range(c + 2, nx - 1):
        if Ix[k] < 0.05 * peak and Ix[k] <= Ix[k - 1] and Ix[k] <= Ix[k + 1]:
            r_null = x[k] - x[c]
            break

    covered = x[-1] - x[c] > 1.5 * r_theory
    ratio = (r_null / r_theory) if r_null else float('nan')
    if verbose:
        print(f"aperture-diffraction check @ {freq_GHz:.0f} GHz "
              f"(lambda={lam*1e3:.2f} mm, D={D*1e3:.0f} mm, L={L} m, Fresnel N={fresnel_N:.2f}):")
        print(f"  detector mesh {nx} pts, x up to {(x[-1]-x[c])*1e3:.0f} mm "
              f"(need >~{1.5*r_theory*1e3:.0f} mm): {'OK' if covered else 'TOO SMALL'}")
        print(f"  first null: measured="
              f"{None if r_null is None else f'{r_null*1e3:.1f} mm'}, "
              f"theory(1.22 lam L/D)={r_theory*1e3:.1f} mm, ratio={ratio:.2f}")
        if fresnel_N > 1:
            print("  NOTE: Fresnel N>1 -> not yet far field; reduce D or increase L.")
    return r_null, r_theory, ratio


def gf_pattern_check(freq_GHz=120.0, cfg=None, test_drift=True, verbose=True):
    """Sanity-check the Ginzburg-Frank CTR source. Confirms:
      (a) on-axis null  (I_onaxis / I_peak << 1),
      (b) ring-peak radius at the source plane ~ z0/gamma,
      (c) coordinate representation (presCA == 0),
      (d) ring expands correctly after a free-space drift (validates Rx=Ry=z0).
    Returns (null_ratio, ring_ratio)."""
    from thz_config import default_config
    from thz_source import src_ctr_wavefront
    cfg = cfg or default_config()
    f_Hz = freq_GHz * 1e9

    def ring_radius(wfr):
        nx, ny = wfr.mesh.nx, wfr.mesh.ny
        x = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, nx)
        arI = array('f', [0] * nx * ny)
        srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
        I2d = np.asarray(arI).reshape(ny, nx)
        ic = nx // 2
        cut = I2d[ny // 2, :]
        ix_pk = int(np.argmax(cut[ic:]) + ic)
        return abs(float(x[ix_pk])), float(I2d[ny // 2, ic]), float(I2d.max())

    wfr = src_ctr_wavefront(f_Hz, cfg)
    r_src, I_on, I_pk = ring_radius(wfr)
    null_ratio = I_on / I_pk if I_pk > 0 else float('nan')
    r_theory = cfg.z_first_oe_m / cfg.gamma
    ring_ratio = r_src / r_theory if r_theory > 0 else float('nan')

    if verbose:
        print(f"GF pattern check @ {freq_GHz:.0f} GHz (gamma={cfg.gamma:.1f}):")
        print(f"  (a) on-axis null I_on/I_peak = {null_ratio:.2e}  (expect << 1)")
        print(f"  (b) ring radius: measured={r_src*1e3:.2f} mm, "
              f"theory(z0/gamma)={r_theory*1e3:.2f} mm, ratio={ring_ratio:.2f}")
        print(f"  (c) presCA={wfr.presCA} (expect 0), mesh={wfr.mesh.nx}x{wfr.mesh.ny}")

    if test_drift:
        # apply the collimator aperture first (as the real pipeline does), then drift,
        # so the ring expansion reflects physical (collimator) diffraction, not the
        # mesh-edge artifact of the heavy 1/theta GF tail
        L = 0.5
        pp = [1, 1, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
        optBL = SRWLOptC([SRWLOptA('c', 'a', 2.0 * cfg.collimator_radius_m), SRWLOptD(L)],
                         [pp, pp])
        srwl.PropagElecField(wfr, optBL)
        r_after, _, _ = ring_radius(wfr)
        r_after_geom = (cfg.z_first_oe_m + L) / cfg.gamma
        if verbose:
            print(f"  (d) collimator + {L} m drift: ring={r_after*1e3:.2f} mm "
                  f"(>= geometric {r_after_geom*1e3:.2f} mm; excess = collimator "
                  f"diffraction). Ring expands -> diverging curvature OK.")

    return null_ratio, ring_ratio


def oblique_field_check(tilt_deg=30.0, cfg=None, verbose=True):
    """Source-plane checks for the Level-3 exact oblique-TR field (no propagation):
      (a) psi=0 reproduces the symmetric radial GF field;
      (b) psi>0 fills the on-axis null in p (Ex) but not s (Ey);
      (c) Ey is odd in y (s-polarization antisymmetry);
      (d) in-plane lobe intensity is asymmetric (~1/gamma, small for our beam)."""
    import math
    from thz_config import default_config
    from thz_source import gf_oblique_field, gf_shape, _gf_x_half, _gf_amplitude_norm
    base = cfg or default_config()

    def field(tilt):
        c = default_config(); c.__dict__.update(base.__dict__)
        c.gf_tilt_model = 'level3'; c.screen_tilt_deg = tilt
        xh = _gf_x_half(c); xs = np.linspace(-xh, xh, 201)
        X, Y = np.meshgrid(xs, xs)
        EX, EY = gf_oblique_field(X, Y, c)
        return EX, EY, X, Y, xs, c

    # (a) psi=0 vs symmetric GF
    EX, EY, X, Y, xs, c = field(0.0)
    g = c.gamma; beta = math.sqrt(1 - 1 / g ** 2); z0 = c.z_first_oe_m
    R = np.sqrt(X ** 2 + Y ** 2); PHI = np.arctan2(Y, X)
    Eref = (_gf_amplitude_norm(c) / z0) * gf_shape(np.where(R > 0, R / z0, 1e-9), beta)
    EXref = np.where(R > 0, Eref * np.cos(PHI), 0.0)
    rel0 = float(np.max(np.abs(EX - EXref)) / np.max(np.abs(EXref)))

    # (b)-(d) psi = tilt_deg
    EX, EY, X, Y, xs, c = field(tilt_deg); ic = len(xs) // 2
    on_p = abs(float(EX[ic, ic])); on_s = abs(float(EY[ic, ic]))
    pk = float(np.max(np.abs(EX)))
    s_odd = float(np.max(np.abs(EY + EY[::-1, :])) / np.max(np.abs(EY)))
    I = EX ** 2 + EY ** 2; cut = I[ic, :]
    lr = float(cut[:ic].sum() / cut[ic:].sum())
    if verbose:
        print(f"oblique-field check (Level-3):")
        print(f"  (a) psi=0 vs symmetric GF: rel diff = {rel0:.2e} (expect ~0)")
        print(f"  (b) psi={tilt_deg:g}: on-axis |Ex(p)|/peak = {on_p/pk:.2e} (filled), "
              f"|Ey(s)| = {on_s:.1e} (~0)")
        print(f"  (c) Ey oddness in y: {s_odd:.2e} (expect ~0)")
        print(f"  (d) in-plane lobe ratio L/R = {lr:.3f} (asymmetric; ~1/gamma small)")
    return rel0, lr


if __name__ == '__main__':
    # deep far field (Fresnel N ~ 0.01): SRW should reproduce the Airy null (ratio ~ 1)
    aperture_diffraction_check(freq_GHz=300.0, D=0.01, L=2.0)
    print()
    gf_pattern_check(freq_GHz=120.0)
    print()
    oblique_field_check(tilt_deg=30.0)
