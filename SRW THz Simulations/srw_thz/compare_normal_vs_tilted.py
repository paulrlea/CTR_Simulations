#!/usr/bin/env python
"""
Source-plane comparison of two CTR source models on a shared grid / frequency:

  A) NORMAL-INCIDENCE Ginzburg-Frank, NO diffraction
     foil perpendicular to the beam; symmetric radial TR ring, on-axis null, infinite foil.
     E(theta) ~ gf_shape(theta) e_r,  gf_shape = beta sin(theta)/(1 - beta^2 cos^2 theta).

  B) TILTED (45 deg) + finite-foil DIFFRACTION
     analytic 45-deg oblique-TR vector field (true p/s polarization, elliptical footprint)
     multiplied by the far-zone finite-foil form factor H(theta_ell).

Physics to read off (a = 12.7 mm = 0.5" foil, gamma ~ 117 -> a/r_eff < 1 across the whole
10-400 GHz band, so diffraction bites band-wide):
  - tilt      -> ELLIPTICAL footprint (45 deg foreshortening: cos45 in theta_x -> ~1.41x wider in x)
  - diffraction -> the near-axis TR ring (theta ~ 1/gamma) is strongly SUPPRESSED, the hard central
                   null is PARTIALLY FILLED, and energy is REWEIGHTED to larger angles + edge
                   fringes; the source is also absolutely much DIMMER at low f.
Shown at 400 GHz (top of band, a ~ r_eff -> mildest diffraction, ring best survives); suppression
is stronger at lower f. The physical collimator acceptance angle is drawn for reference (the mesh
extends past it on purpose to avoid a spurious diffracting edge).
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from thz_config import default_config
from thz_source import (gf_shape, _gf_amplitude_norm, gf_analytic_field,
                        finite_foil_formfactor, _gf_x_half)

OUT = os.path.join(_HERE, 'results_finitefoil')
os.makedirs(OUT, exist_ok=True)

F_HZ = 400e9                       # top of band: a ~ r_eff, mildest diffraction, ring survives best
cfg = default_config()
g = cfg.gamma
beta = math.sqrt(1 - 1 / g ** 2)
z0 = cfg.z_first_oe_m
a = cfg.foil_radius_m
th_coll = cfg.collimator_radius_m / z0        # collimator acceptance half-angle
r_coll_mm = cfg.collimator_radius_m * 1e3

# shared source grid
xh = _gf_x_half(cfg)
n = 401
xs = np.linspace(-xh, xh, n)
X, Y = np.meshgrid(xs, xs)
R = np.sqrt(X * X + Y * Y)
thx, thy = X / z0, Y / z0

# ---------- A) normal-incidence GF, no diffraction (symmetric radial) ----------
A = _gf_amplitude_norm(cfg)
theta = R / z0
with np.errstate(divide='ignore', invalid='ignore'):
    Eamp = (A / z0) * gf_shape(theta, beta)
    PHI = np.arctan2(Y, X)
    EXa = np.where(R > 0, Eamp * np.cos(PHI), 0.0)
    EYa = np.where(R > 0, Eamp * np.sin(PHI), 0.0)
Ia = EXa ** 2 + EYa ** 2

# ---------- B) tilted 45 deg + finite-foil diffraction ----------
cfgB = default_config()
cfgB.gf_tilt_model = 'analytic'
cfgB.screen_tilt_deg = 45.0
cfgB.finite_foil_diffraction = True
EXb, EYb = (arr.astype(float) for arr in gf_analytic_field(X, Y, cfgB))
cpsi = math.cos(math.radians(cfgB.screen_tilt_deg))
theta_ell = np.sqrt((cpsi * thx) ** 2 + thy ** 2)
Hb = finite_foil_formfactor(theta_ell, F_HZ, cfgB)
Ib = (EXb * Hb) ** 2 + (EYb * Hb) ** 2

# A and B share the SAME analytic amplitude normalization A/z0, so intensities are directly
# comparable in absolute terms. Report the key contrasts.
vA = Ia.max()
mid = n // 2
# collimator-limited energy (what actually propagates) for an honest energy ratio
within = R <= cfg.collimator_radius_m
E_A = Ia[within].sum()
E_B = Ib[within].sum()
print(f"=== source comparison @ {F_HZ/1e9:.0f} GHz  (gamma={g:.1f}, z0={z0:.3f} m, a={a*1e3:.1f} mm, "
      f"collimator theta*gamma={th_coll*g:.1f}) ===")
print(f"  A normal GF:  ring at r={xs[mid:][np.argmax(Ia[mid,mid:])]*1e3:.2f} mm (symmetric), "
      f"on-axis |E|^2/peak = {Ia[mid,mid]/vA:.1e} (hard null)")
print(f"  B tilt+diff:  on-axis |E|^2/A-peak = {Ib[mid,mid]/vA:.2e} (null partly filled)")
print(f"  B/A intensity at A's ring pixel = {Ib[mid, mid+np.argmax(Ia[mid,mid:])]/vA:.3f} "
      f"(ring suppressed by diffraction)")
print(f"  collimator-limited energy B/A = {E_B/E_A:.3f}")

# ============================ figure ============================
fig = plt.figure(figsize=(15, 9))
ext = [xs[0] * 1e3, xs[-1] * 1e3, xs[0] * 1e3, xs[-1] * 1e3]
th_grid = np.linspace(0, 2 * np.pi, 200)


def draw_coll(ax):
    ax.plot(r_coll_mm * np.cos(th_grid), r_coll_mm * np.sin(th_grid), 'w--', lw=1.0, alpha=0.8)


# Panel 1: A, normalized to A peak
axA = fig.add_subplot(2, 3, 1)
imA = axA.imshow(Ia / vA, origin='lower', extent=ext, cmap='viridis', vmax=1.0)
draw_coll(axA)
axA.set_title('A) normal incidence GF, no diffraction\n$|E|^2$ / A-peak — symmetric ring, hard null')
axA.set_xlabel('x [mm]'); axA.set_ylabel('y [mm]'); fig.colorbar(imA, ax=axA, fraction=0.046)

# Panel 2: B, SAME absolute scale as A (shows absolute suppression)
axB = fig.add_subplot(2, 3, 2)
imB = axB.imshow(Ib / vA, origin='lower', extent=ext, cmap='viridis', vmax=1.0)
draw_coll(axB)
axB.set_title('B) tilted 45$\\degree$ + diffraction\n$|E|^2$ / A-peak (SAME scale) — absolutely dimmer')
axB.set_xlabel('x [mm]'); axB.set_ylabel('y [mm]'); fig.colorbar(imB, ax=axB, fraction=0.046)

# Panel 3: B self-normalized (reveals reshaped structure)
axC = fig.add_subplot(2, 3, 3)
imC = axC.imshow(Ib / Ib.max(), origin='lower', extent=ext, cmap='magma', vmax=1.0)
draw_coll(axC)
axC.set_title('B) self-normalized $|E|^2$\nelliptical, null filled, energy reweighted outward')
axC.set_xlabel('x [mm]'); axC.set_ylabel('y [mm]'); fig.colorbar(imC, ax=axC, fraction=0.046)

# Panel 4: radial lineouts, A-normalized, log-y (absolute suppression + null fill)
axD = fig.add_subplot(2, 3, 4)
r_mm = xs[mid:] * 1e3
axD.semilogy(r_mm, np.maximum(Ia[mid, mid:] / vA, 1e-6), lw=2, label='A normal GF (x)')
axD.semilogy(r_mm, np.maximum(Ib[mid, mid:] / vA, 1e-6), lw=2, label='B tilt+diff (x)')
axD.semilogy(r_mm, np.maximum(Ib[mid:, mid] / vA, 1e-6), '--', lw=1.6, label='B tilt+diff (y)')
axD.axvline(r_coll_mm, color='k', ls=':', lw=1, label=f'collimator {r_coll_mm:.0f} mm')
axD.set_title(f'radial lineout $|E|^2$/A-peak (log) @ {F_HZ/1e9:.0f} GHz')
axD.set_xlabel('r [mm]'); axD.set_ylabel('$|E|^2$ / A-peak'); axD.set_ylim(1e-4, 2)
axD.legend(fontsize=8); axD.grid(alpha=0.3, which='both')

# Panel 5: B self-normalized angular lineouts x vs y -> ellipticity
axE = fig.add_subplot(2, 3, 5)
thg = xs[mid:] / z0 * g
axE.plot(thg, Ib[mid, mid:] / Ib.max(), lw=2, label='B along $\\theta_x$ (tilt plane)')
axE.plot(thg, Ib[mid:, mid] / Ib.max(), '--', lw=1.8, label='B along $\\theta_y$')
axE.axvline(th_coll * g, color='k', ls=':', lw=1, label='collimator')
axE.set_title('B ellipticity: $\\theta_x$ broadened $\\approx1.41\\times$ vs $\\theta_y$')
axE.set_xlabel('$\\theta\\,\\gamma$'); axE.set_ylabel('$|E|^2$ (self-norm)')
axE.set_xlim(0, th_coll * g * 1.1); axE.legend(fontsize=8); axE.grid(alpha=0.3)

# Panel 6: ingredients of B
axF = fig.add_subplot(2, 3, 6)
thp = np.linspace(1e-4, th_coll * 1.1, 500)
axF.plot(thp * g, (gf_shape(thp, beta) / gf_shape(thp, beta).max()) ** 2, lw=2,
         label='GF $|shape|^2$ (norm)')
axF.plot(thp * g, finite_foil_formfactor(thp, F_HZ, cfgB), lw=2, label=f'H($\\theta$) @ {F_HZ/1e9:.0f} GHz')
axF.axhline(1.0, color='k', lw=0.7, ls=':')
axF.axvline(th_coll * g, color='k', ls=':', lw=1, label='collimator')
axF.set_title('B = GF shape $\\times$ form factor H\n(H suppresses the ring, recovers at large $\\theta$)')
axF.set_xlabel('$\\theta\\,\\gamma$'); axF.set_ylabel('value'); axF.legend(fontsize=8); axF.grid(alpha=0.3)

fig.suptitle('CTR source: NORMAL-INCIDENCE GF (no diffraction)  vs  TILTED 45$\\degree$ + FINITE-FOIL DIFFRACTION',
             fontsize=13, y=0.995)
fig.tight_layout()
out = os.path.join(OUT, 'compare_normal_vs_tilted.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f"\nwrote {out}")
