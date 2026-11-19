#!/usr/bin/env python
"""
End-to-end comparison (FINITE_FOIL_PLAN.md Sec 7, check 7): propagate the PREVIOUS infinite-foil
CTR source and the NEW finite-foil source through the identical interferometer beamline and compare
detector-plane observables -- spatial map / ring, detected spectrum, power capture, recovered
spectrum.  Reads the two completed run directories (no propagation here).

    previous (infinite foil): results_interf_analytic45      (analytic 45, golay)
    new      (finite   foil): results_interf_finitefoil45    (analytic 45, golay, --finite-foil)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.join(_HERE, 'results_interf_analytic45')      # PREVIOUS: infinite foil
FIN = os.path.join(_HERE, 'results_interf_finitefoil45')    # NEW: finite foil
OUT = os.path.join(_HERE, 'results_interf_finitefoil45')

for d, tag in ((INF, 'infinite'), (FIN, 'finite')):
    if not os.path.isdir(d):
        sys.exit(f"missing run dir ({tag} foil): {d}")


def load_txt(d, name):
    return np.loadtxt(os.path.join(d, name))


spec_i = load_txt(INF, 'spectrum.txt')            # f[GHz], P[arb]
spec_f = load_txt(FIN, 'spectrum.txt')
pc_i = load_txt(INF, 'power_capture.txt')          # r[mm], captured fraction
pc_f = load_txt(FIN, 'power_capture.txt')
rs_i = load_txt(INF, 'recovered_spectrum.txt')     # f, S_direct, S_recovered
rs_f = load_txt(FIN, 'recovered_spectrum.txt')
map_i = np.load(os.path.join(INF, 'spatial_map.npz'))
map_f = np.load(os.path.join(FIN, 'spatial_map.npz'))

# ---- console summary ----
def peak_ghz(s):
    return s[np.argmax(s[:, 1]), 0]
print("=== end-to-end infinite vs finite foil ===")
print(f"  detected-spectrum peak:  infinite={peak_ghz(spec_i):.1f} GHz   finite={peak_ghz(spec_f):.1f} GHz")
print(f"  ring radius:             infinite={float(map_i['ring_radius_m'])*1e3:.2f} mm   "
      f"finite={float(map_f['ring_radius_m'])*1e3:.2f} mm")
ap = float(map_i['golay_aperture_m']) * 1e3
def cap_at(pc, r_mm):
    return np.interp(r_mm, pc[:, 0], pc[:, 1])
print(f"  power captured @ {ap:.1f} mm:  infinite={cap_at(pc_i, ap)*100:.1f}%   finite={cap_at(pc_f, ap)*100:.1f}%")
# total detected energy ratio (finite/infinite), trapezoid over f
E_i = np.trapezoid(spec_i[:, 1], spec_i[:, 0])
E_f = np.trapezoid(spec_f[:, 1], spec_f[:, 0])
print(f"  total detected energy finite/infinite = {E_f/E_i:.3f}  (finite foil radiates less low-f)")

# ============================ figure ============================
fig = plt.figure(figsize=(15, 9))

# spatial maps (shared colour scale = infinite max)
Ii, If = map_i['I'], map_f['I']
xi = map_i['x'] * 1e3
ext = [xi[0], xi[-1], xi[0], xi[-1]]
vmax = Ii.max()
axA = fig.add_subplot(2, 3, 1)
imA = axA.imshow(Ii, origin='lower', extent=ext, cmap='viridis', vmax=vmax)
axA.set_title(f"PREVIOUS infinite foil\ndetector $|E|^2$  (ring {float(map_i['ring_radius_m'])*1e3:.2f} mm)")
axA.set_xlabel('x [mm]'); axA.set_ylabel('y [mm]'); fig.colorbar(imA, ax=axA, fraction=0.046)
axB = fig.add_subplot(2, 3, 2)
imB = axB.imshow(If, origin='lower', extent=ext, cmap='viridis', vmax=vmax)
axB.set_title(f"NEW finite foil (a=12.7 mm)\ndetector $|E|^2$ (same scale)")
axB.set_xlabel('x [mm]'); axB.set_ylabel('y [mm]'); fig.colorbar(imB, ax=axB, fraction=0.046)

# radial lineout of the two maps along +x
mid = Ii.shape[0] // 2
r_mm = xi[mid:]
axC = fig.add_subplot(2, 3, 3)
axC.plot(r_mm, Ii[mid, mid:] / vmax, lw=2, label='infinite')
axC.plot(r_mm, If[mid, mid:] / vmax, lw=2, label='finite')
axC.axvline(ap, color='k', ls=':', lw=1, label=f'Golay ap {ap:.1f} mm')
axC.set_title('detector radial lineout (+x)'); axC.set_xlabel('r [mm]')
axC.set_ylabel('$|E|^2$ (norm to infinite)'); axC.legend(); axC.grid(alpha=0.3)

# detected spectrum overlay
axD = fig.add_subplot(2, 3, 4)
axD.plot(spec_i[:, 0], spec_i[:, 1] / spec_i[:, 1].max(), lw=2, label='infinite')
axD.plot(spec_f[:, 0], spec_f[:, 1] / spec_i[:, 1].max(), lw=2, label='finite')
axD.set_title('detected spectrum (norm to infinite peak)'); axD.set_xlabel('f [GHz]')
axD.set_ylabel('P [arb]'); axD.legend(); axD.grid(alpha=0.3)

# power capture overlay
axE = fig.add_subplot(2, 3, 5)
axE.plot(pc_i[:, 0], pc_i[:, 1] * 100, lw=2, label='infinite')
axE.plot(pc_f[:, 0], pc_f[:, 1] * 100, lw=2, label='finite')
axE.axvline(ap, color='k', ls=':', lw=1, label=f'Golay ap {ap:.1f} mm')
axE.set_title('power capture vs aperture'); axE.set_xlabel('aperture radius [mm]')
axE.set_ylabel('captured power [%]'); axE.legend(); axE.grid(alpha=0.3)

# recovered spectrum overlay (from autocorrelation)
axF = fig.add_subplot(2, 3, 6)
axF.plot(rs_i[:, 0], rs_i[:, 2] / np.nanmax(np.abs(rs_i[:, 2])), lw=2, label='infinite (recovered)')
axF.plot(rs_f[:, 0], rs_f[:, 2] / np.nanmax(np.abs(rs_f[:, 2])), lw=2, label='finite (recovered)')
axF.set_title('recovered spectrum (interferogram FT)'); axF.set_xlabel('f [GHz]')
axF.set_ylabel('S (norm)'); axF.legend(); axF.grid(alpha=0.3)

fig.suptitle('CTR interferometer end-to-end: PREVIOUS infinite-foil vs NEW finite-foil source '
             '(analytic 45$\\degree$, Golay)', fontsize=13, y=0.995)
fig.tight_layout()
out = os.path.join(OUT, 'compare_finitefoil_e2e.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f"\nwrote {out}")
