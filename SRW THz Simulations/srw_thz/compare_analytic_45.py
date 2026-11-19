#!/usr/bin/env python
"""
Compare the Level-3 tilted-GF simulation source to the supplied closed-form 45-deg TR field

  E_45(r,w) ∝ [ sinφ/(1-β cosφ sinθ) ] e_φ
            + [ ((β-cosθ)cosφ - sinθ) / ((1-β cosθ)(1-β cosφ sinθ)) ] e_θ

Claim to test: this is exactly the image-charge amplitude that gf_oblique_field builds,
  A(n) = T(n,β) - T(n,β_i),  T(n,b) = (n(n·b)-b)/(1-n·b),
for a 45-deg foil (electron β = β ẑ, image β_i = β x̂), decomposed into (e_θ, e_φ).

Part 1 (analytic, frame-fixed): show A = -β · [given bracket] to machine precision.
Part 2 (code tie-in): call gf_oblique_field(ψ=45°) and show its |E|² source pattern
        matches the given equation's |E|² (correlation ~1, constant ratio).
"""
import sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/paulrleaa/Documents/1_Research/Suli2026/srw_thz")
from thz_config import default_config
from thz_source import gf_oblique_field, _gf_x_half

cfg = default_config(); cfg.gf_tilt_model = 'level3'; cfg.screen_tilt_deg = 45.0
g = cfg.gamma; b = math.sqrt(1 - 1/g**2); z0 = cfg.z_first_oe_m


def Tterm(nx, ny, nz, bx, by, bz):
    ndb = nx*bx + ny*by + nz*bz
    return ((nx*ndb - bx)/(1-ndb), (ny*ndb - by)/(1-ndb), (nz*ndb - bz)/(1-ndb))


def given_bracket(th, ph):
    """(B_theta, B_phi) of the supplied equation (image along +x, electron along +z)."""
    D1 = 1 - b*np.cos(th)
    D2 = 1 - b*np.cos(ph)*np.sin(th)
    B_ph = np.sin(ph)/D2
    B_th = ((b - np.cos(th))*np.cos(ph) - np.sin(th))/(D1*D2)
    return B_th, B_ph


# ============ Part 1: analytic identity  A(n) = -β · B ============
th = np.linspace(0.05, math.pi-0.05, 240)
ph = np.linspace(0.02, 2*math.pi-0.02, 241)
TH, PH = np.meshgrid(th, ph)
nx, ny, nz = np.sin(TH)*np.cos(PH), np.sin(TH)*np.sin(PH), np.cos(TH)
# A = T(n, βẑ) - T(n, βx̂)
ex1, ey1, ez1 = Tterm(nx, ny, nz, 0, 0, b)      # electron term (β along z)
ex2, ey2, ez2 = Tterm(nx, ny, nz, b, 0, 0)      # image term (β_i along x, 45°)
Ax, Ay, Az = ex1-ex2, ey1-ey2, ez1-ez2
# spherical unit vectors, decompose A
eth = (np.cos(TH)*np.cos(PH), np.cos(TH)*np.sin(PH), -np.sin(TH))
eph = (-np.sin(PH), np.cos(PH), np.zeros_like(PH))
A_th = Ax*eth[0] + Ay*eth[1] + Az*eth[2]
A_ph = Ax*eph[0] + Ay*eph[1] + Az*eph[2]
A_r  = Ax*nx + Ay*ny + Az*nz                    # should be ~0 (transverse)
B_th, B_ph = given_bracket(TH, PH)

good = (np.abs(B_th) > 1e-3) & (np.abs(B_ph) > 1e-3)
r_th = (A_th/B_th)[good]
r_ph = (A_ph/B_ph)[good]
print("=== Part 1: analytic identity  A(n) = T(n,βẑ) - T(n,βx̂)  vs  given bracket B ===")
print(f"  expect A_theta/B_theta = A_phi/B_phi = -β = {-b:.6f} everywhere")
print(f"  A_theta/B_theta : mean={r_th.mean():.6f}  max|dev from -β|={np.max(np.abs(r_th+b)):.2e}")
print(f"  A_phi/B_phi     : mean={r_ph.mean():.6f}  max|dev from -β|={np.max(np.abs(r_ph+b)):.2e}")
print(f"  transverse check max|A·n| = {np.max(np.abs(A_r)):.2e}  (expect ~0)")

# ============ Part 2: the actual simulation source vs the equation ============
xh = _gf_x_half(cfg); n = 221
xs = np.linspace(-xh, xh, n); X, Y = np.meshgrid(xs, xs)
EX, EY = (a.astype(float) for a in gf_oblique_field(X, Y, cfg))   # SIMULATION source
I_code = EX**2 + EY**2

# build the given equation's |E|² on the SAME source grid using the same plane projection.
# code (ψ=+45°) has image along -x; the given eq has image along +x -> mirror in x, so we
# evaluate the eq with shat=+x and compare (auto-detecting the x-mirror = tilt-sign convention).
shat = np.array([1.0, 0, 0]); es = np.array([0, 1.0, 0])
ep = np.cross(shat, [0, 1, 0]); ep = ep/np.linalg.norm(ep)         # = (0,0,1)
nX = shat[0] + (X/z0)*ep[0] + (Y/z0)*es[0]
nY = shat[1] + (X/z0)*ep[1] + (Y/z0)*es[1]
nZ = shat[2] + (X/z0)*ep[2] + (Y/z0)*es[2]
nn = np.sqrt(nX**2 + nY**2 + nZ**2); nX, nY, nZ = nX/nn, nY/nn, nZ/nn
th_g = np.arccos(np.clip(nZ, -1, 1)); ph_g = np.arctan2(nY, nX)
Bt, Bp = given_bracket(th_g, ph_g)
I_given = Bt**2 + Bp**2

# correlate (handle the x-mirror tilt-sign convention)
def corr(a, b_):
    a = a.ravel(); b_ = b_.ravel()
    return np.corrcoef(a, b_)[0, 1]
r_same = corr(I_code, I_given)
r_flip = corr(I_code, np.fliplr(I_given))
flip = r_flip > r_same
Ig = np.fliplr(I_given) if flip else I_given
r = max(r_same, r_flip)
ratio = I_code[I_code > 0.2*I_code.max()].sum() / Ig[I_code > 0.2*I_code.max()].sum()
print("\n=== Part 2: gf_oblique_field(ψ=45°)  vs  given equation, |E|² over source grid ===")
print(f"  intensity-pattern correlation r = {r:.5f}"
      f"{'  (x-mirror applied = tilt-direction convention)' if flip else ''}")
print(f"  constant scale (code/eq, ring region) = {ratio:.3e}  (absolute-normalization only)")

# figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ext = [xs[0]*1e3, xs[-1]*1e3, xs[0]*1e3, xs[-1]*1e3]
im0 = ax[0].imshow(I_code, origin='lower', extent=ext, cmap='viridis', aspect='equal')
ax[0].set_title('Simulation: gf_oblique_field (ψ=45°)\n$|E|^2$'); ax[0].set_xlabel('x [mm]'); ax[0].set_ylabel('y [mm]')
fig.colorbar(im0, ax=ax[0], fraction=0.046)
im1 = ax[1].imshow(Ig/Ig.max()*I_code.max(), origin='lower', extent=ext, cmap='viridis', aspect='equal')
ax[1].set_title('Given equation $E_{45°}$\n$|E|^2$ (scaled)'); ax[1].set_xlabel('x [mm]'); ax[1].set_ylabel('y [mm]')
fig.colorbar(im1, ax=ax[1], fraction=0.046)
m = I_code > 0.02*I_code.max()
ax[2].scatter((Ig[m]/Ig.max()).ravel(), (I_code[m]/I_code.max()).ravel(), s=4, alpha=0.3)
ax[2].plot([0, 1], [0, 1], 'r--', lw=1)
ax[2].set_title(f'pixelwise (r = {r:.4f})'); ax[2].set_xlabel('given eq |E|² (norm)'); ax[2].set_ylabel('simulation |E|² (norm)')
ax[2].grid(alpha=0.3)
fig.suptitle('Level-3 tilted-GF simulation vs supplied 45° TR equation  —  identical up to overall constant',
             fontsize=12, y=1.00)
fig.tight_layout()
out = '/home/paulrleaa/Documents/1_Research/Suli2026/srw_thz/results_l3/compare_analytic_45.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f"\nwrote {out}")
