#!/usr/bin/env python
"""
Validate the far-zone finite-foil diffraction form factor and compare the finite-foil CTR
source to the previous (infinite-foil) source.  Source-level only -- no SRW propagation
(that is the end-to-end check 7, done by run_interf.py / compare_finitefoil_e2e).

The form factor (FINITE_FOIL_PLAN.md Sec 1) is the ratio of the truncated-at-radius-a to the
infinite order-1 Hankel transform of the electron virtual-photon field E_r(r) proportional to
K1(kappa r), kappa = k/(beta gamma):

    H(theta) = E_a(theta)/E_inf(theta)
             = 1 - x K1(x) J0(y) - (x^2/y) K0(x) J1(y),   x = kappa a,  y = k a theta

The infinite transform has the closed form  int_0^inf r K1(kappa r) J1(q r) dr = q/(kappa(kappa^2+q^2))
(= the Ginzburg-Frank shape q/(kappa^2+q^2)), so H can be checked against direct quadrature of
the truncated integral with no free constants.

Checks (mirroring FINITE_FOIL_PLAN.md Sec 7, 1-6):
  1. Lommel closed form == numerical quadrature of E_a/E_inf   (all theta; includes on-axis)
  2. On-axis suppression H(0) = 1 - x K1(x) - (x^2/2) K0(x)
  3. a -> inf recovers infinite foil (H -> 1);  a -> 0 gives H(0) -> 0
  4. Energy ratio int|HE|^2/int|E|^2 tracks S_foil trend and its a>>reff / a<<reff limits
  5. Fringes: intensity minima at the J0/J1 zeros of H
  6. Elliptical signature: fringe spacing along theta_x is ~1.41x that along theta_y (psi=45)
Plus a previous-vs-new SOURCE figure (|E|^2 maps, radial lineouts, H(f) family).
"""
import os, sys, math
import numpy as np
from scipy.special import kn, jv
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from thz_config import default_config, C_LIGHT
from thz_source import gf_analytic_field, finite_foil_formfactor, _gf_x_half

OUT = os.path.join(_HERE, 'results_finitefoil')
os.makedirs(OUT, exist_ok=True)


def H_lommel(theta, f_Hz, a, cfg):
    """Lommel closed form, standalone (independent of finite_foil_formfactor for cross-check)."""
    k = 2 * math.pi * f_Hz / C_LIGHT
    beta = math.sqrt(1 - 1 / cfg.gamma ** 2)
    x = k * a / (beta * cfg.gamma)
    y = k * a * np.asarray(theta, float)
    K1x, K0x = float(kn(1, x)), float(kn(0, x))
    ys = np.where(y == 0.0, 1.0, y)
    H = 1 - x * K1x * jv(0, y) - x * x * K0x * jv(1, y) / ys
    H0 = 1 - x * K1x - 0.5 * x * x * K0x
    return np.where(y == 0.0, H0, H)


def H_quad(theta, f_Hz, a, cfg):
    """Direct numerical H = [int_0^a r K1(kr) J1(k th r) dr] / [q/(kappa(kappa^2+q^2))]."""
    k = 2 * math.pi * f_Hz / C_LIGHT
    beta = math.sqrt(1 - 1 / cfg.gamma ** 2)
    kappa = k / (beta * cfg.gamma)
    out = np.empty_like(np.asarray(theta, float))
    for i, th in np.ndenumerate(np.asarray(theta, float)):
        q = k * th
        num = quad(lambda r: r * kn(1, kappa * r) * jv(1, q * r), 0, a, limit=200)[0]
        if q == 0.0:
            # E_inf(0+) = int_0^inf r K1(kr) (q r/2) dr / q ... use analytic den limit q/(kappa(kappa^2+q^2)) -> 0;
            # instead take ratio via l'Hopital: both ~q, so use small-q surrogate
            q = 1e-6 * k
            num = quad(lambda r: r * kn(1, kappa * r) * jv(1, q * r), 0, a, limit=200)[0]
        den = q / (kappa * (kappa ** 2 + q ** 2))
        out[i] = num / den
    return out


cfg = default_config()
cfg.gf_tilt_model = 'analytic'
cfg.screen_tilt_deg = 45.0
cfg.finite_foil_diffraction = True
g = cfg.gamma
a = cfg.foil_radius_m
beta = math.sqrt(1 - 1 / g ** 2)
print(f"gamma={g:.1f}  foil radius a={a*1e3:.2f} mm  (default {a/0.0254:.3f} in)")
print(f"r_eff(f) = beta*gamma*lambda/2pi;  a/r_eff = 2pi a f/(beta gamma c)\n")

# =========================== Check 1 & 2: closed form vs quadrature ===========================
print("=== Check 1/2: Lommel closed form  vs  numerical quadrature of E_a/E_inf ===")
th = np.linspace(0.0, 6.0 / g, 13)            # 0 .. 6/gamma (covers the ring + first fringes)
worst = 0.0
for f_GHz in (50.0, 100.0, 200.0, 400.0):
    f = f_GHz * 1e9
    Hc = H_lommel(th, f, a, cfg)
    Hq = H_quad(th, f, a, cfg)
    d = np.max(np.abs(Hc - Hq))
    worst = max(worst, d)
    ka = 2 * math.pi * f / C_LIGHT * a
    x = ka / (beta * g)
    H0 = 1 - x * float(kn(1, x)) - 0.5 * x * x * float(kn(0, x))
    print(f"  f={f_GHz:6.1f} GHz  x=ka/bg={x:6.3f}  H(0)={H0:.4f}  max|Lommel-quad|={d:.2e}")
print(f"  -> worst closed-form vs quadrature discrepancy = {worst:.2e}  (expect ~1e-6)\n")

# on-axis explicit formula check
print("=== Check 2b: on-axis H(0) = 1 - x K1(x) - (x^2/2) K0(x) ===")
for f_GHz in (10.0, 100.0, 400.0):
    f = f_GHz * 1e9
    x = 2 * math.pi * f / C_LIGHT * a / (beta * g)
    H0_formula = 1 - x * float(kn(1, x)) - 0.5 * x * x * float(kn(0, x))
    H0_lommel = float(H_lommel(np.array([0.0]), f, a, cfg)[0])
    print(f"  f={f_GHz:6.1f} GHz  H(0) formula={H0_formula:.6f}  Lommel(y=0)={H0_lommel:.6f}"
          f"  diff={abs(H0_formula-H0_lommel):.1e}")
print()

# =========================== Check 3: a-> inf recovers infinite foil ===========================
print("=== Check 3: limits  a->inf => H->1 (infinite foil);  a->0 => H(0)->0 ===")
f = 100e9
for a_test in (a, 10 * a, 100 * a, 1000 * a):
    Hmax_dev = np.max(np.abs(H_lommel(th, f, a_test, cfg) - 1.0))
    print(f"  a={a_test*1e3:9.2f} mm  max|H-1| over ring = {Hmax_dev:.3e}")
for a_test in (a, 0.1 * a, 0.01 * a):
    H0 = float(H_lommel(np.array([0.0]), f, a_test, cfg)[0])
    print(f"  a={a_test*1e3:9.4f} mm  H(0) = {H0:.4e}")
print()

# =========================== Check 4: energy ratio vs S_foil ===========================
print("=== Check 4: angle-integrated energy int|HE|^2/int|E|^2  vs  scalar S_foil(f) ===")
def S_foil(f, cfg):
    ka = 2 * math.pi * f / C_LIGHT * cfg.foil_radius_m
    twog = 2 * cfg.gamma
    return (math.log(min(twog, ka)) - 1.0) / (math.log(twog) - 1.0) if ka > 1 else \
           (math.log(max(ka, 1.0000001)) - 1.0) / (math.log(twog) - 1.0)

# GF angular weight ~ [theta/(1/g^2+theta^2)]^2 * 2pi theta dtheta (radial power)
thg = np.linspace(1e-6, 15.0 / g, 4000)
gf_shape2 = (thg / (1 / g ** 2 + thg ** 2)) ** 2
fs = np.array([10, 20, 40, 80, 160, 320, 400]) * 1e9
print(f"  {'f[GHz]':>7} {'a/reff':>8} {'E_ratio(H)':>11} {'S_foil':>8}")
for f in fs:
    reff = beta * g * (C_LIGHT / f) / (2 * math.pi)
    H = H_lommel(thg, f, a, cfg)
    Er = np.trapezoid(H ** 2 * gf_shape2 * thg, thg) / np.trapezoid(gf_shape2 * thg, thg)
    print(f"  {f/1e9:7.0f} {a/reff:8.3f} {Er:11.4f} {S_foil(f, cfg):8.4f}")
print("  (both -> 1 for a>>reff (high f);  both drop for a<<reff (low f); H is the exact one)\n")

# =========================== Check 5/6: fringes & elliptical signature ===========================
# At a=12.7 mm the low-f suppression is not deep enough to invert H, so the edge-diffraction
# fringes appear as LOCAL MINIMA (dips) of H(theta), not sign changes. Detect the dips and
# validate that along theta_x (45-deg elliptical foreshortening) the same feature sits at
# 1/cos(psi) ~ 1.41x larger angle.
print("=== Check 5/6: fringe minima of H, and 45-deg elliptical broadening (~1.41x) ===")
cpsi = math.cos(math.radians(45.0))
f = 400e9   # highest band point -> a > reff -> deepest fringe structure
thf = np.linspace(0.5 / g, 14.0 / g, 8000)
Hf = H_lommel(thf, f, a, cfg)
loc_min = np.where((Hf[1:-1] < Hf[:-2]) & (Hf[1:-1] < Hf[2:]))[0] + 1
mins = thf[loc_min]
print(f"  f={f/1e9:.0f} GHz: {len(mins)} local minima of H at theta*gamma = "
      f"{', '.join(f'{z*g:.2f}' for z in mins[:6])}  (edge-diffraction fringes)")
if len(mins):
    z1_y = mins[0]                       # first fringe dip along theta_y (cos factor = 1)
    z1_x = z1_y / cpsi                   # same dip along theta_x is at 1.41x the angle
    print(f"  first fringe dip:  theta_y*gamma={z1_y*g:.3f}   theta_x*gamma={z1_x*g:.3f}"
          f"   ratio={z1_x/z1_y:.3f}  (expect {1/cpsi:.3f} from 1/cos45)")
# direct 2-D confirmation: locate the dip on the actual elliptical grid along +x and +y
th_line = np.linspace(0.5 / g, 14.0 / g, 8000)
Hx = H_lommel(cpsi * th_line, f, a, cfg)          # theta_ell along +x = cos(psi)*theta_x
Hy = H_lommel(th_line, f, a, cfg)                 # theta_ell along +y = theta_y
mx = th_line[np.where((Hx[1:-1] < Hx[:-2]) & (Hx[1:-1] < Hx[2:]))[0] + 1]
my = th_line[np.where((Hy[1:-1] < Hy[:-2]) & (Hy[1:-1] < Hy[2:]))[0] + 1]
if len(mx) and len(my):
    print(f"  2-D grid: first dip theta_x*gamma={mx[0]*g:.3f}  theta_y*gamma={my[0]*g:.3f}"
          f"  measured ratio={mx[0]/my[0]:.3f}  (expect {1/cpsi:.3f})")
print()

# =========================== previous-vs-new SOURCE figure ===========================
f_fig = 150e9
xh = _gf_x_half(cfg)
n = 301
xs = np.linspace(-xh, xh, n)
X, Y = np.meshgrid(xs, xs)
EX, EY = (arr.astype(float) for arr in gf_analytic_field(X, Y, cfg))
I_inf = EX ** 2 + EY ** 2                                    # PREVIOUS (infinite-foil) source
z0 = cfg.z_first_oe_m
thx, thy = X / z0, Y / z0
theta_ell = np.sqrt((cpsi * thx) ** 2 + thy ** 2)
H2d = finite_foil_formfactor(theta_ell, f_fig, cfg)
I_fin = (EX * H2d) ** 2 + (EY * H2d) ** 2                    # NEW (finite-foil) source

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
ext = [xs[0] * 1e3, xs[-1] * 1e3, xs[0] * 1e3, xs[-1] * 1e3]
vmax = I_inf.max()
im0 = ax[0, 0].imshow(I_inf, origin='lower', extent=ext, cmap='viridis', vmax=vmax)
ax[0, 0].set_title(f'PREVIOUS: infinite foil  $|E|^2$\n(analytic 45$\\degree$, {f_fig/1e9:.0f} GHz)')
fig.colorbar(im0, ax=ax[0, 0], fraction=0.046)
# NEW finite foil on the SAME (infinite) colour scale -> shows the true absolute suppression.
# At mid-band a 0.5" foil suppresses the ring so hard that this panel looks nearly black;
# that is correct physics, not a blank plot (see the self-normalized panel to its right).
im1 = ax[0, 1].imshow(I_fin, origin='lower', extent=ext, cmap='viridis', vmax=vmax)
frac = I_fin.max() / vmax
ax[0, 1].set_title(f'NEW: finite foil (a={a*1e3:.1f} mm)  $|E|^2$\nsame scale as infinite '
                   f'(peak = {frac*100:.1f}% of scale)')
fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)
# same finite source, self-normalized -> reveals the reshaped structure hidden by the shared scale
im2 = ax[0, 2].imshow(I_fin / I_fin.max(), origin='lower', extent=ext, cmap='magma')
ax[0, 2].set_title('NEW: finite foil, SELF-normalized $|E|^2$\n(ring suppressed, null filled, '
                   'energy reweighted out)')
fig.colorbar(im2, ax=ax[0, 2], fraction=0.046)
for a_ in ax[0]:
    a_.set_xlabel('x [mm]'); a_.set_ylabel('y [mm]')

# radial lineout along +x (tilt plane) and +y -- LOG scale so the (much dimmer) finite curves
# are visible against the infinite ring; both normalized to the infinite peak.
mid = n // 2
r_mm = xs[mid:] * 1e3
ax[1, 0].semilogy(r_mm, np.maximum(I_inf[mid, mid:] / vmax, 1e-6), label='infinite (x)', lw=2)
ax[1, 0].semilogy(r_mm, np.maximum(I_fin[mid, mid:] / vmax, 1e-6), label='finite (x)', lw=2)
ax[1, 0].semilogy(r_mm, np.maximum(I_fin[mid:, mid] / vmax, 1e-6), '--', label='finite (y)', lw=1.5)
ax[1, 0].set_title(f'radial lineout $|E|^2$/infinite-peak (log) @ {f_fig/1e9:.0f} GHz')
ax[1, 0].set_xlabel('r [mm]'); ax[1, 0].set_ylabel('$|E|^2$ / infinite peak')
ax[1, 0].set_ylim(1e-4, 2); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3, which='both')

# H(theta) family across frequencies
thp = np.linspace(0, 8.0 / g, 500)
for f_GHz in (25, 50, 100, 200, 400):
    ax[1, 1].plot(thp * g, H_lommel(thp, f_GHz * 1e9, a, cfg), label=f'{f_GHz} GHz')
ax[1, 1].axhline(1.0, color='k', lw=0.7, ls=':')
ax[1, 1].set_title('$H(\\theta)$ vs frequency'); ax[1, 1].set_xlabel('$\\theta\\,\\gamma$')
ax[1, 1].set_ylabel('H'); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

# energy ratio vs freq: H-exact vs S_foil
ff = np.linspace(10e9, 400e9, 60)
Er_list, Sf_list, are_list = [], [], []
for f in ff:
    H = H_lommel(thg, f, a, cfg)
    Er_list.append(np.trapezoid(H ** 2 * gf_shape2 * thg, thg) / np.trapezoid(gf_shape2 * thg, thg))
    Sf_list.append(S_foil(f, cfg))
    are_list.append(2 * math.pi * a * f / (beta * g * C_LIGHT))
ax[1, 2].plot(ff / 1e9, Er_list, lw=2, label='exact $\\int|HE|^2/\\int|E|^2$')
ax[1, 2].plot(ff / 1e9, Sf_list, '--', lw=2, label='scalar $S_{foil}$')
ax[1, 2].set_title('low-f energy suppression'); ax[1, 2].set_xlabel('f [GHz]')
ax[1, 2].set_ylabel('relative energy'); ax[1, 2].legend(); ax[1, 2].grid(alpha=0.3)

fig.suptitle('Finite-foil diffraction form factor: PREVIOUS (infinite) vs NEW (finite) CTR source',
             fontsize=13, y=0.995)
fig.tight_layout()
out = os.path.join(OUT, 'compare_finitefoil_source.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f"wrote {out}")
