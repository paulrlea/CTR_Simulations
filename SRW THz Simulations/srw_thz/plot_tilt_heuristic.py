#!/usr/bin/env python
"""
Illustrate the Level-1 tilted-screen lobe heuristic
    lobe(theta_x) = max(0, 1 + k * tan(psi) * tanh(gamma * theta_x))
and why each factor was chosen. Writes source_tilt_heuristic.png.

    python plot_tilt_heuristic.py
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thz_config import default_config
from thz_source import gf_shape

cfg = default_config()
g = cfg.gamma
beta = math.sqrt(1.0 - 1.0 / g ** 2)

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

# -- Panel A: the tilt-strength factor tan(psi) vs the rigorous C(psi) ----------
psi = np.linspace(0, np.radians(80), 400)
tan_fac = np.tan(psi)
C_rig = np.sin(2 * psi) / (1 + beta * np.cos(2 * psi))      # image-charge result
a = ax[0, 0]
a.plot(np.degrees(psi), tan_fac, lw=2, label=r'heuristic  $\tan\psi$')
a.plot(np.degrees(psi), C_rig, lw=2, ls='--',
       label=r'rigorous  $\sin 2\psi/(1+\beta\cos 2\psi)$')
a.axhline(0, color='k', lw=0.5)
a.set_xlabel(r'tilt $\psi$ [deg]'); a.set_ylabel('tilt-strength factor')
a.set_ylim(0, 4); a.set_title(r'(A) tilt factor: $\tan\psi$ IS the exact $\beta\to1$ form')
a.legend(fontsize=9)
a.text(2, 3.2, r'curves coincide: ' '\n'
       r'$\sin 2\psi/(1{+}\beta\cos 2\psi)\to\tan\psi$' '\n'
       r'as $\beta\to1$ (differ only as $\psi\to90^\circ$)', fontsize=8, color='gray')

# -- Panel B: tanh(u) vs linear u  (u = gamma*theta_x) -------------------------
u = np.linspace(-4, 4, 400)
a = ax[0, 1]
a.plot(u, np.tanh(u), lw=2, label=r'$\tanh(\gamma\theta_x)$  (used)')
a.plot(u, u, lw=2, ls='--', label=r'linear $\gamma\theta_x$  (rejected)')
a.axhline(1, color='gray', lw=0.7, ls=':'); a.axhline(-1, color='gray', lw=0.7, ls=':')
a.axhline(0, color='k', lw=0.5); a.axvline(0, color='k', lw=0.5)
a.set_ylim(-3, 3); a.set_xlabel(r'$\gamma\,\theta_x$  (angle in units of $1/\gamma$)')
a.set_ylabel('odd shape factor')
a.set_title('(B) angular shape: odd, linear near axis, bounded')
a.legend(fontsize=9, loc='upper left')
a.text(0.5, -2.6, 'linear runs away;\n$\\tanh$ saturates at $\\pm1$', fontsize=8, color='gray')

# -- Panel C: full lobe multiplier for several tilts (k=1) ---------------------
a = ax[1, 0]
for psdeg, c in [(0, 'C0'), (15, 'C1'), (30, 'C2'), (45, 'C3')]:
    lobe = np.maximum(0.0, 1.0 + 1.0 * math.tan(math.radians(psdeg)) * np.tanh(u))
    a.plot(u, lobe, lw=2, color=c, label=f'$\\psi={psdeg}^\\circ$')
a.axhline(1, color='k', lw=0.5, ls=':')
a.set_xlabel(r'$\gamma\,\theta_x$'); a.set_ylabel(r'lobe$(\theta_x)$ (brightness $\times$)')
a.set_title(r'(C) lobe multiplier $\max(0,\,1+k\tan\psi\tanh\gamma\theta_x)$, $k=1$')
a.legend(fontsize=9); a.text(-3.8, 1.75, 'clamped at 0\non the dim side\n(45°)',
                             fontsize=8, color='gray')

# -- Panel D: effect on the GF intensity cut (ring -> crescent) ----------------
a = ax[1, 1]
th = u / g                                              # theta_x [rad]
I_sym = gf_shape(np.abs(th), beta) ** 2
I_sym_n = I_sym / I_sym.max()
a.plot(u, I_sym_n, lw=2, color='k', label=r'symmetric ($\psi=0$)')
for psdeg, c in [(30, 'C2'), (45, 'C3')]:
    lobe = np.maximum(0.0, 1.0 + 1.0 * math.tan(math.radians(psdeg)) * np.tanh(u))
    I_tilt = (gf_shape(np.abs(th), beta) * lobe) ** 2
    a.plot(u, I_tilt / I_sym.max(), lw=2, color=c, label=f'$\\psi={psdeg}^\\circ$ (lobe-weighted)')
a.set_xlabel(r'$\gamma\,\theta_x$'); a.set_ylabel('intensity along $x$ (norm.)')
a.set_title('(D) effect on the GF lobes: ring cut -> crescent')
a.legend(fontsize=9)
a.text(-3.8, 1.5, 'one lobe up,\nother suppressed', fontsize=8, color='gray')

fig.suptitle('Level-1 tilted-screen lobe heuristic: why  '
             r'$\mathrm{lobe}=\max(0,\,1+k\,\tan\psi\,\tanh(\gamma\theta_x))$',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = 'source_tilt_heuristic.png'
fig.savefig(out, dpi=140); plt.close(fig)
print(f"wrote {out}  (gamma={g:.1f}, 1/gamma={1e3/g:.2f} mrad)")
