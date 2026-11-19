#!/usr/bin/env python
"""
Compare the two THz source models AT THE FOIL (angular emission pattern):
  - Gaussian surrogate  (--src gaussian): on-axis maximum, linear (horizontal) pol.
  - Ginzburg-Frank CTR  (--src ctr):      on-axis null + ring at ~1/gamma, radial pol.

Both angular shapes are frequency-independent in these models (the detected spectrum
comes from the bunch form factor F(f) and the beamline, not the emission shape), so this
is a single figure. Writes source_angular_distributions.png.

    python plot_sources.py
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thz_config import default_config
from thz_source import gf_shape


def main():
    cfg = default_config()
    g = cfg.gamma
    beta = math.sqrt(1.0 - 1.0 / g ** 2)
    inv_g_mrad = 1e3 / g                       # 1/gamma in mrad

    # angular grid (mrad), out to ~4/gamma
    th_max = 4.0 / g
    th = np.linspace(-th_max, th_max, 401)
    TX, TY = np.meshgrid(th, th)
    TH = np.sqrt(TX ** 2 + TY ** 2)
    PHI = np.arctan2(TY, TX)

    # --- Ginzburg-Frank: amplitude from the exact shape; intensity = |.|^2 ---
    A_gf = gf_shape(TH, beta)
    I_gf = A_gf ** 2
    I_gf /= I_gf.max()

    # --- Gaussian surrogate: intensity rms angular width sigma_th = 1/(4*div_match*gamma)
    # (from sigma_r = div_match*lambda*gamma/pi and the Gaussian space-angle product
    #  sigma_x*sigma_theta = lambda/4pi; lambda cancels -> frequency-independent). ---
    sig_th = 1.0 / (4.0 * cfg.div_match * g)
    I_g = np.exp(-TH ** 2 / (2.0 * sig_th ** 2))
    I_g /= I_g.max()
    A_g = np.sqrt(I_g)

    th_mrad = th * 1e3
    ext = [th_mrad[0], th_mrad[-1], th_mrad[0], th_mrad[-1]]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    # quiver grids (coarse) showing polarization direction
    qs = np.linspace(-th_max, th_max, 13)
    QX, QY = np.meshgrid(qs, qs)
    QT = np.sqrt(QX ** 2 + QY ** 2); QP = np.arctan2(QY, QX)

    # (a) Gaussian
    im0 = ax[0].imshow(I_g, origin='lower', extent=ext, cmap='inferno', vmin=0, vmax=1)
    qg = np.exp(-QT ** 2 / (4.0 * sig_th ** 2))          # field amplitude
    ax[0].quiver(QX * 1e3, QY * 1e3, qg, np.zeros_like(qg), color='cyan',
                 scale=12, width=0.005, pivot='mid')
    ax[0].set_title('Gaussian surrogate\n(on-axis peak, linear pol.)')

    # (b) Ginzburg-Frank
    im1 = ax[1].imshow(I_gf, origin='lower', extent=ext, cmap='inferno', vmin=0, vmax=1)
    qa = gf_shape(QT, beta)
    qa = np.divide(qa, np.nanmax(qa))
    ax[1].quiver(QX * 1e3, QY * 1e3, qa * np.cos(QP), qa * np.sin(QP), color='cyan',
                 scale=12, width=0.005, pivot='mid')
    ax[1].set_title('Ginzburg-Frank CTR\n(on-axis null, radial pol.)')

    for a in ax[:2]:
        a.set_xlabel(r'$\theta_x$ [mrad]'); a.set_ylabel(r'$\theta_y$ [mrad]')
        for s in (-1, 1):
            a.add_patch(plt.Circle((0, 0), inv_g_mrad, fill=False, ls='--',
                                   ec='white', lw=0.8, alpha=0.6))
    fig.colorbar(im1, ax=ax[1], fraction=0.046, label='intensity (norm.)')

    # (c) radial line-outs
    r = np.linspace(0, th_max, 400)
    lo_gf = gf_shape(r, beta) ** 2; lo_gf /= lo_gf.max()
    lo_g = np.exp(-r ** 2 / (2.0 * sig_th ** 2))
    ax[2].plot(r * 1e3, lo_g, label='Gaussian surrogate', lw=2)
    ax[2].plot(r * 1e3, lo_gf, label='Ginzburg-Frank', lw=2)
    ax[2].axvline(inv_g_mrad, color='k', ls=':', lw=1)
    ax[2].text(inv_g_mrad * 1.03, 0.5, r'$1/\gamma$', fontsize=10)
    ax[2].set_xlabel(r'$\theta$ [mrad]'); ax[2].set_ylabel('intensity (norm.)')
    ax[2].set_title(f'Angular emission (γ={g:.0f}, 1/γ={inv_g_mrad:.1f} mrad)')
    ax[2].legend(); ax[2].set_xlim(0, th_max * 1e3)

    fig.suptitle('THz source angular distribution at the foil  '
                 '(frequency-independent shape)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'source_angular_distributions.png'
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}  (1/gamma = {inv_g_mrad:.2f} mrad, Gaussian sigma_theta = "
          f"{sig_th*1e3:.2f} mrad)")


if __name__ == '__main__':
    main()
