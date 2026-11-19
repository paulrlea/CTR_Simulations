#!/usr/bin/env python
"""
Compare the CTR angular-emission distribution for two foil orientations, using the
SAME shared prefactor  e*beta*exp(-i w r/c) / (4 pi^2 eps0 c r)  (dropped; it cancels):

  NORMAL incidence (Ginzburg-Frank), e_theta only, azimuthally symmetric:
      E_perp(theta) = 2 sin(theta) / (1 - beta^2 cos^2 theta)  * e_theta

  45-degree incidence (backward TR from a tilted foil), mixed e_theta / e_phi,
  azimuthally ASYMMETRIC in the plane of incidence:
      E_phi   =  sin(phi) / (1 - beta cos(phi) sin(theta))
      E_theta = [ (beta - cos theta) cos(phi) - sin(theta) ]
                / [ (1 - beta cos theta)(1 - beta cos(phi) sin(theta)) ]

We map the observation direction to Cartesian angle coordinates
  theta_x = theta cos(phi),  theta_y = theta sin(phi)   [mrad]
and plot the emitted intensity |E|^2 = |E_theta|^2 + |E_phi|^2.

Both maps are normalized to the COMMON peak (max over both) so the difference map is
physically meaningful: it shows how tilting the foil to 45 deg redistributes the CTR.

Writes:
  ctr_incidence_sidebyside.png   (the two distributions, shared colorbar)
  ctr_incidence_difference.png   (I_45deg - I_perp, diverging colorbar)

    python plot_ctr_incidence_compare.py
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thz_config import default_config


def e_perp_normal(theta, beta):
    """Normal-incidence Ginzburg-Frank: (E_theta, E_phi). e_phi component is zero."""
    ct = np.cos(theta)
    st = np.sin(theta)
    e_th = 2.0 * st / (1.0 - beta ** 2 * ct ** 2)
    e_ph = np.zeros_like(e_th)
    return e_th, e_ph


def e_45deg(theta, phi, beta):
    """45-degree incidence backward TR: (E_theta, E_phi)."""
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    denom_phi = 1.0 - beta * cp * st                      # (1 - beta cos phi sin theta)
    e_ph = sp / denom_phi
    e_th = ((beta - ct) * cp - st) / ((1.0 - beta * ct) * denom_phi)
    return e_th, e_ph


def polar_to_cart(e_th, e_ph, phi):
    """Project (e_theta, e_phi) onto the (theta_x, theta_y) plane for quiver display.
    In the small-angle projection e_theta -> radial (cos phi, sin phi),
    e_phi -> azimuthal (-sin phi, cos phi)."""
    ux = e_th * np.cos(phi) - e_ph * np.sin(phi)
    uy = e_th * np.sin(phi) + e_ph * np.cos(phi)
    return ux, uy


def main():
    cfg = default_config()
    g = cfg.gamma
    beta = math.sqrt(1.0 - 1.0 / g ** 2)
    inv_g_mrad = 1e3 / g                          # 1/gamma in mrad

    # angular grid (rad); 45-deg lobe is shifted/distorted, so go out to ~6/gamma
    th_max = 6.0 / g
    n = 501
    ax_rad = np.linspace(-th_max, th_max, n)
    TX, TY = np.meshgrid(ax_rad, ax_rad)
    TH = np.sqrt(TX ** 2 + TY ** 2)
    PHI = np.arctan2(TY, TX)

    # intensities |E|^2 = |E_theta|^2 + |E_phi|^2
    eth_n, eph_n = e_perp_normal(TH, beta)
    I_n = eth_n ** 2 + eph_n ** 2

    eth_45, eph_45 = e_45deg(TH, PHI, beta)
    I_45 = eth_45 ** 2 + eph_45 ** 2

    # common normalization -> difference map is meaningful
    cmax = max(I_n.max(), I_45.max())
    I_n /= cmax
    I_45 /= cmax

    th_mrad = ax_rad * 1e3
    ext = [th_mrad[0], th_mrad[-1], th_mrad[0], th_mrad[-1]]

    # coarse quiver grid for polarization direction
    qs = np.linspace(-th_max, th_max, 15)
    QX, QY = np.meshgrid(qs, qs)
    QT = np.sqrt(QX ** 2 + QY ** 2)
    QP = np.arctan2(QY, QX)

    def draw_ring(a):
        a.add_patch(plt.Circle((0, 0), inv_g_mrad, fill=False, ls='--',
                               ec='white', lw=0.9, alpha=0.7))

    # ----------------------------------------------------------------- figure 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    im0 = ax[0].imshow(I_n, origin='lower', extent=ext, cmap='inferno',
                       vmin=0, vmax=1)
    q_eth, q_eph = e_perp_normal(QT, beta)
    ux, uy = polar_to_cart(q_eth, q_eph, QP)
    mag = np.hypot(ux, uy); mag[mag == 0] = 1.0
    ax[0].quiver(QX * 1e3, QY * 1e3, ux / mag, uy / mag, color='cyan',
                 scale=22, width=0.005, pivot='mid', alpha=0.85)
    ax[0].set_title('Normal incidence (Ginzburg-Frank)\n'
                    r'$e_\theta$ only, symmetric ring, radial pol.')

    im1 = ax[1].imshow(I_45, origin='lower', extent=ext, cmap='inferno',
                       vmin=0, vmax=1)
    q_eth, q_eph = e_45deg(QT, QP, beta)
    ux, uy = polar_to_cart(q_eth, q_eph, QP)
    mag = np.hypot(ux, uy); mag[mag == 0] = 1.0
    ax[1].quiver(QX * 1e3, QY * 1e3, ux / mag, uy / mag, color='cyan',
                 scale=22, width=0.005, pivot='mid', alpha=0.85)
    ax[1].set_title(r'$45^\circ$ incidence (backward TR)'
                    '\n'
                    r'$e_\theta+e_\phi$, asymmetric, mixed pol.')

    for a in ax:
        a.set_xlabel(r'$\theta_x$ [mrad]')
        a.set_ylabel(r'$\theta_y$ [mrad]')
        draw_ring(a)
        a.set_aspect('equal')
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.02,
                 label='intensity |E|$^2$ (norm. to common peak)')
    fig.suptitle(f'CTR angular distribution: normal vs. 45$^\\circ$ incidence '
                 f'($\\gamma$={g:.0f}, 1/$\\gamma$={inv_g_mrad:.1f} mrad)',
                 fontsize=12)
    out1 = 'ctr_incidence_sidebyside.png'
    fig.savefig(out1, dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------- figure 2
    diff = I_45 - I_n
    dlim = np.max(np.abs(diff))
    fig2, ax2 = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    im = ax2.imshow(diff, origin='lower', extent=ext, cmap='RdBu_r',
                    vmin=-dlim, vmax=dlim)
    draw_ring(ax2)
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$\theta_x$ [mrad]')
    ax2.set_ylabel(r'$\theta_y$ [mrad]')
    ax2.set_title(r'Difference  $I_{45^\circ} - I_{\perp}$'
                  '\n(red: 45$^\\circ$ emits more; blue: less)')
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.02,
                  label=r'$\Delta$ intensity (norm. to common peak)')
    out2 = 'ctr_incidence_difference.png'
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)

    print(f"wrote {out1} and {out2}")
    print(f"  gamma={g:.1f}  1/gamma={inv_g_mrad:.2f} mrad  th_max={th_max*1e3:.1f} mrad")
    print(f"  peak ratio I_45/I_perp (raw) = {I_45.max()/I_n.max():.3f}")


if __name__ == '__main__':
    main()
