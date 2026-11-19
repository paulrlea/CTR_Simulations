#!/usr/bin/env python
"""
Beampath schematic for the CTR-THz beamline, drawn straight from the config
(no SRW run).  Side view: z [m] horizontal, transverse extent [cm] vertical.

NOTE: axes are NOT the same scale (z in m, transverse in cm) and the mm-scale
window thickness is exaggerated for visibility -- this is a labeled schematic,
not a to-scale optical drawing or a computed beam envelope.

    python thz_layout.py            # -> beamline_layout.png from default config
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

from thz_config import default_config
from thz_beamline import detector_z_m


def _elements(cfg):
    """Return ordered list of dicts: z[m], kind, label, half-aperture[m]."""
    els = []
    z = 0.0
    els.append(dict(z=z, kind='foil', label='CTR foil\n(source)',
                    half=cfg.collimator_radius_m))
    z += cfg.L_foil_to_collimator_m
    els.append(dict(z=z, kind='aperture', half=cfg.collimator_radius_m,
                    label=f'collimator\n⌀{2*cfg.collimator_radius_m*100:.1f} cm'))
    if cfg.window_enabled:
        z += cfg.L_collimator_to_window_m
        els.append(dict(z=z, kind='window', half=cfg.window_clear_ap_radius_m,
                        label=f'sapphire window\n{cfg.window_thickness_m*1e3:.2f} mm, '
                              f'⌀{2*cfg.window_clear_ap_radius_m*100:.1f} cm'))
    if cfg.mirror_enabled:
        z += cfg.L_window_to_mirror_m
        els.append(dict(z=z, kind='lens', half=1.3*cfg.collimator_radius_m,
                        label=f'OAP1 collimate\nf={cfg.mirror_focal_length_m:.2f} m'))
        if cfg.mirror2_enabled:
            z += cfg.L_mirror1_to_mirror2_m
            els.append(dict(z=z, kind='lens', half=1.3*cfg.collimator_radius_m,
                            label=f'OAP2 focus\nf={cfg.mirror2_focal_length_m:.2f} m'))
    z = detector_z_m(cfg)
    els.append(dict(z=z, kind='detector', half=cfg.det_active_radius_m,
                    label='EO sensor\n(TFLN)'))
    return els


def draw_beamline(cfg, path_png='beamline_layout.png'):
    els = _elements(cfg)
    ymax_cm = max(e['half'] for e in els) * 100 * 1.7        # vertical span [cm]
    z_end = els[-1]['z']
    jaw_top = ymax_cm
    win_w = 0.02 * z_end                                     # exaggerated slab width [m]

    fig, ax = plt.subplots(figsize=(11, 4.4))

    # faint illustrative beam corridor (cartoon, not computed): expand from the
    # foil to the collimator/OAP1, ~collimated between the OAPs, focus to the sensor
    rc = cfg.collimator_radius_m * 100
    rdet = cfg.det_active_radius_m * 100
    lens_zs = [e['z'] for e in els if e['kind'] == 'lens']
    xs = [0.0, els[1]['z']] + lens_zs + [z_end]
    ys = [0.4*rc, rc] + [rc] * len(lens_zs) + [rdet]
    ax.fill(xs + xs[::-1], ys + [-y for y in ys[::-1]],
            color='gold', alpha=0.12, zorder=0)

    for e in els:
        z, half_cm = e['z'], e['half'] * 100
        if e['kind'] == 'foil':
            ax.add_patch(Rectangle((z - 0.004*z_end, -half_cm), 0.008*z_end, 2*half_cm,
                                   facecolor='dimgray', edgecolor='k', hatch='///', zorder=3))
        elif e['kind'] == 'aperture':
            # two jaws leaving a gap of 2*half
            ax.add_patch(Rectangle((z - 0.004*z_end, half_cm), 0.008*z_end, jaw_top-half_cm,
                                   facecolor='slategray', edgecolor='k', zorder=3))
            ax.add_patch(Rectangle((z - 0.004*z_end, -jaw_top), 0.008*z_end, jaw_top-half_cm,
                                   facecolor='slategray', edgecolor='k', zorder=3))
        elif e['kind'] == 'window':
            ax.add_patch(Rectangle((z - win_w/2, -half_cm), win_w, 2*half_cm,
                                   facecolor='deepskyblue', edgecolor='navy', alpha=0.55, zorder=3))
            # mount beyond the clear aperture
            ax.add_patch(Rectangle((z - win_w/2, half_cm), win_w, jaw_top-half_cm,
                                   facecolor='lightsteelblue', edgecolor='navy', alpha=0.7, zorder=3))
            ax.add_patch(Rectangle((z - win_w/2, -jaw_top), win_w, jaw_top-half_cm,
                                   facecolor='lightsteelblue', edgecolor='navy', alpha=0.7, zorder=3))
        elif e['kind'] == 'lens':
            ax.add_patch(Ellipse((z, 0), width=0.018*z_end, height=2*half_cm,
                                 facecolor='mediumpurple', edgecolor='indigo', alpha=0.8, zorder=3))
        elif e['kind'] == 'detector':
            ax.add_patch(Rectangle((z - 0.006*z_end, -1.2), 0.012*z_end, 2.4,
                                   facecolor='firebrick', edgecolor='k', zorder=3))
        ax.text(z, jaw_top*1.04, e['label'], ha='center', va='bottom', fontsize=8.5)

    # distance arrows between consecutive elements
    yA = -jaw_top * 1.18
    for a, b in zip(els[:-1], els[1:]):
        ax.annotate('', xy=(b['z'], yA), xytext=(a['z'], yA),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1))
        ax.text(0.5*(a['z']+b['z']), yA - 0.06*jaw_top,
                f"{(b['z']-a['z'])*100:.0f} cm", ha='center', va='top', fontsize=8)

    ax.axhline(0, color='k', lw=0.6, ls=':', zorder=1)       # optical axis
    ax.set_xlim(-0.04*z_end, 1.06*z_end)
    ax.set_ylim(-jaw_top*1.5, jaw_top*1.45)
    ax.set_xlabel('z  [m]   (beam direction →)')
    ax.set_ylabel('transverse  [cm]')
    ax.set_title('CTR-THz beampath (schematic, not to scale)')
    ax.text(0.99, 0.02,
            'axes not same scale; window thickness exaggerated; beam corridor illustrative',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color='gray')
    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    plt.close(fig)
    return path_png


if __name__ == '__main__':
    out = draw_beamline(default_config())
    print(f"wrote {out}")
