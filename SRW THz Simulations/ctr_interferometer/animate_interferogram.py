#!/usr/bin/env python
"""
Animate the Golay-cell interferogram as the delay stage scans.

The Golay is a single-element power detector: at each delay-stage position it returns ONE
number (time-integrated power). This animation sweeps the stage and reveals the reading it
produces at each step -- the CTR field-autocorrelation interferogram building up as the scan
crosses zero path difference (ZPD).

Top panel : the delay-stage retro mirror sliding along the scan (ZPD marked).
Bottom    : the Golay signal (normalised to its baseline) traced out vs relative delay,
            with a live readout of stage position, delay, and the current reading.

Reads <out>/interferogram.txt (rel_delay[ps], stage_travel[mm], P_golay[arb]) written by
run_interf.py -- no re-propagation. Writes <out>/interferogram.gif.

    source ../.venv/bin/activate
    python animate_interferogram.py [out_dir]     # default out_dir = results_interf
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from interf_config import default_interf_config
from thz_config import C_LIGHT

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results_interf'
N_FRAMES = 160
FPS = 20

cfg = default_interf_config()
etalon_ps = 2.0 * cfg.window_n * cfg.window_thickness_m / C_LIGHT * 1e12   # window recurrence

# ---- load the interferogram (rel delay [ps], stage [mm], Golay power [arb]) ----
tau, stage, P = np.loadtxt(os.path.join(OUT, 'interferogram.txt'), unpack=True)
wing = np.abs(tau) > 0.6 * np.abs(tau).max()          # flat baseline in the wings
base = np.median(P[wing])
Pn = P / base                                          # -> baseline = 1, burst peaks ~2
idx = np.linspace(0, len(tau) - 1, N_FRAMES).astype(int)

fig, (axtop, axg) = plt.subplots(2, 1, figsize=(8.2, 5.2),
                                 gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.38})

# ---- top panel: delay-stage rail with the sliding retro mirror ----
axtop.set_xlim(tau.min(), tau.max()); axtop.set_ylim(-1, 1)
axtop.axhline(0, color='0.6', lw=2, zorder=1)
axtop.plot([0.0], [0.0], marker='v', color='#b8470f', ms=11, zorder=3)      # ZPD at tau=0
axtop.text(0.0, 0.55, 'ZPD', color='#b8470f', ha='center', fontsize=8, fontweight='bold')
axtop.annotate('', xy=(tau.max() * 0.6, 0.62), xytext=(tau.min() * 0.6, 0.62),
               arrowprops=dict(arrowstyle='->', color='#1f6fd1', lw=1.2))
axtop.text(tau.min() * 0.9, -0.7, 'delay-stage retro mirror', fontsize=8, color='#1f6fd1')
mirror, = axtop.plot([tau[0]], [0.0], marker='s', color='#1f6fd1', ms=15, zorder=4)
cur_t = axtop.axvline(tau[0], color='#1f6fd1', ls='--', lw=1.0, zorder=2)
axtop.set_yticks([]); axtop.set_xticks([])
for sp in axtop.spines.values():
    sp.set_visible(False)
axtop.set_title('Golay interferometer — delay-stage scan (what the detector reads)',
                fontsize=12, fontweight='bold')

# ---- bottom panel: the Golay signal building up ----
axg.plot(tau, Pn, color='0.85', lw=1.0, zorder=1)                     # full curve, faint
line, = axg.plot([], [], color='#d12f2f', lw=1.7, zorder=3)          # revealed so far
dot, = axg.plot([], [], marker='o', color='#d12f2f', ms=7, zorder=4)  # current reading
cur_g = axg.axvline(tau[0], color='#1f6fd1', ls='--', lw=1.0, zorder=2)
axg.axvline(0.0, color='#b8470f', ls=':', lw=0.9)
for s in (-etalon_ps, etalon_ps):
    axg.axvline(s, color='#2a8', ls=':', lw=0.9)
axg.text(etalon_ps, Pn.max() * 1.03, 'window etalon\nrecurrence',
         color='#2a8', fontsize=7, ha='center', va='bottom')
axg.set_xlim(tau.min(), tau.max()); axg.set_ylim(0, Pn.max() * 1.12)
axg.set_xlabel('relative delay  tau [ps]     (ZPD at tau = 0)')
axg.set_ylabel('Golay signal / baseline')
axg.grid(alpha=0.3)
readout = axg.text(0.015, 0.96, '', transform=axg.transAxes, va='top', ha='left',
                   fontsize=9, family='monospace',
                   bbox=dict(fc='white', ec='0.7', alpha=0.85))


def update(k):
    i = idx[k]
    line.set_data(tau[:i + 1], Pn[:i + 1])
    dot.set_data([tau[i]], [Pn[i]])
    cur_g.set_xdata([tau[i], tau[i]]); cur_t.set_xdata([tau[i], tau[i]])
    mirror.set_data([tau[i]], [0.0])
    zpd = '  <-- ZPD' if abs(tau[i]) < (tau[1] - tau[0]) else ''
    readout.set_text(f"stage = {stage[i]:+6.1f} mm\n"
                     f"delay = {tau[i]:+6.1f} ps{zpd}\n"
                     f"Golay = {Pn[i]:4.2f} x baseline")
    return line, dot, cur_g, cur_t, mirror, readout


anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 // FPS, blit=False)
out_gif = os.path.join(OUT, 'interferogram.gif')
anim.save(out_gif, writer=PillowWriter(fps=FPS))
plt.close(fig)
print(f"wrote {out_gif}  ({len(idx)} frames @ {FPS} fps)")
