#!/usr/bin/env python
"""
Schematic beampath for the CTR/OTR interferometer.

Elements in order of incidence (from the as-built list):
  1. Mo transition target (1" solid Mo mirror)  -- backward TR source
  2. drift
  3. Fused-silica / quartz window (MDC Vacuum)   -- vacuum boundary
  4. drift
  5. 4" first-surface mirror, Si overcoat        -- "L3 monitor only" (insertable)
  6. drift
  7. Pellicle beamsplitter 3" 30/30 (ETP-NPIN 30/30, 2 um)
  8. VARIABLE drift  (delay arm)
  9. Flat mirror 75 mm, protected Al (01 MFG 028) -- on translation/delay stage,
     NORMAL INCIDENCE: retro-reflects the radiation BACK toward the beamsplitter
 10. Off-axis parabolic mirror, gold (A8037-305)  -- sits BEHIND the beamsplitter
     (opposite side from the flat mirror); focuses the recombined beam to detector

Interferometer topology (Michelson-type):
  - The pellicle BS splits the incoming beam into two arms.
  - The 75 mm flat sits at the end of the delay arm at normal incidence and sends
    the light straight back to the BS (round trip).
  - The transmitted (reference) arm is a stub -- its fixed mirror was not specified.
  - The recombined beam exits the BS on the side opposite the flat mirror and is
    picked up by the OAP, which focuses it onto the detector.

Geometry is SCHEMATIC: 45 deg folds and representative spacings (no distances
were specified).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

BEAM = '#d12f2f'      # THz / transition-radiation beam
EBEAM = '#1f6fd1'     # electron beam
GOLD = '#d9a521'
ALU = '#9aa3ad'
MO = '#6f6f78'
FS = '#9fc6e8'

# ---- node coordinates (scene units; not to scale) ----
T  = (0.0, 0.0)     # Mo target
W  = (4.0, 0.0)     # window
M1 = (8.0, 0.0)     # 4" FS mirror (L3 monitor)
P  = (8.0, 4.0)     # pellicle BS
M2 = (12.5, 4.0)    # flat 75 mm mirror (delay stage, normal incidence / retro)
M3 = (8.0, 7.2)     # flat mirror for reference arm
RS = (8.0, 6.4)     # reference-arm stub end (transmitted through BS)
O  = (4.0, 4.0)     # OAP -- behind the BS, opposite side from M2
D  = (4.0, 7.2)     # detector (OAP focus)
EB = (0.0, -2.6)    # e-beam start


def beam(ax, a, b, color=BEAM, lw=2.6, ls='-', alpha=1.0, arrow=True):
    style = '-|>' if arrow else '-'
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=color, ls=ls, alpha=alpha,
                                 shrinkA=0, shrinkB=0, zorder=3))


def mirror(ax, c, length=1.5, color='0.4', lw=7, curved=False, u=(1, 1), cup=(1, -1)):
    """Mirror bar of given length centred at c, oriented along unit direction u.

    u = (1, 1)  -> 45 deg fold, slope +1
    u = (1,-1)  -> 45 deg fold, slope -1
    u = (0, 1)  -> vertical bar (normal-incidence to a horizontal beam)
    u = (1, 0)  -> horizontal bar (normal-incidence to a vertical beam)

    curved=True draws a concave focusing surface; `cup` is the direction the
    concave side opens toward (the middle of the bar recedes opposite to it).
    """
    u = np.asarray(u, float); u /= np.linalg.norm(u)
    c = np.asarray(c, float)
    p0 = c - (length / 2) * u
    p1 = c + (length / 2) * u
    if curved:
        cup = np.asarray(cup, float); cup /= np.linalg.norm(cup)
        t = np.linspace(-1, 1, 30)
        base = p0 + np.outer(np.linspace(0, 1, 30), (p1 - p0))
        pts = base - (0.18 * (1 - t**2))[:, None] * cup
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw,
                solid_capstyle='round', zorder=5)
    else:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
                solid_capstyle='round', zorder=5)


fig, ax = plt.subplots(figsize=(12.6, 9.4))

# vacuum region (source side up to the window)
ax.add_patch(FancyBboxPatch((-1.8, -3.0), 6.0, 4.4,
             boxstyle='round,pad=0.02,rounding_size=0.2',
             fc='#eaf3fb', ec='#b9d6ef', lw=1.0, zorder=0))
ax.text(-1.6, 1.05, 'vacuum (beamline)', color='#3a78b5', fontsize=9, style='italic')

# ---- beam path (order of incidence) ----
beam(ax, EB, (T[0], T[1]-0.18), color=EBEAM, lw=2.4, ls=(0, (5, 3)))   # e-beam in
beam(ax, T, W)                       # target -> window
beam(ax, W, M1)                      # window -> L3 mirror
beam(ax, M1, P)                      # L3 mirror -> pellicle BS  (input port)

# delay arm: BS -> flat mirror and back (round trip, drawn as two offset passes)
beam(ax, (P[0] + 0.14, P[1]+0.14), (M2[0], M2[1]+0.14), color=BEAM)          # forward  ->
beam(ax, (M2[0], M2[1]-0.14), (P[0], P[1]-0.14), color=BEAM)          # return   <-

# transmitted (reference) arm -- fixed mirror 
beam(ax, (P[0] - 0.14, P[1] + 0.14), (M3[0] - 0.14, M3[1]-0.14), color=BEAM)           # forward  ->
beam(ax, (M3[0] + 0.14, M3[1]-0.14), (P[0] + 0.14, P[1] + 0.14), color=BEAM)          # return   <-

# recombined output: exits BS on the side opposite the flat mirror -> OAP
beam(ax, P, O, color=BEAM)                                           # BS -> OAP
beam(ax, O, D)                                                       # OAP -> detector
# focusing cone into detector (converging upward from OAP to focus)
ax.plot([O[0]-0.5, D[0]], [O[1]+0.5, D[1]], color=BEAM, lw=1.0, alpha=0.5)
ax.plot([O[0]+0.5, D[0]], [O[1]+0.5, D[1]], color=BEAM, lw=1.0, alpha=0.5)

# ---- elements ----
mirror(ax, T, length=1.6, color=MO, lw=9)                 # Mo target (45 deg fold)
# window (slab perpendicular to +x beam)
ax.add_patch(Rectangle((W[0]-0.10, W[1]-0.8), 0.20, 1.6, fc=FS, ec='#5b8fc0',
                       lw=1.2, zorder=5))
mirror(ax, M1, color=FS, lw=8)                            # 4" FS fold mirror
# pellicle BS (thin membrane) -- dashed thin bar slope +1
d = 1.7/2/np.sqrt(2)
ax.plot([P[0]-d, P[0]+d], [P[1]-d, P[1]+d], color='#2b2b2b', lw=2.0,
        ls=(0, (4, 2)), zorder=5)
mirror(ax, M2, length=1.7, color=ALU, lw=8, u=(0, 1))     # flat retro mirror (vertical)
mirror(ax, M3, length=1.7, color=ALU, lw=8, u=(1, 0))
mirror(ax, O, color=GOLD, lw=9, curved=True, u=(1, -1), cup=(1, 1))   # OAP (slope -1)
# detector (facing the incoming, upward-focused beam)
ax.add_patch(Rectangle((D[0]-0.7, D[1]-0.12), 1.4, 0.5, fc='#2c2c2c',
                       ec='k', zorder=5))

# delay-stage double arrow: flat mirror translates ALONG the beam axis (+/- x)
ax.add_patch(FancyArrowPatch((M2[0]-0.95, 2.6), (M2[0]+0.35, 2.6),
                             arrowstyle='<|-|>', mutation_scale=12,
                             lw=1.6, color='#444', zorder=6))
ax.plot([M2[0], M2[0]], [M2[1]-0.85, 2.6], color='#888', lw=0.8, ls=':', zorder=4)

# ---- labels ----
def lab(x, y, txt, ha='left', va='center', fs=9.5, c='k', bold=False):
    ax.text(x, y, txt, ha=ha, va=va, fontsize=fs, color=c,
            fontweight=('bold' if bold else 'normal'), zorder=8)

lab(EB[0]+0.25, EB[1]+0.5, 'e$^-$ beam', c=EBEAM, bold=True)
lab(T[0]-1.0, T[1]-1.2, 'Mo transition target\n1" dia, MM-1012-M-UC',
    ha='center', va='top', fs=9)
lab(W[0], W[1]-1.0, 'fused-silica window\n(MDC Vacuum)', ha='center', va='top', fs=9)
lab(M1[0]+0.6, M1[1]-0.9, '4" 1st-surface mirror\nSi overcoat\n"Only at L3 monitor position"',
    ha='left', va='top', fs=9)
lab(P[0]-0.35, P[1]+0.85, 'pellicle BS 3" 30/30\nETP-NPIN 30/30, 2 µm',
    ha='right', va='bottom', fs=9)
lab(RS[0]+0.35, RS[1]-0.15, 'transmitted arm\n 1.854"',
    ha='left', va='center', fs=8.2, c='#888')
lab(M2[0]+0.5, M2[1], 'flat mirror 75 mm\nprotected Al, 01 MFG 028\n(delay stage — retro)',
    ha='left', va='center', fs=9)
lab(M2[0]-0.3, 2.15, 'delay scan', ha='center', va='center', fs=8.5, c='#444')
lab(O[0]-0.55, O[1]-0.35, 'OAP (gold)\nA8037-305', ha='right', va='top', fs=9)
lab(D[0]+0.75, D[1], 'detector', ha='left', va='center', fs=9)

# drift / arm labels
def drift(a, b, txt, dx=0, dy=0, c='0.35'):
    mx, my = (a[0]+b[0])/2+dx, (a[1]+b[1])/2+dy
    ax.text(mx, my, txt, ha='center', va='center', fontsize=8.5, color=c,
            style='italic', zorder=8)
drift(W, M1, '7.25"', dy=0.32)
drift(T, W, '7.25"', dy=0.32)
drift(M1, P, '13.519"', dx=-0.5)
drift(P, M2, 'variable delay arm range:\n 1.342 to 2.342"', dy=0.5, c='#b8470f')
drift(P, O, 'recombined\noutput 3.244"', dy=-0.55)
drift(O, D, '3.165"', dx=0.75)

ax.set_title('CTR / OTR Interferometer — beampath (schematic, not to scale)',
             fontsize=13, fontweight='bold')


ax.set_xlim(-2.5, 16.0)
ax.set_ylim(-3.6, 8.4)
ax.set_aspect('equal')
ax.axis('off')
fig.tight_layout()
out = '/home/paulrleaa/Documents/1_Research/Suli2026/ctr_interferometer/beampath.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print('wrote', out)
