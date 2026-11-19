"""
EO Detection — Step-by-Step Explanation
========================================

Walks through the BPD → E_THz conversion pipeline used in
THz_Sensor_Characterization.ipynb, with three explicit steps:

  Step 1.  Processed BPD voltage  →  differential optical power (intensity)
  Step 2.  Optical power  →  EO phase shift Δφ, with static bias φ₀
  Step 3.  Δφ  →  THz E-field via Wilke et al. (Sci. Rep. 14, 4822, 2024)

The script reproduces the analysis chain on the July 31 1 nC fine-scan
(typical operating point, modest charge → small-signal regime safe) and
saves a set of pedagogical PNGs under EO_Detection_Explained_figs/.

References
----------
- Wilke, Monahan, Toroghi, Rabiei, Hine, "Thin-film lithium niobate
  electro-optic terahertz wave detector", Sci. Rep. 14, 4822 (2024).
- ThorLabs PDB440x/PDB450x Operation Manual (M0009-510-1043, v2.4).
- PersoNota_BeamNetUSAnalysis.pdf (Piot, July 2025) §3.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.constants import c, epsilon_0

# ─── Paths ──────────────────────────────────────────────────────────────────
WORKTREE = '/home/paulrleaa/Documents/1_Research/Suli2026/SULI_THZ_SUM26/.claude/worktrees/unified-thz-analysis'
FIG_DIR  = os.path.join(WORKTREE, 'Analysis', 'EO_Detection_Explained_figs')
DATA     = os.path.join(WORKTREE, 'Data_raw', 'July31_', 'finescaneQ1nC_wh5_6.npz')
os.makedirs(FIG_DIR, exist_ok=True)


# ─── Physical constants  ────────────────────────────────────────────────────
# (Match THz_Sensor_Characterization.ipynb so numbers are directly comparable.)
RF_HZ        = 81.25e6
T_LASER_NS   = 1e9 / RF_HZ          # 12.31 ns laser period
LAMBDA_PROBE = 1550e-9              # m
N_E_LN       = 2.15                 # LiNbO3 extraordinary index at 1550 nm (Wilke)
R33          = 30.9e-12             # m/V    (Wilke)
L_MZI        = 600e-6               # m      (Wilke MZI arm length)
N_RF_FUSED   = 1.95                 # SiO2 THz-frequency index (substrate)
N_OPT_GUIDE  = 2.40                 # TFLN waveguide group index at 1550 nm

# PDB450C-AC at 1550 nm (the device per PersoNota §3)
R_PD_1550    = 1.0                  # A/W (InGaAs peak responsivity)
G_OPTIONS    = [1e3, 1e4, 1e5, 1e6, 1e7]   # V/A switch positions (high-Z)
BW_AT_GAIN   = [150e6, 45e6, 4e6, 0.3e6, 0.1e6]  # Hz, −3 dB BW at each gain
# AC-coupled rejects DC (≤100 Hz cutoff, typ <5 Hz) but passes the
# 81.25 MHz laser pulse train, so the "baseline" period-fill captures
# the unmodulated pulse-train response, not a literal DC level.

# BPD scope reduction (from the existing notebook)
BPD_DT_NS        = 20e-3            # 20 ps/sample
SIGNAL_START_NS  = 45.0
SIGNAL_END_NS    = 75.0
FFT_SIGMA_GHZ    = 0.35
FFT_ORDER        = 6
PROBE_HALF_WIN   = 25


# ============================================================================
# Helpers (mirror THz_Sensor_Characterization.ipynb)
# ============================================================================
def period_fill_indices(t_ns, s0, s1):
    dt = (t_ns[-1] - t_ns[0]) / (len(t_ns) - 1)
    T  = T_LASER_NS / dt
    i0, i1 = int(round(s0/dt)), int(round(s1/dt))
    idx = np.arange(i0, i1)
    nl  = np.ceil((idx + 1 - i0) / T).astype(int)
    nr  = np.ceil((i1 - idx)       / T).astype(int)
    return i0, i1, np.round(idx - nl*T).astype(int), np.round(idx + nr*T).astype(int)

def fft_lowpass(arr, dt_ns):
    n = arr.shape[1]
    freqs = np.fft.rfftfreq(n, d=dt_ns)
    H = np.exp(-0.5 * (freqs / FFT_SIGMA_GHZ) ** (2 * FFT_ORDER))
    return np.fft.irfft(np.fft.rfft(arr, axis=1) * H, n=n, axis=1)


# ============================================================================
# Load the 1 nC fine scan and reduce one step's BPD trace
# ============================================================================
d = np.load(DATA)
t_ns   = 1e9 * d['x'][0, :]
phase  = d['phase']
BPDwf  = d['BPDwf']                                  # (steps, 10, 5000) V
delay_ps = (phase - phase[0]) / RF_HZ / 360 * 1e12   # ps

# Step-averaged traces
traces = BPDwf.mean(axis=1)                          # (steps, 5000)

# Period-fill baseline & residual
dt = (t_ns[-1] - t_ns[0]) / (len(t_ns) - 1)
i0, i1, sl, sr = period_fill_indices(t_ns, SIGNAL_START_NS, SIGNAL_END_NS)
baseline = traces.copy()
baseline[:, i0:i1] = 0.5 * (traces[:, sl] + traces[:, sr])
residual = fft_lowpass(traces - baseline, dt)

# Probe time = sample where the residual has highest variance across delay
tp = int(residual.var(axis=0).argmax())
t_probe = t_ns[tp]
win = slice(max(0, tp - PROBE_HALF_WIN), min(residual.shape[1], tp + PROBE_HALF_WIN))
signed_amp = residual[:, win].mean(axis=1)           # (steps,) V

# Pick the step with largest |signed_amp| → the EO peak step we'll use throughout
peak_step = int(np.argmax(np.abs(signed_amp)))
V_baseline_at_peak = baseline[peak_step, tp]         # V (unmodulated pulse-train level)
V_total_at_peak    = traces[peak_step, tp]           # V (with THz modulation)
ΔV_peak            = signed_amp[peak_step]            # V (period-fill-subtracted)

print(f'Loaded 1 nC fine scan: {BPDwf.shape}')
print(f'Probe sample idx tp = {tp}  (t = {t_probe:.2f} ns)')
print(f'EO peak step = {peak_step}  (Δτ = {delay_ps[peak_step]:+.2f} ps)')
print(f'  V_baseline (period-fill) at probe = {V_baseline_at_peak*1e3:+.2f} mV')
print(f'  V_total    (raw)         at probe = {V_total_at_peak*1e3:+.2f} mV')
print(f'  ΔV         (signed amp)         = {ΔV_peak*1e3:+.2f} mV')


# ============================================================================
# Common figure helpers
# ============================================================================
plt.rcParams.update({
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'figure.dpi':       110,
    'savefig.dpi':      170,
    'savefig.bbox':     'tight',
    'figure.facecolor': 'white',
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'legend.fontsize':  9,
})

def savefig(fig, name):
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out)
    print(f'  saved: {out}')
    plt.close(fig)


# ============================================================================
# Pipeline overview figure (drawn first, references all 3 steps)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.axis('off')

boxes = [
    (0.04, 'V_BPD(τ)\n(scope, AC-coupled,\n50 Ω terminated)'),
    (0.28, 'ΔP_diff(τ)\n= P_x − P_y\n(differential intensity)'),
    (0.52, 'Δφ(τ)\n(EO phase shift,\nbias-corrected)'),
    (0.76, 'E_THz(τ)\n(physical field\nincident on TFLN)'),
]
colors = ['#9ecae1', '#fdae6b', '#a1d99b', '#bcbddc']
for (x, label), col in zip(boxes, colors):
    bb = mpatches.FancyBboxPatch((x, 0.30), 0.16, 0.42,
        boxstyle='round,pad=0.02', facecolor=col, edgecolor='k', lw=1.2)
    ax.add_patch(bb)
    ax.text(x + 0.08, 0.51, label, ha='center', va='center', fontsize=10)

# Arrows + step labels
for x0, x1, txt, sub in [
    (0.20, 0.28, 'STEP 1', 'V → ΔP\nvia G·ℛ'),
    (0.44, 0.52, 'STEP 2', 'ΔP → Δφ\nvia BPD cosine\n+ bias φ₀'),
    (0.68, 0.76, 'STEP 3', 'Δφ → E_THz\nvia Wilke 2024\nhalf-wave field E_π'),
]:
    ax.add_patch(FancyArrowPatch((x0, 0.51), (x1, 0.51),
        arrowstyle='-|>', mutation_scale=22, color='k', lw=1.5))
    ax.text((x0+x1)/2, 0.80, txt, ha='center', fontsize=10, weight='bold')
    ax.text((x0+x1)/2, 0.18, sub,  ha='center', fontsize=8.5, color='dimgray')

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('EO detection pipeline — measured scope voltage to incident THz electric field',
             fontsize=12, pad=8)
savefig(fig, '00_pipeline_overview.png')


# ============================================================================
# STEP 1 — V_BPD → optical power (intensity)
# ============================================================================
# The PDB450C-AC datasheet gives the explicit formula for the RF OUTPUT:
#
#     V_RF,OUT  =  (P_x − P_y) · ℛ(λ) · G_eff
#                =  ΔP_diff      · ℛ(λ) · G_eff
#
# where ℛ(λ) is the InGaAs responsivity (A/W) and G_eff is the *effective*
# transimpedance gain seen by the load:
#     G_eff(high-Z) = G_switch            (datasheet value)
#     G_eff(50 Ω)   = G_switch / 2        (voltage-divider with internal 50 Ω)
#
# AC coupling rejects DC (≤100 Hz cutoff) but the 81.25 MHz laser pulse train
# passes through, so the "baseline" of the trace is the unmodulated-pulse
# response, not a literal DC.  The period-fill subtraction in the existing
# notebook removes that pulse-train carrier, leaving:
#
#     ΔV(τ) = V_BPD(τ) − V_baseline(τ)
#           = G_eff · ℛ(λ) · ΔP_diff(τ)
#
# Inversion: ΔP_diff(τ) = ΔV(τ) / (G_eff · ℛ(λ))

# ── Plot 1a — raw trace, baseline, residual at the EO peak step
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

ax = axes[0]
ax.plot(t_ns, 1e3*traces[peak_step],   'k-',  lw=1.0, label='raw V_BPD (10-shot avg)')
ax.plot(t_ns, 1e3*baseline[peak_step], 'C0--',lw=1.4,
        label='period-fill baseline (81.25 MHz laser pulse train, AC-passed)')
ax.axvspan(SIGNAL_START_NS, SIGNAL_END_NS, color='gold', alpha=0.18, label='signal window [45, 75] ns')
ax.set_ylabel('V_BPD (mV)')
ax.set_title(f'STEP 1 — raw scope trace at EO peak step  (1 nC scan, Δτ = {delay_ps[peak_step]:+.2f} ps)')
ax.legend(loc='lower right', fontsize=9)

ax = axes[1]
ax.plot(t_ns, 1e3*residual[peak_step], 'C3-', lw=1.2,
        label='ΔV(t) = V_BPD − baseline   (FFT low-pass filtered)')
ax.axvspan(t_ns[win.start], t_ns[win.stop-1], color='C3', alpha=0.18,
           label=f'probe window: ±{PROBE_HALF_WIN} samples about tp = {t_probe:.2f} ns')
ax.axvline(t_probe, color='k', lw=0.8, ls=':')
ax.set_ylabel('Residual ΔV (mV)')
ax.legend(loc='upper right', fontsize=9)

# Bottom: convert ΔV → ΔP_diff for several plausible gain settings
ax = axes[2]
ax.plot(t_ns, 1e3*residual[peak_step], 'C3-', lw=1.4, label='ΔV (mV) — left axis')
ax2 = ax.twinx()
for G, ls, col in [(1e3, '-', 'C0'), (1e4, '--', 'C2'), (1e5, ':', 'C4')]:
    G_eff = G/2
    ΔP_uW = 1e6 * residual[peak_step] / (G_eff * R_PD_1550)
    ax2.plot(t_ns, ΔP_uW, ls, color=col, lw=1.4,
             label=f'G_switch = 10$^{int(np.log10(G))}$ V/A   →   G_eff(50Ω) = {G_eff:.0e} V/A')
ax.set_xlim(SIGNAL_START_NS, SIGNAL_END_NS)
ax.set_xlabel('Scope time t (ns)')
ax.set_ylabel('ΔV (mV)', color='C3')
ax2.set_ylabel('ΔP_diff = P_x − P_y   (µW)   — right axis')
ax.tick_params(axis='y', labelcolor='C3')
# Two separate legends so they don't fight each other
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9,
           title='Right axis: gain-dependent intensity  (ΔP = ΔV / (G_eff · ℛ))')

plt.tight_layout()
savefig(fig, '01_step1_voltage_to_intensity.png')


# ── Plot 1b — PDB450C-AC gain table visualised
fig, ax = plt.subplots(figsize=(9, 4.8))
G_arr = np.array(G_OPTIONS)
G_eff = G_arr / 2
P_full_scale_uW = 1e6 * 2.3 / (G_eff * R_PD_1550)   # 50 Ω max swing ±2.3 V
P_full_g7_uW    = 1e6 * 5.0 / (G_OPTIONS[-1]/2 * R_PD_1550)  # G=10⁷ → ±5 V
P_full_scale_uW[-1] = P_full_g7_uW

# Convert observed ΔV peak (mV) into ΔP_diff for each gain
ΔP_obs_uW = 1e6 * abs(ΔV_peak) / (G_eff * R_PD_1550)

ax.bar(range(len(G_arr)), ΔP_obs_uW, color='C3', alpha=0.75,
       label=f'Observed |ΔP_diff| (this scan, |ΔV|={1e3*abs(ΔV_peak):.1f} mV)')
ax.bar(range(len(G_arr)), P_full_scale_uW, color='C0', alpha=0.25,
       label='Full-scale ΔP at 50 Ω clipping')
for i, (g, bw) in enumerate(zip(G_arr, BW_AT_GAIN)):
    ax.text(i, 1.05*max(ΔP_obs_uW[i], P_full_scale_uW[i]),
            f'BW = {bw/1e6:.1f} MHz', ha='center', fontsize=8)
ax.set_xticks(range(len(G_arr)))
ax.set_xticklabels([f'10$^{int(np.log10(g))}$' for g in G_arr])
ax.set_xlabel('PDB450C-AC gain switch (V/A, datasheet high-Z value)')
ax.set_ylabel('Differential optical power |ΔP_diff| (µW)')
ax.set_yscale('log')
ax.set_title('STEP 1 — observed ΔV mapped to ΔP_diff across all gain switch positions')
ax.legend(loc='upper right')
ax.text(0.02, 0.95,
        'Knowing the actual gain switch position is required to fix\n'
        'the absolute intensity scale.  Bandwidth shrinks with gain.',
        transform=ax.transAxes, va='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
plt.tight_layout()
savefig(fig, '01_step1_gain_table.png')


# ============================================================================
# STEP 2 — Intensity → phase difference + static bias φ₀
# ============================================================================
# In a balanced MZI with two complementary outputs (PersoNota §3, Wilke §1):
#
#     P_x = I₀ · cos²(φ₀ + Δφ)
#     P_y = I₀ · sin²(φ₀ + Δφ)
#
# I₀  : total optical power reaching the BPD pair (P_x + P_y = I₀)
# φ₀  : *static bias* — the residual MZI imbalance + r₂₂ / fabrication phase,
#       sets the operating point on the cosine curve.
# Δφ  : THz-induced extra phase, what we want.
#
# Subtracting the two photocurrents:
#
#     ΔP_diff  =  P_x − P_y  =  I₀ · cos[ 2(φ₀ + Δφ) ]                (★)
#
# This is the "BPD cosine".  The BPD output voltage is V = G·ℛ·ΔP_diff.
#
# Small-signal expansion around φ₀:
#
#     ΔP_diff  ≈  I₀ · [ cos(2φ₀)   −  2 sin(2φ₀) · Δφ  +  𝒪(Δφ²) ]
#
# Define the bias factor  s ≡ sin(2φ₀) ∈ [0, 1].  The slope of the
# response to Δφ is maximised at  φ₀ = π/4  (quadrature, s = 1).
#
# Calibration: the static cosine offset is
#
#     V_baseline  ≡  ⟨V_BPD⟩_no-THz  =  G · ℛ · I₀ · cos(2φ₀)
#     V_max       =                       G · ℛ · I₀         (cos = 1)
#
# Therefore:
#
#     cos(2φ₀) = V_baseline / V_max
#     sin(2φ₀) = √(1 − (V_baseline/V_max)²)
#
# To pin V_max you need an auxiliary measurement of I₀ — either:
#   (a) the MONITOR+/MONITOR− outputs (sum gives I₀ since they're DC-coupled),
#   (b) a separate power meter at the device input + the −13 dB insertion loss,
#   (c) a deliberate bias scan that sweeps the cosine full-range and reads off
#       both V_max and V_min so V_max = (V_max_measured − V_min_measured)/2.
#
# In the AWA data the MONITOR ports were NOT recorded, so φ₀ is presently
# *unknown*.  The existing notebook implicitly assumes quadrature (s = 1).

# ── Plot 2a — the BPD cosine, with bias regimes annotated
phi_full = np.linspace(0, np.pi, 600)                # full period (0 → π in φ₀)
y_full   = np.cos(2*phi_full)
fig, ax = plt.subplots(figsize=(9, 5.4))
ax.plot(np.degrees(phi_full), y_full, 'k-', lw=1.5,
        label='ΔP_diff / I₀  =  cos[ 2(φ₀ + Δφ) ]')

# Mark special operating points
for phi_deg, name, col in [
    (0,   'φ₀ = 0  (in-phase)\nmax intensity\ns = sin 2φ₀ = 0', 'C1'),
    (45,  'φ₀ = π/4  (QUADRATURE)\nzero crossing\ns = 1  ⇒ max Δφ-sensitivity', 'C2'),
    (90,  'φ₀ = π/2  (anti-phase)\nmin intensity\ns = 0', 'C3'),
]:
    ax.axvline(phi_deg, color=col, ls='--', lw=1)
    ax.plot(phi_deg, np.cos(2*np.radians(phi_deg)), 'o', color=col, ms=9)
    y_text = np.cos(2*np.radians(phi_deg)) + (0.18 if phi_deg != 45 else 0.22)
    ax.text(phi_deg, y_text, name, ha='center', fontsize=8.5, color=col,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=col))

# Highlight the linear-response slope at quadrature
phi_q = np.pi/4
delta = np.linspace(-0.18, 0.18, 50)
tangent_y = np.cos(2*phi_q) - 2*np.sin(2*phi_q)*delta
ax.plot(np.degrees(phi_q + delta), tangent_y, 'C2-', lw=2.5, alpha=0.55,
        label='small-signal slope at quadrature:  −2·s  (here s=1)')

ax.set_xlabel('Operating point φ₀ + Δφ  (degrees)')
ax.set_ylabel('Normalised differential power  ΔP_diff / I₀')
ax.set_title('STEP 2 — the BPD cosine: how the operating bias φ₀ sets the Δφ-sensitivity')
ax.legend(loc='lower left', fontsize=9)
ax.set_xlim(0, 180); ax.set_ylim(-1.18, 1.32)
ax.axhline(0, color='gray', lw=0.5)
plt.tight_layout()
savefig(fig, '02_step2_bpd_cosine.png')


# ── Plot 2b — small-signal sensitivity for several bias choices
fig, ax = plt.subplots(figsize=(10, 5.5))
dphi_mrad = np.linspace(-50, 50, 200)
for phi0_deg, col in zip([15, 30, 45, 60, 75], plt.cm.viridis(np.linspace(0.15, 0.85, 5))):
    phi0 = np.radians(phi0_deg)
    s = np.sin(2*phi0)
    ΔP_norm = np.cos(2*phi0 + 2*1e-3*dphi_mrad) - np.cos(2*phi0)
    ax.plot(dphi_mrad, 1e3*ΔP_norm, color=col, lw=1.8,
            label=f'φ₀ = {phi0_deg}°   (s = sin 2φ₀ = {s:.2f})')
ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('Induced phase shift Δφ (mrad)')
ax.set_ylabel('Small-signal differential power\n[ΔP_diff(Δφ) − ΔP_diff(0)] / I₀  × 10³')
ax.set_title('STEP 2 — small-signal response slope set by sin(2φ₀)\n'
             'Steepest slope (and best Δφ sensitivity) at quadrature, φ₀ = 45°')
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
savefig(fig, '02_step2_smallsignal_slopes.png')


# ── Plot 2c — how V_baseline encodes φ₀ (single big panel, cleaner annotation)
fig, ax = plt.subplots(figsize=(11, 6))
phi0_axis = np.linspace(0, np.pi/2, 400)
ax.plot(np.degrees(phi0_axis), np.cos(2*phi0_axis), 'k-',  lw=2,
        label='cos(2φ₀) = V_baseline / V_max   ← what you read off')
ax.plot(np.degrees(phi0_axis), np.sin(2*phi0_axis), 'C2-', lw=2,
        label='sin(2φ₀) = s   (bias factor, scales Δφ → ΔV gain)')

# Demo: if V_max were 180/220/280 mV, what φ₀ does our V_baseline=175 mV give?
V_baseline_demo = V_baseline_at_peak
V_max_hyps = [(0.18, 'C0'),  (0.22, 'C1'),  (0.28, 'C3')]
for j, (V_max_hyp, col) in enumerate(V_max_hyps):
    r = V_baseline_demo / V_max_hyp
    if abs(r) > 1: continue
    phi0_inf = 0.5 * np.arccos(r)
    s_inf    = np.sin(2*phi0_inf)
    ax.axvline(np.degrees(phi0_inf), color=col, ls='--', lw=1.2, alpha=0.75)
    ax.plot(np.degrees(phi0_inf), r,      'o', color=col, ms=10, mec='k')
    ax.plot(np.degrees(phi0_inf), s_inf,  's', color=col, ms=10, mec='k')

# Stacked text box (no overlap)
text_lines = [f'V_baseline (measured) = {V_baseline_demo*1e3:.0f} mV', '']
for V_max_hyp, col in V_max_hyps:
    r = V_baseline_demo / V_max_hyp
    if abs(r) > 1:
        text_lines.append(f'V_max = {V_max_hyp*1e3:.0f} mV → r>1, infeasible')
    else:
        phi0_inf = 0.5 * np.arccos(r)
        s_inf    = np.sin(2*phi0_inf)
        text_lines.append(
            f'If V_max = {V_max_hyp*1e3:>3.0f} mV  →  '
            f'φ₀ = {np.degrees(phi0_inf):4.1f}°,  s = {s_inf:.2f}')
ax.text(0.97, 0.05, '\n'.join(text_lines),
        transform=ax.transAxes, ha='right', va='bottom',
        family='monospace', fontsize=10,
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.95, boxstyle='round,pad=0.4'))

ax.set_xlabel('Inferred static bias φ₀ (degrees)')
ax.set_ylabel('cos(2φ₀)  /  sin(2φ₀)')
ax.set_title('STEP 2 — extracting φ₀ from the unmodulated baseline\n'
             '(circles: V_bl/V_max points on cos curve;  squares: corresponding s on sin curve)')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, 90); ax.set_ylim(-0.05, 1.1)
ax.axhline(0, color='gray', lw=0.5)
plt.tight_layout()
savefig(fig, '02_step2_bias_calculation.png')


# ── Plot 2c — combining into the K_BPD calibration
# K_BPD ≡ ∂V/∂Δφ at the operating point  =  2 · G_eff · ℛ · I₀ · sin(2φ₀)
fig, ax = plt.subplots(figsize=(9, 5.4))

G_eff_demo = 1e4 / 2     # assume gain switch 10⁴ V/A into 50 Ω
P_arm_grid = np.geomspace(10, 2000, 200) * 1e-6   # 10 µW – 2 mW per arm
I0_grid    = 2 * P_arm_grid                       # total = 2 × per-arm at quadrature

for s, col in zip([0.3, 0.5, 0.7, 1.0],
                  plt.cm.plasma(np.linspace(0.15, 0.85, 4))):
    K = 2 * G_eff_demo * R_PD_1550 * I0_grid * s
    ax.plot(1e6*P_arm_grid, K, color=col, lw=1.6,
            label=f's = sin 2φ₀ = {s:.1f}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Per-diode CW power I₀/2 (µW)')
ax.set_ylabel('BPD calibration  K_BPD = ∂V/∂Δφ  (V/rad)')
ax.set_title(f'STEP 2 — small-signal calibration K_BPD = 2·G_eff·ℛ·I₀·sin(2φ₀)\n'
             f'(here G_switch = 10⁴ V/A, 50 Ω; ℛ(1550 nm) = {R_PD_1550:.1f} A/W)')

# Mark the assumption used in the existing notebook
ax.axvline(100, color='gray', ls=':', lw=1)
ax.text(110, 1.5, 'notebook assumes\nP_arm = 100 µW', fontsize=8, color='gray')

ax.legend()
plt.tight_layout()
savefig(fig, '02_step2_K_BPD_calibration.png')


# ============================================================================
# STEP 3 — Phase shift → THz E-field (Wilke 2024)
# ============================================================================
# Wilke et al. Eq. (1) — for collinear THz and probe, X-cut LN, TE mode along
# the in-plane extraordinary axis:
#
#     Δφ(ω_RF)  =  (ω_opt / c) · T_RF(ω_RF) · n_e³ · r₃₃ · E_THz · l
#               =  (2π / λ_probe) · T_RF(ω_RF) · n_e³ · r₃₃ · E_THz · l
#
# Variables (all explicit):
#   ω_opt    = 2π c / λ_probe      probe optical angular frequency
#   λ_probe                        = 1550 nm  (telecom band)
#   c                              vacuum speed of light
#   n_e                            = 2.15     bulk LN extraordinary index @1550 nm
#   r₃₃                            = 30.9 pm/V    LN electro-optic coefficient
#   l                              = 600 µm   MZI arm length
#   E_THz                          THz electric field along ê_extraordinary
#   T_RF(ω_RF)                     velocity-mismatch transfer function (Eq. 2):
#
#     T_RF(ω_RF) = sinc[ ω_RF (n_RF − n_opt) l / (2c) ]
#                = sin(x) / x  with  x = ω_RF (n_RF − n_opt) l / (2c)
#
#     n_RF       THz-frequency refractive index of the SUBSTRATE (SiO2: 1.95)
#     n_opt      group index of the optical mode in the TFLN waveguide (2.4)
#
# (Wilke's substrate carries the THz wave since the LN thin film is much
#  thinner than the THz wavelength — the field "sees" SiO2.)
#
# Rewriting as half-wave field E_π:
#
#     Δφ = π · (E_THz / E_π)
#     E_π = λ_probe / [ 2 · T_RF · n_e³ · r₃₃ · l ]
#
# At low frequency (T_RF → 1) with Wilke's parameters this gives 4.2 MV/m.

# ── Plot 3a — the transfer function T_RF(ω) full-page
f_GHz = np.linspace(0.001, 1500, 4000)
omega = 2 * np.pi * f_GHz * 1e9
x = omega * (N_RF_FUSED - N_OPT_GUIDE) * L_MZI / (2 * c)
T_RF = np.sinc(x / np.pi)           # numpy sinc = sin(πx)/(πx)
modulation_response = T_RF**2

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(f_GHz, T_RF, 'k-', lw=2,
        label='T_RF(ω) = sinc[ ω·(n_RF − n_opt)·l / (2c) ]   (Wilke Eq. 2)')
ax.plot(f_GHz, modulation_response, 'C3--', lw=2,
        label='|T_RF|²   (Wilke Fig. 2 modulation response)')
ax.axhline(1/np.sqrt(2), color='gray', ls=':', lw=0.8, alpha=0.6)
ax.axhline(0.5,          color='gray', ls=':', lw=0.8, alpha=0.6)
idx_3dB_T  = np.argmin(np.abs(T_RF - 1/np.sqrt(2)))
idx_3dB_T2 = np.argmin(np.abs(T_RF**2 - 0.5))
ax.axvline(f_GHz[idx_3dB_T2], color='C3', ls=':', lw=1)
ax.text(f_GHz[idx_3dB_T2]+25, 0.55,
        f'f_3dB(|T_RF|²) ≈ {f_GHz[idx_3dB_T2]:.0f} GHz\n(Wilke quotes ≈ 640 GHz)',
        fontsize=10, color='C3',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='C3'))
ax.set_xlabel('THz frequency f_RF (GHz)')
ax.set_ylabel('Transfer function')
ax.set_xlim(0, 1500); ax.set_ylim(0, 1.05)
ax.set_title(f'STEP 3 — phase-matching transfer T_RF(f)\n'
             f'l = {L_MZI*1e6:.0f} µm,  n_RF = {N_RF_FUSED} (SiO₂ THz),  '
             f'n_opt = {N_OPT_GUIDE} (TFLN waveguide group index)')
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
savefig(fig, '03_step3_transfer_function.png')


# ── Plot 3b — variables inventory as a standalone, large figure
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.axis('off')
table_data = [
    ['Symbol', 'Meaning', 'Value (this setup)', 'Source / note'],
    ['λ_probe',      'probe wavelength',               '1550 nm',
                                                        'telecom band, AWA c-fiber laser'],
    ['ω_opt = 2πc/λ', 'probe optical angular freq.',  f'{2*np.pi*c/LAMBDA_PROBE:.3e} rad/s',
                                                        '— derived —'],
    ['n_e',           'LN extraordinary index @ 1550 nm', f'{N_E_LN}',
                                                        'Wilke / Palik'],
    ['r₃₃',           'LiNbO₃ EO coefficient',          f'{R33*1e12:.1f} pm/V',
                                                        'Yariv & Yeh; Wilke'],
    ['l',             'MZI interaction-arm length',    f'{L_MZI*1e6:.0f} µm',
                                                        'Wilke device geometry'],
    ['n_opt',         'TFLN waveguide group index',     f'{N_OPT_GUIDE}',
                                                        'Wilke Fig. 2'],
    ['n_RF',          'SiO₂ THz refractive index',      f'{N_RF_FUSED} (≈ 600 GHz)',
                                                        'Palik'],
    ['T_RF(ω)',       'velocity-mismatch sinc',         'sinc[ω·(n_RF−n_opt)·l/(2c)]',
                                                        'Wilke Eq. 2'],
    ['E_π',           'half-wave field (Wilke Eq. 3)',  'λ/(2·T_RF·n_e³·r₃₃·l)',
                                                        ''],
    ['',              '       …at T_RF → 1',           f'≈ {LAMBDA_PROBE/(2*1*N_E_LN**3*R33*L_MZI)/1e6:.2f} MV/m',
                                                        'matches Wilke 4.2 MV/m'],
    ['Δφ',            'EO phase shift (small-signal)', 'π · E_THz / E_π',
                                                        'inverted: E_THz = (E_π/π)·Δφ'],
]
table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                 colWidths=[0.15, 0.32, 0.27, 0.26])
table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.7)
for j in range(4): table[(0, j)].set_facecolor('#888888'); table[(0, j)].set_text_props(color='white', weight='bold')
for i in range(1, len(table_data)):
    for j in range(4):
        table[(i, j)].set_facecolor('#f5f5f5' if i % 2 else '#ffffff')
ax.set_title('Variable inventory for the Wilke et al. EO model\n'
             'Δφ(ω_RF) = (2π/λ_probe) · T_RF(ω_RF) · n_e³ · r₃₃ · l · E_THz',
             pad=14, fontsize=12)
plt.tight_layout()
savefig(fig, '03_step3_variables_table.png')


# ── Plot 3b — Δφ ↔ E_THz line, with the AWA data overlaid
# Compute E_π and convert the measured signed-amp curve to Δφ, then E_THz.
# Use the *quadrature* assumption (s=1) AND show how the answer shifts if s<1.

T_RF_at_DC = 1.0    # low-freq limit — Wilke quotes 4.2 MV/m here
E_pi_DC = LAMBDA_PROBE / (2 * T_RF_at_DC * N_E_LN**3 * R33 * L_MZI)

# Demo gain & power assumptions
G_eff_demo = 1e4 / 2
P_arm_demo = 100e-6
I0_demo    = 2 * P_arm_demo
K_BPD_quad = 2 * G_eff_demo * R_PD_1550 * I0_demo * 1.0
K_BPD_half = 2 * G_eff_demo * R_PD_1550 * I0_demo * 0.5

dphi_q = signed_amp / K_BPD_quad           # rad
dphi_h = signed_amp / K_BPD_half
E_q    = dphi_q * E_pi_DC / np.pi           # V/m
E_h    = dphi_h * E_pi_DC / np.pi

# Plot 3c — the linear Δφ↔E_THz map (single big panel)
fig, ax = plt.subplots(figsize=(10, 5.8))
E_grid_kVm = np.linspace(-200, 200, 400)
dphi_grid_mrad = 1e3 * np.pi * (E_grid_kVm * 1e3) / E_pi_DC
ax.plot(dphi_grid_mrad, E_grid_kVm, 'k-', lw=2,
        label=f'E_THz = (E_π / π) · Δφ            E_π(T_RF=1) = {E_pi_DC/1e6:.2f} MV/m')
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
dphi_q_max_mrad = 1e3*np.abs(dphi_q).max()
ax.axvspan(-dphi_q_max_mrad, dphi_q_max_mrad, color='C2', alpha=0.15,
           label=f'measured Δφ range (this 1 nC scan, s=1):  ±{dphi_q_max_mrad:.2f} mrad')
# Mark the EO peak point
ax.plot(1e3*dphi_q[peak_step], 1e-3*E_q[peak_step], 'o',
        color='C3', ms=11, mec='k',
        label=f'EO peak step:  Δφ = {1e3*dphi_q[peak_step]:+.1f} mrad → E_THz = {1e-3*E_q[peak_step]:+.1f} kV/m')
ax.set_xlabel('Phase shift Δφ (mrad)')
ax.set_ylabel('E_THz (kV/m)')
ax.set_title('STEP 3 — linear Δφ ↔ E_THz map (Wilke half-wave relation)')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(-30, 30); ax.set_ylim(-45, 45)
plt.tight_layout()
savefig(fig, '03_step3_phase_to_field_map.png')

# Plot 3d — the recovered E_THz(τ) profile for this scan
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.plot(delay_ps, 1e-3*E_q, 'C0-',  lw=1.8,
        label='E_THz(τ),  φ₀ = quadrature  (s=1, best-case bias)')
ax.plot(delay_ps, 1e-3*E_h, 'C3--', lw=1.8,
        label='E_THz(τ),  s = 0.5  (sub-optimal bias → doubles inferred E)')
ax.axhline(0, color='k', lw=0.5)
ax.axvline(delay_ps[peak_step], color='gray', ls=':', lw=1,
           label=f'EO peak delay = {delay_ps[peak_step]:+.1f} ps')
ax.set_xlabel('Laser–beam delay τ (ps)')
ax.set_ylabel('Recovered E_THz (kV/m)')
ax.set_title('STEP 3 — measured V_BPD(τ) propagated all the way to E_THz(τ)\n'
             f'(1 nC fine scan,  G_switch = 10⁴ V/A into 50 Ω,  P_arm = 100 µW assumed)')
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
savefig(fig, '03_step3_E_THz_vs_delay.png')


# ============================================================================
# END-TO-END SUMMARY — apply all 3 steps to the 1 nC scan and tabulate
# ============================================================================
ΔP_at_peak_uW = 1e6 * abs(ΔV_peak) / (G_eff_demo * R_PD_1550)
Δφ_at_peak_rad  = abs(ΔV_peak) / K_BPD_quad
Δφ_at_peak_mrad = 1e3 * Δφ_at_peak_rad
E_at_peak_Vm    = Δφ_at_peak_rad * E_pi_DC / np.pi
E_at_peak_kVm   = E_at_peak_Vm / 1e3

print('\n' + '='*72)
print('END-TO-END PROPAGATION (1 nC fine scan, EO peak step)')
print('='*72)
print(f'STEP 1:  |ΔV|     = {1e3*abs(ΔV_peak):8.2f} mV   '
      f'(from scope, period-fill-subtracted, FFT-LP filtered)')
print(f'         |ΔP|     = {ΔP_at_peak_uW:8.2f} µW   '
      f'(/ G_eff·ℛ, G=10⁴ V/A 50 Ω, ℛ=1.0 A/W @ 1550 nm)')
print(f'STEP 2:  K_BPD    = {K_BPD_quad:8.4f} V/rad '
      f'(2·G·ℛ·I₀·s, s=1 quadrature, I₀=200 µW assumed)')
print(f'         |Δφ|     = {Δφ_at_peak_mrad:8.3f} mrad ({Δφ_at_peak_mrad*1e-3*180/np.pi*1000:.2f} mdeg)')
print(f'STEP 3:  E_π      = {E_pi_DC/1e6:8.3f} MV/m  '
      f'(λ/(2·T_RF·n_e³·r₃₃·l), T_RF=1)')
print(f'         |E_THz|  = {E_at_peak_kVm:8.2f} kV/m  (Δφ · E_π / π)')
print('='*72)
print('Note: the static bias factor s = sin(2φ₀) divides Δφ and hence E_THz.')
print('      Without MONITOR readouts or a bias scan, s is currently unknown;')
print('      values reported assume quadrature (s = 1, best case).')

# Final summary figure: chained-arrow diagram showing the actual propagation
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.axis('off')

stages = [
    ('V_BPD',     f'|ΔV|\n= {1e3*abs(ΔV_peak):.2f} mV',         '#9ecae1',
     'scope reading\n(period-fill subtracted,\nFFT low-pass filtered)'),
    ('ΔP_diff',   f'|ΔP|\n= {ΔP_at_peak_uW:.2f} µW',          '#fdae6b',
     '÷ (G_eff · R_PD)\nG_eff = G_switch/2 (50 Ω)\nG_switch = 1e4 V/A → 5e3 V/A\nR_PD(1550) = 1.0 A/W'),
    ('Δφ',        f'|Δφ|\n= {Δφ_at_peak_mrad:.2f} mrad',      '#a1d99b',
     '÷ K_BPD\nK_BPD = 2·G_eff·R_PD·I₀·sin(2φ₀)\nI₀ = 200 µW assumed\ns = 1 (quadrature assumed)'),
    ('E_THz',     f'|E_THz|\n= {E_at_peak_kVm:.1f} kV/m',     '#bcbddc',
     '× E_π / π\nE_π = λ/(2·T_RF·n_e³·r₃₃·l)\n     = 4.21 MV/m  (T_RF = 1)'),
]
x_positions = [0.06, 0.31, 0.56, 0.81]
for (sym, val, col, formula), x in zip(stages, x_positions):
    # Coloured pill with value
    bb = mpatches.FancyBboxPatch((x, 0.45), 0.16, 0.30,
        boxstyle='round,pad=0.02', facecolor=col, edgecolor='k', lw=1.4)
    ax.add_patch(bb)
    ax.text(x + 0.08, 0.66, sym, ha='center', va='center',
            fontsize=15, weight='bold')
    ax.text(x + 0.08, 0.54, val, ha='center', va='center', fontsize=11)
    # Operation label below
    ax.text(x + 0.08, 0.22, formula, ha='center', va='top',
            fontsize=9, family='monospace',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.3'))

# Arrows between stages
for i in range(3):
    x0 = x_positions[i] + 0.16
    x1 = x_positions[i+1]
    ax.add_patch(FancyArrowPatch((x0, 0.60), (x1, 0.60),
        arrowstyle='-|>', mutation_scale=26, color='k', lw=2))
    op_label = ['STEP 1', 'STEP 2', 'STEP 3'][i]
    ax.text((x0+x1)/2, 0.83, op_label, ha='center', fontsize=11, weight='bold')

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title(f'End-to-end propagation on the 1 nC fine scan (EO peak step, Δτ = {delay_ps[peak_step]:+.1f} ps)\n'
             'quadrature bias assumed (s = 1)',
             fontsize=12, pad=10)
plt.tight_layout()
savefig(fig, '04_end_to_end_propagation.png')

print('\nAll figures written to:', FIG_DIR)
