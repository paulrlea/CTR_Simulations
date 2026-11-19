# Delay-stage phase calculation

How the moving delay-stage mirror is represented in the CTR-interferometer simulation
(`interf_golay.py`), and why it needs no re-propagation.

## Summary

The delay stage is modelled **analytically as a per-frequency phase**, not by physically
translating the retro mirror and re-running SRW at each position. A pure longitudinal
path-length change is a global (transverse-position-independent) phase factor on that arm's
field, and a global complex factor **commutes with linear propagation** — so the whole
delay scan is a cheap cosine sum over one already-propagated field per frequency.

## 1. The physical delay

The 75 mm flat sits at normal incidence at the end of one Michelson arm and retro-reflects.
Translating it by `Δx` along the beam axis changes that arm's **round-trip** path by `2Δx`,
i.e. a time delay

```
τ = 2 Δx / c            (Δx = stage travel,  c = speed of light)
```

so the plots carry both a delay axis `τ [ps]` and a stage-travel axis `x = cτ/2 [mm]`.

## 2. Delay as a per-frequency phase

Work one monochromatic component at a time (the whole sim is a frequency sweep). A time
delay `τ` is a phase in the frequency domain:

```
E(t − τ)   ⟷   Ẽ(f) · exp(i 2π f τ)
```

Because the mirror is **flat and at normal incidence**, every ray in the arm — whatever its
transverse position `(x,y)` — picks up the *same* extra path `2Δx` (paraxially). The delay is
therefore a **piston phase**, uniform across the wavefront:

```
E_delay-arm(x, y, f; τ) = E_arm(x, y, f) · exp(i 2π f τ)
```

with no `(x,y)` dependence in the phase factor. (The only position dependence dropped is the
`2Δx·θ²/2` curvature term for a ray at angle `θ` — a second-order wavefront change over the
scan; negligible here.)

## 3. Recombination and the Golay reading

For a **balanced (symmetric) Michelson**, both arms deliver the *same* detector-plane field.
Each output amplitude carries the beamsplitter factor `r·t` (delay arm: reflect `r` → retro →
transmit `t`; reference arm: transmit `t` → mirror → reflect `r`) plus the mirror
reflectivities and window transmission — all already folded into the single propagated field
`E_det(x,y,f)` via `channel_response_interf`. Taking the reference arm as the phase origin,
the recombined field at delay `τ` is

```
E_rec(x, y, f; τ) = E_det(x, y, f) · [ 1 + exp(i 2π f τ) ]
```

The Golay cell reads **time-integrated, polarization-blind power over its aperture** (radius
`R`). Using `|1 + e^{iφ}|² = 2(1 + cos φ)` and Parseval (∫|E(t)|²dt = ∫|Ẽ(f)|²df, the
time-integration → a sum over frequencies):

```
P(τ) = Σ_f  w(f) ∫_{r<R} |E_det(x,y,f)|² · |1 + e^{i2πfτ}|²  dA
     = Σ_f  2 S(f) [ 1 + cos(2π f τ) ]

     with   S(f) = w(f) · ∫_{r<R} |E_det|² dA      (aperture-integrated detected spectrum)
            w(f) = |F(f)|² Q²                        (bunch form factor × charge)
```

So the interferogram is the DC term `Σ_f S(f)` plus the cosine transform of `S(f)`. In code
the overall factor of 2 is dropped (units are arbitrary):

```python
# interf_golay.interferogram(records, cfg, R_ap, taus_s)
S = detected_spectrum(records, cfg, R_ap)          # S(f) = w(f)·P_det(f,R)
P(τ) = S.sum() + Σ_f S(f)·cos(2π f τ)              # DC + AC(τ)
```

At `τ = 0` all frequencies add in phase → `P = 2 Σ S` (peak, "zero path difference"); at large
`τ` the cosines wash out → `P → Σ S` (the flat DC baseline). The modulation between them **is**
the interferogram.

## 4. Why no re-propagation is needed

`exp(i 2π f τ)` is a *global* (space-independent) complex scalar. Fresnel propagation is a
linear operator `L`, and a scalar commutes with it: `L[a·E] = a·L[E]`. So applying the delay
phase *before* propagation (at the moving mirror) or *after* propagation (at the detector)
gives the identical result. We therefore:

1. Propagate each frequency through the beamline **once** → `E_det(x,y,f)` (the nominal
   delay-arm geometry, including the 75 mm retro aperture, is baked into this one pass); then
2. Apply the `[1 + e^{i2πfτ}]` factor analytically for every `τ` on the grid.

This is the same trick used for the window transmission and the beamsplitter split (all global
per-frequency complex factors applied at extraction). Each delay point costs one cosine
evaluation — the 411-point scan is essentially free relative to the propagation.

## 5. Delay grid and stage mapping

`interf_golay.delay_grid(cfg, freqs, tau_max_ps, n_tau)` builds a symmetric grid
`τ ∈ [−τ_max, +τ_max]`:

- **Step** `Δτ = 0.8 / (2 f_max)` — finer than the Nyquist limit for the highest frequency
  (f_max = 400 GHz → Δτ ≈ 1 ps, stage step ≈ 0.15 mm), so the fastest fringe is resolved.
- **Range** default `τ_max = 1 / (2 Δf)` where `Δf` is the frequency-grid spacing
  (Δf ≈ 2.4 GHz → τ_max ≈ 205 ps → ±30.75 mm of stage travel), enough to resolve the
  spectral features. Override with `--tau-max-ps` / `--n-tau`.

## 6. Spectrum recovery (Wiener–Khinchin)

The τ-modulated part of `P(τ)` is the cosine transform of `S(f)`, so the CTR power spectrum is
recovered by inverting it — exactly what a real diagnostic does with the measured interferogram:

```python
# interf_golay.recover_spectrum(taus_s, P, f_grid)
ac = P − mean(P)                                    # strip the DC baseline
S_rec(f) = Σ_τ ac(τ) cos(2π f τ) Δτ
```

`recovered_spectrum.png` overlays `S_rec(f)` on the directly-integrated `S(f)`; they match,
confirming the analysis chain. `S(f)` carries the bunch form factor `|F(f)|²`, which is the
longitudinal-bunch information the diagnostic exists to measure.

## 6b. Arm-length asymmetry and the ZPD stage offset

The reference arm (BS → fixed flat) is **1.854"** and the delay arm's nominal length
(BS → retro, `L_bs_to_delay`) is **1.342"**, so the reference is longer by
`ΔL = L_ref − L_bs_to_delay = 0.512" = 13.0 mm`. The interferogram plotted vs *relative
delay* `τ` (and the recovered spectrum) are **unchanged** by this — `τ` is defined as the
optical path difference between the arms, so ZPD is at `τ = 0` by construction. What the
asymmetry sets is the **physical stage position of ZPD**: the delay stage must extend the
delay arm by `ΔL` to equalise the paths, so

```
x_stage(τ) = ΔL + c·τ/2          ⇒   ZPD (τ=0) at x_stage = ΔL = 13.0 mm  (τ₀ = 2ΔL/c = 86.8 ps)
```

**Two distinct quantities (mind the factor of 2):**

| quantity | value | note |
|---|---|---|
| one-way arm length difference `ΔL` | 0.512" = 13.0 mm | `L_ref − L_bs_to_delay` |
| **delay-stage travel** to reach ZPD | **0.512" = 13.0 mm** | move the retro by `ΔL` |
| **round-trip optical path difference** | **1.024" = 26.0 mm = 2ΔL** | both arms are round-trip |
| delay offset `τ₀ = OPD/c` | 86.8 ps | includes the ×2 |

The **round-trip does double the path** — the reference beam goes BS→mirror→BS, so its extra
length counts twice: `OPD = 2·L_ref − 2·L_bs_to_delay = 2ΔL = 1.024"`, and `τ₀ = 2ΔL/c` already
carries that factor. But the **stage travel is only 0.512"**, not 1.024": moving the retro by
`x` changes *its* round-trip path by `2x`, so `x = ΔL = 0.512"` already supplies the full
`2ΔL = 1.024"` of path needed to cancel the OPD. In short: stage 0.512" ⇒ path 1.024" ⇒ 86.8 ps.

This is exactly the beam transport being unchanged: the arm difference (26 mm round trip) is
2.8% of the ~0.94 m detector path, so `E_det` is a good approximation for both arms — **no
re-propagation is needed**, only the stage-axis offset (`save_interferogram`'s
`stage_zero_offset_m`). Practically: the operator finds the central burst at a 13 mm (0.512")
stage displacement, not at the nominal position. At ZPD the two arm paths are equal, so the
beams are equally diffracted → the ideal-overlap (full-visibility) assumption is well justified
right where the burst is measured.

## 7. Assumptions & limitations

- **Reference-arm length is now the measured 1.854"** (was assumed symmetric earlier); it enters
  only as the ZPD stage offset above. A residual assumption is that `E_ref ≈ E_delay` in beam
  size at recombination — exact at ZPD (equal paths), and off by ≤2.8% path elsewhere.
- **Ideal spatial overlap** → 100% fringe visibility. A real interferometer has <100% from
  wavefront/pointing mismatch between the arms; this model does not degrade the contrast.
- **Pure temporal shift**: the piston-phase form drops the `~Δx·θ²` curvature term, i.e. the
  small change in beam size/diffraction from physically moving the mirror over the ~±31 mm
  scan (cm out of a ~0.9 m path). Negligible for the coherence envelope a bunch-length
  diagnostic measures.
- **Amplitude-splitting (Michelson)** combination math. A polarizing (wire-grid / Martin–
  Puplett) interferometer would add polarization bookkeeping, but the delay-as-phase step is
  identical.

## Code references

| item | location |
|---|---|
| interferogram `P(τ) = Σ_f S(f)[1+cos2πfτ]` | `interf_golay.interferogram` |
| aperture-integrated `S(f)` | `interf_golay.detected_spectrum` / `per_freq_aperture_power` |
| delay grid, `τ = 2x/c` mapping | `interf_golay.delay_grid`, `save_interferogram` |
| spectrum recovery (cosine transform) | `interf_golay.recover_spectrum` |
| beamsplitter/window/mirror factor in `E_det` | `interf_beamline.channel_response_interf` |
