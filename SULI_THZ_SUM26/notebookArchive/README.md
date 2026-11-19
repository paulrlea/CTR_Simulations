# SULI_THZ_SUM26

SULI intern project -- Summer 2026.

Analysis of electro-optic (EO) sampling data from BeamNetUS experiments at the Argonne Wakefield Accelerator (AWA), measuring balanced-photodiode (BPD) traces vs. laser delay to characterize bunch and wake signals. Includes companion fiber-laser timing and jitter studies for the RPI/PARTOW integrated THz sensor.

## Notebooks and scripts

- `analysisdata_experiment.ipynb` — main analysis notebook. Loads `.npy` scan files, averages every 4 shots, and plots BPD - Reference vs. delay for dipole-on / dipole-off configurations.
- `jitter_study0Xing.ipynb` — preliminary analysis of fiber-laser timing/jitter data (channels: Ch1 BPM, Ch2 SD, Ch3 FPD). Works with the `.npy` files under `fiberlaser/`.
- `PostP_fromWanming.py` — original post-processing script from Wanming Liu (`wmliu`). Reads a `.npy` scan dict (`phase`, `peak`), converts phase to delay using a 81.25 MHz reference, and writes paired `.csv` / `.png` outputs.

## Data

- `Data_raw/July30_/` — raw scan data from 2025-07-30. Coarse/fine scans plus fiber-laser tests with the CTR screen in/out (no beam).
- `Data_raw/July31_/` — raw scan data from 2025-07-31. Coarse and fine scans across a charge sweep (0.25, 0.5, 1, 2.5, 4, 5, 7, 10 nC), including fiber-resync variants and dipole on/off configurations.
- `Data_raw/Aug01/` — raw scan data from 2025-08-01. Fine and quick scans at 1 nC under various optical configurations (BBO / NoBBO, dipole on/off, iris, rotated polarization).
- `fiberlaser/` — fiber-laser timing and jitter measurements:
  - `CoarsTiming_Measurement*_0.npy` — coarse-timing baseline and BD / BD+10m variants.
  - `CT_Monday_SD_0.npy`, `CT_Monday_SDFPD_{0..5}.npy` — Monday coarse-timing runs with the SD and SD+FPD configurations.
  - `JitterMeasurement_{0..3}.npy` — jitter measurement series.

Each scan `.npy` is a pickled dict with keys:
- `phase` — array of phase set-points in degrees
- `peak` — per-shot array where index 3 is the reference channel and index 4 is the EO-modulated BPD channel

Delay (ns) is computed as `(phase - phase[0]) * 1e9 / 81.25e6 / 360`.

Many scans have companion `.npz`, `.csv`, and `.png` files; `.npz.gz` archives are gzipped bundles of multiple scans.

## Archive key reference

### Data array keys (all `.npz` files)

| Key | Full name | Meaning |
|-----|-----------|---------|
| `x` | Delay axis | Time delay in nanoseconds, converted from phase set-points |
| `BPDwf` | Balanced Photodiode Waveform | Per-shot waveform from the balanced photodiode detector (the EO sampling signal) |
| `ictwf` | Integrating Current Transformer Waveform | Per-shot charge waveform from the ICT, used to normalize by bunch charge |
| `phase` | Phase set-points | Raw phase values in degrees sent to the timing system |

### Fiberlaser file keys (`fiberlaser/*.npy`)

| Key | Full name | Meaning |
|-----|-----------|---------|
| `AWAVXI11:Tek1:wf1/2/3` | AWA VXI-11 Tektronix scope 1, waveform channel 1/2/3 | EPICS process variable: raw waveform from each oscilloscope channel |
| `AWAVXI11:Tek1:wf1/2/3_ts` | ...timestamp | Acquisition timestamp for the corresponding waveform |
| `AWAVXI11:Tek1:x` | Time axis | Horizontal time axis for the oscilloscope waveforms |
| `note` | Human-readable note | String describing what is plugged into each channel |

### Detector / optical components

| Acronym | Meaning |
|---------|---------|
| `BPD` | Balanced Photodiode |
| `BPM` | Beam Position Monitor |
| `BBO` | Beta Barium Borate — nonlinear crystal used for second-harmonic generation / frequency doubling |
| `NoBBO` | No BBO crystal in the optical path |
| `OneBBO` | One BBO crystal inserted |
| `SD` / `schottkyDiode` | Schottky Diode — a fast RF/microwave diode detector |
| `FPD` | Fast Photodiode |
| `BD` | Balanced Detector — used in fiber-laser coarse timing runs as an alternative to the Schottky Diode |
| `CTR` | Coherent Transition Radiation — radiation emitted when the beam passes through a foil screen |
| `scrn` / `scrnIn` / `scrnOut` | Screen (the CTR foil screen), in or out of the beam path |
| `Vitara-T` | Coherent Vitara-T Ti:Sapphire laser oscillator (the reference laser) |
| `Iris` | Iris aperture inserted in the optical path |
| `NoSD` | No Schottky Diode in the detection chain |

### Scan types and conditions

| Acronym | Meaning |
|---------|---------|
| `FineScan` | Fine delay scan — small step size, narrow range |
| `CoarseScan` / `CoarsTiming` | Coarse delay/timing scan — large step size, wide range (`Coars` is a typo of `Coarse`) |
| `CT` | Coarse Timing |
| `eQ` | Nominal charge Q — labels the set charge for that scan (e.g. `eQ10nC` = 10 nC bunch charge) |
| `wh` | Waveform/window height parameter recorded alongside the charge (e.g. `wh4_0` = value 4.0; `_` is a decimal point) |
| `0p5` / `0p25` / `4p5` | Decimal notation — the `p` is the decimal point (e.g. `0p5` = 0.5) |
| `nC` | Nanocoulombs — unit of bunch charge |
| `DipoleOn` / `DipoleOff` | Dipole magnet switched on or off to steer/deflect the beam |
| `Shifted` | Delay axis has been shifted (zero-referenced to the signal peak or edge) |
| `rotated45` / `rotated90` | Polarization rotation of 45° or 90° applied to the probe beam |
| `FiberResync` | Fiber laser resynchronization — the fiber laser timing was re-locked before this scan |
| `jitter_0Xing` | Jitter measured at the zero-crossing of the waveform |
| `jitter_finalEdge` / `jitter_finalEdgelarger` | Jitter measured at the final rising/falling edge, with a larger window variant |
| `longscan` | Extended-range or higher-shot-count scan |
| `Resync` / `Roughscan` | Fiber resync followed by a rough (coarse) scan to re-find the signal |
| `BD10m` / `BD10mPlus` | Balanced Detector with 10 m of additional fiber delay / more than 10 m |

### Accelerator / facility

| Acronym | Meaning |
|---------|---------|
| `AWA` | Argonne Wakefield Accelerator |
| `VXI11` | VXI-11 — network-based instrument control protocol (TCP/IP equivalent of GPIB) |
| `Tek1` | Tektronix oscilloscope #1 |

## Generated figures

- `BPDwf_combined.png` — combined BPD waveform plot.
- `coarsetiming.png`, `coarsetiming_SD.png` — coarse-timing results (baseline and SD configuration).
- `timingandscan.png` — timing/scan overview figure.
