# Using the SRW installation on this machine

**SRW** = *Synchrotron Radiation Workshop* (O. Chubar et al., BNL — `github.com/ochubar/SRW`).
It computes synchrotron/undulator/bending-magnet radiation and propagates **partially- or
fully-coherent wavefronts** through optical systems (Fourier optics). On this machine it is
used as the **wavefront-propagation engine** for the AWA CTR-THz study (`srw_thz/`).

This document describes the specific install in `~/Documents/1_Research/Suli2026` and how to
drive it. It is written against the state of the machine on 2026-06-25 — verify versions with
the commands in §1 if something looks off.

---

## 0. TL;DR

```bash
cd ~/Documents/1_Research/Suli2026
source .venv/bin/activate              # the PYTHONPATH-isolated project venv
python -c "from srwpy.srwlib import *; print(srwl.UtiVer())"   # -> 4.10

# run a bundled example (headless):
cd SRW/env/python/srwpy/examples
MPLBACKEND=Agg python SRWLIB_Example07.py

# the real worked example on this machine:
cd ~/Documents/1_Research/Suli2026/srw_thz
python run_thz.py            # full THz beamline run -> results/
```

Always import from the **`srwpy`** package (`from srwpy.srwlib import *`), not a bare
`from srwlib import *`. Always run inside the project `.venv`.

---

## 1. What is installed

| Component | Value |
|---|---|
| Python package | **`srwpy` 4.2.1** (PyPI prebuilt wheel) |
| Compiled C++ core | `srwlpy.so` — `srwl.UtiVer()` reports **"4.10"** |
| Python | **3.13.9** (system `/usr/bin/python3`, symlinked into the venv) |
| Virtualenv | `~/Documents/1_Research/Suli2026/.venv` |
| Runtime deps | numpy 2.5, scipy 1.18, matplotlib 3.11, h5py 3.16, pillow, scikit-learn |

The package lives at `.venv/lib64/python3.13/site-packages/srwpy/`. The key files are
`srwlib.py` (the ~690 KB Python API you import), `srwlpy.so` (the compiled FFT/propagation
core), plus helper modules `srwl_bl.py`, `srwl_uti_*.py`, `uti_plot*.py`, `uti_math.py`.

Verify the install at any time:

```bash
source .venv/bin/activate
pip show srwpy
python - <<'PY'
from srwpy import srwlpy, srwlib
print("core version:", srwlpy.UtiVer())          # 4.10
print("srwlib at   :", srwlib.__file__)
PY
```

### Two copies of SRW exist — know which one runs

1. **The installed, working one** → `.venv/.../site-packages/srwpy/` (the wheel, with the
   compiled `.so`). **This is what `import srwpy` uses.**
2. **A full source clone** → `SRW/` (~1.2 GB, the git repo). This is *source only* — it is
   **not** on the Python path and is **not** compiled in place. Use it for: reading C++/Python
   source, the bundled example scripts (`SRW/env/python/srwpy/examples/`), the built HTML docs
   (`SRW/docs/build/html/index.html`), and as the tree you'd `pip install .` from if you ever
   need to rebuild (see §10).

---

## 2. Directory layout

```
Suli2026/
├── .venv/                          # project virtualenv  ← always activate this
│   └── lib64/python3.13/site-packages/srwpy/   # the installed, compiled SRW
├── SRW/                            # full source clone (reference only, ~1.2 GB)
│   ├── cpp/                        #   C++ core source
│   ├── env/python/                 #   Python packaging + examples + tests
│   │   ├── srwpy/examples/         #   24 SRWLIB_Example*.py + data_example_*/
│   │   ├── tests/                  #   pytest suite (test_example01..21)
│   │   ├── setup.py, requirements*.txt
│   ├── docs/build/html/            #   pre-built Sphinx/Doxygen HTML docs
│   └── ext_lib/                    #   bundled FFTW + libs for source builds
└── srw_thz/                        # ★ the real project using SRW as a propagator
    ├── thz_config.py  thz_source.py  thz_beamline.py
    ├── thz_extract.py  run_thz.py    thz_verify.py ...
    └── README.md  WRITEUP.md  FINDINGS.md
```

---

## 3. Quick start: minimal end-to-end script

This is the canonical SRW wavefront-propagation idiom (Gaussian beam → aperture → drift →
intensity), which mirrors `SRWLIB_Example07.py` and `srw_thz/thz_source.py`:

```python
from array import array
from srwpy.srwlib import (SRWLGsnBm, SRWLWfr, SRWLOptA, SRWLOptD,
                          SRWLOptL, SRWLOptC, srwl)

# 1) ---- define a source (here a Gaussian beam) -------------------------------
gsn = SRWLGsnBm()
gsn.x = gsn.y = gsn.z = 0.0          # waist position
gsn.xp = gsn.yp = 0.0               # angles
gsn.avgPhotEn = 4.136e-4            # photon energy [eV]  (here ~100 GHz THz)
gsn.pulseEn   = 1.0                 # pulse energy [J] (arbitrary -> relative units)
gsn.polar     = 1                  # 1 = linear horizontal
gsn.sigX = gsn.sigY = 2.0e-3        # rms waist size [m]
gsn.mx = gsn.my = 0                # transverse Gauss-Hermite mode orders

# 2) ---- allocate the wavefront mesh and compute the initial field ------------
wfr = SRWLWfr()
wfr.allocate(1, 100, 100)           # (n_photon_energies, nx, ny)
wfr.mesh.zStart = 1.0               # plane where the field is first defined [m]
wfr.mesh.eStart = wfr.mesh.eFin = gsn.avgPhotEn
wfr.mesh.xStart, wfr.mesh.xFin = -0.02, 0.02     # [m]
wfr.mesh.yStart, wfr.mesh.yFin = -0.02, 0.02
# copy source moments into the wavefront's particle-beam slot (examples do this):
wfr.partBeam.partStatMom1.x  = gsn.x
wfr.partBeam.partStatMom1.y  = gsn.y
wfr.partBeam.partStatMom1.z  = gsn.z
wfr.partBeam.partStatMom1.xp = gsn.xp
wfr.partBeam.partStatMom1.yp = gsn.yp

sampFact = 1.0                      # mesh oversampling (bigger = finer & heavier)
srwl.CalcElecFieldGaussian(wfr, gsn, [sampFact])

# 3) ---- build the beamline: elements + per-element propagation params --------
ap   = SRWLOptA('c', 'a', 0.05)     # circular ('c') aperture ('a'), 50 mm diameter
lens = SRWLOptL(0.5, 0.5)           # ideal thin lens, fx = fy = 0.5 m
drift= SRWLOptD(0.5)                # 0.5 m of free space

pp_def = [1, 1, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]   # see §5
optBL = SRWLOptC([ap, lens, drift],
                 [pp_def, pp_def, pp_def])

# 4) ---- propagate -----------------------------------------------------------
srwl.PropagElecField(wfr, optBL)    # mutates wfr in place

# 5) ---- extract a 2D intensity map ------------------------------------------
arI = array('f', [0] * (wfr.mesh.nx * wfr.mesh.ny))
srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
# arI is now a flat float array (row-major, ny rows × nx cols)
```

---

## 4. The core API, object by object

Everything hangs off the `srwl` driver object and a handful of structures from `srwpy.srwlib`.

### Sources
- **`SRWLGsnBm`** — coherent Gaussian beam. Fields: `x,y,z,xp,yp`, `avgPhotEn` [eV],
  `pulseEn` [J], `polar` (1=lin-H, 2=lin-V, 3/4=±45°, 5/6=circ), `sigX,sigY` [m],
  `sigT` [s], `mx,my` (mode orders). Computed by `srwl.CalcElecFieldGaussian`.
- **`SRWLPartBeam` / `SRWLPartBeamStatMom`** — a real electron beam (energy, current,
  emittance, Twiss/moments). Used with magnetic fields for true SR.
- **`SRWLMagFldU` / `SRWLMagFldC` / `SRWLMagFldM`** — undulator / container / multipole
  (e.g. bending magnet) magnetic fields. Drive `srwl.CalcElecFieldSR`,
  `srwl.CalcStokesUR`, `srwl.CalcPowDenSR`, etc.

> Note: SRW has **no transition-radiation source**. `srw_thz/` synthesizes the CTR source
> (Gaussian surrogate, with a planned Ginzburg–Frank analytic field) and feeds it in as a
> custom wavefront — see §8.

### Wavefront
- **`SRWLWfr`** — the electric field on a mesh. `wfr.allocate(ne, nx, ny)` then set
  `wfr.mesh.{eStart,eFin, xStart,xFin, yStart,yFin, zStart}`. After computation the complex
  field lives in `wfr.arEx` / `wfr.arEy` as interleaved `[Re, Im, Re, Im, …]` floats,
  indexed `arEx[2*(ix + iy*nx)]`. `wfr.mesh.{nx,ny}` give the (possibly auto-resized) grid.

### Optical elements (build a beamline by listing these)
| Class | Element |
|---|---|
| `SRWLOptD(L)` | drift space of length `L` [m] |
| `SRWLOptA(shape, ap_or_obst, Dx, Dy=0)` | aperture/obstacle — `'c'`/`'r'` shape, `'a'`/`'o'` |
| `SRWLOptL(Fx, Fy, x=0, y=0)` | ideal thin lens (focal lengths [m]) — used for OAP mirrors |
| `SRWLOptT` | generic transmission element (amplitude + optical-path map) |
| `SRWLOptMir*`, `SRWLOptG`, `SRWLOptCryst` | mirrors, gratings, crystals |
| **`SRWLOptC([elems...], [pp...])`** | **the container** = ordered elements + their propagation params |

### The driver `srwl`
| Call | Purpose |
|---|---|
| `srwl.CalcElecFieldGaussian(wfr, gsn, [sampFact])` | fill `wfr` from a Gaussian beam |
| `srwl.CalcElecFieldSR(wfr, eBeam, magFld, prec)` | fill `wfr` from electron beam + field (true SR) |
| `srwl.PropagElecField(wfr, optC)` | propagate `wfr` through the beamline (in place) |
| `srwl.CalcIntFromElecField(arI, wfr, pol, type, dep, e, x, y)` | extract intensity/phase/field (see §6) |
| `srwl.CalcStokesUR`, `srwl.CalcPowDenSR` | Stokes params / power density |
| `srwl.UtiVer()` | core version string |

---

## 5. Propagation parameters (the `pp` arrays) — the part everyone gets wrong

Each optical element in an `SRWLOptC` gets a **parameter list** controlling resampling around
that element. The documented slots (12 commonly used; the core accepts up to 17):

```
[0]  Auto-resize mesh BEFORE propagation (0/1)
[1]  Auto-resize mesh AFTER  propagation (0/1)
[2]  Relative precision for auto-resizing (1.0 = nominal)
[3]  Semi-analytic treatment of the quadratic (leading) phase term (0/1)  ← set 1 on drifts
[4]  Resize on the Fourier side via FFT (0/1)
[5]  Horizontal RANGE      modification factor (1.0 = unchanged)
[6]  Horizontal RESOLUTION modification factor
[7]  Vertical   RANGE      modification factor
[8]  Vertical   RESOLUTION modification factor
[9]  Type of wavefront shift before resizing (not implemented)
[10] New horizontal center after shift (not implemented)
[11] New vertical   center after shift (not implemented)
```

Practical recipes (these are exactly what `srw_thz/thz_beamline.py` uses):

```python
# auto-resize on, semi-analytic phase on — a safe general default:
PP_DRIFT  = [1, 1, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
# refine resolution at the detector by 1.3× in each axis:
PP_FINAL  = [1, 1, 1.0, 1, 0, 1.0, 1.3, 1.0, 1.3, 0, 0, 0]
# auto-resize OFF, grow range a fixed 2× per drift (bounded mesh — see §7):
PP_CAP_DR = [0, 0, 1.0, 1, 0, 2.0, 1.0, 2.0, 1.0, 0, 0, 0]
```

**Key tradeoff:** auto-resize (`[0]=[1]=1`) keeps the beam contained and sampled as it
diffracts, but at long wavelengths (THz) it can balloon the mesh to tens of millions of
points or hang. The fixed-growth, auto-resize-off recipe bounds the mesh at the cost of
mildly clipping deeply-diffracting low-frequency beams. See `srw_thz/README.md` §"Resolution
& performance" for the full discussion.

---

## 6. Extracting results: `CalcIntFromElecField`

```python
srwl.CalcIntFromElecField(arOut, wfr, pol, intType, depType, e, x, y)
```

- **`pol`** (polarization): `0`=lin-H, `1`=lin-V, … `6`=**total** (most common for intensity).
- **`intType`**: `0`=single-electron (coherent), `1`=multi-e, `4`=phase, `5`=Re(E), `6`=Im(E), …
  (For a coherent Gaussian/THz field, `0` for components, `6` for total intensity per examples.)
- **`depType`**: what the output array varies over — `0`=vs energy, `1`=vs x, `2`=vs y,
  `3`=**vs x AND y (2D map)**, …
- **`e, x, y`**: the fixed coordinates for the cut (use `wfr.mesh.eStart`, `0`, `0`).
- **`arOut`**: a pre-sized `array('f', [0]*N)`. For a 2D map `N = wfr.mesh.nx*wfr.mesh.ny`.

Common patterns from this codebase:

```python
# 2D total-intensity map:
arI = array('f', [0]*(wfr.mesh.nx*wfr.mesh.ny))
srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)

# complex on-axis field via Re/Im (srw_thz/thz_extract.py):
reE = array('f', [0]*(nx*ny)); imE = array('f', [0]*(nx*ny))
srwl.CalcIntFromElecField(reE, wfr, 0, 5, 0, wfr.mesh.eStart, 0., 0.)  # Re(E), lin-H
srwl.CalcIntFromElecField(imE, wfr, 0, 6, 0, wfr.mesh.eStart, 0., 0.)  # Im(E), lin-H

# or read the raw interleaved field directly:
import numpy as np
arEx = np.asarray(wfr.arEx, dtype=float).reshape(wfr.mesh.ny, wfr.mesh.nx, 2)
Ex_complex = arEx[..., 0] + 1j*arEx[..., 1]
```

Build the physical axes with
`np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx)` (and likewise for y).

---

## 7. Running the bundled examples

24 example scripts ship in `SRW/env/python/srwpy/examples/` with their data folders:

```bash
source ~/Documents/1_Research/Suli2026/.venv/bin/activate
cd ~/Documents/1_Research/Suli2026/SRW/env/python/srwpy/examples
MPLBACKEND=Agg python SRWLIB_Example07.py        # headless (no window)
python SRWLIB_ExamplesRunAll.py                  # run them all
```

What each one demonstrates:

| Script | Topic |
|---|---|
| Example01 | Electron trajectory in a 3D APPLE-II undulator field |
| Example02 | Trajectory in a segmented planar undulator |
| Example03 | Undulator radiation from one electron (ellipsoidal motion) |
| Example04 | Undulator radiation **electric field** (single electron) |
| Example05 | Trajectory + spontaneous emission |
| Example06 (+_PETRA) | **Spectral flux** of undulator radiation |
| **Example07** | **Gaussian X-ray beam through optics (CRL)** ← best propagation template |
| Example08 | Partially-coherent UR focusing with a CRL |
| Example09 | Gaussian beam through a beamline with imperfect mirrors |
| Example10 | Emission + propagation of partially-coherent undulator radiation |
| Example11 | Spectral flux for a finite-emittance beam |
| Example12 | Wavefront propagation, soft-X-ray UR |
| Example13 | Bending-magnet SR through a simple optical scheme |
| Example14 | Gaussian beam through a simple scheme |
| Example15–16 | RadTrack-project examples (T. Shaftan) |
| Example17–19 | Coherent X-ray (Gaussian) scattering, incl. 3D nanoparticle ensembles |
| Example20 | 4D cross-spectral density / mutual intensity (partial coherence) |
| Example21 | Wavefront propagation through an optical layout |

For propagation work (the THz use case), **Example07, 09, 14, 21** are the most relevant
templates. For true source modeling, **Example06/11 (flux)** and **04/10 (field+propagation)**.

There is also a pytest suite in `SRW/env/python/tests/` (`test_example01..21`, `test_imports`)
you can run from `SRW/env/python/` to sanity-check the install:
```bash
cd ~/Documents/1_Research/Suli2026/SRW/env/python && python -m pytest tests/test_imports.py
```

---

## 8. The worked example on this machine: `srw_thz/`

`srw_thz/` is a complete, documented project that uses SRW purely as the propagation engine
for the AWA 1550-nm EO-sampling THz experiment. It is the best on-machine reference for
"how do I actually structure an SRW study." Read `srw_thz/README.md` and `WRITEUP.md` first.

Beamline: radiative foil (CTR source) → collimator aperture → sapphire window → OAP1
(collimating) → OAP2 (focusing) → EO-sensor plane.

```bash
cd ~/Documents/1_Research/Suli2026/srw_thz
source ../.venv/bin/activate
python run_thz.py                                 # full run -> results/
python run_thz.py --src gaussian --nfreq 161      # explicit knobs
python run_thz.py --window-mode wedge --out wedge # etalon-mitigation variant
python thz_layout.py                              # beampath schematic only (no SRW run)
python thz_etalon.py                              # sapphire-etalon metrics/plots (no SRW run)
python thz_verify.py                              # Airy far-field propagation sanity check
```

Module map (each file is a clean illustration of one SRW concept):

| File | What it shows |
|---|---|
| `thz_source.py` | building an `SRWLGsnBm`/`SRWLWfr` source per frequency; a memory guard on `sampFact` |
| `thz_beamline.py` | assembling `SRWLOptC` from `SRWLOptA/D/L` with `pp` arrays; complex per-frequency transmission |
| `thz_extract.py` | every `CalcIntFromElecField` pattern + reading raw `wfr.arEx` |
| `run_thz.py` | a real driver: argparse CLI, per-frequency monochromatic sweep, assembly |
| `thz_config.py` | a dataclass holding all parameters (`default_config(**overrides)`) |

Key modeling trick worth copying: a **flat window is a spatial no-op**, so it is modeled as
(a) a clear aperture in the SRW beamline plus (b) an analytic complex transmission applied
per-frequency *outside* SRW — exact, because a global complex factor commutes with linear
propagation. This is how the sapphire Fabry-Pérot etalon is reproduced cheaply.

---

## 9. Plotting & headless operation

- **Headless (no display / over SSH / in scripts):** set `MPLBACKEND=Agg`. SRW's own
  `uti_plot` helpers (`uti_plot1d`, `uti_plot2d`, `uti_plot_show`) wrap matplotlib; with the
  Agg backend they render to files instead of windows. `srw_thz` forces Agg in `run_thz.py`.
- **Interactive:** just `python SRWLIB_Example07.py` from a desktop session — windows pop up.
- Intensity arrays from `CalcIntFromElecField` are flat `array('f', …)`; reshape to
  `(ny, nx)` for `matplotlib.pyplot.imshow`, using the mesh extents for `extent=`.

---

## 10. Rebuilding SRW from source (only if you must)

The installed wheel already has a working compiled core, so **you normally never need this.**
You'd only rebuild to modify the C++ core or get a different version. Source build needs the
Python headers, which are **not currently installed**:

```bash
sudo dnf install python3-devel          # provides Python.h (currently missing)
cd ~/Documents/1_Research/Suli2026/SRW/env/python
source ../../../.venv/bin/activate
pip install .                            # CMake builds bundled FFTW (ext_lib/) + the core
```

If you only need a clean reinstall of the working version, just:
```bash
pip install --force-reinstall srwpy==4.2.1
```

---

## 11. Gotchas & troubleshooting

- **Always activate the venv.** `source .venv/bin/activate`. The venv also shields you from
  a past system-wide `PYTHONPATH` pollution issue (Hyprland `ml4w.conf` forced system
  site-packages onto `PYTHONPATH`, which broke venv imports). If imports pull the wrong
  numpy/scipy, check `echo $PYTHONPATH` is empty.
- **Import path:** use `from srwpy.srwlib import *` (package-qualified). A bare
  `import srwlib` only works from inside the package dir and will bite you.
- **`ModuleNotFoundError: srwlpy` or a `.so` load error** → you're on the wrong Python
  (not the venv) or on a machine without the compiled core. Confirm with
  `python -c "from srwpy import srwlpy; print(srwlpy.UtiVer())"`.
- **Runs hang or eat all RAM at low frequency / long wavelength.** This is SRW auto-resize
  ballooning the mesh as the beam diffracts. Mitigations: turn auto-resize off and grow the
  range a fixed factor per drift (`pp[0]=pp[1]=0`, `pp[5]=pp[7]=2.0`); cap the source
  sampling factor; or crop the frequency range. See `srw_thz` `README.md` and `thz_source.py`'s
  `_guarded_sampfact`.
- **Blank plots in scripts** → you forgot `MPLBACKEND=Agg` (or to call `uti_plot_show()`).
- **Units:** lengths in **metres**, photon energy in **eV** (THz → use
  `E[eV] = 4.1357e-15 × f[Hz]`), angles in **radians**. Intensity is in SRW's
  ph/s/mm²/0.1%bw conventions for SR sources, or relative units for a normalized Gaussian.

---

## 12. Further documentation

- **Pre-built local HTML docs:** open `SRW/docs/build/html/index.html` in a browser
  (also `installation.html`, `usage.html`, plus a Doxygen C++ API and a Sphinx-gallery
  cookbook). These were generated with the `sphinx`/`breathe`/`numpydoc` toolchain in the venv.
- **In-code reference:** `srwlib.py` docstrings are thorough — e.g. `help(SRWLOptC)` documents
  every `pp` slot; `help(srwl.CalcIntFromElecField)` documents the extraction flags.
- **Upstream:** `github.com/ochubar/SRW` (README.txt in `SRW/` mirrors it), and the SRW papers
  linked from `SRW/docs/build/html/literature.html`.
- **On-machine worked example & physics notes:** `srw_thz/README.md`, `srw_thz/WRITEUP.md`,
  `srw_thz/FINDINGS.md`.
