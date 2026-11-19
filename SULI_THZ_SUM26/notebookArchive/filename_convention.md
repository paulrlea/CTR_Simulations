## Scan Filename Convention

Filenames follow the pattern:

**`{ScanType}_{Charge}_{NDfilters}_{DateTime}[_{Step}].npy/.npz`**

| Field | Position | Format | Meaning |
|-------|----------|--------|---------|
| **Scan type** | 1st | `FineScan` or `CoarseScan` | Resolution/range of the delay stage sweep |
| **Charge** | 2nd | `#nC`, `Q#nC`, or `#p#nC` | Bunch charge (`p` = decimal point, `Q` prefix optional) |
| **ND filter wheels** | 3rd | `#_#` | Position of each of two neutral-density filter wheels (7 positions each) |
| **Date/time** | 4th | `YYYY_M_D_H_m` | Scan start time |
| **Step number** | 5th *(days 1–2 only)* | integer | Index of individual phase step (before files were combined) |

**Note:** On days 1–2 (July 30–31), each phase step was saved as a separate `.npy` file. A post-processing script later merged them into single `.npz` files. From day 3 onward, data was saved directly as combined `.npz` archives.
