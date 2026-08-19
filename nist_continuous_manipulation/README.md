# NIST Continuous Manipulation — Peg-in-Hole Task

This folder replays a standardized-procedure mobile-manipulator test and applies
the certificate methods to the recorded outcomes (category **C3** in the paper).
A wheeled mobile manipulator performs a peg-in-hole task (**Fig. 3c**).

The outcomes come from the open NIST Continuous Mobile Manipulator Performance
Measurement Dataset:

Aboul-Enein, Medeiros, Shah, Li-Baboud, Bostelman, Virts. *Continuous Mobile
Manipulator Performance Measurement Data.* National Institute of Standards and
Technology, 2024.

- Publication: https://www.nist.gov/publications/continuous-mobile-manipulator-performance-measurement-data
- Data: https://data.nist.gov/od/id/mds2-3187

## Reproducibility

### Measures

Three measures are certified, one row per run in `data/measurement.csv`
(48 runs):

| column | measure | normalization bounds | reported in |
| --- | --- | --- | --- |
| `Interception Rate` | fraction of trials where the end effector intercepts the reflector target | `[0, 1]` | percentage points |
| `Search Time` | time to complete one interception | `[0, 12]` | seconds |
| `distance` | average per-marker `l2` distance between the end effector and the ground-truth fiducial position | `[0, 30]` | millimetres |

`distance` is measured between the EOAT rigid body, tracked by the OTS
motion-capture system, and the ground-truth marker position, averaged over a
run's markers.

Normalization is applied only while computing certificates; reported and plotted
widths are converted back to the units above. The bounds are fixed a priori, not
read off the data.

### Run

```bash
python demo.py
python plot_demo.py
```

### Outputs

Everything is written to `data/`:

| file | contents | paper figure |
| --- | --- | --- |
| `normalization_metadata.csv` | bounds plus raw and normalized statistics per measure | — |
| `certificate_widths.csv` | per-`n` bounds and widths for every method and bank | — |
| `summary.csv` | final and mean widths, plus snapshots at n = 5, 10, 20, 30, 48 | — |
| `width_curves.png` | certificate width vs. samples | **Fig. 6a** |
| `normalized_sequences.png` | the normalized measurement sequences | — |

### Dataset

There are two ways to obtain the data.

**1. Use the preprocessed measurements (recommended).** `data/measurement.csv`
is the already-processed result; no raw data or scripts needed.

**2. Rebuild from the raw NIST archive.** Download *Continuous Mobile
Manipulator Performance Experiment 06-07-2022* from
https://data.nist.gov/od/id/mds2-3187 and arrange the files as below. Only these
four sources are used — the rest of the archive (Rosbag, Videos, Matlab Data,
Time Synchronization) is not needed.

| NIST source (under `Nist/mds2-3187/Continuous Mobile Manipulator Experiment 06-07-2022/`) | Put it at |
| --- | --- |
| `Analysis/Continuous_Mobile_Manipulator_Experiment_Factorial_06-07-2022_By_Run_Order.csv` | `data/` |
| `Pre-Test_Data/Cart_Transporter_Map_to_OTS_Registration_5-12-2022/OTS_RMMA_GT 05-12-2022/rmma_fiducials_gt.csv` | `data/raw/` |
| `Data/Continuous_Mobile_Manipulator_Experiment_Run_Order_06-07-2022.xlsx` | `data/raw/` |
| `Data/Replicate 1` … `Replicate 6` (the per-replicate run folders) | `data/raw/` |

Resulting layout the scripts expect:

```
data/
├── Continuous_Mobile_Manipulator_Experiment_Factorial_06-07-2022_By_Run_Order.csv   # filled in place (output)
├── parse_log_events.py
├── fill_run_distances.py
└── raw/
    ├── rmma_fiducials_gt.csv                                            # ground-truth marker positions
    ├── Continuous_Mobile_Manipulator_Experiment_Run_Order_06-07-2022.xlsx   # run -> Side mapping
    └── Replicate 1 … Replicate 6/
        ├── Program Output/run_tests_nist_output_*.txt                  # run log (event timestamps)
        └── OTS/CSV/Custom Axis Convention/Take *.csv                   # mocap takes (EOAT trajectory)
```

Then process all 6 replicates and fill the factorial CSV:

```bash
python3 data/fill_run_distances.py
```

This writes a per-event `events.csv` into each `data/raw/Replicate N/` folder and
adds or updates the `distance` column (average error in mm, one value per run) in
the factorial CSV. It prints a per-run summary and warns if a run's `Side` in the
factorial CSV disagrees with the run-order workbook.

The `distance` pipeline works as follows:

1. **Parse events** — scan the run log for `Searching for targetMarker <id>`
   lines, recording the ROS timestamp and marker index (`0`–`5`).
2. **Match to a take** — OTS takes are sorted by capture start time (which equals
   run order, `run_number` 1–8); each event's timestamp is mapped to the take it
   falls in and to a frame/CSV line via the take's frame rate.
3. **Read the EOAT position** — the `EOAT` rigid body's Position X/Y columns are
   read from the take CSV at that frame.
4. **Look up the ground truth** — the run's `Side` (from the run-order workbook)
   selects the physical marker: Side 1 → `Marker2`–`Marker7`, Side 2 →
   `Marker8`–`Marker13`. Its ground-truth position is the mean of the
   `DRMMA_GT:MarkerN.X/.Y` samples in `rmma_fiducials_gt.csv`.
5. **Distance error** — `distance_error_mm = hypot(eoat - marker_gt)`, then
   averaged over a run's 6 markers for the factorial `distance` column.

## Method

No simulator is matched to this platform — the sim-to-real methods are supplied 
with the synthetic banks from **C1**.

- `CONFIDENCE = 0.95`, `HORIZONS = (5, 10, 20, 30, 48)`.
- Vincent et al. is run at gaps 0.05 / 0.10 / 0.20 / 0.30, against
  `BetaSkewed(5, 1)` for `Interception Rate`, `BetaSkewed(2, 3)` for
  `Search Time`, and `BetaSkewed(2, 2)` for `distance`.

## Notes

- **X-axis sign convention** — the `rmma_fiducials_gt.csv` export uses the
  opposite X-axis sign to the OTS take's "Custom Axis Convention" frame the EOAT
  is read from. `load_marker_gt` negates the ground-truth X to bring both into
  the same frame; without this the error is dominated by a spurious X offset
  (inflated to hundreds of mm, worst on Side 1 where `|X|` is large). After
  correction all errors fall in a realistic ~5–30 mm range.
- **Run-order alignment** — takes are assigned to runs purely by capture-start
  order. The batch script cross-checks each run's `Side` against the run-order
  workbook as a sanity check; all 6 replicates currently agree with no
  mismatches.
