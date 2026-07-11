# The certificate for NIST Continuous Manipulation Task

This module computes the **end-effector positioning error** for the NIST
Continuous Mobile Manipulator experiment. For every "Searching for
targetMarker" event in a run, it resolves where the arm's end-effector
(EOAT rigid body, tracked by the OTS motion-capture system) was at that
instant and compares it to the ground-truth fiducial marker position. The
per-marker L2 distance is then averaged over each run's 6 markers and written
into the factorial results table as a new `distance` column.

## Dataset

We provide 2 methods to obtain the dataset.

### 1. Download the preprocessed dataset (recommended)

`measurement.csv` — the already-processed results; no raw data or scripts
needed.

### 2. Download the raw dataset

Download the raw dataset (Continuous Mobile Manipulator Performance
Experiment 06-07-2022): https://data.nist.gov/od/id/mds2-3187

Then arrange the files into the layout below (paths are relative to this
`nist_continuous_manipulation/` folder). Only the four sources listed are
used by the code — the rest of the NIST archive (Rosbag, Videos, Matlab Data,
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

## Usage

Process **all 6 replicates** and fill the factorial CSV:

```bash
cd nist_continuous_manipulation
python3 data/fill_run_distances.py
```

This writes a per-event `events.csv` into each `data/raw/Replicate N/` folder
and adds/updates the `distance` column (average error in mm, one value per
run) in the factorial CSV. It prints a per-run summary and warns if a run's
`Side` in the factorial CSV disagrees with the run-order workbook.

Process a **single replicate** (also useful for inspecting individual events):

```bash
python3 data/parse_log_events.py "data/raw/Replicate 1/Program Output/run_tests_nist_output_06_07_2022_09_48_56.txt"
```

By default this writes `events.csv` into that replicate's folder. The OTS
directory, run-order workbook, ground-truth CSV, and output path can all be
overridden with flags (`--ots-dir`, `--run-order`, `--fiducials-gt`,
`-o/--output`); run with `-h` for details.

## How it works

1. **Parse events** — scan the run log for `Searching for targetMarker <id>`
   lines, recording the ROS timestamp and marker index (`0`–`5`).
2. **Match to a take** — OTS takes are sorted by capture start time (which
   equals run order, `run_number` 1–8); each event's timestamp is mapped to
   the take it falls in and to a frame/CSV line via the take's frame rate.
3. **Read the EOAT position** — the `EOAT` rigid body's Position X/Y columns
   are read from the take CSV at that frame.
4. **Look up the ground truth** — the run's `Side` (from the run-order
   workbook) selects the physical marker: Side 1 → `Marker2`–`Marker7`,
   Side 2 → `Marker8`–`Marker13`. Its ground-truth position is the mean of the
   `DRMMA_GT:MarkerN.X/.Y` samples in `rmma_fiducials_gt.csv`.
5. **Distance error** — `distance_error_mm = hypot(eoat - marker_gt)`, then
   averaged over a run's 6 markers for the factorial `distance` column.

## Outputs

**`data/raw/Replicate N/events.csv`** — one row per marker-search event:

| column | meaning |
| --- | --- |
| `line_no` | line in the run log where the event appeared |
| `event` | event type (`searching_for_marker`) |
| `ros_time` | ROS timestamp of the event |
| `marker_id` | targetMarker index (0–5) |
| `run_name` | OTS take the event falls in |
| `side` | robot side for this run (1 or 2) |
| `marker_name` | resolved physical marker (`Marker2`–`Marker13`) |
| `frame_number` | frame in the take at the event time |
| `csv_line` | line in the take CSV read for the EOAT position |
| `eoat_x_mm`, `eoat_y_mm` | end-effector position from the mocap take |
| `marker_gt_x_mm`, `marker_gt_y_mm` | ground-truth marker position |
| `distance_error_mm` | L2 distance between EOAT and ground truth |

**Factorial CSV** — the `distance` column holds the per-run mean of
`distance_error_mm` (mm), aligned by `(Replicate, Run)`.

## Notes

- **X-axis sign convention** — the `rmma_fiducials_gt.csv` export uses the
  opposite X-axis sign to the OTS take's "Custom Axis Convention" frame the
  EOAT is read from. `load_marker_gt` negates the ground-truth X to bring both
  into the same frame; without this the error is dominated by a spurious
  X offset (inflated to hundreds of mm, worst on Side 1 where |X| is large).
  After correction all errors fall in a realistic ~5–30 mm range.
- **Run-order alignment** — takes are assigned to runs purely by capture-start
  order. The batch script cross-checks each run's `Side` against the run-order
  workbook as a sanity check; all 6 replicates currently agree with no
  mismatches.
