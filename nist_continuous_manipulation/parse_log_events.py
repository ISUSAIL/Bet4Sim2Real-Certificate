"""Extract timestamped 'Searching for targetMarker' events from a
run_tests_nist_output log file, and resolve each one to the OTS mocap
take, frame number, and CSV line it falls in.
"""
import argparse
import csv
import linecache
import re
import zipfile
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

SS_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

EVENT_PATTERNS = {
    "searching_for_marker": re.compile(
        r"\[INFO\]\s*\[(?P<ros_time>\d+\.\d+)\]:\s*Searching for targetMarker\s*(?P<marker_id>\d+)"
    ),
}

# Number of header lines in the OTS CSV before the "Frame 0" data row
# (Format Version / blank / Type / Name / ID / Rotation-Position / Frame,Time header).
CSV_HEADER_LINES = 7

# The OTS capture start times are recorded in local wall-clock time at NIST (Maryland).
OTS_TZ = ZoneInfo("America/New_York")


def parse_events(log_path: Path):
    events = []
    with log_path.open("r", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = ANSI_RE.sub("", raw_line)
            for event_name, pattern in EVENT_PATTERNS.items():
                match = pattern.search(line)
                if not match:
                    continue
                events.append(
                    {
                        "line_no": line_no,
                        "event": event_name,
                        "ros_time": float(match.group("ros_time")),
                        "marker_id": int(match.group("marker_id")),
                    }
                )
    events.sort(key=lambda e: e["line_no"])
    return events


def _read_take_header(csv_path: Path):
    with csv_path.open("r", errors="replace") as f:
        first_line = f.readline()
    fields = first_line.strip().split(",")
    info = dict(zip(fields[0::2], fields[1::2]))

    start_dt = datetime.strptime(info["Capture Start Time"], "%Y-%m-%d %I.%M.%S.%f %p")
    start_dt = start_dt.replace(tzinfo=OTS_TZ)

    frame_rate = float(info["Capture Frame Rate"])
    total_frames = int(info["Total Frames in Take"])
    return {
        "path": csv_path,
        "name": csv_path.stem,
        "start_epoch": start_dt.timestamp(),
        "frame_rate": frame_rate,
        "total_frames": total_frames,
        "duration": total_frames / frame_rate,
    }


def load_takes(ots_dir: Path):
    takes = [_read_take_header(p) for p in ots_dir.glob("*.csv")]
    takes.sort(key=lambda t: t["start_epoch"])
    # Takes are captured in run order, so chronological position == run number.
    for run_number, take in enumerate(takes, start=1):
        take["run_number"] = run_number
    return takes


def _xlsx_rows(xlsx_path: Path):
    """Yield each worksheet row of a single-sheet .xlsx as a {column_letter: value} dict."""
    with zipfile.ZipFile(xlsx_path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.iter(f"{SS_NS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{SS_NS}t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in sheet.iter(f"{SS_NS}row"):
        cells = {}
        for c in row.findall(f"{SS_NS}c"):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find(f"{SS_NS}v")
            val = v.text if v is not None else None
            if c.get("t") == "s" and val is not None:
                val = shared[int(val)]
            cells[col] = val
        rows.append(cells)
    return rows


def parse_run_order(xlsx_path: Path):
    """Parse the run-order workbook into {replicate_number: {run_number: side}}.

    The sheet is laid out as repeated blocks: a 'Replicate N' banner row, a
    'Run | Search Method | Base speed | Side' header, then the 8 run rows.
    """
    result = {}
    current_rep = None
    for cells in _xlsx_rows(xlsx_path):
        a = (cells.get("A") or "").strip()
        rep_match = re.fullmatch(r"Replicate (\d+)", a)
        if rep_match:
            current_rep = int(rep_match.group(1))
            result[current_rep] = {}
        elif current_rep is not None and a.isdigit():
            side = cells.get("D")
            if side is not None:
                result[current_rep][int(a)] = int(side)
    return result


def match_take(ros_time: float, takes):
    for take in takes:
        if take["start_epoch"] <= ros_time <= take["start_epoch"] + take["duration"]:
            return take
    return None


def find_rigid_body_xy_columns(csv_path: Path, body_name: str):
    """Return (x_col, y_col) indices for a named Rigid Body's Position X/Y,
    excluding that body's individual "Rigid Body Marker" columns."""
    type_row = linecache.getline(str(csv_path), 3).rstrip("\n").split(",")
    name_row = linecache.getline(str(csv_path), 4).rstrip("\n").split(",")
    kind_row = linecache.getline(str(csv_path), 6).rstrip("\n").split(",")
    axis_row = linecache.getline(str(csv_path), 7).rstrip("\n").split(",")

    x_col = y_col = None
    for i, (t, n, k, a) in enumerate(zip(type_row, name_row, kind_row, axis_row)):
        if t == "Rigid Body" and n == body_name and k == "Position":
            if a == "X":
                x_col = i
            elif a == "Y":
                y_col = i
    if x_col is None or y_col is None:
        raise ValueError(f"Could not find Position X/Y columns for rigid body {body_name!r} in {csv_path}")
    return x_col, y_col


def read_xy_at_line(csv_path: Path, csv_line: int, x_col: int, y_col: int):
    row = linecache.getline(str(csv_path), csv_line).rstrip("\n").split(",")
    return float(row[x_col]), float(row[y_col])


# Physical marker name = f"Marker{base + targetMarker index}", with base per side:
#   Side 1: Marker2..Marker7  (idx 0..5)
#   Side 2: Marker8..Marker13 (idx 0..5)
SIDE_MARKER_BASE = {1: 2, 2: 8}


def marker_name_for(side, marker_id):
    base = SIDE_MARKER_BASE.get(side)
    if base is None:
        return None
    return f"Marker{base + marker_id}"


def load_marker_gt(gt_path: Path):
    """Parse rmma_fiducials_gt.csv into {marker_name: (x_mm, y_mm)}.

    Columns are named 'DRMMA_GT:MarkerN.X' / '.Y'; each column has several
    rows and the ground-truth position is their mean.
    """
    with gt_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_map = {}  # column index -> (marker_name, axis)
        for i, h in enumerate(header):
            m = re.match(r"DRMMA_GT:(Marker\d+)\.([XY])", h.strip())
            if m:
                col_map[i] = (m.group(1), m.group(2))

        acc = {}  # marker_name -> {"X": [...], "Y": [...]}
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            for i, (marker, axis) in col_map.items():
                val = row[i].strip()
                if val:
                    acc.setdefault(marker, {"X": [], "Y": []})[axis].append(float(val))

    return {
        marker: (
            sum(d["X"]) / len(d["X"]) if d["X"] else None,
            sum(d["Y"]) / len(d["Y"]) if d["Y"] else None,
        )
        for marker, d in acc.items()
    }


def annotate_with_take(events, takes, run_side_map=None, marker_gt=None, body_name="EOAT"):
    run_side_map = run_side_map or {}
    marker_gt = marker_gt or {}
    xy_columns_cache = {}
    for e in events:
        take = match_take(e["ros_time"], takes)
        if take is None:
            e.update(
                run_name=None, frame_number=None, csv_line=None,
                eoat_x_mm=None, eoat_y_mm=None, side=None, marker_name=None,
                marker_gt_x_mm=None, marker_gt_y_mm=None,
            )
            continue
        offset_sec = e["ros_time"] - take["start_epoch"]
        frame_number = round(offset_sec * take["frame_rate"])
        csv_line = frame_number + CSV_HEADER_LINES + 1

        if take["path"] not in xy_columns_cache:
            xy_columns_cache[take["path"]] = find_rigid_body_xy_columns(take["path"], body_name)
        x_col, y_col = xy_columns_cache[take["path"]]
        eoat_x_mm, eoat_y_mm = read_xy_at_line(take["path"], csv_line, x_col, y_col)

        side = run_side_map.get(take["run_number"])
        marker_name = marker_name_for(side, e["marker_id"])
        marker_gt_x_mm, marker_gt_y_mm = marker_gt.get(marker_name, (None, None))
        e.update(
            run_name=take["name"],
            frame_number=frame_number,
            csv_line=csv_line,
            eoat_x_mm=eoat_x_mm,
            eoat_y_mm=eoat_y_mm,
            side=side,
            marker_name=marker_name,
            marker_gt_x_mm=marker_gt_x_mm,
            marker_gt_y_mm=marker_gt_y_mm,
        )
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to run_tests_nist_output_*.txt")
    parser.add_argument(
        "--ots-dir",
        type=Path,
        default=None,
        help="Path to the 'OTS/CSV/Custom Axis Convention' directory holding the take CSVs "
        "(default: derived from log_path's Replicate folder)",
    )
    default_run_order = Path(__file__).parent / "data" / "Continuous_Mobile_Manipulator_Experiment_Run_Order_06-07-2022.xlsx"
    parser.add_argument(
        "--run-order",
        type=Path,
        default=default_run_order,
        help=f"Run-order .xlsx used to map run number -> Side (default: {default_run_order})",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=None,
        help="Replicate number (default: parsed from the 'Replicate N' folder in log_path)",
    )
    default_gt = Path(__file__).parent / "data" / "rmma_fiducials_gt.csv"
    parser.add_argument(
        "--fiducials-gt",
        type=Path,
        default=default_gt,
        help=f"Ground-truth marker positions CSV (default: {default_gt})",
    )
    default_output = Path(__file__).parent / "data" / "events.csv"
    parser.add_argument(
        "-o", "--output", type=Path, default=default_output,
        help=f"CSV output path (default: {default_output})",
    )
    args = parser.parse_args()

    ots_dir = args.ots_dir
    if ots_dir is None:
        replicate_dir = args.log_path.parent.parent
        ots_dir = replicate_dir / "OTS" / "CSV" / "Custom Axis Convention"

    replicate = args.replicate
    if replicate is None:
        for part in args.log_path.parts:
            m = re.fullmatch(r"Replicate (\d+)", part)
            if m:
                replicate = int(m.group(1))
                break
    if replicate is None:
        parser.error("Could not infer replicate number from log_path; pass --replicate")

    run_side_map = parse_run_order(args.run_order).get(replicate, {})
    marker_gt = load_marker_gt(args.fiducials_gt)

    events = parse_events(args.log_path)
    takes = load_takes(ots_dir)
    events = annotate_with_take(events, takes, run_side_map=run_side_map, marker_gt=marker_gt)

    print(f"Replicate {replicate}  run->side: {run_side_map}")
    for e in events:
        print(
            f"line {e['line_no']}: t={e['ros_time']:.6f}  marker_id={e['marker_id']}  "
            f"run={e['run_name']}  side={e['side']}  marker={e['marker_name']}  "
            f"frame={e['frame_number']}  csv_line={e['csv_line']}  "
            f"eoat_x_mm={e['eoat_x_mm']}  eoat_y_mm={e['eoat_y_mm']}  "
            f"marker_gt_x_mm={e['marker_gt_x_mm']}  marker_gt_y_mm={e['marker_gt_y_mm']}"
        )

    if args.output:
        fieldnames = [
            "line_no",
            "event",
            "ros_time",
            "marker_id",
            "run_name",
            "side",
            "marker_name",
            "frame_number",
            "csv_line",
            "eoat_x_mm",
            "eoat_y_mm",
            "marker_gt_x_mm",
            "marker_gt_y_mm",
        ]
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)
        print(f"\nWrote {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
