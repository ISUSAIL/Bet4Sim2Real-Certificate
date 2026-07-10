"""Extract timestamped 'Searching for targetMarker' events from a
run_tests_nist_output log file, and resolve each one to the OTS mocap
take, frame number, and CSV line it falls in.
"""
import argparse
import csv
import linecache
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

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
    return takes


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


def annotate_with_take(events, takes, body_name="EOAT"):
    xy_columns_cache = {}
    for e in events:
        take = match_take(e["ros_time"], takes)
        if take is None:
            e.update(run_name=None, frame_number=None, csv_line=None, eoat_x_mm=None, eoat_y_mm=None)
            continue
        offset_sec = e["ros_time"] - take["start_epoch"]
        frame_number = round(offset_sec * take["frame_rate"])
        csv_line = frame_number + CSV_HEADER_LINES + 1

        if take["path"] not in xy_columns_cache:
            xy_columns_cache[take["path"]] = find_rigid_body_xy_columns(take["path"], body_name)
        x_col, y_col = xy_columns_cache[take["path"]]
        eoat_x_mm, eoat_y_mm = read_xy_at_line(take["path"], csv_line, x_col, y_col)

        e.update(
            run_name=take["name"],
            frame_number=frame_number,
            csv_line=csv_line,
            eoat_x_mm=eoat_x_mm,
            eoat_y_mm=eoat_y_mm,
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

    events = parse_events(args.log_path)
    takes = load_takes(ots_dir)
    events = annotate_with_take(events, takes)

    for e in events:
        print(
            f"line {e['line_no']}: t={e['ros_time']:.6f}  marker_id={e['marker_id']}  "
            f"run={e['run_name']}  frame={e['frame_number']}  csv_line={e['csv_line']}  "
            f"eoat_x_mm={e['eoat_x_mm']}  eoat_y_mm={e['eoat_y_mm']}"
        )

    if args.output:
        fieldnames = [
            "line_no",
            "event",
            "ros_time",
            "marker_id",
            "run_name",
            "frame_number",
            "csv_line",
            "eoat_x_mm",
            "eoat_y_mm",
        ]
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)
        print(f"\nWrote {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
