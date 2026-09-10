"""Command line interface for the Student Exercise Monitoring System.

Runs a monitored session in an OpenCV window, saves it to SQLite and writes
the PDF/CSV report. This is the no-Streamlit path, and the easiest way to
process the sample clips in videos/.

Examples
--------
    python run.py --student-id S001 --name "Asha Rao" --exercise squat --reps 5
    python run.py --student-id S002 --name "Dev Patel" --exercise pushup \
        --reps 5 --source videos/count_pushup.mp4
"""

import argparse
import sys

import cv2 as cv

from config import ALL_EXERCISES, get_exercise
from database import init_db, save_session
from report import format_console_report, save_reports
from videoStream import CaptureError, ExerciseSession


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor and count a student's exercise repetitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--student-id", required=True, help="student identifier")
    parser.add_argument("--name", required=True, help="student's full name")
    parser.add_argument(
        "--exercise",
        required=True,
        choices=sorted(ALL_EXERCISES),
        help="exercise to monitor",
    )
    parser.add_argument(
        "--reps", type=int, required=True, help="required number of valid repetitions"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="webcam index (e.g. 0) or path to a video file",
    )
    parser.add_argument("--db", default=None, help="override the SQLite database path")
    parser.add_argument(
        "--reports-dir", default=None, help="override the report output directory"
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="process without showing a window (useful for batch runs)",
    )
    parser.add_argument(
        "--no-skeleton", action="store_true", help="hide the pose skeleton overlay"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.reps < 1:
        print("error: --reps must be at least 1", file=sys.stderr)
        return 2

    try:
        exercise = get_exercise(args.exercise)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Exercise      : {exercise.name}")
    print(f"Student       : {args.name} ({args.student_id})")
    print(f"Required reps : {args.reps} valid")
    print(f"Source        : {args.source}")
    if not args.no_window:
        print("Press ESC or Q to stop early.\n")

    window = f"{exercise.name} - {args.name}"

    try:
        with ExerciseSession(
            student_id=args.student_id,
            student_name=args.name,
            exercise=exercise,
            required=args.reps,
            source=args.source,
            draw_skeleton=not args.no_skeleton,
        ) as session:
            for frame, _ in session.frames():
                if args.no_window:
                    continue

                cv.imshow(window, frame)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC or Q
                    session.stop()

            record = session.to_record()
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    finally:
        cv.destroyAllWindows()

    init_db(args.db)
    record.session_id = save_session(record, args.db)
    paths = save_reports(record, args.reports_dir)

    print()
    print(format_console_report(record))
    print(f"\nSaved to database (session_id={record.session_id})")
    print(f"  PDF : {paths['pdf']}")
    print(f"  CSV : {paths['csv']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
