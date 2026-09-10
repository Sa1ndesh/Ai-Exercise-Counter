"""Camera-free checks for the exercise monitoring system.

Exercises the rep counter with synthetic joint angles, then round-trips a
session through SQLite and both report formats. Run it after installing
dependencies to confirm the logic works before pointing a camera at anything.

    python selftest.py
"""

import os
import sys
import tempfile

from config import PUSHUP, SQUAT, get_exercise
from database import fetch_session, fetch_sessions, save_session, student_summary
from exercises import RepCounter
from pose_engine import calc_angle, mediapipe_available, mediapipe_error
from report import build_csv_bytes, build_pdf_bytes, format_console_report

_failures: list[str] = []


def check(label: str, actual, expected) -> None:
    ok = actual == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {actual!r}" + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        _failures.append(f"{label}: got {actual!r}, expected {expected!r}")


def rep(counter: RepCounter, deepest: float) -> None:
    """Drive one full up -> down -> up cycle reaching `deepest` degrees."""

    cfg = counter.config
    for angle in (cfg.up_angle + 10, cfg.down_detect_angle - 5, deepest,
                  cfg.down_detect_angle - 5, cfg.up_angle + 5):
        counter.update(angle)


def test_angle_maths() -> None:
    print("\nJoint angle calculation")
    check("straight line is 180 deg", round(calc_angle([0, 0], [1, 0], [2, 0])), 180)
    check("right angle is 90 deg", round(calc_angle([0, 1], [0, 0], [1, 0])), 90)
    check("folded is 0 deg", round(calc_angle([1, 0], [0, 0], [1, 0])), 0)


def test_valid_and_invalid_reps() -> None:
    print("\nRep counting and form validation (squat, 5 required)")
    counter = RepCounter(SQUAT, required=5)
    check("starts in standing state", counter.state, "STANDING")
    check("starts at zero", counter.completed, 0)

    rep(counter, 80)    # deep  -> valid
    rep(counter, 95)    # deep  -> valid
    rep(counter, 125)   # shallow -> invalid

    check("completed counts every cycle", counter.completed, 3)
    check("valid counts good form only", counter.valid, 2)
    check("invalid counts shallow reps", counter.invalid, 1)
    check("accuracy is valid/completed", counter.accuracy, 66.7)
    check("not finished yet", counter.finished, False)
    check("remaining uses valid reps", counter.remaining, 3)


def test_completion_uses_valid_reps() -> None:
    print("\nCompletion rule (push-up, 2 required)")
    counter = RepCounter(PUSHUP, required=2)

    rep(counter, 120)   # shallow
    rep(counter, 120)   # shallow
    check("shallow reps do not complete the session", counter.finished, False)
    check("two invalid recorded", counter.invalid, 2)

    rep(counter, 80)
    check("still short after one valid", counter.finished, False)
    rep(counter, 85)
    check("finishes on the second valid rep", counter.finished, True)
    check("status is COMPLETED", counter.status, "COMPLETED")
    check("counting stops once finished", counter.valid, 2)

    rep(counter, 70)    # ignored: session already over
    check("extra reps ignored after completion", counter.completed, 4)


def test_pose_loss_holds_state() -> None:
    print("\nPose loss handling")
    counter = RepCounter(SQUAT, required=3)
    counter.update(SQUAT.up_angle + 5)
    counter.update(SQUAT.down_detect_angle - 20)

    for _ in range(30):
        counter.update(None)    # person walked out of frame

    check("state held while untracked", counter.state, "SQUATTING")
    check("no phantom reps", counter.completed, 0)

    counter.update(60)
    counter.update(SQUAT.up_angle + 5)
    check("rep completes after pose returns", counter.completed, 1)
    check("and is valid", counter.valid, 1)


def test_partial_descent_is_not_a_rep() -> None:
    print("\nPartial movement")
    counter = RepCounter(SQUAT, required=3)
    for angle in (175, 170, 168, 172, 175):   # never bends past detection
        counter.update(angle)
    check("no rep from standing sway", counter.completed, 0)
    check("state stays standing", counter.state, "STANDING")


def test_exercise_lookup() -> None:
    print("\nExercise lookup")
    check("by display name", get_exercise("Push-Up").key, "pushup")
    check("by key", get_exercise("squat").key, "squat")
    check("case insensitive", get_exercise("PUSHUP").key, "pushup")
    try:
        get_exercise("cartwheel")
        check("unknown exercise raises", False, True)
    except ValueError:
        check("unknown exercise raises", True, True)


def test_database_and_reports() -> None:
    print("\nDatabase and report round-trip")
    counter = RepCounter(SQUAT, required=2)
    rep(counter, 80)
    rep(counter, 130)
    rep(counter, 85)

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")

        from database import SessionRecord

        record = SessionRecord(
            student_id="S001",
            student_name="Test Student",
            exercise_type=SQUAT.name,
            required_repetitions=2,
            completed_repetitions=counter.completed,
            valid_repetitions=counter.valid,
            invalid_repetitions=counter.invalid,
            duration=42.5,
            status=counter.status,
        )
        session_id = save_session(record, db)
        check("insert returns an id", session_id > 0, True)

        loaded = fetch_session(session_id, db)
        check("row is retrievable", loaded is not None, True)
        check("name persisted", loaded.student_name, "Test Student")
        check("valid reps persisted", loaded.valid_repetitions, 2)
        check("invalid reps persisted", loaded.invalid_repetitions, 1)
        check("status persisted", loaded.status, "COMPLETED")
        check("duration formatted", loaded.duration_text, "00:42")

        check("listing finds the session", len(fetch_sessions(db_path=db)), 1)
        check(
            "filtering by student works",
            len(fetch_sessions(student_id="S001", db_path=db)),
            1,
        )
        check(
            "filtering by other student is empty",
            len(fetch_sessions(student_id="NOPE", db_path=db)),
            0,
        )
        check("student summary counts sessions", student_summary("S001", db)["sessions"], 1)

        pdf = build_pdf_bytes(loaded)
        check("pdf is non-empty", len(pdf) > 1000, True)
        check("pdf has a PDF header", pdf[:4], b"%PDF")

        csv_bytes = build_csv_bytes(loaded)
        text = csv_bytes.decode("utf-8")
        check("csv has header and one row", len(text.strip().splitlines()), 2)
        check("csv mentions the student", "Test Student" in text, True)
        check("csv includes accuracy column", "accuracy" in text.splitlines()[0], True)

        check("console report renders", "STUDENT EXERCISE SESSION REPORT" in
              format_console_report(loaded), True)


def main() -> int:
    print("=" * 62)
    print("Student Exercise Monitoring System - self test".center(62))
    print("=" * 62)

    test_angle_maths()
    test_valid_and_invalid_reps()
    test_completion_uses_valid_reps()
    test_pose_loss_holds_state()
    test_partial_descent_is_not_a_rep()
    test_exercise_lookup()
    test_database_and_reports()

    print("\n" + "-" * 62)
    if mediapipe_available():
        print("mediapipe : available (live pose detection ready)")
    else:
        print("mediapipe : UNAVAILABLE")
        for line in (mediapipe_error() or "").splitlines():
            print(f"            {line}")
        print("            logic above still verified; camera features unavailable.")

    print("-" * 62)
    if _failures:
        print(f"\n{len(_failures)} CHECK(S) FAILED:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
