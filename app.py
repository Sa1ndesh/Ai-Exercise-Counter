"""Streamlit interface for the AI-Based Student Exercise Monitoring System.

Flow: setup -> live session -> report, plus a session history page.

Run with:
    streamlit run app.py
"""

import inspect
import os
import tempfile
import time

import pandas as pd
import streamlit as st

from config import EXERCISES, get_exercise
from database import SessionRecord, fetch_sessions, init_db, save_session, student_summary
from report import (
    CSV_COLUMNS,
    build_csv_bytes,
    build_csv_bytes_many,
    build_pdf_bytes,
    report_basename,
    report_rows,
)
from pose_engine import mediapipe_available, mediapipe_error
from videoStream import CaptureError, ExerciseSession

st.set_page_config(
    page_title="Student Exercise Monitor",
    page_icon="🏋️",
    layout="wide",
)

# st.image renamed use_column_width -> use_container_width in Streamlit 1.41.
# Detect it so the app works on either side of that change.
_IMAGE_WIDTH = (
    {"use_container_width": True}
    if "use_container_width" in inspect.signature(st.image).parameters
    else {"use_column_width": True}
)


# ---------------------------------------------------------------- state ----


def init_state() -> None:
    defaults = {
        "stage": "setup",
        "page": "Session",
        "student_id": "",
        "student_name": "",
        "exercise_name": next(iter(EXERCISES)),
        "required": 10,
        "source_kind": "Webcam",
        "camera_index": 0,
        "video_path": None,
        "video_is_temp": False,
        "live": None,        # snapshot of the running session
        "record": None,      # saved SessionRecord
        "stop": False,
        "error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go(stage: str) -> None:
    st.session_state.stage = stage


def request_stop() -> None:
    """Button callback: clicking it starts a new run, which ends the frame loop."""

    st.session_state.stop = True


def cleanup_video() -> None:
    """Delete an uploaded temp file once we are done with it."""

    path = st.session_state.get("video_path")
    if path and st.session_state.get("video_is_temp") and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass
    st.session_state.video_path = None
    st.session_state.video_is_temp = False


# ---------------------------------------------------------------- setup ----


def page_setup() -> None:
    st.title("🏋️ Student Exercise Monitor")
    st.caption(
        "Enter the student's details, choose an exercise and a repetition target. "
        "Only repetitions performed through the full range of motion count towards the target."
    )

    with st.form("session_setup"):
        left, right = st.columns(2)

        with left:
            st.subheader("Student")
            student_id = st.text_input(
                "Student ID", value=st.session_state.student_id, placeholder="e.g. S001"
            )
            student_name = st.text_input(
                "Student Name", value=st.session_state.student_name, placeholder="e.g. Sandesh Birannavar"
            )

        with right:
            st.subheader("Exercise")
            exercise_name = st.selectbox(
                "Exercise Type",
                list(EXERCISES),
                index=list(EXERCISES).index(st.session_state.exercise_name),
            )
            required = st.number_input(
                "Required Repetitions",
                min_value=1,
                max_value=200,
                value=int(st.session_state.required),
                step=1,
                help="Number of correctly-performed repetitions needed to complete the session.",
            )

        st.divider()
        st.subheader("Video source")
        source_kind = st.radio(
            "Read frames from",
            ["Webcam", "Upload video file"],
            index=0 if st.session_state.source_kind == "Webcam" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )

        cam_col, up_col = st.columns(2)
        with cam_col:
            camera_index = st.number_input(
                "Camera index", min_value=0, max_value=8,
                value=int(st.session_state.camera_index), step=1,
                help="0 is the default webcam. Try 1 or 2 if you have several cameras.",
            )
        with up_col:
            upload = st.file_uploader(
                "Video file", type=["mp4", "mov", "avi", "mkv"],
                help="Useful for testing without a camera - try the clips in videos/.",
            )

        cfg = get_exercise(exercise_name)
        st.info(
            f"**{cfg.name}** is measured at the "
            f"**{cfg.joint[1].lower()}** ({'-'.join(p.lower() for p in cfg.joint)}). "
            f"A repetition counts as valid when the angle passes "
            f"**{cfg.down_valid_angle:.0f}°** and returns above **{cfg.up_angle:.0f}°**.",
            icon="ℹ️",
        )

        submitted = st.form_submit_button("▶ Start session", type="primary", use_container_width=True)

    if not submitted:
        return

    problems = []
    if not student_id.strip():
        problems.append("Student ID is required.")
    if not student_name.strip():
        problems.append("Student Name is required.")
    if source_kind == "Upload video file" and upload is None:
        problems.append("Choose a video file, or switch the source to Webcam.")

    if problems:
        for problem in problems:
            st.error(problem, icon="⚠️")
        return

    cleanup_video()
    st.session_state.update(
        student_id=student_id.strip(),
        student_name=student_name.strip(),
        exercise_name=exercise_name,
        required=int(required),
        source_kind=source_kind,
        camera_index=int(camera_index),
        live=None,
        record=None,
        stop=False,
        error=None,
    )

    if source_kind == "Upload video file":
        suffix = os.path.splitext(upload.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(upload.getbuffer())
            st.session_state.video_path = handle.name
        st.session_state.video_is_temp = True

    go("live")
    st.rerun()


# ----------------------------------------------------------------- live ----


def page_live() -> None:
    exercise = get_exercise(st.session_state.exercise_name)
    required = st.session_state.required

    st.title(f"● Live session — {exercise.name}")
    st.caption(
        f"{st.session_state.student_name} ({st.session_state.student_id}) — "
        f"target {required} valid repetitions"
    )

    if st.session_state.error:
        st.error(st.session_state.error, icon="🚫")
        if st.button("← Back to setup"):
            cleanup_video()
            st.session_state.error = None
            go("setup")
            st.rerun()
        return

    # Clicking Stop starts a fresh run and kills the frame loop of the previous
    # one, so the save code below it never executes. That case lands here on the
    # next run: finalise from the snapshot the loop mirrored into session_state
    # rather than opening the camera again.
    if st.session_state.stop:
        st.session_state.stop = False
        finalise_interrupted()
        return

    # Rendered before the loop so it stays clickable while frames stream.
    st.button("⏹ Stop and save", on_click=request_stop, type="secondary")

    metrics = st.container()
    frame_slot = st.empty()

    source = (
        st.session_state.video_path
        if st.session_state.source_kind == "Upload video file"
        else st.session_state.camera_index
    )

    session = None
    try:
        session = ExerciseSession(
            student_id=st.session_state.student_id,
            student_name=st.session_state.student_name,
            exercise=exercise,
            required=required,
            source=source,
        )
        session.open()
    except (CaptureError, ImportError) as exc:
        st.session_state.error = str(exc)
        if session is not None:
            session.close()
        st.rerun()
        return

    slots = _metric_slots(metrics)
    last_ui = 0.0

    try:
        for frame, snapshot in session.frames():
            # Mirror progress into session_state *before* touching any widget:
            # a Stop click raises out of the next st.* call, and this is what
            # survives to be saved.
            st.session_state.live = snapshot

            frame_slot.image(frame, channels="BGR", **_IMAGE_WIDTH)

            # Throttle the metric widgets; the video itself updates every frame.
            now = time.time()
            if now - last_ui > 0.12 or snapshot["finished"]:
                _render_metrics(slots, snapshot)
                last_ui = now
    finally:
        # Runs whether the loop ended normally or Streamlit killed it.
        st.session_state.live = session.snapshot()
        session.close()

    _finalise(session.to_record())


def finalise_interrupted() -> None:
    """Save a session whose frame loop was cut short by the Stop button."""

    snapshot = st.session_state.live
    if not snapshot or snapshot["completed"] == 0 and snapshot["elapsed"] < 0.5:
        # Stopped before anything was measured - nothing worth recording.
        cleanup_video()
        go("setup")
        st.rerun()
        return

    _finalise(_record_from_snapshot(snapshot))


def _record_from_snapshot(snapshot: dict) -> SessionRecord:
    return SessionRecord(
        student_id=st.session_state.student_id,
        student_name=st.session_state.student_name,
        exercise_type=snapshot["exercise"],
        required_repetitions=snapshot["required"],
        completed_repetitions=snapshot["completed"],
        valid_repetitions=snapshot["valid"],
        invalid_repetitions=snapshot["invalid"],
        duration=round(snapshot["elapsed"], 2),
        status=snapshot["status"],
    )


def _finalise(record: SessionRecord) -> None:
    """Persist the session and move to the report page."""

    record.session_id = save_session(record)
    st.session_state.record = record
    cleanup_video()
    st.session_state.stop = False
    go("report")
    st.rerun()


def _metric_slots(container):
    with container:
        columns = st.columns(6)
    return [column.empty() for column in columns]


def _render_metrics(slots, snapshot: dict) -> None:
    minutes, seconds = divmod(int(snapshot["elapsed"]), 60)

    slots[0].metric("Required", snapshot["required"])
    slots[1].metric(
        "Valid",
        snapshot["valid"],
        delta=f"{snapshot['remaining']} to go" if snapshot["remaining"] else "target met",
        delta_color="normal" if snapshot["remaining"] else "off",
    )
    slots[2].metric("Completed", snapshot["completed"])
    slots[3].metric("Invalid", snapshot["invalid"])
    slots[4].metric("State", snapshot["state"] if snapshot["tracked"] else "NO POSE")
    slots[5].metric("Timer", f"{minutes:02d}:{seconds:02d}")


# --------------------------------------------------------------- report ----


def page_report() -> None:
    record = st.session_state.record
    if record is None:
        go("setup")
        st.rerun()
        return

    completed = record.status == "COMPLETED"
    st.title("Session report")

    if completed:
        st.success(
            f"Target reached — {record.valid_repetitions} of "
            f"{record.required_repetitions} valid repetitions.",
            icon="✅",
        )
    else:
        short = max(0, record.required_repetitions - record.valid_repetitions)
        st.warning(
            f"Session incomplete — {short} more valid "
            f"repetition{'s' if short != 1 else ''} needed.",
            icon="⚠️",
        )

    top = st.columns(5)
    top[0].metric("Required", record.required_repetitions)
    top[1].metric("Completed", record.completed_repetitions)
    top[2].metric("Valid", record.valid_repetitions)
    top[3].metric("Invalid", record.invalid_repetitions)
    top[4].metric("Form accuracy", f"{record.accuracy}%")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Details")
        st.dataframe(
            pd.DataFrame(report_rows(record), columns=["Field", "Value"]),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.subheader("Downloads")
        stem = report_basename(record)
        st.download_button(
            "⬇ Download PDF report",
            data=build_pdf_bytes(record),
            file_name=f"{stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download CSV",
            data=build_csv_bytes(record),
            file_name=f"{stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(f"Stored in the database as session #{record.session_id}.")

        summary = student_summary(record.student_id)
        if summary["sessions"] > 1:
            st.metric(
                f"All sessions for {record.student_id}",
                f"{summary['sessions']} sessions",
                delta=f"{summary['accuracy']}% average form accuracy",
                delta_color="off",
            )

    st.divider()
    actions = st.columns(2)
    if actions[0].button("🔄 New session", type="primary", use_container_width=True):
        st.session_state.record = None
        st.session_state.live = None
        go("setup")
        st.rerun()
    if actions[1].button("📊 View history", use_container_width=True):
        st.session_state.page = "History"
        st.rerun()


# -------------------------------------------------------------- history ----


def page_history() -> None:
    st.title("📊 Session history")

    sessions = fetch_sessions()
    if not sessions:
        st.info("No sessions recorded yet. Run a session to populate this page.", icon="ℹ️")
        return

    student_ids = sorted({s.student_id for s in sessions})
    chosen = st.selectbox("Filter by student", ["All students"] + student_ids)

    if chosen != "All students":
        sessions = [s for s in sessions if s.student_id == chosen]
        summary = student_summary(chosen)
        columns = st.columns(4)
        columns[0].metric("Sessions", summary["sessions"])
        columns[1].metric("Valid reps", summary["valid"])
        columns[2].metric("Invalid reps", summary["invalid"])
        columns[3].metric("Form accuracy", f"{summary['accuracy']}%")

    frame = pd.DataFrame([s.as_dict() for s in sessions])
    frame["accuracy"] = [s.accuracy for s in sessions]
    frame["duration"] = frame["duration"].round(1)
    frame = frame[CSV_COLUMNS + ["accuracy"]]

    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "session_id": st.column_config.NumberColumn("#", width="small"),
            "student_id": "Student ID",
            "student_name": "Name",
            "exercise_type": "Exercise",
            "required_repetitions": st.column_config.NumberColumn("Req.", width="small"),
            "completed_repetitions": st.column_config.NumberColumn("Done", width="small"),
            "valid_repetitions": st.column_config.NumberColumn("Valid", width="small"),
            "invalid_repetitions": st.column_config.NumberColumn("Invalid", width="small"),
            "duration": st.column_config.NumberColumn("Seconds", format="%.1f"),
            "date_time": "Date and time",
            "status": "Status",
            "accuracy": st.column_config.NumberColumn("Accuracy %", format="%.1f"),
        },
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.download_button(
            "⬇ Export these sessions as CSV",
            data=build_csv_bytes_many(sessions),
            file_name="exercise_sessions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with right:
        labels = {
            f"#{s.session_id} — {s.student_name} — {s.exercise_type} "
            f"({s.valid_repetitions}/{s.required_repetitions})": s
            for s in sessions
        }
        picked = st.selectbox("Re-download a PDF report", list(labels))
        record = labels[picked]
        st.download_button(
            "⬇ Download PDF",
            data=build_pdf_bytes(record),
            file_name=f"{report_basename(record)}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ----------------------------------------------------------------- main ----


def sidebar() -> str:
    with st.sidebar:
        st.header("Exercise Monitor")

        options = ["Session", "History"]
        current = st.session_state.get("page", "Session")
        page = st.radio("Page", options, index=options.index(current), label_visibility="collapsed")
        st.session_state.page = page

        st.divider()
        if st.session_state.stage == "live":
            st.caption("A session is in progress.")
        elif st.session_state.record is not None:
            st.caption(f"Last session: #{st.session_state.record.session_id}")

        st.caption(
            "Repetitions are counted from MediaPipe Pose joint angles. "
            "Shallow repetitions are recorded as invalid and do not count "
            "towards the target."
        )
    return page


def main() -> None:
    init_state()
    init_db()

    if not mediapipe_available():
        st.title("🏋️ Student Exercise Monitor")
        st.error("Pose detection is unavailable.", icon="🚫")
        st.code(mediapipe_error() or "mediapipe could not be loaded.", language="text")
        st.markdown(
            "Install the supported version, then restart this app:\n\n"
            '```bash\npip install "mediapipe==0.10.21"\n```\n\n'
            "The rest of the system (database, reports) works without it — "
            "run `python selftest.py` to verify."
        )
        return

    page = sidebar()

    if page == "History":
        page_history()
        return

    stage = st.session_state.stage
    if stage == "live":
        page_live()
    elif stage == "report":
        page_report()
    else:
        page_setup()


if __name__ == "__main__":
    main()
