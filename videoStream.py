"""Frame loop that ties capture, pose detection and rep counting together.

`ExerciseSession.frames()` is a generator yielding one annotated frame at a
time. Both front-ends consume it: the Streamlit app pushes each frame into an
image placeholder, the CLI shows it in an OpenCV window. All exercise logic
therefore lives in exactly one place.
"""

import os
import time
from typing import Iterator

import cv2 as cv

from config import ExerciseConfig
from database import SessionRecord
from exercises import RepCounter
from overlay import draw_dashboard
from pose_engine import PoseEstimator

# How long a "rep counted" message stays on screen, in seconds.
MESSAGE_TTL = 1.4


class CaptureError(RuntimeError):
    """Raised when the requested webcam or video file cannot be opened."""


def open_capture(source) -> cv.VideoCapture:
    """Open a webcam index or a video file path, or raise CaptureError."""

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, str) and not os.path.exists(source):
        raise CaptureError(f"Video file not found: {source}")

    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        if isinstance(source, int):
            raise CaptureError(
                f"Could not open camera index {source}. Check that a webcam is "
                "connected and not in use by another application."
            )
        raise CaptureError(f"Could not open video source: {source}")

    return cap


class ExerciseSession:
    """One monitored exercise session, from first frame to final record."""

    def __init__(
        self,
        student_id: str,
        student_name: str,
        exercise: ExerciseConfig,
        required: int,
        source=0,
        mirror: bool | None = None,
        draw_skeleton: bool = True,
    ):
        self.student_id = student_id
        self.student_name = student_name
        self.exercise = exercise
        self.required = required
        self.source = source
        self.draw_skeleton = draw_skeleton

        self.is_webcam = isinstance(source, int) or (
            isinstance(source, str) and source.isdigit()
        )
        # Mirroring a webcam feed makes the overlay feel natural to the student;
        # recorded footage is left as filmed.
        self.mirror = self.is_webcam if mirror is None else mirror

        self.counter = RepCounter(exercise, required)
        self.elapsed = 0.0
        self.frame_count = 0

        self._cap: cv.VideoCapture | None = None
        self._pose: PoseEstimator | None = None
        self._started: float | None = None
        self._fps = 0.0
        self._message: str | None = None
        self._message_ok = True
        self._message_until = 0.0
        self._stop = False

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "ExerciseSession":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:
        self._cap = open_capture(self.source)
        self._fps = self._cap.get(cv.CAP_PROP_FPS) or 0.0
        self._pose = PoseEstimator()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def stop(self) -> None:
        """Ask the generator to finish after the current frame."""

        self._stop = True

    # -- main loop ---------------------------------------------------------

    def frames(self) -> Iterator[tuple["cv.Mat", dict]]:
        """Yield `(annotated_bgr_frame, snapshot_dict)` until the session ends.

        Ends when the target is reached, the source is exhausted, or stop()
        is called.
        """

        if self._cap is None or self._pose is None:
            self.open()

        while not self._stop:
            ret, frame = self._cap.read()
            # The original project crashed here at the end of a video, because
            # it used `frame` without checking `ret`.
            if not ret or frame is None:
                break

            if self.mirror:
                frame = cv.flip(frame, 1)

            if self._started is None:
                self._started = time.time()

            self.frame_count += 1
            self.elapsed = self._compute_elapsed()

            reading = self._pose.process(frame, self.exercise)
            event = self.counter.update(reading.angle)

            if event is not None:
                if event.valid:
                    self._set_message(f"REP {self.counter.valid} COUNTED", ok=True)
                else:
                    self._set_message(f"TOO SHALLOW ({event.deepest_angle:.0f} deg)", ok=False)

            if self.draw_skeleton:
                self._pose.draw_landmarks(frame, reading.landmarks)

            snapshot = self.snapshot(reading)
            draw_dashboard(frame, snapshot)

            yield frame, snapshot

            if self.counter.finished:
                break

    # -- state -------------------------------------------------------------

    def snapshot(self, reading=None) -> dict:
        """Everything the dashboard and the UI metrics need for this frame."""

        data = {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "exercise": self.exercise.name,
            "required": self.required,
            "completed": self.counter.completed,
            "valid": self.counter.valid,
            "invalid": self.counter.invalid,
            "state": self.counter.state,
            "status": self.counter.status,
            "finished": self.counter.finished,
            "accuracy": self.counter.accuracy,
            "remaining": self.counter.remaining,
            "elapsed": self.elapsed,
            "angle": reading.angle if reading is not None else self.counter.last_angle,
            "tracked": reading.tracked if reading is not None else False,
            "side": reading.side if reading is not None else None,
        }

        if self._message and time.time() < self._message_until:
            data["message"] = self._message
            data["message_ok"] = self._message_ok

        return data

    def to_record(self) -> SessionRecord:
        """Build the database row for this session."""

        return SessionRecord(
            student_id=self.student_id,
            student_name=self.student_name,
            exercise_type=self.exercise.name,
            required_repetitions=self.required,
            completed_repetitions=self.counter.completed,
            valid_repetitions=self.counter.valid,
            invalid_repetitions=self.counter.invalid,
            duration=round(self.elapsed, 2),
            status=self.counter.status,
        )

    # -- internals ---------------------------------------------------------

    def _compute_elapsed(self) -> float:
        """Wall clock for a live camera, media time for a recorded file."""

        if not self.is_webcam and self._fps > 0:
            return self.frame_count / self._fps
        return time.time() - (self._started or time.time())

    def _set_message(self, text: str, ok: bool) -> None:
        self._message = text
        self._message_ok = ok
        self._message_until = time.time() + MESSAGE_TTL


def mp_stream(cap, exercise, counter=None, phase=None) -> None:
    """Backwards-compatible entry point for the original project's API.

    Kept so older scripts calling `videoStream.mp_stream(cap, 'squat', ...)`
    still work; new code should use ExerciseSession instead.
    """

    from config import get_exercise

    cfg = get_exercise(exercise) if isinstance(exercise, str) else exercise
    session = ExerciseSession("-", "Demo", cfg, required=10 ** 6, source=0)
    session._cap = cap
    session._pose = PoseEstimator()
    session.mirror = False

    try:
        for frame, _ in session.frames():
            cv.imshow(cfg.name, frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        if session._pose is not None:
            session._pose.close()
            session._pose = None
        session._cap = None  # the caller owns this capture object
