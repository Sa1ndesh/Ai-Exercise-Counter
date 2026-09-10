"""MediaPipe Pose wrapper: landmark extraction and joint angle measurement.

MediaPipe is imported lazily so that the pure-logic modules (config, exercises,
database, report) stay importable — and testable — on a machine where the
mediapipe wheel is not installed.
"""

import sys
from contextlib import contextmanager
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from config import MIN_VISIBILITY, ExerciseConfig

_mp = None
_mp_pose = None
_mp_drawing = None
_mp_styles = None


INSTALL_HINT = 'pip install "mediapipe>=0.10.9,<0.10.30"'

_MISSING = object()


@contextmanager
def _tensorflow_hidden():
    """Hide tensorflow from mediapipe for the duration of its import.

    mediapipe imports tensorflow purely for documentation decorators, guarding
    it with `except ModuleNotFoundError`. A tensorflow that is installed but
    unusable — most often a protobuf version mismatch — raises a plain
    ImportError instead, which slips past that guard and takes mediapipe down
    with it.

    Putting None in sys.modules turns that into the ModuleNotFoundError
    mediapipe already handles, so it falls back to its no-op decorators. If
    tensorflow is already imported it is clearly working, so leave it alone.
    The original state is restored on the way out, and real use of tensorflow
    elsewhere is unaffected.
    """

    if "tensorflow" in sys.modules:
        yield
        return

    previous = sys.modules.get("tensorflow", _MISSING)
    sys.modules["tensorflow"] = None
    try:
        yield
    finally:
        if previous is _MISSING:
            sys.modules.pop("tensorflow", None)
        else:
            sys.modules["tensorflow"] = previous


def _load_mediapipe():
    """Import mediapipe on first use, with an actionable error if it is unusable.

    This project uses the legacy `mediapipe.solutions.pose` API. That API was
    removed in mediapipe 0.10.30 in favour of the Tasks API, so a too-new
    install imports fine but has no `solutions` attribute at all — hence the
    AttributeError branch below.
    """

    global _mp, _mp_pose, _mp_drawing, _mp_styles
    if _mp_pose is not None:
        return _mp_pose

    try:
        # Both the import and the first `solutions` access can pull in the
        # optional tensorflow dependency, so cover them together.
        with _tensorflow_hidden():
            import mediapipe as mp

            pose = mp.solutions.pose
            drawing = mp.solutions.drawing_utils
            styles = mp.solutions.drawing_styles
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"mediapipe could not be imported ({exc}).\n"
            f"Install a supported version with:  {INSTALL_HINT}"
        ) from exc
    except AttributeError as exc:  # pragma: no cover - depends on the environment
        version = getattr(sys.modules.get("mediapipe"), "__version__", "unknown")
        raise ImportError(
            f"mediapipe {version} does not provide the legacy 'solutions.pose' API "
            "(it was removed in 0.10.30).\n"
            f"Install a supported version with:  {INSTALL_HINT}"
        ) from exc

    _mp = mp
    _mp_pose = pose
    _mp_drawing = drawing
    _mp_styles = styles
    return _mp_pose


def mediapipe_available() -> bool:
    """True when pose detection can actually run. Used by the UI to degrade nicely."""

    try:
        _load_mediapipe()
    except ImportError:
        return False
    return True


def mediapipe_error() -> str | None:
    """The reason pose detection is unavailable, or None when it works."""

    try:
        _load_mediapipe()
    except ImportError as exc:
        return str(exc)
    return None


def calc_angle(a, b, c) -> float:
    """Angle at point `b` formed by the points `a-b-c`, in degrees (0-180).

    Unchanged from the original project's exercises.calc_angle.
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


@dataclass
class PoseReading:
    """One frame's worth of pose information.

    `angle` is None whenever the measurement cannot be trusted — no person in
    frame, or the relevant joints are occluded. Callers must treat None as
    "hold current state" rather than as a value.
    """

    angle: float | None
    side: str | None
    visibility: float
    landmarks: object | None

    @property
    def tracked(self) -> bool:
        return self.angle is not None


class PoseEstimator:
    """Thin wrapper around mediapipe's Pose solution.

    Adds two things the original project lacked: automatic left/right side
    selection (it was hardcoded to the right side) and a visibility gate so
    occluded joints do not produce phantom repetitions.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_visibility: float = MIN_VISIBILITY,
    ):
        pose_module = _load_mediapipe()
        self.min_visibility = min_visibility
        self._pose = pose_module.Pose(
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._pose.close()

    def process(self, frame_bgr, exercise: ExerciseConfig) -> PoseReading:
        """Run pose detection on a BGR frame and measure the exercise's joint angle."""

        image = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self._pose.process(image)

        if results.pose_landmarks is None:
            return PoseReading(angle=None, side=None, visibility=0.0, landmarks=None)

        landmarks = results.pose_landmarks.landmark
        side, visibility = self._best_side(landmarks, exercise.joint)

        if visibility < self.min_visibility:
            return PoseReading(
                angle=None, side=side, visibility=visibility, landmarks=results.pose_landmarks
            )

        points = [self._point(landmarks, side, name) for name in exercise.joint]
        angle = round(calc_angle(*points), 1)
        return PoseReading(
            angle=angle, side=side, visibility=visibility, landmarks=results.pose_landmarks
        )

    def draw_landmarks(self, frame_bgr, landmarks) -> None:
        """Draw the pose skeleton onto `frame_bgr` in place."""

        if landmarks is None:
            return
        _mp_drawing.draw_landmarks(
            frame_bgr,
            landmarks,
            _mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=_mp_styles.get_default_pose_landmarks_style(),
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _index(side: str, name: str) -> int:
        return _mp_pose.PoseLandmark[f"{side}_{name}"].value

    @classmethod
    def _point(cls, landmarks, side: str, name: str) -> list[float]:
        lm = landmarks[cls._index(side, name)]
        return [lm.x, lm.y]

    @classmethod
    def _best_side(cls, landmarks, joint: tuple[str, str, str]) -> tuple[str, float]:
        """Pick whichever body side mediapipe can see more clearly.

        Returns the side and its mean landmark visibility across the triplet.
        """

        scores = {}
        for side in ("LEFT", "RIGHT"):
            visibilities = [landmarks[cls._index(side, name)].visibility for name in joint]
            scores[side] = float(np.mean(visibilities))

        side = max(scores, key=scores.get)
        return side, scores[side]
