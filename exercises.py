"""Repetition counting with form validation.

The counter is a two-state machine (up <-> down). Every completed
`up -> down -> up` cycle increments `completed`; it is then classified as
either `valid` or `invalid` depending on the deepest joint angle reached
during that cycle:

    deepest angle <= config.down_valid_angle   ->  valid
    otherwise                                  ->  invalid (range too short)

A session finishes when `valid` reaches the required number of repetitions,
so shallow repetitions never bring a student closer to completion.
"""

from dataclasses import dataclass, field

from config import ExerciseConfig


@dataclass
class RepEvent:
    """Emitted the moment a repetition is completed, for on-screen feedback."""

    index: int          # 1-based repetition number
    valid: bool
    deepest_angle: float


@dataclass
class RepCounter:
    """Tracks state, counts and form quality for one exercise session."""

    config: ExerciseConfig
    required: int

    completed: int = 0
    valid: int = 0
    invalid: int = 0
    finished: bool = False

    state: str = field(init=False)
    last_angle: float | None = field(default=None, init=False)
    last_event: RepEvent | None = field(default=None, init=False)
    _deepest: float = field(default=180.0, init=False)

    def __post_init__(self):
        if self.required < 1:
            raise ValueError("required repetitions must be at least 1")
        # A session begins from the extended position: standing, or at the top
        # of a push-up. The original project seeded the push-up counter at -1
        # to compensate for starting in the wrong state; that is not needed.
        self.state = self.config.up_state

    @property
    def remaining(self) -> int:
        return max(0, self.required - self.valid)

    @property
    def accuracy(self) -> float:
        """Share of completed repetitions that had acceptable form, 0-100."""

        if self.completed == 0:
            return 0.0
        return round(100.0 * self.valid / self.completed, 1)

    @property
    def status(self) -> str:
        return "COMPLETED" if self.finished else "INCOMPLETE"

    def update(self, angle: float | None) -> RepEvent | None:
        """Feed one frame's joint angle into the state machine.

        Pass None when the pose is not reliably tracked; the counter then holds
        its current state instead of counting noise. Returns a RepEvent on the
        frame a repetition completes, otherwise None.
        """

        self.last_event = None

        if angle is None or self.finished:
            return None

        self.last_angle = angle
        cfg = self.config

        if self.state == cfg.up_state:
            if angle < cfg.down_detect_angle:
                self.state = cfg.down_state
                self._deepest = angle
            return None

        # In the down state: remember how deep this repetition went.
        self._deepest = min(self._deepest, angle)

        if angle < cfg.up_angle:
            return None

        # Returned to the top: the repetition is complete.
        self.state = cfg.up_state
        self.completed += 1
        is_valid = self._deepest <= cfg.down_valid_angle

        if is_valid:
            self.valid += 1
        else:
            self.invalid += 1

        event = RepEvent(index=self.completed, valid=is_valid, deepest_angle=self._deepest)
        self.last_event = event
        self._deepest = 180.0

        if self.valid >= self.required:
            self.finished = True

        return event

    def snapshot(self) -> dict:
        """Plain-dict view of the counter, for the dashboard and session records."""

        return {
            "state": self.state,
            "completed": self.completed,
            "valid": self.valid,
            "invalid": self.invalid,
            "required": self.required,
            "remaining": self.remaining,
            "accuracy": self.accuracy,
            "angle": self.last_angle,
            "finished": self.finished,
            "status": self.status,
        }
