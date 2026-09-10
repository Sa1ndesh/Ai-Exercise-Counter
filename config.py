"""Exercise definitions and angle thresholds.

Each exercise is described by a single joint triplet and four angle thresholds.
The split between `down_detect_angle` and `down_valid_angle` is what lets the
system report Completed / Valid / Invalid as three separate numbers:

    down_detect_angle   joint bent this far  -> a repetition ATTEMPT started
    down_valid_angle    deepest angle of the attempt must clear this to be VALID

Angles are in degrees and are measured at the middle joint of the triplet
(elbow for a push-up, knee for a squat), so smaller means more bent.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExerciseConfig:
    """Everything the rep counter needs to know about one exercise."""

    key: str                     # cli/database identifier, e.g. "pushup"
    name: str                    # display name, e.g. "Push-Up"
    joint: tuple[str, str, str]  # landmark triplet, angle taken at the middle one
    up_state: str                # label for the extended state
    down_state: str              # label for the flexed state
    up_angle: float              # angle >= this  -> back in the up state
    down_detect_angle: float     # angle <  this  -> a rep attempt has begun
    down_valid_angle: float      # deepest angle must be <= this to count as valid
    cue: str                     # what to tell a student whose rep was too shallow

    @property
    def states(self) -> tuple[str, str]:
        return self.up_state, self.down_state


PUSHUP = ExerciseConfig(
    key="pushup",
    name="Push-Up",
    joint=("SHOULDER", "ELBOW", "WRIST"),
    up_state="UP",
    down_state="DOWN",
    up_angle=160.0,
    down_detect_angle=130.0,
    down_valid_angle=95.0,
    cue="Lower your chest until your elbows bend past 90 degrees.",
)

SQUAT = ExerciseConfig(
    key="squat",
    name="Squat",
    joint=("HIP", "KNEE", "ANKLE"),
    up_state="STANDING",
    down_state="SQUATTING",
    up_angle=165.0,
    down_detect_angle=140.0,
    down_valid_angle=100.0,
    cue="Sit deeper until your thighs are close to parallel with the floor.",
)

# Pull-up is kept from the original project but is not offered in the UI,
# because the specification only calls for push-ups and squats.
PULLUP = ExerciseConfig(
    key="pullup",
    name="Pull-Up",
    joint=("SHOULDER", "ELBOW", "WRIST"),
    up_state="HANGING",
    down_state="PULLED",
    up_angle=160.0,
    down_detect_angle=120.0,
    down_valid_angle=80.0,
    cue="Pull until your elbows are fully bent and your chin clears the bar.",
)

# Exercises selectable in the Streamlit interface, keyed by display name.
EXERCISES: dict[str, ExerciseConfig] = {
    PUSHUP.name: PUSHUP,
    SQUAT.name: SQUAT,
}

# Every exercise the engine can run, keyed by cli identifier.
ALL_EXERCISES: dict[str, ExerciseConfig] = {
    cfg.key: cfg for cfg in (PUSHUP, SQUAT, PULLUP)
}


def get_exercise(name: str) -> ExerciseConfig:
    """Look an exercise up by either display name ("Push-Up") or key ("pushup")."""

    if name in EXERCISES:
        return EXERCISES[name]

    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in ALL_EXERCISES:
        return ALL_EXERCISES[key]

    valid = ", ".join(sorted(ALL_EXERCISES))
    raise ValueError(f"Unknown exercise {name!r}. Choose one of: {valid}")


# A landmark must be at least this visible before its angle is trusted.
MIN_VISIBILITY = 0.5

# Database and report output locations, relative to the project folder.
DB_PATH = "data/sessions.db"
REPORT_DIR = "reports"
