"""SQLite persistence for exercise sessions.

One table, `exercise_sessions`, holding one row per completed or abandoned
session. Column names follow the project specification exactly.
"""

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS exercise_sessions (
    session_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id            TEXT    NOT NULL,
    student_name          TEXT    NOT NULL,
    exercise_type         TEXT    NOT NULL,
    required_repetitions  INTEGER NOT NULL,
    completed_repetitions INTEGER NOT NULL,
    valid_repetitions     INTEGER NOT NULL,
    invalid_repetitions   INTEGER NOT NULL,
    duration              REAL    NOT NULL,
    date_time             TEXT    NOT NULL,
    status                TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_student ON exercise_sessions (student_id);
CREATE INDEX IF NOT EXISTS idx_sessions_datetime ON exercise_sessions (date_time);
"""


@dataclass
class SessionRecord:
    """One exercise session, ready to be written to the database."""

    student_id: str
    student_name: str
    exercise_type: str
    required_repetitions: int
    completed_repetitions: int
    valid_repetitions: int
    invalid_repetitions: int
    duration: float
    status: str
    date_time: str = ""
    session_id: int | None = None

    def __post_init__(self):
        if not self.date_time:
            self.date_time = datetime.now().isoformat(timespec="seconds")

    @property
    def accuracy(self) -> float:
        """Share of completed repetitions with acceptable form, 0-100."""

        if self.completed_repetitions == 0:
            return 0.0
        return round(100.0 * self.valid_repetitions / self.completed_repetitions, 1)

    @property
    def duration_text(self) -> str:
        minutes, seconds = divmod(int(round(self.duration)), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def as_dict(self) -> dict:
        return asdict(self)


def _resolve(db_path: str | None) -> str:
    return db_path or DB_PATH


@contextmanager
def connect(db_path: str | None = None):
    """Open a connection with dict-like rows, committing on clean exit."""

    path = _resolve(db_path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create the table and indexes if they do not already exist."""

    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def save_session(record: SessionRecord, db_path: str | None = None) -> int:
    """Insert a session and return its generated session_id."""

    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO exercise_sessions (
                student_id, student_name, exercise_type,
                required_repetitions, completed_repetitions,
                valid_repetitions, invalid_repetitions,
                duration, date_time, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.student_id,
                record.student_name,
                record.exercise_type,
                record.required_repetitions,
                record.completed_repetitions,
                record.valid_repetitions,
                record.invalid_repetitions,
                record.duration,
                record.date_time,
                record.status,
            ),
        )
        record.session_id = int(cursor.lastrowid)
        return record.session_id


def fetch_sessions(
    student_id: str | None = None,
    limit: int | None = None,
    db_path: str | None = None,
) -> list[SessionRecord]:
    """Return stored sessions, newest first, optionally filtered by student."""

    init_db(db_path)
    query = "SELECT * FROM exercise_sessions"
    params: list = []

    if student_id:
        query += " WHERE student_id = ?"
        params.append(student_id)

    query += " ORDER BY session_id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [_to_record(row) for row in rows]


def fetch_session(session_id: int, db_path: str | None = None) -> SessionRecord | None:
    """Return a single session by id, or None if it does not exist."""

    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM exercise_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

    return _to_record(row) if row else None


def student_summary(student_id: str, db_path: str | None = None) -> dict:
    """Aggregate totals across every session for one student."""

    sessions = fetch_sessions(student_id=student_id, db_path=db_path)
    if not sessions:
        return {"sessions": 0, "valid": 0, "invalid": 0, "completed": 0, "accuracy": 0.0}

    valid = sum(s.valid_repetitions for s in sessions)
    invalid = sum(s.invalid_repetitions for s in sessions)
    completed = sum(s.completed_repetitions for s in sessions)
    return {
        "sessions": len(sessions),
        "valid": valid,
        "invalid": invalid,
        "completed": completed,
        "accuracy": round(100.0 * valid / completed, 1) if completed else 0.0,
    }


def _to_record(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        student_id=row["student_id"],
        student_name=row["student_name"],
        exercise_type=row["exercise_type"],
        required_repetitions=row["required_repetitions"],
        completed_repetitions=row["completed_repetitions"],
        valid_repetitions=row["valid_repetitions"],
        invalid_repetitions=row["invalid_repetitions"],
        duration=row["duration"],
        date_time=row["date_time"],
        status=row["status"],
    )
