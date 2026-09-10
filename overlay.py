"""On-frame dashboard rendering.

`draw_dashboard` paints the real-time panel required by the specification onto
a video frame. It is shared by the Streamlit interface and the CLI, so both
show identical information.
"""

import cv2 as cv

_FONT = cv.FONT_HERSHEY_SIMPLEX

_WHITE = (255, 255, 255)
_MUTED = (190, 190, 190)
_ACCENT = (7, 183, 248)     # BGR amber
_GOOD = (90, 200, 90)
_BAD = (60, 60, 235)
_PANEL = (35, 28, 22)


def _put(frame, text, origin, scale=0.5, colour=_WHITE, thickness=1):
    cv.putText(frame, text, origin, _FONT, scale, colour, thickness, cv.LINE_AA)


def _panel(frame, x, y, width, height, alpha=0.62):
    """Blend a translucent rectangle so text stays readable over any footage."""

    x2, y2 = x + width, y + height
    h, w = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x or y2 <= y:
        return

    region = frame[y:y2, x:x2]
    overlay = region.copy()
    overlay[:] = _PANEL
    cv.addWeighted(overlay, alpha, region, 1 - alpha, 0, region)


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def draw_dashboard(frame, info: dict) -> None:
    """Draw the real-time dashboard onto `frame` in place.

    Expected keys in `info`:
        student_id, student_name, exercise, required, completed, valid,
        invalid, state, elapsed, angle (optional), tracked (optional),
        finished (optional), message (optional), message_ok (optional)
    """

    height, width = frame.shape[:2]

    # ---- header: student and exercise identity -------------------------
    _panel(frame, 0, 0, width, 78)
    _put(frame, f"ID {info['student_id']}", (12, 22), 0.55, _ACCENT, 2)
    _put(frame, str(info["student_name"])[:34], (12, 44), 0.6, _WHITE, 2)
    _put(frame, str(info["exercise"]).upper(), (12, 66), 0.5, _MUTED, 1)

    timer = format_duration(info.get("elapsed", 0.0))
    _put(frame, timer, (width - 96, 30), 0.7, _WHITE, 2)
    _put(frame, "SESSION TIME", (width - 110, 50), 0.35, _MUTED, 1)

    # ---- counters ------------------------------------------------------
    _panel(frame, 0, 86, 196, 150)
    _put(frame, "REPETITIONS", (12, 106), 0.42, _MUTED, 1)

    valid = info["valid"]
    required = info["required"]
    _put(frame, f"{valid}/{required}", (12, 146), 1.25, _ACCENT, 3)
    _put(frame, "VALID / REQUIRED", (12, 166), 0.35, _MUTED, 1)

    _put(frame, f"Completed : {info['completed']}", (12, 190), 0.45, _WHITE, 1)
    _put(frame, f"Valid     : {valid}", (12, 208), 0.45, _GOOD, 1)
    _put(frame, f"Invalid   : {info['invalid']}", (12, 226), 0.45, _BAD, 1)

    # ---- exercise state ------------------------------------------------
    _panel(frame, 0, height - 62, width, 62)
    tracked = info.get("tracked", True)
    state_colour = _GOOD if tracked else _BAD
    state_text = str(info["state"]) if tracked else "NO POSE DETECTED"
    _put(frame, state_text, (12, height - 36), 0.72, state_colour, 2)

    angle = info.get("angle")
    if angle is not None:
        _put(frame, f"joint angle {angle:.0f} deg", (12, height - 12), 0.44, _MUTED, 1)

    # ---- transient feedback / completion banner ------------------------
    if info.get("finished"):
        _banner(frame, "EXERCISE COMPLETED", _GOOD)
    elif info.get("message"):
        colour = _GOOD if info.get("message_ok", True) else _BAD
        text = str(info["message"])
        (text_w, _), _ = cv.getTextSize(text, _FONT, 0.62, 2)
        x = max(12, (width - text_w) // 2)
        _panel(frame, x - 12, height - 108, text_w + 24, 34)
        _put(frame, text, (x, height - 84), 0.62, colour, 2)


def _banner(frame, text: str, colour) -> None:
    """Centred banner used when the session target has been reached."""

    height, width = frame.shape[:2]
    (text_w, text_h), _ = cv.getTextSize(text, _FONT, 1.0, 3)
    x = max(0, (width - text_w) // 2)
    y = height // 2

    _panel(frame, x - 20, y - text_h - 18, text_w + 40, text_h + 36, alpha=0.75)
    cv.rectangle(frame, (x - 20, y - text_h - 18), (x + text_w + 20, y + 18), colour, 2)
    _put(frame, text, (x, y), 1.0, colour, 3)
