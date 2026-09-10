# AI-Based Student Exercise Monitoring System

Counts a student's push-ups and squats from a webcam or video file using
MediaPipe Pose, validates the form of every repetition, stores each session in
SQLite and produces a PDF/CSV report.

## What it does

1. Teacher enters Student ID, name, exercise and a repetition target
2. Webcam (or an uploaded clip) streams into MediaPipe Pose
3. The relevant joint angle is measured every frame
4. A state machine counts repetitions and grades each one
5. A live dashboard shows progress; the session ends automatically at the target
6. The session is saved to SQLite and exported as PDF and CSV

## Valid vs invalid repetitions

Each exercise defines two thresholds, which is what separates the three counters:

| Counter | Meaning |
|---|---|
| **Completed** | every `up → down → up` cycle detected |
| **Valid** | cycles that reached the required depth |
| **Invalid** | cycles that were too shallow |

A session completes when **valid** repetitions reach the target, so a shallow
repetition is recorded and shown but never brings the student closer to finishing.

| Exercise | Joint measured | Rep detected below | Valid below | Back up above |
|---|---|---|---|---|
| Push-Up | shoulder–elbow–wrist | 130° | 95° | 160° |
| Squat | hip–knee–ankle | 140° | 100° | 165° |

Thresholds live in [config.py](config.py) — adjust them there for a stricter or
more lenient standard.

## Install

```bash
pip install -r requirements.txt
```

> **MediaPipe version matters.** This project uses the legacy
> `mediapipe.solutions.pose` API, which was **removed in mediapipe 0.10.30**.
> `requirements.txt` pins `<0.10.30` for that reason. A newer install imports
> successfully but has no `solutions` attribute, and the app will tell you so
> instead of crashing.

## Run

Web interface (recommended):

```bash
streamlit run app.py
```

Command line, no Streamlit — the quickest way to check everything works, using
one of the bundled clips:

```bash
python run.py --student-id S001 --name "Test Student" --exercise squat --reps 5 --source videos/count_squat.mp4
```

With a webcam, press `ESC` or `Q` to stop early:

```bash
python run.py --student-id S001 --name "Asha Rao" --exercise pushup --reps 10 --source 0
```

Verify the logic without a camera or MediaPipe:

```bash
python selftest.py
```

## Files

| File | Purpose |
|---|---|
| [app.py](app.py) | Streamlit interface: setup → live → report, plus history |
| [run.py](run.py) | Command line runner |
| [config.py](config.py) | Exercise definitions and angle thresholds |
| [pose_engine.py](pose_engine.py) | MediaPipe wrapper, angle measurement, side selection |
| [exercises.py](exercises.py) | `RepCounter` — state machine and form validation |
| [videoStream.py](videoStream.py) | `ExerciseSession` — the shared capture/detect/count loop |
| [overlay.py](overlay.py) | On-frame real-time dashboard |
| [database.py](database.py) | SQLite storage (`data/sessions.db`) |
| [report.py](report.py) | PDF and CSV generation |
| [selftest.py](selftest.py) | Camera-free checks |

## Database

One table, `exercise_sessions`, written to `data/sessions.db`:

`session_id`, `student_id`, `student_name`, `exercise_type`,
`required_repetitions`, `completed_repetitions`, `valid_repetitions`,
`invalid_repetitions`, `duration`, `date_time`, `status`

Reports are written to `reports/` by the CLI; the Streamlit app serves them as
download buttons.

## Notes on accuracy

- The side (left/right) with better landmark visibility is chosen automatically
  each frame, so the student may face either way.
- When the pose is lost or joints are occluded, the counter **holds its state**
  rather than counting noise; the dashboard shows `NO POSE DETECTED`.
- Film from the side. A front-on camera cannot see elbow or knee bend reliably.
- Ensure the whole body is in frame, with even lighting.
