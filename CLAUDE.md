# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Implementation follows the 13-step order in `prompt`. Step 1 (landmark extraction, `src/extracao_pose.py`) is done and validated. The spec documents (in Portuguese) are the source of truth:

- `readme.md` — full project scope: problem statement, scientific grounding, V1 scope, DB schema, pipeline, validation methodology, references
- `prompt` — development instructions distilled from the scope, including the required implementation order

Read both before implementing anything. Documentation, comments, code identifiers, and user-facing text are in Portuguese; the DB schema uses Portuguese names (`exercicio`, `serie`, `repeticao`, etc.).

## Commands

```powershell
# Always use the project venv (Python 3.13)
.\.venv\Scripts\python.exe src\extracao_pose.py data\videos_brutos\<video>.mp4 [--modelo heavy] [--sem-video-debug]
```

Outputs go to `data/landmarks/<video>_landmarks.csv` (one row per frame, wide format) and `data/debug/` (annotated video + sample PNG frames). The MediaPipe `.task` model auto-downloads to `modelos_mediapipe/` on first run.

## Environment gotchas

- **The mediapipe wheel for Python 3.13 is the slim Tasks-API-only build**: no `mp.solutions`, no `mediapipe.framework.formats.landmark_pb2`, no drawing utils. Use `mediapipe.tasks.python.vision.PoseLandmarker` and draw overlays with plain OpenCV (`CONEXOES_POSE` in `extracao_pose.py` replicates POSE_CONNECTIONS).
- Don't add `opencv-python` to requirements — mediapipe already pulls `opencv-contrib-python`; both installed together causes duplicate-cv2 conflicts.
- Windows console defaults to cp850; scripts call `sys.stdout.reconfigure(encoding="utf-8")` for accented output.

## Critical measurement fact (affects step 4 — velocity)

**World landmarks have their origin at the hip midpoint, so the hip midpoint's world coordinates are always ≈ (0,0,0).** The spec's original phrasing ("vertical displacement of the hip midpoint via world landmarks") is physically impossible as written — verified empirically (hip-midpoint world-y amplitude of 0.001 m over a full lifting video, while ankle world-y spans ~1.2 m). Concentric velocity must instead be measured as the hip's vertical movement **relative to a ground-anchored landmark (ankles)**, i.e. d/dt of the hip-to-ankle vertical distance in world coordinates (equivalently, the ankle's world-y signal, inverted). Document this as a justified deviation from the spec.

## What RepDecay is

A portfolio project that estimates proximity to muscular failure (RIR — reps in reserve) in the free squat from recorded video, using MediaPipe Pose world landmarks as a proxy for barbell velocity. This is velocity-based training (VBT) without the $400–$2,000 hardware. Everything must use free tools only — do not suggest paid alternatives.

**Stack (fixed):** Python 3.11+, MediaPipe Pose, OpenCV, NumPy/Pandas, SciPy (`savgol_filter`, `find_peaks`), SQLite, scikit-learn (linear regression), Streamlit or Tableau Public for the dashboard. Jupyter for exploration, `.py` scripts for the final pipeline.

## Non-negotiable methodological constraints

These come from the spec and matter more than "making it work":

1. **Validated vs. exploratory must stay explicit.** The velocity→RIR relationship and markerless pose estimation error margins are literature-backed. Using the *hip landmark* as a proxy for the barbell path is this project's own exploratory adaptation — no published study does it. Code comments, docs, and the final README must never present the exploratory part as scientifically validated. The project includes a mandatory self-validation step (automatic velocity vs. manual frame-by-frame marking on 5–10 reps, with the error documented).
2. **No data leakage in validation.** Train/test split is always **by session (date)**, never by individual repetition — reps in the same set are correlated. Evaluation metric: MAE between `rir_estimado` and `rir_real`.
3. **Real RIR labels, not perceived.** `rir_real` is generated retroactively only from sets taken to actual failure (last rep = 0, second-to-last = 1, …). Sets not taken to failure get only `rir_estimado` from the model.
4. **Model fallback logic:** per-user/per-exercise personal regression when enough failure-labeled history exists; otherwise a populational baseline from Paulsen et al. (2025) (~0.70 m/s at 79% 1RM → ~0.49 m/s near 89% 1RM in the squat).
5. **DB schema is fixed as a starting point** (see `readme.md` §5 / `prompt`). Adjustments are allowed but must be justified. It is deliberately structured for multiple exercises even though V1 only populates squat.
6. **V1 scope is closed.** In: one exercise (free squat, side camera, recorded video), concentric velocity in m/s via world landmarks, robust rep detection (Savitzky-Golay smoothing + `find_peaks` with prominence), manual logging of weight/reps/failure, simple vs. multi-variable model comparison, dashboard. Out (document as next steps, don't build): other exercises, real-time webcam, neural networks, mobile app.

## Implementation order

Follow the 13-step order in `prompt` (§"Ordem de implementação sugerida") — MediaPipe extraction test first, then knee angle + smoothing, rep detection, velocity, self-validation, DB, manual logging, RIR labeling, models, session-split validation, dashboard, final README. Build step by step; at each step explain the technical decisions and show evidence it works (values, plots) before moving on.

## Signal-processing specifics

- Rep boundaries come from the knee angle signal; depth = minimum knee angle.
- Concentric velocity = vertical displacement of the hip midpoint from **world landmarks** (3D coordinates in real meters, origin at hip midpoint) — not normalized landmarks.
- Smooth angle and vertical-position signals with Savitzky-Golay before peak detection; use `scipy.signal.find_peaks` with prominence to avoid phantom reps.
