# Model-assisted annotation: what changes when a human reviews pre-labels

A small experiment on one night street video (Bangkok traffic, Pixabay Content License):
YOLOv8n pre-labels, human review in Label Studio, and an independent from-scratch audit,
with everything stored in SQLite and compared with SQL.

## Pipeline

1. `extract_frames.py`: 30 evenly spaced frames from the video (943 frames total)
2. `prelabel.py`: YOLOv8n pre-labels (conf >= 0.25, 6 classes: person, bicycle, car, motorcycle, bus, truck), 338 boxes
3. Review in Label Studio: accept, edit, relabel, delete or add boxes
4. `build_db.py`: compares model output with the reviewed result, writes `annotation_review.db`
5. `make_audit_tasks.py`: picks 8 random frames (seed 42) for an independent from-scratch pass
6. `compare_audit.py`, `analyze_extras.py`, `check_confident.py`: compare model, review and from-scratch labels
7. `make_video.py`: demo video with boxes colored by confidence

## Results

**Review of pre-labels (30 frames):** 337 of 338 boxes accepted, 1 edited, 0 deleted, 0 added, 0 relabeled.
Total review time 7.7 minutes.

**From-scratch audit (8 frames):** 53 boxes drawn without seeing the predictions.
Time: 15.5 minutes from scratch vs 4.2 minutes reviewing the same frames (about 3.7x).

Model boxes compared with my from-scratch labels (IoU >= 0.5):

| | Boxes |
|---|---|
| Same place and class | 40 |
| Same place, different class | 8 |
| Found only from scratch (all people in a dense crowd) | 5 |
| Model boxes not in my from-scratch labels | 67 |

Share of model boxes that I also drew from scratch:

| Model confidence | Share |
|---|---|
| >= 0.70 | 88% (35 of 40) |
| 0.45 to 0.70 | 22% (7 of 32) |
| < 0.45 | 14% (6 of 43) |

By box size: small 10% (4 of 39), medium 42% (16 of 38), large 74% (28 of 38).

Most frequent class disagreements: person vs bus (3), bus vs car (2), bus vs truck (1).

## What I take from this

- Reviewing pre-labels was much faster, but I accepted almost everything (337 of 338).
  The from-scratch pass shows my standards were different in the two passes.
- Many of the "extra" model boxes are small, distant or partly visible objects that I
  skipped when labeling from scratch. That is a labeling-guideline difference, not
  necessarily a model error.
- Rider plus motorcycle is a good example: the model draws two boxes (person and motorcycle),
  my from-scratch labels used one. Without written rules, even one person labels inconsistently.
- Confidence was a useful signal of whether I would also draw the box. A threshold near 0.7
  looks worth testing, but 8 frames cannot prove it.

## Limitations

- One video, one night scene, 30 frames, 8 audited frames.
- One annotator did every pass. The from-scratch labels are a second opinion, not ground truth.
- I had already reviewed the audit frames with predictions before labeling them from scratch,
  so the audit was blind to the predictions only, not to the images.
- Times come from Label Studio and include pauses.
- Matching uses IoU 0.5 with one-to-one assignment, which is rough in crowds.

## Data

The video is not included (Pixabay Content License, "Traffic, Road, Bangkok" by Expatsiam).
Frames and the video are excluded by `.gitignore`.