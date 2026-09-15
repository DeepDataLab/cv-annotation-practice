# IoU Evaluation — September 14, 2026

Comparing YOLO's predictions against manually annotated ground truth, using a self-written IoU (Intersection over Union) function instead of checking each frame by eye.

## Files

- `iou_check.py` — runs YOLO on two still frames and compares the result against manual CVAT annotation
- `instances_default.json` — the ground truth export from CVAT (COCO format) for the car frame

## What calculate_iou does

Takes two bounding boxes in `[x1, y1, x2, y2]` format, finds the overlapping rectangle, and divides its area by the combined area of both boxes. Returns 0 if there is no overlap.

Note: CVAT exports boxes in COCO format (`[x, y, width, height]`), which had to be converted to `[x1, y1, x2, y2]` before comparison.

## Results

**Car frame:** IoU = 0.96 against the manual annotation. The model's predicted box aligned very closely with the ground truth.

**Sign frame:** No prediction and no ground-truth object (this frame has no vehicle). No IoU to calculate here, since there's nothing to compare. Interesting on its own: the same object was falsely detected as "bus" elsewhere in the source video, but not on this particular still frame, showing how much a single detection can depend on the visual context of a frame.
