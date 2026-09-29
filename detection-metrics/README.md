# Detection Metrics: Precision, Recall & F1 Evaluation

A small pipeline to measure how well YOLOv8 detections match manually annotated ground truth, using IoU-based matching.

## What this does

Instead of just running object detection and visually checking the output, this project measures detection quality against a human-annotated reference using standard evaluation metrics.

## Pipeline

1. **extract_frames.py** — extracts 3 evenly spaced frames from a short pedestrian video
2. Manual annotation in CVAT / Label Studio — every clearly visible person labeled as ground truth (`ground_truth.json`, COCO format)
3. **run_yolo.py** — runs YOLOv8 on the same frames, saves predictions (`yolo_predictions.json`)
4. **compare_metrics.py** — matches predictions to ground truth using IoU (threshold 0.5), calculates precision, recall, and F1 score
5. **visualize_comparison.py** — draws ground truth (green) and predictions (red) on the same frames for visual comparison

## Results

Across 3 frames:
- Precision: 0.65
- Recall: 0.65
- F1 Score: 0.65

## Key finding

On clearly visible people in the foreground, YOLO's detections closely matched the manual annotations. Most mismatches came from small, blurred figures in the background, cases where the model detected something the annotator chose not to label due to unclear boundaries. This highlights a common gap in real-world QA: model output and human judgment can disagree not because either is "wrong," but because they define detectable objects differently.

## Files

- `extract_frames.py` — frame extraction from video
- `run_yolo.py` — YOLOv8 inference on frames
- `ground_truth.json` — manual annotations (COCO format)
- `yolo_predictions.json` — model predictions
- `compare_metrics.py` — IoU matching and metric calculation
- `visualize_comparison.py` — visual comparison generator
- `comparison_frame_1.jpg`, `comparison_frame_2.jpg`, `comparison_frame_3.jpg` — ground truth vs prediction visualizations
