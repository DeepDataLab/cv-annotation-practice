# Annotation QA Demo: Clean vs Corrupted Ground Truth

A small demonstration of how annotation quality issues can be identified visually using FiftyOne, built after completing the Hands-on Data Centric Visual AI course (UC Davis, Coursera).

## What this does

1. **corrupt_annotations.py** — takes the clean ground truth annotations (from the detection-metrics project) and creates an intentionally corrupted version: some bounding boxes are shifted, others are removed entirely, simulating common annotation errors.
2. **view_in_fiftyone.py** — loads both the clean and corrupted annotations into FiftyOne as separate label fields on the same images, allowing side-by-side visual comparison in the FiftyOne App.

## Why this matters

In data-centric AI, model performance depends as much on the quality of the training data as on the model itself. This exercise demonstrates a simple but practical QA workflow: comparing annotation sets visually to catch misplaced or missing labels before they affect model training or evaluation.

## Files

- `corrupt_annotations.py` — generates a corrupted version of the ground truth
- `view_in_fiftyone.py` — loads both annotation sets into FiftyOne for comparison
- `ground_truth.json` — original clean annotations (COCO format)
- `ground_truth_corrupted.json` — intentionally corrupted annotations
