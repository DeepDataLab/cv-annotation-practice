# Object Detection Practice Log — September 5, 2026

First hands-on experience with object detection and tracking: running a pretrained model (YOLO, Ultralytics) on video, and my own Python analysis of what it produced. This is not a theory writeup (I have not studied YOLO's architecture, non-max suppression, or metrics like mAP) — it is a record of a real experiment: what I ran, what I saw, which limitations I found, and how I handled them.

## 1. Setup

Downloaded a short traffic video (Pixabay, ~1 minute, 1920x1080). The goal was not to build a detection system, but to see how a pretrained model behaves on real video, and to dig into its mistakes.

Tools: Python, the `ultralytics` library (pretrained YOLO model), `opencv-python`.

## 2. Final script

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.track(source="cars.mp4", save=True, tracker="bytetrack.yaml")

CONF_THRESHOLD = 0.5

unique_ids = set()
for r in results:
    if r.boxes.id is not None:
        for obj_id, conf in zip(r.boxes.id.tolist(), r.boxes.conf.tolist()):
            if conf >= CONF_THRESHOLD:
                unique_ids.add(int(obj_id))

print(f"Unique tracked objects (confidence >= {CONF_THRESHOLD}): {len(unique_ids)}")
```

`model.track()` is a ready-made library function that handles both detection and tracking (assigning IDs to objects across frames) using the ByteTrack algorithm. Everything below it (`CONF_THRESHOLD`, the loop, the `set()`) was written and added independently, on top of the pretrained model.

## 3. First run, no filter

3000 frames processed. The terminal printed per-frame detections line by line, for example:

```
384x640 6 cars, 1 bus, 1 truck, 33.9ms
```

The first version of the script (without `CONF_THRESHOLD`) reported:

```
Unique vehicles tracked: 139
```

A correction worth noting: the variable and print statement name were imprecise. The code collects IDs for **all** tracked objects (cars, a bus, a person, a motorcycle, anything the model detects), not only "vehicles." The gap between what the code actually counts and how I labeled it is itself a useful lesson: a metric's name being accurate matters as much as the number being correct.

## 4. Three issues found

### 4.1 False positive
A roadside sign/camera mounted on a pole was classified as a `bus`, but with **confidence 0.48** (in an earlier run, as low as 0.19) — the model itself was far from certain. Confidence score (0 to 1) reflects how sure the model is about a given prediction; a low value is a signal that the prediction deserves scrutiny, not blind trust.

### 4.2 Classification instability
The same van was tagged as `truck` from a distance, then as `car` once it got closer and more visual detail was available. One physical object, different predictions across frames. This is a known, documented limitation of detection on video, not an isolated glitch.

### 4.3 Inflated count (track fragmentation)
Objects occasionally disappeared from tracking for a few frames (occlusion, distance, motion blur) and were assigned a new ID upon reappearing, meaning the same vehicle could be counted more than once.

An important terminology correction I made to myself: the drop from 139 to 107 after filtering does **not** by itself prove ID switching (where the tracker confuses two different objects with each other). It shows that part of the counted IDs came from low-confidence detections. Proving an actual ID switch would require following a specific object frame by frame and confirming that a new ID really refers to the same physical object, which I did not do. For that reason, the more careful term used here is track fragmentation, not ID switch.

## 5. Fix applied

Added `CONF_THRESHOLD = 0.5`, a filter that discards low-confidence IDs before they are included in the count.

Result: **139 -> 107**.

Important: the video output itself (boxes, labels) did not change. `model.track(save=True)` renders the video independently of any later analysis, so the false `bus` detection is still visible in the video. The filter only affected how I analyzed the results, not what the model drew. This distinction, raw model output versus my analysis on top of it, was the main takeaway of the whole experiment.

## 6. Terms clarified through practice

- **Confidence score**: the model's certainty in a specific prediction (0 to 1)
- **Classification instability**: one object, different predicted classes across frames
- **Track fragmentation**: an object's ID is lost and a new one created for the same object (as opposed to ID switch, where the ID is confused between two different objects)
- **IoU (Intersection over Union)**: noted as a next step, the metric needed to programmatically compare predicted boxes against ground truth

## 7. Next step

Take a few specific frames with errors from this video (the false `bus`, the truck/car flip), manually annotate the correct label in CVAT, and compare predicted output against ground truth directly.
