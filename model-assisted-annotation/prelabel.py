import json
import os
from ultralytics import YOLO

FRAMES_DIR = "frames"
MODEL = "yolov8n.pt"
CONF = 0.25
KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

model = YOLO(MODEL)
names = model.names

ls_tasks = []   # для Label Studio
raw_rows = []   # для SQL

for fname in sorted(os.listdir(FRAMES_DIR)):
    if not fname.endswith(".jpg"):
        continue
    path = os.path.join(FRAMES_DIR, fname)
    result = model.predict(path, conf=CONF, verbose=False)[0]
    h, w = result.orig_shape

    ls_results = []
    for box in result.boxes:
        label = names[int(box.cls)]
        if label not in KEEP:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)

        raw_rows.append({
            "frame": fname, "label": label, "conf": round(conf, 4),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
        })
        ls_results.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "score": conf,
            "value": {
                "x": x1 / w * 100,
                "y": y1 / h * 100,
                "width": (x2 - x1) / w * 100,
                "height": (y2 - y1) / h * 100,
                "rotation": 0,
                "rectanglelabels": [label],
            },
        })

    ls_tasks.append({
        "data": {"image": f"/data/local-files/?d=frames/{fname}"},
        "predictions": [{"model_version": "yolov8n", "result": ls_results}],
    })

with open("prelabels_ls.json", "w") as f:
    json.dump(ls_tasks, f, indent=2)
with open("prelabels_raw.json", "w") as f:
    json.dump(raw_rows, f, indent=2)

print(f"Frames: {len(ls_tasks)}, total pre-labeled boxes: {len(raw_rows)}")