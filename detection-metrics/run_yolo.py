from ultralytics import YOLO
import json
import os

model = YOLO("yolov8n.pt")

FRAMES_DIR = "frames"
OUTPUT_JSON = "yolo_predictions.json"

all_predictions = {}

for filename in sorted(os.listdir(FRAMES_DIR)):
    if not filename.endswith(".jpg"):
        continue

    frame_path = os.path.join(FRAMES_DIR, filename)
    results = model(frame_path)

    boxes_data = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        boxes_data.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        })

    all_predictions[filename] = boxes_data
    print(f"{filename}: {len(boxes_data)} detections")

with open(OUTPUT_JSON, "w") as f:
    json.dump(all_predictions, f, indent=2)

print(f"Saved predictions to {OUTPUT_JSON}")