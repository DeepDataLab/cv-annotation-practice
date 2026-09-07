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
