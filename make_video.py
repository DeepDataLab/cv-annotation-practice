import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "street.mp4"
OUT = "confidence_demo.mp4"
MAX_FRAMES = 300          # около 10-12 секунд
W, H = 1280, 720
CONF = 0.25
KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

# цвета в BGR (бирюзовый, золотой, коралловый, тёмно-синий фон)
TEAL, GOLD, CORAL, NAVY = (166, 184, 20), (66, 197, 245), (68, 68, 239), (40, 22, 10)
WHITE = (240, 240, 240)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def tier(conf):
    if conf >= 0.70:
        return 0, TEAL
    if conf >= 0.45:
        return 1, GOLD
    return 2, CORAL


model = YOLO("yolov8n.pt")
names = model.names

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

n = 0
while n < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (W, H))
    result = model.predict(frame, conf=CONF, verbose=False)[0]

    counts = [0, 0, 0]
    for box in result.boxes:
        label = names[int(box.cls)]
        if label not in KEEP:
            continue
        conf = float(box.conf)
        t, color = tier(conf)
        counts[t] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{conf:.2f}", (x1 + 3, y1 + 18), FONT, 0.55, color, 2)

    # панель-легенда
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (470, 175), NAVY, -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    cv2.putText(frame, "YOLOv8n pre-labels by confidence", (35, 52), FONT, 0.7, WHITE, 2)
    rows = [("conf >= 0.70", TEAL, counts[0]),
            ("conf 0.45 - 0.70", GOLD, counts[1]),
            ("conf < 0.45", CORAL, counts[2])]
    for i, (text, color, c) in enumerate(rows):
        y = 88 + i * 32
        cv2.rectangle(frame, (35, y - 16), (55, y + 4), color, -1)
        cv2.putText(frame, f"{text}: {c}", (70, y), FONT, 0.65, WHITE, 2)

    writer.write(frame)
    n += 1
    if n % 50 == 0:
        print("frames processed:", n)

# финальная карточка
card = np.full((H, W, 3), NAVY, dtype=np.uint8)
lines = [
    ("Model boxes I also drew from scratch", 1.1, WHITE, 2),
    ("conf >= 0.70:      88%   (35 of 40)", 1.0, TEAL, 2),
    ("conf 0.45 - 0.70:  22%   (7 of 32)", 1.0, GOLD, 2),
    ("conf < 0.45:       14%   (6 of 43)", 1.0, CORAL, 2),
    ("8 frames, one annotator. A pattern, not proof.", 0.75, WHITE, 1),
]
y = 230
for text, scale, color, th in lines:
    cv2.putText(card, text, (140, y), FONT, scale, color, th, cv2.LINE_AA)
    y += 80
for _ in range(int(fps * 3.5)):
    writer.write(card)

cap.release()
writer.release()
print("Saved:", OUT)