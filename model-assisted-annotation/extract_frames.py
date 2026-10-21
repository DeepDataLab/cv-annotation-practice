import cv2
import os

VIDEO = "street.mp4"
N_FRAMES = 30

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = total // N_FRAMES
os.makedirs("frames", exist_ok=True)

saved = 0
for i in range(N_FRAMES):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imwrite(f"frames/frame_{i:03d}.jpg", frame)
    saved += 1

cap.release()
print(f"Total frames in video: {total}, saved: {saved}")