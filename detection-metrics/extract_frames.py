import cv2
import os

VIDEO_PATH = "input_video/pedestrians.mp4"
OUTPUT_DIR = "frames"
NUM_FRAMES = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Total frames in video: {total_frames}")

frame_indices = [
    int(total_frames * (i + 1) / (NUM_FRAMES + 1))
    for i in range(NUM_FRAMES)
]

print(f"Selected frame indices: {frame_indices}")

for i, frame_index in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    if success:
        output_path = os.path.join(OUTPUT_DIR, f"frame_{i+1}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to read frame at index {frame_index}")

cap.release()