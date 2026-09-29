import cv2
import json

def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]

with open("ground_truth.json", "r") as f:
    gt_data = json.load(f)

with open("yolo_predictions.json", "r") as f:
    pred_data = json.load(f)

image_id_to_filename = {}
for img in gt_data["images"]:
    filename = img["file_name"].split("/")[-1]
    filename = filename.split("-")[-1]
    image_id_to_filename[img["id"]] = filename

gt_by_image = {}
for ann in gt_data["annotations"]:
    filename = image_id_to_filename[ann["image_id"]]
    gt_by_image.setdefault(filename, []).append(coco_bbox_to_xyxy(ann["bbox"]))

for filename in ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]:
    image_path = f"frames/{filename}"
    image = cv2.imread(image_path)

    gt_boxes = gt_by_image.get(filename, [])
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    predictions = pred_data.get(filename, [])
    for pred in predictions:
        if pred["class"] != "person":
            continue
        x1, y1, x2, y2 = [int(v) for v in pred["bbox"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    output_path = f"comparison_{filename}"
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")