import json

IOU_THRESHOLD = 0.5

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area


def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


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

total_tp = 0
total_fp = 0
total_fn = 0

for filename, predictions in pred_data.items():
    gt_boxes = gt_by_image.get(filename, [])
    pred_boxes = [p["bbox"] for p in predictions if p["class"] == "person"]

    matched_gt = set()
    tp = 0

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_index = -1
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = i

        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_gt.add(best_gt_index)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    print(f"{filename}: GT={len(gt_boxes)}, Predicted={len(pred_boxes)}, TP={tp}, FP={fp}, FN={fn}")

    total_tp += tp
    total_fp += fp
    total_fn += fn

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print()
print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")