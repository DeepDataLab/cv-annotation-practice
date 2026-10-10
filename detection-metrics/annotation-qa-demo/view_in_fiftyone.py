import fiftyone as fo
import json

def coco_bbox_to_fiftyone(bbox, img_width, img_height):
    x, y, w, h = bbox
    return [x / img_width, y / img_height, w / img_width, h / img_height]

with open("ground_truth.json", "r") as f:
    clean_data = json.load(f)

with open("ground_truth_corrupted.json", "r") as f:
    corrupted_data = json.load(f)

image_id_to_filename = {}
for img in clean_data["images"]:
    filename = img["file_name"].split("/")[-1]
    filename = filename.split("-")[-1]
    image_id_to_filename[img["id"]] = {
        "filename": filename,
        "width": img["width"],
        "height": img["height"],
    }

clean_by_image = {}
for ann in clean_data["annotations"]:
    info = image_id_to_filename[ann["image_id"]]
    clean_by_image.setdefault(info["filename"], []).append(ann)

corrupted_by_image = {}
for ann in corrupted_data["annotations"]:
    info = image_id_to_filename[ann["image_id"]]
    corrupted_by_image.setdefault(info["filename"], []).append(ann)

dataset_name = "annotation-qa-demo"
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)

dataset = fo.Dataset(dataset_name)

for filename, info in [(v["filename"], v) for v in image_id_to_filename.values()]:
    filepath = filename
    sample = fo.Sample(filepath=filepath)

    clean_dets = []
    for ann in clean_by_image.get(filename, []):
        bbox = coco_bbox_to_fiftyone(ann["bbox"], info["width"], info["height"])
        clean_dets.append(fo.Detection(label="person", bounding_box=bbox))
    sample["ground_truth_clean"] = fo.Detections(detections=clean_dets)

    corrupted_dets = []
    for ann in corrupted_by_image.get(filename, []):
        bbox = coco_bbox_to_fiftyone(ann["bbox"], info["width"], info["height"])
        label = "person (shifted)" if ann.get("_corruption") == "shifted" else "person"
        corrupted_dets.append(fo.Detection(label=label, bounding_box=bbox))
    sample["ground_truth_corrupted"] = fo.Detections(detections=corrupted_dets)

    dataset.add_sample(sample)

session = fo.launch_app(dataset)
session.wait()