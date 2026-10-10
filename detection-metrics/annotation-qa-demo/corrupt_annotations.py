import json
import random
import copy

random.seed(42)

with open("ground_truth.json", "r") as f:
    data = json.load(f)

corrupted = copy.deepcopy(data)
annotations = corrupted["annotations"]

num_to_shift = max(1, len(annotations) // 4)
shift_indices = random.sample(range(len(annotations)), num_to_shift)

for idx in shift_indices:
    bbox = annotations[idx]["bbox"]
    shift_x = random.randint(40, 90)
    shift_y = random.randint(40, 90)
    annotations[idx]["bbox"] = [bbox[0] + shift_x, bbox[1] + shift_y, bbox[2], bbox[3]]
    annotations[idx]["_corruption"] = "shifted"

num_to_remove = max(1, len(annotations) // 6)
remaining_indices = [i for i in range(len(annotations)) if i not in shift_indices]
remove_indices = random.sample(remaining_indices, min(num_to_remove, len(remaining_indices)))

corrupted["annotations"] = [
    ann for i, ann in enumerate(annotations) if i not in remove_indices
]

print(f"Total original annotations: {len(data['annotations'])}")
print(f"Shifted (misplaced) boxes: {len(shift_indices)}")
print(f"Removed (missing) boxes: {len(remove_indices)}")
print(f"Total corrupted annotations: {len(corrupted['annotations'])}")

with open("ground_truth_corrupted.json", "w") as f:
    json.dump(corrupted, f, indent=2)

print("Saved: ground_truth_corrupted.json")