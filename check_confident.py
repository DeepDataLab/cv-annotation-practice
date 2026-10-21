import json
import os
import cv2

RAW = "prelabels_raw.json"
SCRATCH = "audit_ls.json"
MATCH_IOU = 0.5
os.makedirs("check", exist_ok=True)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0


with open(SCRATCH) as f:
    tasks = json.load(f)
scratch = {}
for t in tasks:
    frame = t["data"]["image"].split("/")[-1].split("?")[0]
    anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        continue
    lst = []
    for r in anns[-1]["result"]:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        W, H = r["original_width"], r["original_height"]
        x1 = v["x"] / 100 * W
        y1 = v["y"] / 100 * H
        lst.append((x1, y1, x1 + v["width"] / 100 * W, y1 + v["height"] / 100 * H))
    scratch[frame] = lst

with open(RAW) as f:
    raw = json.load(f)

for frame, ref in scratch.items():
    flagged = []
    for r in raw:
        if r["frame"] != frame or r["conf"] < 0.7:
            continue
        mb = (r["x1"], r["y1"], r["x2"], r["y2"])
        if max((iou(mb, s) for s in ref), default=0) < MATCH_IOU:
            flagged.append(r)
    if not flagged:
        continue
    img = cv2.imread(f"frames/{frame}")
    for s in ref:  # твоя разметка с нуля: зелёные
        cv2.rectangle(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])),
                      (0, 200, 0), 2)
    for r in flagged:  # уверенные боксы модели, которых нет у тебя: красные
        cv2.rectangle(img, (int(r["x1"]), int(r["y1"])),
                      (int(r["x2"]), int(r["y2"])), (0, 0, 255), 3)
        cv2.putText(img, f'{r["label"]} {r["conf"]:.2f}',
                    (int(r["x1"]), max(20, int(r["y1"]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(f"check/{frame}", img)
    print(frame, "flagged:", len(flagged))