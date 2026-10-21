import json
import sqlite3

RAW = "prelabels_raw.json"
SCRATCH = "audit_ls.json"
DB = "annotation_review.db"
MATCH_IOU = 0.5


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
        lst.append({"label": v["rectanglelabels"][0],
                    "box": (x1, y1, x1 + v["width"] / 100 * W,
                            y1 + v["height"] / 100 * H)})
    scratch[frame] = lst

with open(RAW) as f:
    raw = json.load(f)

con = sqlite3.connect(DB)
con.executescript("""
DROP TABLE IF EXISTS audit_model_boxes;
CREATE TABLE audit_model_boxes (
    frame TEXT, label TEXT, conf REAL, area REAL,
    in_scratch INTEGER, scratch_label TEXT
);
""")

for frame, ref in scratch.items():
    mboxes = [r for r in raw if r["frame"] == frame]
    pairs = []
    for i, m in enumerate(mboxes):
        mb = (m["x1"], m["y1"], m["x2"], m["y2"])
        for j, s in enumerate(ref):
            v = iou(mb, s["box"])
            if v >= MATCH_IOU:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_i, used_j, match = set(), set(), {}
    for v, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        match[i] = j
    for i, m in enumerate(mboxes):
        area = (m["x2"] - m["x1"]) * (m["y2"] - m["y1"])
        if i in match:
            con.execute("INSERT INTO audit_model_boxes VALUES (?,?,?,?,?,?)",
                        (frame, m["label"], m["conf"], area, 1,
                         ref[match[i]]["label"]))
        else:
            con.execute("INSERT INTO audit_model_boxes VALUES (?,?,?,?,?,?)",
                        (frame, m["label"], m["conf"], area, 0, None))
con.commit()


def show(title, sql):
    print(f"\n{title}")
    for row in con.execute(sql):
        print(" ", row)


show("By confidence: (bucket, boxes, also drawn from scratch, share)", """
SELECT CASE WHEN conf >= 0.7 THEN 'high >=0.7'
            WHEN conf >= 0.45 THEN 'mid 0.45-0.7'
            ELSE 'low <0.45' END AS bucket,
       COUNT(*), SUM(in_scratch), ROUND(1.0 * SUM(in_scratch) / COUNT(*), 2)
FROM audit_model_boxes GROUP BY bucket ORDER BY MIN(conf) DESC""")

show("By box size: (tier, boxes, also drawn from scratch, share)", """
WITH t AS (SELECT *, NTILE(3) OVER (ORDER BY area) AS tier FROM audit_model_boxes)
SELECT CASE tier WHEN 1 THEN 'small' WHEN 2 THEN 'medium' ELSE 'large' END,
       COUNT(*), SUM(in_scratch), ROUND(1.0 * SUM(in_scratch) / COUNT(*), 2)
FROM t GROUP BY tier ORDER BY tier""")

show("By class: (class, boxes, also drawn from scratch, share)", """
SELECT label, COUNT(*), SUM(in_scratch),
       ROUND(1.0 * SUM(in_scratch) / COUNT(*), 2)
FROM audit_model_boxes GROUP BY label ORDER BY 2 DESC""")

show("Class disagreements: (scratch said, model said, count)", """
SELECT scratch_label, label, COUNT(*) FROM audit_model_boxes
WHERE in_scratch = 1 AND scratch_label != label GROUP BY 1, 2 ORDER BY 3 DESC""")

show("Confident boxes missing from scratch, check them by eye: (frame, class, conf)", """
SELECT frame, label, ROUND(conf, 2) FROM audit_model_boxes
WHERE in_scratch = 0 AND conf >= 0.7 ORDER BY conf DESC LIMIT 10""")

con.close()