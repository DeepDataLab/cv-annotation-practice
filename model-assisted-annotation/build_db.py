import json
import os
import sqlite3

RAW = "prelabels_raw.json"
EXPORT = "annotations_ls.json"
DB = "annotation_review.db"
MATCH_IOU = 0.5   # бокс считается тем же объектом
SAME_IOU = 0.97   # выше этого бокс не менялся


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0


# --- модель (до) ---
with open(RAW) as f:
    raw = json.load(f)
model = {}
for r in raw:
    model.setdefault(r["frame"], []).append(r)

# --- человек (после) ---
with open(EXPORT) as f:
    tasks = json.load(f)

final = {}
seconds = {}
for t in tasks:
    frame = t["data"]["image"].split("/")[-1].split("?")[0]
    anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        continue
    ann = anns[-1]
    seconds[frame] = ann.get("lead_time", 0)
    boxes = []
    for r in ann["result"]:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        W, H = r["original_width"], r["original_height"]
        x1 = v["x"] / 100 * W
        y1 = v["y"] / 100 * H
        x2 = x1 + v["width"] / 100 * W
        y2 = y1 + v["height"] / 100 * H
        boxes.append({"label": v["rectanglelabels"][0],
                      "box": (x1, y1, x2, y2)})
    final[frame] = boxes

# --- база ---
if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
con.executescript("""
CREATE TABLE frames (
    frame TEXT PRIMARY KEY,
    review_seconds REAL
);
CREATE TABLE boxes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame TEXT,
    model_label TEXT,
    model_conf REAL,
    final_label TEXT,
    status TEXT,
    iou REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL
);
""")

for frame in sorted(final):
    con.execute("INSERT INTO frames VALUES (?, ?)", (frame, seconds[frame]))
    m_list = model.get(frame, [])
    f_list = final[frame]

    pairs = []
    for i, m in enumerate(m_list):
        mb = (m["x1"], m["y1"], m["x2"], m["y2"])
        for j, fb in enumerate(f_list):
            s = iou(mb, fb["box"])
            if s >= MATCH_IOU:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)

    used_m, used_f, match = set(), set(), {}
    for s, i, j in pairs:
        if i in used_m or j in used_f:
            continue
        used_m.add(i)
        used_f.add(j)
        match[i] = (j, s)

    for i, m in enumerate(m_list):
        mb = (m["x1"], m["y1"], m["x2"], m["y2"])
        if i in match:
            j, s = match[i]
            fb = f_list[j]
            if fb["label"] != m["label"]:
                status = "relabeled"
            elif s < SAME_IOU:
                status = "edited"
            else:
                status = "accepted"
            con.execute(
                "INSERT INTO boxes (frame, model_label, model_conf, final_label,"
                " status, iou, x1, y1, x2, y2) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (frame, m["label"], m["conf"], fb["label"], status, round(s, 4),
                 *fb["box"]))
        else:
            con.execute(
                "INSERT INTO boxes (frame, model_label, model_conf, final_label,"
                " status, iou, x1, y1, x2, y2) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (frame, m["label"], m["conf"], None, "deleted", None, *mb))

    for j, fb in enumerate(f_list):
        if j not in used_f:
            con.execute(
                "INSERT INTO boxes (frame, model_label, model_conf, final_label,"
                " status, iou, x1, y1, x2, y2) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (frame, None, None, fb["label"], "added", None, *fb["box"]))

con.commit()

print("Frames:", con.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
print("Total review time (min):",
      round(con.execute("SELECT SUM(review_seconds) FROM frames").fetchone()[0] / 60, 1))
print("\nStatus counts:")
for row in con.execute(
        "SELECT status, COUNT(*) FROM boxes GROUP BY status ORDER BY 2 DESC"):
    print(" ", row)
con.close()