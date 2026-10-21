import json
import sqlite3

RAW = "prelabels_raw.json"
REVIEW = "annotations_ls.json"
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


def load_ls(path):
    with open(path) as f:
        tasks = json.load(f)
    boxes, seconds = {}, {}
    for t in tasks:
        frame = t["data"]["image"].split("/")[-1].split("?")[0]
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        ann = anns[-1]
        seconds[frame] = ann.get("lead_time", 0)
        lst = []
        for r in ann["result"]:
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            W, H = r["original_width"], r["original_height"]
            x1 = v["x"] / 100 * W
            y1 = v["y"] / 100 * H
            x2 = x1 + v["width"] / 100 * W
            y2 = y1 + v["height"] / 100 * H
            lst.append({"label": v["rectanglelabels"][0], "box": (x1, y1, x2, y2)})
        boxes[frame] = lst
    return boxes, seconds


def compare(ref, other):
    pairs = []
    for i, r in enumerate(ref):
        for j, o in enumerate(other):
            s = iou(r["box"], o["box"])
            if s >= MATCH_IOU:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_r, used_o, rows = set(), set(), []
    for s, i, j in pairs:
        if i in used_r or j in used_o:
            continue
        used_r.add(i)
        used_o.add(j)
        same = ref[i]["label"] == other[j]["label"]
        rows.append(("match" if same else "class_mismatch",
                     ref[i]["label"], other[j]["label"], round(s, 4)))
    for i, r in enumerate(ref):
        if i not in used_r:
            rows.append(("missed", r["label"], None, None))
    for j, o in enumerate(other):
        if j not in used_o:
            rows.append(("extra", None, o["label"], None))
    return rows


# --- данные ---
scratch, scratch_sec = load_ls(SCRATCH)
review, review_sec = load_ls(REVIEW)

with open(RAW) as f:
    raw = json.load(f)
model = {}
for r in raw:
    model.setdefault(r["frame"], []).append(
        {"label": r["label"], "box": (r["x1"], r["y1"], r["x2"], r["y2"])})

frames = sorted(scratch)

# --- база ---
con = sqlite3.connect(DB)
con.executescript("""
DROP TABLE IF EXISTS audit_compare;
DROP TABLE IF EXISTS audit_frames;
CREATE TABLE audit_compare (
    source TEXT,        -- 'model' или 'review'
    frame TEXT,
    outcome TEXT,       -- match / class_mismatch / missed / extra
    scratch_label TEXT,
    other_label TEXT,
    iou REAL
);
CREATE TABLE audit_frames (
    frame TEXT PRIMARY KEY,
    scratch_seconds REAL,
    review_seconds REAL
);
""")

for frame in frames:
    con.execute("INSERT INTO audit_frames VALUES (?,?,?)",
                (frame, scratch_sec[frame], review_sec.get(frame)))
    for source, data in (("model", model), ("review", review)):
        for outcome, sl, ol, s in compare(scratch[frame], data.get(frame, [])):
            con.execute("INSERT INTO audit_compare VALUES (?,?,?,?,?,?)",
                        (source, frame, outcome, sl, ol, s))
con.commit()

# --- отчёт ---
n_scratch = sum(len(scratch[f]) for f in frames)
print(f"Audit frames: {len(frames)}, boxes drawn from scratch: {n_scratch}")
t_s = sum(scratch_sec[f] for f in frames) / 60
t_r = sum(review_sec.get(f, 0) for f in frames) / 60
print(f"Time from scratch: {t_s:.1f} min | time reviewing the same frames: {t_r:.1f} min")

for source in ("model", "review"):
    print(f"\n=== {source} vs from-scratch ===")
    counts = dict(con.execute(
        "SELECT outcome, COUNT(*) FROM audit_compare WHERE source=? GROUP BY outcome",
        (source,)).fetchall())
    m = counts.get("match", 0)
    cm = counts.get("class_mismatch", 0)
    miss = counts.get("missed", 0)
    ex = counts.get("extra", 0)
    print("  counts:", counts)
    if m + cm + miss:
        print(f"  recall vs scratch:    {m / (m + cm + miss):.3f}")
    if m + cm + ex:
        print(f"  precision vs scratch: {m / (m + cm + ex):.3f}")

print("\nMissed by review (found only when annotating from scratch), by class:")
for row in con.execute(
        "SELECT scratch_label, COUNT(*) FROM audit_compare "
        "WHERE source='review' AND outcome='missed' GROUP BY 1 ORDER BY 2 DESC"):
    print(" ", row)

con.close()