import json
import sqlite3

con = sqlite3.connect("annotation_review.db")
rows = con.execute(
    "SELECT label, conf, in_scratch FROM audit_model_boxes").fetchall()
total = con.execute(
    "SELECT COUNT(*) FROM audit_compare WHERE source='model' "
    "AND outcome IN ('match', 'class_mismatch', 'missed')").fetchone()[0]
con.close()

data = {
    "total_scratch": total,
    "boxes": [{"label": l, "conf": round(c, 3), "hit": int(h)}
              for l, c, h in rows],
}
with open("data.json", "w") as f:
    json.dump(data, f)

print("model boxes:", len(rows), "| my from-scratch boxes:", total)