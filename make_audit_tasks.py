import json
import random

random.seed(42)
frames = [f"frame_{i:03d}.jpg" for i in range(30)]
picked = sorted(random.sample(frames, 8))

tasks = [{"data": {"image": f"/data/local-files/?d=frames/{f}"}} for f in picked]

with open("audit_tasks.json", "w") as f:
    json.dump(tasks, f, indent=2)

print("Audit frames:", picked)