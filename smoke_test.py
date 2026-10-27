import json
import os
import sys

from fastapi.testclient import TestClient

from app.config import settings

if "--mock" in sys.argv:
    # .env is loaded with override=True, so force mock mode here rather than via env.
    settings.llm_provider = "mock"

from app.main import app  # noqa: E402

client = TestClient(app)

h = client.get("/api/health")
print("HEALTH:", h.json())

base = os.path.join("sample_data")
with open(os.path.join(base, "question_paper.txt"), "rb") as f1, \
     open(os.path.join(base, "answer_sheet_good.txt"), "rb") as f2, \
     open(os.path.join(base, "rubric.txt"), "rb") as f3, \
     open(os.path.join(base, "model_answers.txt"), "rb") as f4:
    r = client.post(
        "/api/grade",
        files={
            "question_paper": ("question_paper.txt", f1),
            "answer_sheet": ("answer_sheet_good.txt", f2),
            "rubric": ("rubric.txt", f3),
            "model_answers": ("model_answers.txt", f4),
        },
        data={"n_samples": "5", "threshold": "0.6"},
    )

print("STATUS:", r.status_code)
data = r.json()
if r.status_code != 200:
    print("ERROR:", json.dumps(data, indent=2))
    raise SystemExit(1)

print("TOTALS:", data["total_awarded"], "/", data["total_max"])
print("OVERALL CONF:", data["overall_confidence"])
print("FLAGGED:", data["flagged"])
print("WARNINGS:", data["warnings"])
for res in data["results"]:
    g = res["grading"]
    c = res["confidence"]
    print(f"  Q{g['question_id']}: {g['total_awarded']}/{g['max_marks']} conf={c['combined_confidence']:.3f} "
          f"(var={c['score_variance']:.3f} sem={c['semantic_consistency']:.3f}) flagged={c['flagged']} "
          f"samples={res['sampled_totals']}")
