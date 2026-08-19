import json
import sys

sys.path.insert(0, "src")
from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus

raw = json.load(open("notes/scratch/golden_200_raw.json"))
assert len(raw) == 200

scored = []
for r in raw:
    extraction = extract_answer(r["model_prediction"], finish_reason=r.get("finish_reason"))
    if extraction.status != FailureStatus.OK:
        scored.append({
            **r,
            "extraction_status": extraction.status.value,
            "extracted_value": None,
            "auto_equivalent": None,
        })
        continue
    eq = check_equivalent(prediction=extraction.value, gold=r["gold_answer"])
    scored.append({
        **r,
        "extraction_status": extraction.status.value,
        "extracted_value": str(extraction.value),
        "auto_equivalent": eq.equivalent,
        "equivalence_status": eq.status.value,
    })

with open("notes/scratch/golden_200_scored.json", "w") as f:
    json.dump(scored, f, indent=2)

from collections import Counter
status_counts = Counter(s["extraction_status"] for s in scored)
print("extraction status counts:", dict(status_counts))
eq_counts = Counter(s.get("auto_equivalent") for s in scored if s["extraction_status"] == "ok")
print("equivalence verdict counts (extraction ok only):", dict(eq_counts))
print(f"non-ok extraction rate: {sum(v for k,v in status_counts.items() if k != 'ok')}/{len(scored)} = {sum(v for k,v in status_counts.items() if k != 'ok')/len(scored):.1%}")
