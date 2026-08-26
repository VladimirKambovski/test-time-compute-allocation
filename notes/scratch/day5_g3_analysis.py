"""Day 5: G3 AUROC computation from notes/scratch/day5_g3_prm_scores.jsonl.

AUROC(mean_reward -> is_correct) is the primary metric (mean_reward is
prm_primary's own headline aggregate, per hosted-endpoints.yaml's
response schema). final_step_reward is reported alongside as the common
alternate aggregation (last-step score is standard practice for PRMs
whose steps culminate in "here is the final answer") -- not used to
decide the gate, reported for completeness so the choice isn't silently
buried.
"""
import json
from collections import Counter

from sklearn.metrics import roc_auc_score

PATH = "notes/scratch/day5_g3_prm_scores.jsonl"
THRESHOLD = 0.6


def main():
    rows = [json.loads(line) for line in open(PATH) if line.strip()]
    print(f"total rows: {len(rows)}")

    status_counts = Counter(r["answer_status"] for r in rows)
    seg_counts = Counter(r["segmentation_status"] for r in rows)
    prm_counts = Counter(r.get("prm_status", "n/a (segmentation failed)") for r in rows)
    print(f"answer_status: {dict(status_counts)}")
    print(f"segmentation_status: {dict(seg_counts)}")
    print(f"prm_status: {dict(prm_counts)}")

    usable = [r for r in rows if r["is_correct"] is not None and r.get("mean_reward") is not None]
    excluded = len(rows) - len(usable)
    print(f"usable for AUROC (is_correct known AND PRM score present): {len(usable)}/{len(rows)} ({excluded} excluded)")

    y = [1 if r["is_correct"] else 0 for r in usable]
    n_pos, n_neg = sum(y), len(y) - sum(y)
    print(f"class balance: {n_pos} correct / {n_neg} incorrect")

    mean_scores = [r["mean_reward"] for r in usable]
    auroc_mean = roc_auc_score(y, mean_scores)
    print(f"AUROC (mean_reward):       {auroc_mean:.4f}")

    usable_final = [r for r in rows if r["is_correct"] is not None and r.get("final_step_reward") is not None]
    y_final = [1 if r["is_correct"] else 0 for r in usable_final]
    final_scores = [r["final_step_reward"] for r in usable_final]
    auroc_final = roc_auc_score(y_final, final_scores)
    print(f"AUROC (final_step_reward): {auroc_final:.4f}")

    verdict = "PASS" if auroc_mean > THRESHOLD else "FAIL"
    print(f"\nG3 verdict (mean_reward vs {THRESHOLD} threshold): {verdict}")


if __name__ == "__main__":
    main()
