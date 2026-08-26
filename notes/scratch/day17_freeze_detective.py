"""
Day 17: freeze the final Detective model, per docs/brief.md section 16's
Protocol ("Coefficients frozen and tagged before Day 18"). This is the
DEPLOYED model -- fit once on ALL 754 canonical problems (not CV-folded;
CV was for evaluating generalization, this is for production use), then
persisted so the demo (and any future live gateway use) loads it rather
than refitting on every request.

Reuses the exact same canonical-only enumeration fix from
day15_e5_corrected.py -- never trust the glob for problem identity.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "src")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.controller.predictor import DetectiveController  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402


def fetch_math500_all():
    found = {}
    offset = 0
    while offset < 500:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d["rows"]:
            row = r["row"]
            found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def fetch_olympiad_all(wanted_ids):
    wanted = set(wanted_ids)
    found = {}
    offset = 0
    while offset < 674 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=Hothan/OlympiadBench"
               f"&config=OE_TO_maths_en_COMP&split=train&offset={offset}&length=100")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d["rows"]:
            row = r["row"]
            row_id = str(row["id"])
            if row_id in wanted and not row.get("is_multiple_answer") and not row.get("error") and row.get("final_answer"):
                found[row_id] = row["final_answer"][0]
        offset += 100
    return found


def main():
    print("fetching gold answers...", flush=True)
    math_gold = fetch_math500_all()
    oly_ids = [str(x) for x in json.load(open("configs/benchmarks/data/olympiad-a-ids.json"))]
    oly_gold = fetch_olympiad_all(oly_ids)
    print(f"got {len(math_gold)} MATH-500, {len(oly_gold)}/{len(oly_ids)} OlympiadBench-A", flush=True)

    store = PoolStore("results/pools")
    X_dicts, y = [], []

    for pid, gold in math_gold.items():
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="math500",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        X_dicts.append(featurize(Probe(samples=ordered[:4])))
        y.append(oracle_action_label(pool.samples, gold).action)

    for pid in oly_ids:
        if pid not in oly_gold:
            continue
        gold = oly_gold[pid]
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="olympiad-a",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "olympiad-a", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        X_dicts.append(featurize(Probe(samples=ordered[:4])))
        y.append(oracle_action_label(pool.samples, gold).action)

    print(f"total training rows: {len(X_dicts)}", flush=True)
    from collections import Counter
    print(f"class distribution: {dict(Counter(y))}")

    controller = DetectiveController()
    controller.fit(X_dicts, y)

    artifact_path = "results/models/detective_frozen.joblib"
    joblib.dump(controller, artifact_path)

    meta = {
        "frozen_at": "2026-08-26 (Day 17, per docs/brief.md section 16 Protocol: "
                      "'coefficients frozen and tagged before Day 18')",
        "n_training_rows": len(X_dicts),
        "class_distribution": dict(Counter(y)),
        "feature_names": list(FEATURE_NAMES),
        "coefficients_by_class_and_feature": {
            cls: dict(zip(FEATURE_NAMES, controller.model.coef_[i].tolist()))
            for i, cls in enumerate(controller.model.classes_)
        } if len(controller.model.classes_) > 2 else "binary case, see model.coef_ directly",
        "note": "Fit once on ALL 754 canonical P1+P2 problems (not CV-folded -- this is the "
                "deployed model, CV was for evaluating generalization in day15_e5_corrected.py). "
                "macro-AUROC under proper held-out CV was 0.8692 (notes/2026-08-26.md) -- THIS "
                "specific fit's in-sample accuracy will look better than that and should not be "
                "quoted as the real performance number.",
    }
    json.dump(meta, open("results/models/detective_frozen_meta.json", "w"), indent=2)
    print(f"\nfroze model to {artifact_path}")
    print(f"wrote metadata to results/models/detective_frozen_meta.json")


if __name__ == "__main__":
    main()
