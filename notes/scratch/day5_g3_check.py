"""
Day 5: Gate G3 -- do primary-PRM step scores predict correctness (AUROC >
0.6) on a 10-problem subset, before spending PRM-scoring budget on the
full Day-4 pool? Per docs/roadmap.md Day 5 / docs/brief.md's gate table
row G3, fallback order on failure: other segmentation conventions -> next
PRM ladder rung (Skywork fallback, §27.3a) -> G10 mentor decision.

Convention tried first: double_newline, per configs/prms/qwen-math-prm-7b.yaml's
documented order AND day5_segmentation.py's empirical scan (100%
completion coverage for this policy's actual output style, vs 0% for
literal "Step k:" and no known special token).

10-problem subset: the first 10 problem_ids (sorted, deterministic, not
cherry-picked) from the same frozen dev-100 set already used for Day 4's
G1 pool, with all N=32 samples each = 320 (steps, correctness) data
points -- real live calls to the hosted primary PRM
(math-prm.deb11.smoki.mk). N=32, not the N=64 extension, to match the
scale G3 is meant to gate before ("before any scale scoring" = before
scoring the full 3200-sample N=32 pool).

Checkpointed to notes/scratch/day5_g3_prm_scores.jsonl, resumable by
(problem_id, sample_idx) -- a PRM API call is still a real network call
that can fail transiently, same discipline as Day 4's generation scripts.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")
from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402

sys.path.insert(0, "notes/scratch")
from day5_prm_client import HostedPRMClient  # noqa: E402
from day5_segmentation import segment  # noqa: E402

POOL_PATH = "notes/scratch/day4_pool.jsonl"
IDS_PATH = "configs/benchmarks/data/math500-dev100-ids.json"
OUT_PATH = Path("notes/scratch/day5_g3_prm_scores.jsonl")
N_PROBLEMS = 10
N_SAMPLES = 32  # match the primary G1 scale, ignore the N=64 extension for this check
CONVENTION = "double_newline"


def fetch_problem_text(unique_ids):
    """Same HF datasets-server REST pattern as day4_generate_pool.py --
    no `datasets` package dependency, just urllib against the public
    rows API.
    """
    wanted = set(unique_ids)
    found = {}
    offset = 0
    while offset < 500 and len(found) < len(wanted):
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
            f"&config=default&split=test&offset={offset}&length=100"
        )
        d = {}
        for _attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    d = json.load(resp)
                break
            except Exception:
                time.sleep(2)
        rows = d.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            if row["unique_id"] in wanted:
                found[row["unique_id"]] = row["problem"]
        offset += 100
    return found


def load_pool_subset(problem_ids):
    by_problem = {pid: [] for pid in problem_ids}
    with open(POOL_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["problem_id"] in by_problem and rec["sample_idx"] < N_SAMPLES:
                by_problem[rec["problem_id"]].append(rec)
    for pid in by_problem:
        by_problem[pid].sort(key=lambda r: r["sample_idx"])
    return by_problem


def done_keys():
    if not OUT_PATH.exists():
        return set()
    keys = set()
    with open(OUT_PATH) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                keys.add((d["problem_id"], d["sample_idx"]))
    return keys


def main():
    all_ids = json.load(open(IDS_PATH))
    subset_ids = sorted(all_ids)[:N_PROBLEMS]
    print(f"G3 subset: {len(subset_ids)} problems: {subset_ids}")

    problem_text = fetch_problem_text(subset_ids)
    missing = set(subset_ids) - set(problem_text)
    if missing:
        raise RuntimeError(f"could not fetch problem text for: {missing}")

    by_problem = load_pool_subset(subset_ids)
    for pid, recs in by_problem.items():
        if len(recs) != N_SAMPLES:
            raise RuntimeError(f"{pid}: expected {N_SAMPLES} samples in day4_pool.jsonl, got {len(recs)}")

    client = HostedPRMClient()
    already = done_keys()
    total = len(subset_ids) * N_SAMPLES
    n_done = len(already)
    print(f"resuming: {n_done}/{total} already scored")

    with open(OUT_PATH, "a") as out:
        for pid, recs in by_problem.items():
            query = problem_text[pid]
            for rec in recs:
                key = (pid, rec["sample_idx"])
                if key in already:
                    continue

                extraction = extract_answer(rec["model_prediction"], finish_reason=rec.get("finish_reason"))
                if extraction.status == FailureStatus.OK:
                    eq = check_equivalent(prediction=extraction.value, gold=rec["gold_answer"])
                    is_correct = eq.equivalent  # may be None on equivalence_timeout
                    answer_status = "ok" if eq.equivalent is not None else eq.status.value
                else:
                    is_correct = None
                    answer_status = extraction.status.value

                steps = segment(rec["model_prediction"], CONVENTION)
                result = {
                    "problem_id": pid,
                    "sample_idx": rec["sample_idx"],
                    "answer_status": answer_status,
                    "is_correct": is_correct,
                }
                if not steps:
                    result["segmentation_status"] = FailureStatus.STEP_SEGMENTATION_FAILED.value
                    result["mean_reward"] = None
                    result["final_step_reward"] = None
                else:
                    score = client.score(query, steps)
                    result["segmentation_status"] = "ok"
                    result["num_steps"] = score.num_steps
                    if score.ok:
                        result["prm_status"] = "ok"
                        result["mean_reward"] = score.mean_reward
                        result["final_step_reward"] = score.step_rewards[-1] if score.step_rewards else None
                    else:
                        result["prm_status"] = FailureStatus.PRM_SCORE_MISSING.value
                        result["prm_error"] = score.error
                        result["mean_reward"] = None
                        result["final_step_reward"] = None

                out.write(json.dumps(result) + "\n")
                out.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done == total:
                    print(f"{n_done}/{total}")

    print(f"done: {n_done}/{total}")


if __name__ == "__main__":
    main()
