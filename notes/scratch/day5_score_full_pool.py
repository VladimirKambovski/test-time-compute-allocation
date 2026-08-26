"""
Day 5: score the full Day-4 N=32 pool (100 MATH-500 problems x 32
samples = 3200) with the primary hosted PRM, now that G3 has passed on
the 10-problem subset (see day5_g3_check.py / day5_g3_analysis.py:
AUROC(mean_reward) = 0.9934 on double_newline segmentation, rung 1).

This is what makes B3 (PRM-weighted majority vs plain majority,
docs/brief.md line 101) computable at full scale, per Day 5's roadmap
item "Reproduce B3."

Same checkpointed-resumable, ThreadPoolExecutor pattern as
day4_generate_pool.py (concurrency=10 here -- the PRM /score endpoint
answered the 320-call G3 check near-instantly with zero errors, so this
starts more concurrent than the *policy* endpoint's hard-won concurrency=6
lesson from notes/2026-08-21.md, which was specific to that endpoint's
long (4096-token) completions, a different bottleneck entirely. Watched
for the same false-idle misdiagnosis Day 4 flagged: judge by checkpoint
growth, not elapsed time against an assumed rate, before concluding
anything is stuck.)
"""
import concurrent.futures
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
OUT_PATH = Path("notes/scratch/day5_full_pool_prm_scores.jsonl")
N_SAMPLES = 32  # match the primary G1/B3 scale, ignore the N=64 extension
CONVENTION = "double_newline"
MAX_WORKERS = 10


def fetch_all_problem_text(unique_ids):
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


def load_pool(problem_ids):
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


def precompute(task):
    """CPU-only: answer extraction + equivalence + segmentation. Deliberately
    run sequentially on the main thread, NOT inside the ThreadPoolExecutor
    below -- math_verify.parse()'s timeout mechanism uses signal.alarm(),
    which raises `ValueError: signal only works in main thread` if called
    from a worker thread. Found live (first real run of this script):
    every task crashed at the executor stage until this was split out.
    Only the network-bound PRM call (score_one) is safe to parallelize.
    """
    pid, rec, query = task
    extraction = extract_answer(rec["model_prediction"], finish_reason=rec.get("finish_reason"))
    if extraction.status == FailureStatus.OK:
        eq = check_equivalent(prediction=extraction.value, gold=rec["gold_answer"])
        is_correct = eq.equivalent
        answer_status = "ok" if eq.equivalent is not None else eq.status.value
    else:
        is_correct = None
        answer_status = extraction.status.value

    result = {
        "problem_id": pid,
        "sample_idx": rec["sample_idx"],
        "answer_status": answer_status,
        "is_correct": is_correct,
    }
    steps = segment(rec["model_prediction"], CONVENTION)
    return task, result, steps


def score_one(precomputed, client):
    """Network-bound only -- safe to run in a worker thread."""
    (pid, rec, query), result, steps = precomputed
    if not steps:
        result["segmentation_status"] = FailureStatus.STEP_SEGMENTATION_FAILED.value
        result["mean_reward"] = None
        result["final_step_reward"] = None
        return result

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
    return result


def main():
    all_ids = json.load(open(IDS_PATH))
    print(f"fetching problem text for {len(all_ids)} problems...")
    problem_text = fetch_all_problem_text(all_ids)
    missing = set(all_ids) - set(problem_text)
    if missing:
        raise RuntimeError(f"could not fetch problem text for: {missing}")

    by_problem = load_pool(all_ids)
    for pid, recs in by_problem.items():
        if len(recs) != N_SAMPLES:
            raise RuntimeError(f"{pid}: expected {N_SAMPLES} samples, got {len(recs)}")

    already = done_keys()
    total = len(all_ids) * N_SAMPLES
    print(f"{len(already)}/{total} already scored -- resuming" if already else "starting fresh")

    tasks = []
    for pid, recs in by_problem.items():
        query = problem_text[pid]
        for rec in recs:
            if (pid, rec["sample_idx"]) not in already:
                tasks.append((pid, rec, query))
    print(f"{len(tasks)} tasks remaining out of {total}")

    if not tasks:
        print("nothing to do -- already complete")
        return

    print("precomputing extraction/equivalence/segmentation (CPU-only, main thread)...")
    precomputed = [precompute(task) for task in tasks]

    client = HostedPRMClient()
    t0 = time.time()
    n_done = len(already)
    n_failed = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "a") as out, concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(score_one, p, client) for p in precomputed]
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                n_failed += 1
                print(f"task-level exception (not a PRM-status failure, a real crash): {exc!r}")
                continue
            out.write(json.dumps(result) + "\n")
            out.flush()
            n_done += 1
            if n_done % 200 == 0 or n_done == total:
                elapsed = time.time() - t0
                rate = (n_done - len(already)) / elapsed if elapsed > 0 else 0
                remaining = total - n_done
                eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                print(f"  {n_done}/{total} done, {n_failed} crashed, elapsed {elapsed/60:.1f}min, ETA {eta_min:.1f}min")

    print(f"final: {n_done}/{total}, {n_failed} task-level crashes")


if __name__ == "__main__":
    main()
