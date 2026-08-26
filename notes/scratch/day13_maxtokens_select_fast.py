"""
Day 13, max_tokens angle -- fast P1-only version (the P1+P2 combined
version hung/was slow, likely the OlympiadBench gold-fetch or a slow
equivalence check; not needed here since we only want hard MATH-500
problems). Bands computed on P1 alone (500 problems, not the official
754-problem P1+P2 banding) -- close enough to identify clearly-hardest,
most-truncated MATH-500 problems, which is all this needs.
"""
import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"


def fetch_math500_gold_all():
    found = {}
    offset = 0
    while offset < 500:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = {}
        for _ in range(3):
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
            found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def main():
    print("fetching gold...", flush=True)
    gold = fetch_math500_gold_all()
    print(f"got {len(gold)} gold answers", flush=True)

    math_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        if not problem_id.isdigit():
            math_meta.append((pool_id, problem_id))

    store = PoolStore(POOL_ROOT)
    scored = []  # (pid, pass_at_1, n_ok)
    n_processed = 0
    for pool_id, pid in math_meta:
        if pid not in gold:
            continue
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            continue
        c, ok = 0, 0
        for s in pool.samples:
            ext = extract_answer(s.text, finish_reason=s.finish_reason)
            if ext.status == FailureStatus.OK:
                ok += 1
                if check_equivalent(prediction=ext.value, gold=gold[pid]).equivalent:
                    c += 1
        scored.append((pid, c / 32, ok))
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    print(f"\ntotal P1 problems processed: {n_processed}")
    scored.sort(key=lambda t: (t[1], t[0]))  # hardest first

    TARGET_N = 6
    hardest = scored[:TARGET_N]
    print(f"\n{TARGET_N} hardest MATH-500 problems (P1-only banding):")
    for pid, p1, n_ok in hardest:
        print(f"  {pid}  pass@1={p1:.3f}  successful_extractions={n_ok}/32")

    ids_file = "notes/scratch/day13_maxtokens_ablation_ids.json"
    json.dump([pid for pid, _, _ in hardest], open(ids_file, "w"), indent=2)
    print(f"\nwrote {ids_file}")


if __name__ == "__main__":
    main()
