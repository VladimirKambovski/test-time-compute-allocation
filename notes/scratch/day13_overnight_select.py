"""
Overnight extension of the band-0 max_tokens finding: select the next
24 hardest MATH-500 problems (P1-only banding, same fast approach as
day13_maxtokens_select_fast.py), EXCLUDING the 6 already tested tonight
-- maximizes new coverage rather than re-processing known results.
Combined with tonight's 6, gives a systematic n=30 sample of the
hardest MATH-500 tail by morning.
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
ALREADY_TESTED = {
    "test/algebra/2176.json", "test/algebra/297.json", "test/algebra/686.json",
    "test/algebra/892.json", "test/counting_and_probability/10.json",
    "test/counting_and_probability/1003.json",
}


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
    scored = []
    n_processed = 0
    for pool_id, pid in math_meta:
        if pid not in gold or pid in ALREADY_TESTED:
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

    print(f"\ntotal P1 problems processed (excluding already-tested 6): {n_processed}")
    scored.sort(key=lambda t: (t[1], t[0]))

    TARGET_N = 24
    hardest = scored[:TARGET_N]
    print(f"\n{TARGET_N} next-hardest MATH-500 problems:")
    for pid, p1, n_ok in hardest:
        print(f"  {pid}  pass@1={p1:.3f}  successful_extractions={n_ok}/32")

    ids_file = "notes/scratch/day13_overnight_maxtokens_ids.json"
    json.dump([pid for pid, _, _ in hardest], open(ids_file, "w"), indent=2)
    print(f"\nwrote {ids_file}")


if __name__ == "__main__":
    main()
