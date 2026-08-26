"""
Day 4: generate the real P1-ish pool -- 100 MATH-500 problems x N=32
samples, on the hosted Qwen3.5-4B endpoint, at the frozen decode config.

Not the "real" src/marginal_token/pools/ + generation/ infra (that's Day
6's job -- content-addressed store, resumable sweeps, full provenance
metadata block). This is a one-off, but built to not lose hours of work
on a network blip: checkpoints incrementally, resumable by problem_id.

Decode config matches configs/policies/qwen3.5-4b.yaml exactly:
temperature=0.8, top_p=0.95, max_tokens=1024, non-thinking.
"""
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path

API_KEY = os.environ["HOSTED_ENDPOINT_API_KEY"]
ENDPOINT = "https://qwen35-4b-bf16.deb12.smoki.mk/v1/chat/completions"
MODEL_ID = "unsloth/Qwen3.5-4B-GGUF:BF16"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

N = 64  # extending the existing N=32 pool to N=64 via nested prefixes (2026-08-20, isolating
        # whether G1's null result is an N=32 artifact) -- same 100 problems, same decode config,
        # existing sample_idx 0-31 are skipped automatically via the resumability check below,
        # only 32-63 get generated fresh
OUT_PATH = Path("notes/scratch/day4_pool.jsonl")


def fetch_problems():
    ids = set(json.load(open("configs/benchmarks/data/math500-dev100-ids.json")))
    all_rows = []
    offset = 0
    while offset < 500:
        url = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500&config=default&split=test&offset={offset}&length=100"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    d = json.load(resp)
                break
            except Exception:
                time.sleep(2)
        rows = d.get("rows", [])
        if not rows:
            break
        all_rows.extend(r["row"] for r in rows)
        offset += 100
    return [r for r in all_rows if r["unique_id"] in ids]


def generate(problem_text, timeout_s=180):
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ],
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        d = json.load(resp)
    choice = d["choices"][0]
    return {
        "content": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "usage": d.get("usage"),
        "model": d.get("model"),
    }


def load_done():
    """(problem_id, sample_idx) pairs already written to OUT_PATH."""
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["problem_id"], rec["sample_idx"]))
    return done


def append_result(rec):
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    problems = fetch_problems()
    assert len(problems) == 100, f"expected 100 problems, got {len(problems)}"
    by_id = {p["unique_id"]: p for p in problems}

    done = load_done()
    print(f"{len(done)} (problem, sample) pairs already done -- resuming" if done else "starting fresh")

    tasks = []
    for pid, p in by_id.items():
        for i in range(N):
            if (pid, i) not in done:
                tasks.append((pid, i, p["problem"], p["answer"]))
    print(f"{len(tasks)} tasks remaining out of {100 * N}")

    def run_one(task):
        pid, i, problem_text, gold = task
        try:
            gen = generate(problem_text)
            return {"problem_id": pid, "sample_idx": i, "gold_answer": gold,
                     "model_prediction": gen["content"], "finish_reason": gen["finish_reason"],
                     "usage": gen["usage"], "error": None}
        except Exception as e:
            return {"problem_id": pid, "sample_idx": i, "gold_answer": gold,
                     "model_prediction": None, "finish_reason": None, "usage": None, "error": str(e)}

    t0 = time.time()
    completed = 0
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        for rec in ex.map(run_one, tasks):
            if rec["error"]:
                failed.append(rec)
            else:
                append_result(rec)
                completed += 1
            if completed % 100 == 0 and completed > 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = len(tasks) - completed - len(failed)
                eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                print(f"  {completed}/{len(tasks)} done, {len(failed)} failed so far, "
                      f"elapsed {elapsed/60:.1f}min, ETA {eta_min:.1f}min")

    print(f"First pass: {completed} succeeded, {len(failed)} failed in {(time.time()-t0)/60:.1f} min")

    # Retry failures once at lower concurrency, longer timeout (Day 3's lesson).
    if failed:
        print(f"Retrying {len(failed)} failures at concurrency=8, timeout=240s")

        def retry_one(rec):
            try:
                gen = generate(next(p["problem"] for p in problems if p["unique_id"] == rec["problem_id"]),
                                timeout_s=240)
                return {**rec, "model_prediction": gen["content"], "finish_reason": gen["finish_reason"],
                        "usage": gen["usage"], "error": None}
            except Exception as e:
                return {**rec, "error": str(e)}

        retry_completed = 0
        still_failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for rec in ex.map(retry_one, failed):
                if rec["error"]:
                    still_failed.append(rec)
                else:
                    append_result(rec)
                    retry_completed += 1
        print(f"Retry: {retry_completed}/{len(failed)} recovered, {len(still_failed)} still failed")
        if still_failed:
            json.dump(still_failed, open("notes/scratch/day4_still_failed.json", "w"), indent=2)
            print("Unrecoverable failures saved to notes/scratch/day4_still_failed.json")

    total_done = len(load_done())
    print(f"TOTAL in {OUT_PATH}: {total_done}/{100*N}")


if __name__ == "__main__":
    main()
