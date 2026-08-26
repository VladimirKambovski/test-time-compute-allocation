"""
G1 weaker-policy fallback check: 25 MATH-500 problems (stratified
subsample of the same 100 used for the primary 4B check) x N=32, on
Qwen3.5-2B via the hosted endpoint. See notes/2026-08-22.md for the
full reasoning -- this is the "cheap version" run before committing to
a full-scale check.
"""
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path

API_KEY = os.environ["HOSTED_ENDPOINT_API_KEY"]
ENDPOINT = "https://qwen35-2b.deb11.smoki.mk/v1/chat/completions"
MODEL_ID = "unsloth/Qwen3.5-2B-GGUF:BF16"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

N = 32
OUT_PATH = Path("notes/scratch/day4_weaker_policy_pool.jsonl")


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
    return {"content": choice["message"]["content"], "finish_reason": choice["finish_reason"],
            "usage": d.get("usage")}


def load_done():
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
    problems = json.load(open("notes/scratch/math500_weaker_policy_25.json"))
    assert len(problems) == 25
    by_id = {p["unique_id"]: p for p in problems}

    done = load_done()
    print(f"{len(done)} pairs already done" if done else "starting fresh", flush=True)

    tasks = []
    for pid, p in by_id.items():
        for i in range(N):
            if (pid, i) not in done:
                tasks.append((pid, i, p["problem"], p["answer"]))
    print(f"{len(tasks)} tasks remaining out of {25 * N}", flush=True)

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for rec in ex.map(run_one, tasks):
            if rec["error"]:
                failed.append(rec)
            else:
                append_result(rec)
                completed += 1
            if completed % 50 == 0 and completed > 0:
                elapsed = time.time() - t0
                print(f"  {completed}/{len(tasks)} done, {len(failed)} failed, elapsed {elapsed/60:.1f}min", flush=True)

    print(f"First pass: {completed} succeeded, {len(failed)} failed in {(time.time()-t0)/60:.1f} min", flush=True)

    if failed:
        print(f"Retrying {len(failed)} failures at concurrency=6, timeout=300s", flush=True)

        def retry_one(rec):
            try:
                gen = generate(by_id[rec["problem_id"]]["problem"], timeout_s=300)
                return {**rec, "model_prediction": gen["content"], "finish_reason": gen["finish_reason"],
                        "usage": gen["usage"], "error": None}
            except Exception as e:
                return {**rec, "error": str(e)}

        retry_completed = 0
        still_failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for rec in ex.map(retry_one, failed):
                if rec["error"]:
                    still_failed.append(rec)
                else:
                    append_result(rec)
                    retry_completed += 1
        print(f"Retry: {retry_completed}/{len(failed)} recovered, {len(still_failed)} still failed", flush=True)
        if still_failed:
            json.dump(still_failed, open("notes/scratch/day4_weaker_policy_still_failed.json", "w"), indent=2)

    total_done = len(load_done())
    print(f"TOTAL in {OUT_PATH}: {total_done}/{25*N}", flush=True)


if __name__ == "__main__":
    main()
