"""
G1 fallback check on OlympiadBench-A: 30 problems (same set as Day 3's
golden-200/truncation investigation) x N=32, at max_tokens=4096 --
NOT the frozen configs/policies/qwen3.5-4b.yaml default of 1024.

This is a clearly-labeled ONE-OFF diagnostic condition, not a change to
the frozen decode config. Running at 1024 would just reproduce Day 3's
74% truncation finding and give an uninterpretable G1 result -- 4096 is
the value Day 3 already empirically tested (50% completion rate, vs 26%
at 1024), so this reuses a validated number rather than guessing a new
untested one.

Lower concurrency (10) and longer timeout (300s) than the MATH-500 run,
learned from Day 3: long completions under higher concurrency cause a
timeout storm, not a real reliability problem (all recovered on retry at
lower concurrency there too).
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

N = 32
MAX_TOKENS = 4096
OUT_PATH = Path("notes/scratch/day4_olympiad_pool.jsonl")


def fetch_problems():
    # same 30 problems used in golden-200 (Day 3)
    scored = json.load(open("notes/scratch/golden_200_scored.json"))
    ids = sorted(set(s["problem_id"] for s in scored if s["source"] == "olympiad_a"), key=int)
    assert len(ids) == 30, f"expected 30, got {len(ids)}"

    all_rows = {}
    offset = 0
    while offset < 674:
        url = f"https://datasets-server.huggingface.co/rows?dataset=Hothan/OlympiadBench&config=OE_TO_maths_en_COMP&split=train&offset={offset}&length=100"
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
        for r in rows:
            all_rows[str(r["row"]["id"])] = r["row"]
        offset += 100

    return {pid: all_rows[pid] for pid in ids}


def generate(problem_text, timeout_s=450):
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ],
        "max_tokens": MAX_TOKENS,
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
    by_id = fetch_problems()
    done = load_done()
    print(f"{len(done)} (problem, sample) pairs already done" if done else "starting fresh")

    tasks = []
    for pid, p in by_id.items():
        gold = p["final_answer"][0]
        for i in range(N):
            if (pid, i) not in done:
                tasks.append((pid, i, p["question"], gold))
    print(f"{len(tasks)} tasks remaining out of {30 * N}")

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:  # matches Day 3's proven-working
                                                                        # config for max_tokens=4096
                                                                        # (67/67 succeeded at concurrency=6);
                                                                        # 10 caused a live isolated test to
                                                                        # take 154s solo, likely pushing
                                                                        # concurrent latencies past the
                                                                        # original 300s timeout entirely
        for rec in ex.map(run_one, tasks):
            if rec["error"]:
                failed.append(rec)
            else:
                append_result(rec)
                completed += 1
            if completed % 50 == 0 and completed > 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = len(tasks) - completed - len(failed)
                eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                print(f"  {completed}/{len(tasks)} done, {len(failed)} failed so far, "
                      f"elapsed {elapsed/60:.1f}min, ETA {eta_min:.1f}min", flush=True)

    print(f"First pass: {completed} succeeded, {len(failed)} failed in {(time.time()-t0)/60:.1f} min")

    if failed:
        print(f"Retrying {len(failed)} failures at concurrency=5, timeout=450s")

        def retry_one(rec):
            try:
                gen = generate(by_id[rec["problem_id"]]["question"], timeout_s=450)
                return {**rec, "model_prediction": gen["content"], "finish_reason": gen["finish_reason"],
                        "usage": gen["usage"], "error": None}
            except Exception as e:
                return {**rec, "error": str(e)}

        retry_completed = 0
        still_failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for rec in ex.map(retry_one, failed):
                if rec["error"]:
                    still_failed.append(rec)
                else:
                    append_result(rec)
                    retry_completed += 1
        print(f"Retry: {retry_completed}/{len(failed)} recovered, {len(still_failed)} still failed")
        if still_failed:
            json.dump(still_failed, open("notes/scratch/day4_olympiad_still_failed.json", "w"), indent=2)

    total_done = len(load_done())
    print(f"TOTAL in {OUT_PATH}: {total_done}/{30*N}")


if __name__ == "__main__":
    main()
