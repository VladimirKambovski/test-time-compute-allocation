"""
Day 3 scratch script: generate real completions from the hosted Qwen3.5-4B
endpoint against real MATH-500 + OlympiadBench-A problems, run them
through answers/ extraction+equivalence, and dump a candidate
golden_200.json for hand review. This script is NOT part of the shipped
repo -- lives under notes/scratch/, not committed as production code.
"""
import concurrent.futures
import json
import os
import random
import time
import urllib.request

API_KEY = os.environ["HOSTED_ENDPOINT_API_KEY"]
ENDPOINT = "https://qwen35-4b-bf16.deb12.smoki.mk/v1/chat/completions"
MODEL_ID = "unsloth/Qwen3.5-4B-GGUF:BF16"

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def fetch_math500(n_target=35):
    all_rows = []
    offset = 0
    while len(all_rows) < 200 and offset < 500:
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
    # stratify by level (1-5) for diversity
    by_level = {}
    for r in all_rows:
        by_level.setdefault(r["level"], []).append(r)
    rng = random.Random(20260820)
    for bucket in by_level.values():
        rng.shuffle(bucket)
    picked = []
    i = 0
    levels = sorted(by_level)
    while len(picked) < n_target:
        progressed = False
        for lvl in levels:
            if by_level[lvl]:
                picked.append(by_level[lvl].pop())
                progressed = True
                if len(picked) >= n_target:
                    break
        if not progressed:
            break
    return picked


def fetch_olympiad_a(n_target=30):
    with open("configs/benchmarks/data/olympiad-a-ids.json") as f:
        ids = set(json.load(f))
    all_rows = []
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
        all_rows.extend(r["row"] for r in rows)
        offset += 100
    eligible = [
        r for r in all_rows
        if r["id"] in ids and not r["is_multiple_answer"] and not r["error"] and r["final_answer"]
    ]
    rng = random.Random(20260820)
    rng.shuffle(eligible)
    return eligible[:n_target]


def generate(problem_text, temperature=0.8, top_p=0.95, max_tokens=1024):
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        d = json.load(resp)
    choice = d["choices"][0]
    return {
        "content": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "usage": d.get("usage"),
    }


def main():
    math500 = fetch_math500(35)
    olympiad = fetch_olympiad_a(30)
    print(f"fetched {len(math500)} MATH-500, {len(olympiad)} OlympiadBench-A problems")

    tasks = []
    for row in math500:
        for _ in range(3):
            tasks.append(("math500", row["unique_id"], row["problem"], row["answer"]))
    for row in olympiad:
        for _ in range(3):
            gold = row["final_answer"][0]
            tasks.append(("olympiad_a", str(row["id"]), row["question"], gold))

    print(f"total generation tasks: {len(tasks)}")

    results = []

    def run_one(task):
        source, pid, problem, gold = task
        try:
            gen = generate(problem)
            return {
                "source": source, "problem_id": pid, "gold_answer": gold,
                "model_prediction": gen["content"], "finish_reason": gen["finish_reason"],
                "usage": gen["usage"], "error": None,
            }
        except Exception as e:
            return {
                "source": source, "problem_id": pid, "gold_answer": gold,
                "model_prediction": None, "finish_reason": None, "usage": None,
                "error": str(e),
            }

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        for r in ex.map(run_one, tasks):
            results.append(r)
    dt = time.time() - t0
    print(f"generated {len(results)} completions in {dt:.1f}s ({len(results)/dt:.2f} completions/sec)")

    failures = [r for r in results if r["error"]]
    print(f"{len(failures)} generation-level failures: {failures[:3]}")

    with open("notes/scratch/golden_200_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved notes/scratch/golden_200_raw.json")


if __name__ == "__main__":
    main()
