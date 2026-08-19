import concurrent.futures
import json
import os
import time
import urllib.request

API_KEY = os.environ["HOSTED_ENDPOINT_API_KEY"]
ENDPOINT = "https://qwen35-4b-bf16.deb12.smoki.mk/v1/chat/completions"
MODEL_ID = "unsloth/Qwen3.5-4B-GGUF:BF16"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


_OLYMPIAD_CACHE = None


def _load_all_olympiad_rows():
    global _OLYMPIAD_CACHE
    if _OLYMPIAD_CACHE is not None:
        return _OLYMPIAD_CACHE
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
            all_rows[str(r["row"]["id"])] = r["row"]["question"]
        offset += 100
    _OLYMPIAD_CACHE = all_rows
    return all_rows


def fetch_olympiad_problem_text(pid):
    return _load_all_olympiad_rows().get(str(pid))


def generate(problem_text, timeout_s):
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
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        d = json.load(resp)
    dt = time.time() - t0
    choice = d["choices"][0]
    return {"content": choice["message"]["content"], "finish_reason": choice["finish_reason"],
            "usage": d.get("usage"), "latency_s": dt}


def main():
    results = json.load(open("notes/scratch/golden_200_raw.json"))
    failed = [r for r in results if r["error"]]
    print(f"retrying {len(failed)} failed tasks at lower concurrency, longer timeout")

    def run_one(task):
        pid, gold = task["problem_id"], task["gold_answer"]
        problem_text = fetch_olympiad_problem_text(pid)
        if problem_text is None:
            return {**task, "error": "problem text not found in first-rows window"}
        try:
            gen = generate(problem_text, timeout_s=240)
            return {**task, "model_prediction": gen["content"], "finish_reason": gen["finish_reason"],
                    "usage": gen["usage"], "latency_s": gen["latency_s"], "error": None}
        except Exception as e:
            return {**task, "error": str(e)}

    t0 = time.time()
    retried = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run_one, failed):
            retried.append(r)
    dt = time.time() - t0

    ok = [r for r in retried if not r["error"]]
    still_failed = [r for r in retried if r["error"]]
    print(f"retry: {len(ok)}/{len(failed)} succeeded in {dt:.1f}s at concurrency=8, timeout=240s")
    if ok:
        latencies = [r["latency_s"] for r in ok]
        print(f"latency range: {min(latencies):.1f}s - {max(latencies):.1f}s, mean {sum(latencies)/len(latencies):.1f}s")
    print("still failed:", [(r["source"], r["problem_id"], r["error"]) for r in still_failed])

    # merge: replace failed entries with retried ones where they succeeded
    merged = [r for r in results if not r["error"]]
    merged.extend(retried)
    with open("notes/scratch/golden_200_raw.json", "w") as f:
        json.dump(merged, f, indent=2)
    print(f"merged file now has {len(merged)} entries ({len([m for m in merged if not m['error']])} without error)")


if __name__ == "__main__":
    main()
