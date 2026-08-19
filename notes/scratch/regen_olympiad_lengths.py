"""
Re-generate the 67 previously-truncated OlympiadBench-A samples at a much
higher max_tokens (4096) to observe their TRUE completion length --
needed because a sample truncated at 1024 only tells us "needs >1024,"
not the actual length (right-censored data). Same problems, fresh
samples (not a continuation of the original rollout -- that's fine for a
length-distribution estimate, not fine for reusing as golden-200 data).
"""
import concurrent.futures
import json
import os
import time
import urllib.request

API_KEY = os.environ["HOSTED_ENDPOINT_API_KEY"]
ENDPOINT = "https://qwen35-4b-bf16.deb12.smoki.mk/v1/chat/completions"
MODEL_ID = "unsloth/Qwen3.5-4B-GGUF:BF16"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

_CACHE = None


def _load_all_olympiad_rows():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
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
    _CACHE = all_rows
    return all_rows


def generate(problem_text, max_tokens, timeout_s):
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ],
        "max_tokens": max_tokens,
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


def main():
    truncated = json.load(open("notes/scratch/oly_truncated_to_rerun.json"))
    problems = _load_all_olympiad_rows()

    def run_one(task):
        pid = task["problem_id"]
        text = problems.get(pid)
        if text is None:
            return {**task, "error": "problem not found"}
        try:
            gen = generate(text, max_tokens=4096, timeout_s=300)
            return {**task, "model_prediction": gen["content"], "finish_reason": gen["finish_reason"],
                    "usage": gen["usage"], "error": None}
        except Exception as e:
            return {**task, "error": str(e)}

    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(run_one, truncated):
            results.append(r)
    dt = time.time() - t0

    ok = [r for r in results if not r["error"]]
    print(f"{len(ok)}/{len(truncated)} succeeded in {dt:.1f}s at max_tokens=4096")
    still_truncated = [r for r in ok if r["finish_reason"] == "length"]
    print(f"{len(still_truncated)}/{len(ok)} STILL truncated even at 4096 tokens")

    json.dump(results, open("notes/scratch/oly_regen_4096.json", "w"), indent=2)
    print("saved notes/scratch/oly_regen_4096.json")


if __name__ == "__main__":
    main()
