import concurrent.futures
import json
import os
import time
import urllib.request

API_KEY = os.environ["HOSTED_ENDPOINT_API_KEY"]
ENDPOINT = "https://qwen35-4b-bf16.deb12.smoki.mk/v1/chat/completions"
MODEL_ID = "unsloth/Qwen3.5-4B-GGUF:BF16"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def fetch_amc23():
    url = "https://datasets-server.huggingface.co/first-rows?dataset=math-ai/amc23&config=default&split=test"
    with urllib.request.urlopen(url, timeout=30) as resp:
        d = json.load(resp)
    return [r["row"] for r in d["rows"]]


def generate(problem_text):
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.load(resp)
    choice = d["choices"][0]
    return {"content": choice["message"]["content"], "finish_reason": choice["finish_reason"],
            "usage": d.get("usage")}


def main():
    problems = fetch_amc23()
    print(f"fetched {len(problems)} AMC23 problems")

    def run_one(row):
        try:
            gen = generate(row["question"])
            return {"problem_id": row["id"], "gold_answer": row["answer"],
                     "model_prediction": gen["content"], "finish_reason": gen["finish_reason"],
                     "usage": gen["usage"], "error": None}
        except Exception as e:
            return {"problem_id": row["id"], "gold_answer": row["answer"],
                     "model_prediction": None, "finish_reason": None, "usage": None, "error": str(e)}

    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(run_one, problems):
            results.append(r)
    dt = time.time() - t0

    ok = [r for r in results if not r["error"]]
    print(f"{len(ok)}/{len(problems)} generated successfully in {dt:.1f}s")

    finish_reasons = {}
    for r in ok:
        finish_reasons[r["finish_reason"]] = finish_reasons.get(r["finish_reason"], 0) + 1
    print("finish_reason breakdown:", finish_reasons)

    truncated = [r for r in ok if r["finish_reason"] == "length"]
    print(f"completion rate at max_tokens=1024: {len(ok)-len(truncated)}/{len(ok)} = {(len(ok)-len(truncated))/len(ok):.0%}")

    json.dump(results, open("notes/scratch/amc23_results.json", "w"), indent=2)
    print("saved notes/scratch/amc23_results.json")


if __name__ == "__main__":
    main()
