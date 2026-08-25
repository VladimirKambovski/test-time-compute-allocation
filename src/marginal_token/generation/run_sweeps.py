"""
CLI entrypoint wiring a Backend + DecodeConfig + benchmark problem set +
PoolStore into `generation.sweep.run_sweep` -- the `make reproduce-pools`
target the Makefile has declared since Day 1
(`python -m marginal_token.generation.run_sweeps --config configs/pools/`).

Day 6: this is the first real invocation of `run_sweep` against a live
backend (it was deliberately built but not invoked on Day 4's "safe
list" pass -- see the module docstring in `generation/sweep.py`).

Benchmark-problem fetching lives here, not in a new `benchmarks/`
package -- docs/brief.md §22's module boundaries don't list one, and
this is "driver wiring" (turning a benchmark config into concrete
prompts), the same role notes/scratch/day4_generate_pool.py's
`fetch_problems()` played before this was productionized. Currently
only knows how to fetch MATH-500 (HuggingFaceH4/MATH-500 via the public
datasets-server REST API, no `datasets` package dependency, same pattern
already live-verified on Day 4/5) -- add other benchmarks' fetchers here
as they're needed, not speculatively.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from marginal_token.backends.base import DecodeConfig
from marginal_token.backends.hosted_endpoint import HostedQwen35Backend
from marginal_token.generation.sweep import SweepTask, run_sweep
from marginal_token.pools.store import PoolStore

DEFAULT_POOL_ROOT = "results/pools"


def fetch_math500_problems(dataset_id: str, ids_file: str | None) -> list[tuple[str, str]]:
    """Returns [(unique_id, problem_text), ...]. `ids_file`, if given,
    restricts to that subset (e.g. the frozen dev-100 slice); omit it to
    fetch the full 500 (P1's actual scope per docs/brief.md's pool table).
    """
    wanted = set(json.load(open(ids_file))) if ids_file else None
    rows: list[tuple[str, str]] = []
    offset = 0
    while offset < 500:
        url = (
            f"https://datasets-server.huggingface.co/rows?dataset={dataset_id}"
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
        page = d.get("rows", [])
        if not page:
            break
        for r in page:
            row = r["row"]
            if wanted is None or row["unique_id"] in wanted:
                rows.append((row["unique_id"], row["problem"]))
        offset += 100
    return rows


def fetch_olympiad_bench_problems(
    dataset_id: str, config: str, ids_file: str, total_rows: int = 674
) -> list[tuple[str, str]]:
    """Returns [(str(id), question_text), ...] for the frozen slice-A ID
    list. Same schema day4_generate_olympiad.py / build_golden_200.py
    already validated live (Day 3/4): `id` (int), `question` (problem
    text), `final_answer` (a LIST -- gold isn't fetched here, only what
    generation needs; scoring/analysis is responsible for
    `final_answer`/`is_multiple_answer`/`error` filtering later, per
    invariant #6/#7 -- a malformed gold answer is a scoring-time concern,
    never a reason to skip generating a completion in the first place).
    Split is "train" for this dataset (verified Day 3/4 -- OlympiadBench
    doesn't use a "test" split the way MATH-500 does).
    """
    wanted = set(str(i) for i in json.load(open(ids_file)))
    rows: list[tuple[str, str]] = []
    offset = 0
    while offset < total_rows:
        url = (
            f"https://datasets-server.huggingface.co/rows?dataset={dataset_id}"
            f"&config={config}&split=train&offset={offset}&length=100"
        )
        d = {}
        for _attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    d = json.load(resp)
                break
            except Exception:
                time.sleep(2)
        page = d.get("rows", [])
        if not page:
            break
        for r in page:
            row = r["row"]
            row_id = str(row["id"])
            if row_id in wanted:
                rows.append((row_id, row["question"]))
        offset += 100
    return rows


def load_pool_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_backend(backend_ref: str):
    # Only the hosted Qwen3.5-4B endpoint exists as a real Backend
    # implementation so far (local_vllm is "confirmed_available, not yet
    # exercised" per configs/policies/qwen3.5-4b.yaml) -- extend this
    # dispatch when a second one is actually wired up, not before.
    if backend_ref == "policy_primary":
        return HostedQwen35Backend(pool_id="")  # pool_id filled in per-sample by run_sweep's caller
    raise ValueError(f"No Backend implementation wired up for backend_ref={backend_ref!r} yet.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/pools/*.yaml pool config")
    parser.add_argument("--max-workers", type=int, default=None, help="Override the pool config's max_workers")
    parser.add_argument("--pool-root", default=DEFAULT_POOL_ROOT, help="PoolStore root directory")
    args = parser.parse_args(argv)

    cfg = load_pool_config(args.config)
    benchmark = cfg["benchmark"]
    dataset_id = benchmark["dataset_id"]
    print(f"fetching problems for {dataset_id} "
          f"({'subset ' + benchmark['ids_file'] if benchmark.get('ids_file') else 'full set'})...")
    if dataset_id == "HuggingFaceH4/MATH-500":
        problems = fetch_math500_problems(dataset_id, benchmark.get("ids_file"))
    elif dataset_id == "Hothan/OlympiadBench":
        problems = fetch_olympiad_bench_problems(dataset_id, benchmark["config"], benchmark["ids_file"])
    else:
        raise NotImplementedError(f"No fetcher wired up yet for {dataset_id!r}.")
    print(f"{len(problems)} problems loaded")

    tasks = [SweepTask(problem_id=pid, prompt=text) for pid, text in problems]

    decode = cfg["decode"]
    decode_cfg = DecodeConfig(
        temperature=decode["temperature"],
        top_p=decode["top_p"],
        max_tokens=decode["max_tokens"],
        seed=decode.get("seed"),
        thinking_mode=decode.get("thinking_mode", False),
        stop_sequences=tuple(decode.get("stop_sequences", ())),
    )

    backend = build_backend(cfg["backend_ref"])
    store = PoolStore(args.pool_root)

    print(f"pool: policy_ref={cfg['policy_ref']!r} backend_ref={cfg['backend_ref']!r} "
          f"benchmark_id={cfg['benchmark_id']!r} N={cfg['n']}")
    t0 = time.time()
    result = run_sweep(
        tasks=tasks,
        n=cfg["n"],
        backend=backend,
        cfg=decode_cfg,
        store=store,
        policy_ref=cfg["policy_ref"],
        backend_ref=cfg["backend_ref"],
        benchmark_id=cfg["benchmark_id"],
        max_workers=args.max_workers or cfg.get("max_workers", 6),
    )
    elapsed = time.time() - t0
    print(f"completed {result.completed} samples in {elapsed/60:.1f}min, {len(result.failed)} failed")
    if result.failed:
        print(f"first few failures: {result.failed[:5]}")


if __name__ == "__main__":
    main()
