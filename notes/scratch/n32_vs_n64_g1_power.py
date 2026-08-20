"""
Does H1/G1's 8pp threshold still make statistical sense at N=32 vs N=64?

golden-200 can't answer this directly -- it has only 3 samples/problem,
nowhere near N=32 or N=64, so there's no way to empirically measure
sampling variance at those N from it. This is a Monte Carlo simulation of
the actual G1 mechanics instead: per-problem correctness modeled as
Bernoulli(p_i), wrong answers spread across a few common wrong-answer
buckets (not all distinct), majority vote and pass@N computed from that,
oracle-over-{STOP,SAMPLE,SELECT} vs best-fixed-policy gap computed per
simulated pool, repeated many times to see how much NOISIER that gap is
at N=32 vs N=64.

Modeling choices (stated, not fit to data -- there's no data to fit this
to at N=32/64 scale):
- Per-problem correctness probability p_i ~ Beta(a, b), swept across a
  few (a,b) choices to check the conclusion isn't an artifact of one
  arbitrary distribution.
- K=4 distinct "wrong answer" buckets, wrong mass split uniformly among
  them -- a standard simplifying assumption for this kind of majority-
  vote analysis (real wrong answers cluster on a handful of common
  mistakes, not infinitely many unique strings).
- STOP = majority of the first k=4 samples (the free probe).
- SAMPLE = majority of all N samples.
- SELECT ceiling = pass@N (oracle/perfect selection -- proxy since no PRM
  exists yet, per the already-agreed Day 4 methodology).
- Oracle-over-{STOP,SAMPLE,SELECT} per problem = 1 if ANY of the three
  succeeds (oracle always picks a succeeding cheapest action if one
  exists; for the accuracy-gap computation only "does at least one
  succeed" matters, not which).
- Best fixed policy = whichever single action, applied to EVERY problem,
  has the highest average accuracy in that specific simulated pool.
"""

import numpy as np

RNG = np.random.default_rng(20260820)
K_WRONG = 4  # distinct wrong-answer buckets
M_PROBLEMS = 100  # matches G1's actual "100 MATH-500 problems"
N_TRIALS = 3000  # Monte Carlo repeats per (N, distribution) setting


def simulate_pool(p, n, k_probe=4):
    """p: array of shape (n_trials, m_problems) -- per-problem correctness
    prob, same across a trial's M problems but redrawn per trial.
    Returns per-trial (n_trials,) arrays: stop_acc, sample_acc, select_acc
    (all fraction-correct across the M problems in that trial).
    """
    n_trials, m = p.shape
    # For each (trial, problem), draw N samples' correctness and, for
    # wrong ones, which wrong bucket. Vectorized via random draws.
    correct = RNG.random((n_trials, m, n)) < p[:, :, None]
    wrong_bucket = RNG.integers(0, K_WRONG, size=(n_trials, m, n))

    # --- pass@N: at least one correct sample ---
    pass_at_n = correct.any(axis=2)

    # --- maj@N: correct answer has strictly the most votes among N ---
    correct_votes = correct.sum(axis=2)
    # count votes per wrong bucket among the *wrong* samples
    wrong_votes = np.zeros((n_trials, m, K_WRONG), dtype=int)
    for b in range(K_WRONG):
        wrong_votes[:, :, b] = ((~correct) & (wrong_bucket == b)).sum(axis=2)
    max_wrong_votes = wrong_votes.max(axis=2)
    maj_at_n = correct_votes > max_wrong_votes  # strict majority/plurality; ties -> loses (conservative)

    # --- STOP: majority of the first k_probe samples only ---
    probe_correct = correct[:, :, :k_probe]
    probe_wrong_bucket = wrong_bucket[:, :, :k_probe]
    probe_correct_votes = probe_correct.sum(axis=2)
    probe_wrong_votes = np.zeros((n_trials, m, K_WRONG), dtype=int)
    for b in range(K_WRONG):
        probe_wrong_votes[:, :, b] = ((~probe_correct) & (probe_wrong_bucket == b)).sum(axis=2)
    probe_max_wrong = probe_wrong_votes.max(axis=2)
    stop_correct = probe_correct_votes > probe_max_wrong

    stop_acc = stop_correct.mean(axis=1)
    sample_acc = maj_at_n.mean(axis=1)
    select_acc = pass_at_n.mean(axis=1)  # oracle selection ceiling

    return stop_acc, sample_acc, select_acc, stop_correct, maj_at_n, pass_at_n


def analyze(n, beta_a, beta_b, label):
    p = RNG.beta(beta_a, beta_b, size=(N_TRIALS, M_PROBLEMS))
    stop_acc, sample_acc, select_acc, stop_c, samp_c, sel_c = simulate_pool(p, n)

    # oracle-over-{STOP,SAMPLE,SELECT} per problem = succeeds if ANY does.
    # SELECT's pass@N ceiling belongs ONLY here, not in "best fixed policy"
    # below -- pass@N mathematically dominates STOP/SAMPLE on every single
    # problem (majority-of-a-subset can never succeed if the correct
    # answer isn't anywhere in the pool), so including it as a fixed-
    # policy candidate makes oracle == best-fixed trivially, always,
    # regardless of N. This is exactly brief.md's own "selection gap"
    # (pass@k - best REAL selector) -- the oracle side gets credit for
    # the hypothetical ceiling a real PRM might reach; the fixed-policy
    # side only gets credit for policies actually runnable without one.
    oracle_correct = stop_c | samp_c | sel_c
    oracle_acc = oracle_correct.mean(axis=1)

    # best FIXED policy per trial = max of STOP/SAMPLE only -- SELECT has
    # no real (PRM-free) fixed-policy implementation on Day 4.
    fixed_accs = np.stack([stop_acc, sample_acc], axis=1)
    best_fixed_acc = fixed_accs.max(axis=1)

    gap = (oracle_acc - best_fixed_acc) * 100  # percentage points

    return {
        "label": label, "n": n, "mean_gap": gap.mean(), "std_gap": gap.std(),
        "p05": np.percentile(gap, 5), "p95": np.percentile(gap, 95),
        "frac_ge_8pp": (gap >= 8).mean(),
    }


def bootstrap_power(n, beta_a, beta_b, true_gap_target=8, n_reps=500, m=M_PROBLEMS, n_boot=2000):
    """Out of n_reps independent simulated 'real experiments' (each = one
    generated pool of M problems), what fraction produce a paired
    bootstrap CI on the oracle-vs-best-fixed gap that excludes 0? This is
    literally G1's actual accept test, simulated end to end.
    """
    excludes_zero_count = 0
    gaps_observed = []
    for _ in range(n_reps):
        p = RNG.beta(beta_a, beta_b, size=(1, m))
        stop_acc, sample_acc, select_acc, stop_c, samp_c, sel_c = simulate_pool(p, n)
        oracle_correct = (stop_c | samp_c | sel_c)[0]  # (m,) boolean per problem
        fixed = np.stack([stop_c[0], samp_c[0]], axis=1)  # (m, 2) -- STOP/SAMPLE only, see analyze()
        fixed_acc = fixed.mean(axis=0)
        best_policy_idx = fixed_acc.argmax()
        best_policy_correct = fixed[:, best_policy_idx]

        per_problem_diff = oracle_correct.astype(float) - best_policy_correct.astype(float)
        observed_gap = per_problem_diff.mean() * 100
        gaps_observed.append(observed_gap)

        boot_idx = RNG.integers(0, m, size=(n_boot, m))
        boot_gaps = per_problem_diff[boot_idx].mean(axis=1) * 100
        ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])
        if ci_lo > 0:
            excludes_zero_count += 1

    return {
        "n": n, "power": excludes_zero_count / n_reps,
        "mean_observed_gap": np.mean(gaps_observed), "std_observed_gap": np.std(gaps_observed),
    }


print("=== Part 1: gap distribution across Monte Carlo trials, by N and by assumed difficulty distribution ===")
for beta_a, beta_b, dist_label in [(0.5, 0.5, "U-shaped (many clear-easy/clear-hard)"),
                                     (1.0, 1.0, "uniform p in [0,1]"),
                                     (1.0, 2.0, "skewed harder (mean p~0.33)")]:
    for n in (32, 64):
        r = analyze(n, beta_a, beta_b, dist_label)
        print(f"  [{dist_label:38s}] N={n:2d}: mean_gap={r['mean_gap']:5.2f}pp  std={r['std_gap']:5.2f}pp  "
              f"90%CI=[{r['p05']:5.2f},{r['p95']:5.2f}]  P(gap>=8pp)={r['frac_ge_8pp']:.2f}")
    print()

print("=== Part 2: simulated G1 bootstrap test itself -- statistical power to detect the gap ===")
for beta_a, beta_b, dist_label in [(0.5, 0.5, "U-shaped"), (1.0, 1.0, "uniform"), (1.0, 2.0, "skewed harder")]:
    for n in (32, 64):
        r = bootstrap_power(n, beta_a, beta_b, n_reps=300, n_boot=1000)
        print(f"  [{dist_label:15s}] N={n:2d}: mean_observed_gap={r['mean_observed_gap']:5.2f}pp  "
              f"std={r['std_observed_gap']:5.2f}pp  P(bootstrap CI excludes 0)={r['power']:.2f}")
    print()
