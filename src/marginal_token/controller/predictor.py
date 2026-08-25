"""
Detective (the real, learned post-hoc predictor) and Fortune Teller
(the pre-hoc query-text embedding classifier) -- the two LEARNED fixed
policies from docs/brief.md line 276, both multinomial logistic
regression per §16.

Detective conditions on the k=4 probe (`features.featurize()`).
Fortune Teller conditions ONLY on the query text itself, via a local
sentence-transformer embedding, BEFORE any probe sample exists -- this
is the non-negotiable pre-hoc control §16 says H3 is unfalsifiable
without ("without it H3 is unfalsifiable").

**Day 10 scope: a working, fit-and-predict SCAFFOLD.** The rigorous
grouped 5-fold CV, feature ablation, and coefficient freeze are
explicitly Day 14's job (§16's "Protocol" section: "Coefficients frozen
and tagged before Day 18") -- fitting here is a single in-sample fit on
whatever pool is passed in, good enough to prove the pipeline works and
to produce a real (if not yet publication-grade) number, not the final
result.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from marginal_token.controller.base import Budget, Decision, Probe
from marginal_token.controller.features import FEATURE_NAMES, featurize

_ALL_ACTIONS = ("stop", "sample", "select", "search", "abstain")
# SAMPLE and SELECT get the SAME nominal budget grant (the SELECT-vs-SAMPLE
# budget split -- more PRM forwards, fewer raw samples -- is budget/accounting.py's
# job, not a difference in the grant amount itself).
_BUDGET_ACTIONS = ("sample", "select")


class DetectiveController:
    """The real predictor: multinomial logistic regression on the
    backend-independent probe features (agreement/shape/hygiene; the
    confidence group is included when available, imputed by column mean
    when it isn't -- see `fit()`).
    """

    def __init__(self):
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self._impute_values: dict[str, float] = {}

    def fit(self, X: list[dict[str, float]], y: list[str]) -> None:
        if not X:
            raise ValueError("DetectiveController.fit() needs at least one training example")
        arr = np.array([[row[f] for f in FEATURE_NAMES] for row in X], dtype=float)
        # NaN columns (the confidence group, when the pool has no
        # logprobs -- e.g. P1, see backends/hosted_endpoint.py's Day-10
        # fix) are imputed with the column mean rather than dropped or
        # left to crash sklearn. Recorded explicitly (`_impute_values`)
        # so a caller can tell which columns were actually informative
        # vs. imputed to a constant -- an imputed-to-constant column
        # contributes ~nothing to a fitted logistic model, which is the
        # honest reflection of "this feature group wasn't available,"
        # not a hidden assumption.
        all_nan_cols = np.isnan(arr).all(axis=0)
        col_means = np.zeros(arr.shape[1])
        if not all_nan_cols.all():
            with np.errstate(invalid="ignore"):  # expected/handled below, not a real numeric warning
                col_means[~all_nan_cols] = np.nanmean(arr[:, ~all_nan_cols], axis=0)
        # `col_means` for an all-NaN column (e.g. the whole confidence
        # group, when the pool has no logprobs) stays 0.0 -- nothing to
        # impute from.
        nan_mask = np.isnan(arr)
        for row_idx, col_idx in zip(*np.where(nan_mask)):
            arr[row_idx, col_idx] = col_means[col_idx]
        self._impute_values = dict(zip(FEATURE_NAMES, col_means.tolist()))

        # Scale before fitting -- the feature groups span wildly
        # different ranges (top1_vote_fraction in [0,1] vs
        # mean_output_length in the hundreds of tokens), which made
        # lbfgs fail to converge on the real P1 fit (found live,
        # 2026-08-23: `ConvergenceWarning: lbfgs failed to converge`).
        # A constant-0 imputed column (all-NaN confidence group) has
        # zero variance -- StandardScaler leaves it at 0 rather than
        # dividing by zero, which is the correct behavior here (an
        # unavailable feature group should contribute nothing, not an
        # arbitrary scaled value).
        self.scaler = StandardScaler()
        arr_scaled = self.scaler.fit_transform(arr)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(arr_scaled, y)

    def featurize(self, probe: Probe) -> dict[str, float]:
        return featurize(probe)

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        if self.model is None:
            raise RuntimeError("DetectiveController.fit() must be called before decide()")
        feats = self.featurize(probe)
        row = [
            feats[name] if not np.isnan(feats[name]) else self._impute_values[name]
            for name in FEATURE_NAMES
        ]
        row_scaled = self.scaler.transform(np.array([row]))
        probs = self.model.predict_proba(row_scaled)[0]
        class_probs_learned = dict(zip(self.model.classes_, probs.tolist()))
        action = max(class_probs_learned, key=class_probs_learned.get)
        grant = budget.max_tokens if action in _BUDGET_ACTIONS else 0
        full_probs = {a: class_probs_learned.get(a, 0.0) for a in _ALL_ACTIONS}
        return Decision(action=action, budget_grant=grant, class_probs=full_probs,
                          rationale={"policy": "detective", "features": feats})


class FortuneTellerController:
    """The pre-hoc control: multinomial logistic regression on a local
    sentence-transformer embedding of the query text ALONE -- no probe
    evidence, no generation at all before this decision is made. Per
    §16, this is what makes H3 ("evidence beats a pre-hoc guess")
    falsifiable; without this comparator, a good Detective AUROC could
    just mean "some questions are inherently easy," not "the probe
    evidence helped."
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self._embedder = None  # lazy-loaded -- avoid the model download/load cost when unused
        self.model: LogisticRegression | None = None

    def _embedder_instance(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # heavy optional import

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embedder_instance().encode(texts))

    def fit(self, query_texts: list[str], y: list[str]) -> None:
        if not query_texts:
            raise ValueError("FortuneTellerController.fit() needs at least one training example")
        embeddings = self.embed(query_texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(embeddings, y)

    def featurize(self, probe: Probe) -> dict[str, float]:
        # Not the probe -- this policy is pre-hoc by design. The query
        # text must be attached to `probe.features["query_text"]` by the
        # caller; this method exists only to satisfy the shared
        # Controller protocol, and returns the embedding-derived
        # features actually used, for `rationale` transparency.
        return {}

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        if self.model is None:
            raise RuntimeError("FortuneTellerController.fit() must be called before decide()")
        query_text = probe.features.get("query_text")
        if query_text is None:
            raise ValueError(
                "FortuneTellerController requires probe.features['query_text'] -- it decides from the "
                "question text alone, before any probe samples exist."
            )
        embedding = self.embed([query_text])
        probs = self.model.predict_proba(embedding)[0]
        class_probs_learned = dict(zip(self.model.classes_, probs.tolist()))
        action = max(class_probs_learned, key=class_probs_learned.get)
        grant = budget.max_tokens if action in _BUDGET_ACTIONS else 0
        full_probs = {a: class_probs_learned.get(a, 0.0) for a in _ALL_ACTIONS}
        return Decision(action=action, budget_grant=grant, class_probs=full_probs,
                          rationale={"policy": "fortune_teller", "query_text": query_text})
