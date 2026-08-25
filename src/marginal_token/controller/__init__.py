"""featurize(probe) -> decide(budget). The research finding, as code. Single object, consumed identically by replay and gateway -- see tests/test_controller_parity.py.

Day 10: the real controller now exists. `featurize()` (agreement/shape/
hygiene/confidence probe features), `oracle_action_label()` (the 3-way
STOP/SAMPLE/ABSTAIN ground-truth label, per the SELECT-narrowing
decision in notes/2026-08-22.md/2026-08-23.md), a fitted
`DetectiveController` (multinomial logistic regression scaffold -- the
rigorous grouped-CV fit is Day 14's job, not this), and the 6 other
fixed policies from docs/brief.md line 276's E7 list.
`AlwaysStopFakeController` remains for `test_controller_parity.py`'s
pure plumbing check -- it has no research meaning and is never used
elsewhere.
"""

from marginal_token.controller.base import Action, Budget, Controller, Decision, Probe
from marginal_token.controller.features import FEATURE_NAMES, featurize
from marginal_token.controller.oracle_labels import PROBE_K, OracleAction, OracleLabelResult, oracle_action_label
from marginal_token.controller.policies import (
    GamblerController,
    MiserController,
    OracleController,
    SpendthriftController,
    UniformSelectController,
)
from marginal_token.controller.predictor import DetectiveController, FortuneTellerController
from marginal_token.controller.stub import AlwaysStopFakeController

__all__ = [
    "Action",
    "Budget",
    "Controller",
    "Decision",
    "Probe",
    "FEATURE_NAMES",
    "featurize",
    "PROBE_K",
    "OracleAction",
    "OracleLabelResult",
    "oracle_action_label",
    "GamblerController",
    "MiserController",
    "OracleController",
    "SpendthriftController",
    "UniformSelectController",
    "DetectiveController",
    "FortuneTellerController",
    "AlwaysStopFakeController",
]
