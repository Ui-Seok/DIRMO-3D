import json
from pathlib import Path

import numpy as np

from ..io import digest
from ..schemas import ActionKind
from .features import FEATURE_NAMES, features


class LearnedRouter:
    name = "learned"

    def __init__(self, path: str, cost_weight: float = 0.01):
        """``cost_weight`` is utility per wall-clock second, and it is a research choice,
        not a tuned constant. The predicted cost label is in seconds while utility is in
        the objective's own units, so the exchange rate has to be stated deliberately:
        at 0.01 a five-second draw costs 0.05 utility, which already exceeds a typical
        case's ``minimum_improvement``, and the router will prefer STOP to every action.
        Set it against the objective actually in use, and record what was used.
        """
        if cost_weight < 0:
            raise ValueError("cost_weight must be nonnegative")
        self.artifact = json.loads(Path(path).read_text())
        if self.artifact["feature_names"] != FEATURE_NAMES:
            raise ValueError("Learned router feature schema mismatch")
        self.cost_weight = cost_weight

    def predictions(self, x):
        data = self.artifact
        x = (np.asarray(x) - data["mean"]) / data["scale"]
        for i, layer in enumerate(data["layers"]):
            x = x @ np.asarray(layer["weight"]).T + layer["bias"]
            if i < len(data["layers"]) - 1:
                x = np.maximum(x, 0)
        return x * data["label_scale"] + data["label_mean"]

    def select(self, state, actions, case=None, kernel_id=None):
        if case is None:
            raise ValueError("Learned router requires the fixed objective/case for pre-action features")
        if kernel_id != self.artifact["kernel_id"]:
            raise ValueError("Learned router was trained on a different generation kernel")
        if digest(case.objective) != self.artifact.get("objective_sha256"):
            raise ValueError("Learned router was trained against a different objective; its "
                             "utility predictions do not transfer")
        best, best_value = next(a for a in actions if a.kind == ActionKind.STOP), 0.0
        for action in actions:
            if action.kind == ActionKind.STOP:
                continue
            if action.kind.value not in self.artifact["supported_action_kinds"]:
                continue
            if action.generation.integration_steps != self.artifact["integration_steps"]:
                continue  # condition changes are distinct kernels, not silently mixed labels
            improvement, cost = self.predictions(features(state, action, case))
            value = improvement - self.cost_weight * max(0, cost)
            if value > best_value:
                best, best_value = action, value
        return best
