import time

from ..schemas import BudgetLimits


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    resources = ("fm_draws", "fm_nfe", "llm_calls", "llm_tokens", "evaluation_calls")

    def __init__(self, limits: BudgetLimits, used: dict | None = None, wall_used: float = 0):
        self.limits = limits
        self.used = {key: (used or {}).get(key, 0) for key in self.resources}
        self.wall_used = wall_used
        self.start = time.monotonic()
        self.events: list[dict] = []

    @property
    def elapsed(self):
        return self.wall_used + time.monotonic() - self.start

    def remaining(self):
        return {**{k: getattr(self.limits, k) - self.used[k] for k in self.resources},
                "wall_seconds": max(0, self.limits.wall_seconds - self.elapsed)}

    def can_afford(self, **cost):
        return self.elapsed < self.limits.wall_seconds and all(
            0 <= value <= self.remaining()[key] for key, value in cost.items())

    def charge(self, reason: str, **cost):
        # Reserve a strict upper bound BEFORE side effects. Failed attempts remain charged.
        if not self.can_afford(**cost):
            raise BudgetExhausted(f"Budget cannot cover {reason}: {cost}")
        for key, value in cost.items():
            self.used[key] += value
        self.events.append({"reason": reason, "reserved_cost": cost, "wall_seconds": self.elapsed})

    def refund(self, reason: str, **cost):
        if any(v < 0 or v > self.used[k] for k, v in cost.items()):
            raise ValueError("Invalid refund")
        for key, value in cost.items():
            self.used[key] -= value
        self.events.append({"reason": reason, "refund": cost, "wall_seconds": self.elapsed})
