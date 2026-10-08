import math

from ..io import digest
from ..schemas import Evidence


def assemble_evidence(observation) -> list[Evidence]:
    evidence = []
    for source, record in observation.evaluations.items():
        facts = record.values if record.status == "ok" else {"status": record.status, "reason": record.reason}
        for key, value in facts.items():
            evidence.append(Evidence(evidence_id=digest([observation.observation_id, source, key, value])[:20],
                            observation_id=observation.observation_id, source=f"{source}.{key}",
                            fact=f"{key}={value}"))
    return evidence


def wilson(successes: int, attempts: int, z: float = 1.96) -> dict:
    if not 0 <= successes <= attempts:
        raise ValueError("Invalid success counts")
    if attempts == 0:
        return {"attempts": 0, "successes": 0, "interval": None, "uncertainty": "unobserved"}
    p, n = successes / attempts, attempts
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"attempts": n, "successes": successes, "empirical_rate": p,
            "interval": [max(0, center - half), min(1, center + half)], "method": "Wilson 95%",
            "uncertainty": "high; fewer than 10 attempts" if n < 10 else "finite-sample interval"}
