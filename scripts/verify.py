"""CPU unit verification with durable logs. Does not load LLM/FM weights."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
reports = root / "reports"
reports.mkdir(exist_ok=True)
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""
env["PYTHONPATH"] = str(root / "src")
start = time.monotonic()
result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                        cwd=root, env=env, capture_output=True, text=True)
log = result.stdout + result.stderr
(reports / "unit-tests.log").write_text(log)
summary = {"status": "passed" if result.returncode == 0 else "failed", "exit_code": result.returncode,
           "elapsed_seconds": time.monotonic() - start, "python": sys.executable, "cuda_visible_devices": "",
           "scope": "CPU unit tests; injected model fixtures are not real-model integration evidence"}
(reports / "unit-tests.json").write_text(json.dumps(summary, indent=2) + "\n")
print(log)
sys.exit(result.returncode)
