"""Inspect a resident DIRMO LLM through HTTP without loading another model."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_PROMPT = 'Return exactly this JSON object: {"ok": true}. No other text.'
DEFAULT_SYSTEM = "You are a helpful assistant. Follow the requested output format."


def positive_timeout(value: str) -> float:
    timeout = float(value)
    if not 0 < timeout < float("inf"):
        raise argparse.ArgumentTypeError("Timeout must be a positive finite number")
    return timeout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("worker-runtime"))
    parser.add_argument("--health-only", action="store_true", help="Do not run inference")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--timeout", type=positive_timeout, default=600.0,
                        help="Generation request timeout in seconds (default: 600)")
    args = parser.parse_args()
    token = ""
    try:
        manifest = json.loads((args.runtime_dir / "workers.json").read_text())
        record = manifest.get("workers", {}).get("llm")
        if not record:
            raise ValueError("No LLM worker in the runtime manifest; start the LLM worker first")
        token = record["token"]
        url = f"http://127.0.0.1:{int(record['port'])}"

        def request(route: str, payload: dict | None = None, timeout: float = 10) -> dict:
            req = urllib.request.Request(
                url + route,
                data=json.dumps(payload).encode() if payload is not None else None,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            if not isinstance(result, dict):
                raise ValueError("Server response is not a JSON object")
            if "error" in result:
                raise RuntimeError(str(result["error"]))
            return result

        health = request("/health")
        if health.get("role") != "llm" or health.get("instance") != record["instance"]:
            raise ValueError("Server identity differs from the LLM runtime manifest")
        identity = health.get("model_identity", {})
        summary = {
            "server_ok": True,
            "url": url,
            "model_path": identity.get("path"),
            "model_id": identity.get("model_id"),
            "loader": identity.get("loader"),
            "thinking": identity.get("thinking_applied"),
            "gpu_uuid": health.get("actual_gpu_uuid"),
            "pid": health.get("pid"),
        }
        if args.health_only:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0

        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        start = time.monotonic()
        response = request("/propose", {"messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]}, timeout=args.timeout)
        raw = response.get("raw", "")
        generation_ok = response.get("status") == "ok" and isinstance(raw, str) and bool(raw.strip())
        result = {"generation_ok": generation_ok, "status": response.get("status"), "raw": raw,
                  "elapsed_seconds": round(time.monotonic() - start, 3)}
        for name in ("reason", "input_tokens", "output_tokens", "total_tokens",
                     "latency_seconds", "peak_memory_bytes"):
            if name in response:
                result[name] = response[name]
        try:
            parsed = json.loads(raw)
            result["json_valid"] = True
            if args.prompt == DEFAULT_PROMPT:
                result["json_matches_expected"] = (
                    isinstance(parsed, dict) and set(parsed) == {"ok"} and parsed["ok"] is True
                )
        except (ValueError, TypeError):
            result["json_valid"] = False
        print(json.dumps(result, indent=2, ensure_ascii=False))
        # Successful text generation and JSON-format compliance are separate checks.
        return 0 if generation_ok else 1
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if token:
            error = error.replace(token, "<redacted>")
        print(json.dumps({"status": "failed", "error": error}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
