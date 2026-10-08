"""One resident model, one GPU and one serialized localhost HTTP server."""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from .config import load_config
from .gpu import assert_worker_mask
from .schemas import GenerationRequest


def serve(config, role: str, instance: str):
    assert_worker_mask()  # BEFORE any import of torch or model packages
    if role == "llm":
        from .llm.local import LocalLLM
        backend = LocalLLM(config.llm)
    else:
        from .generation.flowr import FlowrBackend
        backend = FlowrBackend(config.fm)
    token = os.environ["DIRMO_HTTP_TOKEN"]
    identity = backend.health()
    actual = identity["device_uuid"]
    if not actual.startswith("GPU-"):
        actual = "GPU-" + actual
    if actual != os.environ["DIRMO_GPU_UUID"]:
        raise RuntimeError(f"Actual torch UUID {actual} differs from the worker CUDA mask")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            import hmac
            return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}")

        def do_GET(self):
            if not self.authorized():
                return self.reply(401, {"error": "Unauthorized"})
            if self.path != "/health":
                return self.reply(404, {"error": "Unknown endpoint"})
            self.reply(200, {**backend.health(), "pid": os.getpid(), "role": role, "instance": instance,
                       "visible_cuda_devices": os.environ["CUDA_VISIBLE_DEVICES"], "actual_gpu_uuid": actual,
                       "physical_gpu_index": os.environ["DIRMO_PHYSICAL_GPU"], "logical_device": "cuda:0",
                       "request_concurrency": 1, "tensor_parallel_size": 1})

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {"error": "Unauthorized"})
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2 * 1024 * 1024:
                return self.reply(413, {"error": "Request too large"})
            try:
                payload = json.loads(self.rfile.read(length))
                if self.path == "/shutdown":
                    self.reply(200, {"stopping": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                elif self.path == "/propose" and role == "llm":
                    self.reply(200, backend.propose(payload["messages"]))
                elif self.path == "/generate" and role == "fm":
                    from .generation.flowr import CapabilityError
                    from .schemas import GenerationResult
                    try:
                        result = backend.generate_fixed_graph(GenerationRequest.model_validate(payload))
                    except CapabilityError as exc:
                        result = GenerationResult(status="capability_error", reason=str(exc), raw_draws=0, nfe=0)
                    self.reply(200, result.model_dump(mode="json"))
                else:
                    self.reply(404, {"error": "Unsupported endpoint for this worker"})
            except Exception as exc:
                self.reply(400, {"error": f"{type(exc).__name__}: {exc}"})

    server = HTTPServer(("127.0.0.1", getattr(config, role).port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["llm", "fm"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()
    serve(load_config(args.config), args.role, args.instance)


if __name__ == "__main__":
    main()
