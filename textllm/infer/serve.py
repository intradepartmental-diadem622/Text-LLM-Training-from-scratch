"""A tiny local HTTP endpoint for the model, built on the standard library only.

POST /generate with {"prompt": "...", "max_new_tokens": 128, "temperature": 0.8}
and get back {"completion": "..."}. Binds to localhost by default; meant for local use,
not the open internet.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from textllm.chat_template import EOT
from textllm.device import describe, pick_device
from textllm.infer.generate import generate
from textllm.infer.sample import validate_sampling
from textllm.runtime import load_model
from textllm.tokenizer import get_tokenizer

MAX_NEW_TOKENS_CAP = 2048


def make_server(ckpt_path: str, tokenizer_spec: str, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Build the server without starting it — lets tests bind port 0 and drive it directly."""
    device = pick_device()
    model, cfg = load_model(ckpt_path, device)
    tokenizer = get_tokenizer(tokenizer_spec)
    # stop at end-of-turn so a chat-tuned model doesn't run past its reply
    stop_ids = {tid for text, tid in getattr(tokenizer, "special", {}).items() if text == EOT}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or "{}")
                if not isinstance(req, dict):
                    raise ValueError("request body must be a JSON object")
                prompt = req.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("'prompt' must be a non-empty string")
                max_new = min(max(1, int(req.get("max_new_tokens", 128))), MAX_NEW_TOKENS_CAP)
                temperature = float(req.get("temperature", 0.8))
                top_k = int(req["top_k"]) if req.get("top_k") is not None else None
                top_p = float(req["top_p"]) if req.get("top_p") is not None else None
                validate_sampling(temperature, top_k, top_p, repetition_penalty=1.0)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                self._reply(400, {"error": str(e)})
                return

            try:
                prompt_ids = tokenizer.encode(prompt)[-cfg.context_length :]
                ids = generate(
                    model,
                    prompt_ids,
                    device,
                    max_new_tokens=max_new,
                    stop_ids=stop_ids,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                self._reply(200, {"completion": tokenizer.decode(ids)})
            except Exception as e:
                self._reply(500, {"error": str(e)})

    return ThreadingHTTPServer((host, port), Handler)


def serve(ckpt_path: str, tokenizer_spec: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    print(f"device: {describe(pick_device())}")
    server = make_server(ckpt_path, tokenizer_spec, host, port)
    print(f"serving on http://{host}:{server.server_address[1]}  (POST /generate)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()
