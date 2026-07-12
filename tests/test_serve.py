import json
import threading
import urllib.request

import torch

from textllm.chat_template import SPECIAL_TOKENS
from textllm.config import Config, ModelConfig, TrainConfig
from textllm.infer.serve import make_server
from textllm.model import Transformer
from textllm.tokenizer import Tokenizer
from textllm.train.loop import save_checkpoint


def test_generate_endpoint_roundtrip(tmp_path):
    tok = Tokenizer()
    tok.train("the cat sat on the mat . " * 40, vocab_size=300)
    tok.add_special(SPECIAL_TOKENS)
    tok.save(tmp_path / "tok.json")

    cfg = Config(
        ModelConfig(vocab_size=512, context_length=64, n_embed=64, n_head=4, n_kv_head=2, n_blocks=2),
        TrainConfig(),
    )
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "model.pt", Transformer(cfg.model), None, 0, cfg)

    server = make_server(str(tmp_path / "model.pt"), str(tmp_path / "tok.json"), port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps({"prompt": "the cat", "max_new_tokens": 4, "temperature": 0}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        assert resp.status == 200
        assert isinstance(body["completion"], str)

        # Unknown paths 404 rather than crash the server.
        bad = urllib.request.Request(f"http://127.0.0.1:{port}/nope", data=b"{}")
        try:
            urllib.request.urlopen(bad, timeout=10)
            assert False, "expected a 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404

        def expect_400(raw: bytes):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=raw)
            try:
                urllib.request.urlopen(req, timeout=10)
                assert False, "expected a 400"
            except urllib.error.HTTPError as e:
                assert e.code == 400
                assert "error" in json.loads(e.read())

        expect_400(b"this is not json")
        expect_400(json.dumps({"prompt": ""}).encode())          # empty prompt
        expect_400(json.dumps({"prompt": "hi", "max_new_tokens": "lots"}).encode())
        expect_400(json.dumps(["not", "an", "object"]).encode())
        expect_400(json.dumps({"prompt": "hi", "top_p": 5}).encode())      # out of range
        expect_400(json.dumps({"prompt": "hi", "temperature": -1}).encode())
    finally:
        server.shutdown()
        thread.join(timeout=5)
