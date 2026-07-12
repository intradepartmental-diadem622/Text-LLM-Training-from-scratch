"""End-to-end: tokenizer -> shards -> pretrain -> generate, all on CPU in a few seconds."""

import torch

from textllm.chat_template import EOT, SPECIAL_TOKENS
from textllm.config import Config, ModelConfig, TrainConfig
from textllm.data import PretrainDataset, build_bin
from textllm.infer.generate import generate
from textllm.runtime import load_model
from textllm.tokenizer import Tokenizer
from textllm.train.pretrain import run_pretrain


def test_pipeline_learns_a_tiny_corpus(tmp_path):
    corpus = "the cat sat on the mat . " * 300

    tok = Tokenizer()
    tok.train(corpus, vocab_size=300)
    tok.add_special(SPECIAL_TOKENS)
    build_bin(tok, [corpus], tmp_path / "train.bin", eot=EOT)

    cfg = Config(
        ModelConfig(vocab_size=512, context_length=64, n_embed=64, n_head=4, n_kv_head=2, n_blocks=2),
        TrainConfig(
            steps=120, batch_size=8, warmup_steps=10, lr=3e-3,
            eval_every=0, save_every=0, device="cpu",
            data_dir=str(tmp_path), out_dir=str(tmp_path / "ckpt"),
        ),
    )

    baseline = _loss_of_fresh_model(cfg, tmp_path)
    run_pretrain(cfg)

    model, model_cfg = load_model(str(tmp_path / "ckpt" / "final.pt"), torch.device("cpu"))
    trained = _dataset_loss(model, cfg)
    assert trained < baseline * 0.7  # training clearly reduced the loss

    out = generate(model, tok.encode("the cat"), torch.device("cpu"), max_new_tokens=8, temperature=0)
    assert len(out) == 8


def _loss_of_fresh_model(cfg, tmp_path):
    from textllm.model import Transformer

    torch.manual_seed(cfg.train.seed)
    return _dataset_loss(Transformer(cfg.model), cfg)


@torch.no_grad()
def _dataset_loss(model, cfg):
    ds = PretrainDataset(f"{cfg.train.data_dir}/train.bin", cfg.model.context_length)
    x, y = ds[0]
    _, loss = model(x[None, :], y[None, :])
    return loss.item()
