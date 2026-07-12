"""Regression tests for the `llm` CLI — the exact surface the README documents.

These call ``main()`` with real argv, so argument wiring, config overrides, and the lazy
imports inside each handler are all exercised, not just the underlying functions.
"""

import pytest

from textllm.cli import main

TINY_OVERRIDES = [
    "--set", "model.vocab_size=512",
    "--set", "model.context_length=64",
    "--set", "model.n_embed=64",
    "--set", "model.n_head=4",
    "--set", "model.n_kv_head=2",
    "--set", "model.n_blocks=2",
    "--set", "train.steps=60",
    "--set", "train.batch_size=8",
    "--set", "train.warmup_steps=5",
    "--set", "train.lr=3e-3",
    "--set", "train.eval_every=0",
    "--set", "train.device=cpu",
]


def test_full_pipeline_via_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("the cat sat on the mat . " * 300)

    main(["tokenizer", "train", "--input", "corpus.txt", "--vocab-size", "300", "--out", "tok.json"])
    main(["data", "--tokenizer", "tok.json", "--input", "corpus.txt", "--out-dir", "data"])
    main(["pretrain", *TINY_OVERRIDES, "--set", "train.data_dir=data", "--set", "train.out_dir=ckpt"])

    main([
        "generate", "--ckpt", "ckpt/final.pt", "--tokenizer", "tok.json",
        "--prompt", "the cat", "--max-new-tokens", "8", "--temperature", "0",
    ])
    out = capsys.readouterr().out
    assert "the cat" in out

    main(["eval", "ppl", "--ckpt", "ckpt/final.pt", "--data", "data/val.bin"])
    assert "perplexity" in capsys.readouterr().out


def test_data_doc_sep_splits_at_document_boundaries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stories = "<|endoftext|>".join(["the cat sat on the mat . " * 5] * 20)
    (tmp_path / "corpus.txt").write_text(stories)

    main(["tokenizer", "train", "--input", "corpus.txt", "--vocab-size", "300", "--out", "tok.json"])
    main(["data", "--tokenizer", "tok.json", "--input", "corpus.txt", "--out-dir", "data",
          "--doc-sep", "<|endoftext|>", "--val-fraction", "0.2"])

    from textllm.data import load_bin
    from textllm.tokenizer import Tokenizer

    tok = Tokenizer.load("tok.json")
    eot = tok.encode("<|end|>")[0]
    train = load_bin("data/train.bin")
    val = load_bin("data/val.bin")
    assert (train == eot).sum() == 16  # one EOT per training document
    assert (val == eot).sum() == 4


def test_malformed_set_flag_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="section.field=value"):
        main(["pretrain", "--set", "train.steps"])


def test_eval_acc_requires_tokenizer(tmp_path):
    with pytest.raises(SystemExit, match="tokenizer"):
        main(["eval", "acc", "--ckpt", "x.pt", "--data", "x.jsonl"])


def test_missing_data_dir_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="llm data"):
        main(["pretrain", *TINY_OVERRIDES, "--set", "train.data_dir=nowhere", "--set", "train.out_dir=ckpt"])


def test_pretrain_rejects_token_file_too_short_for_a_window(tmp_path, monkeypatch):
    import numpy as np
    import pytest

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    np.arange(10, dtype=np.uint16).tofile(tmp_path / "data" / "train.bin")  # context is 64

    with pytest.raises(ValueError, match="context_length"):
        main(["pretrain", *TINY_OVERRIDES, "--set", "train.data_dir=data", "--set", "train.out_dir=ckpt"])
