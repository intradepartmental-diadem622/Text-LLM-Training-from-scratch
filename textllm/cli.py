"""The `llm` command — one entry point for the whole pipeline.

    llm tokenizer train --input corpus.txt --vocab-size 8192 --out tok.json
    llm data --tokenizer tok.json --input corpus.txt --out-dir data
    llm pretrain --config configs/tiny.json
    llm sft      --config configs/tiny.json --base checkpoints/final.pt --tokenizer tok.json --data sft.jsonl
    llm chat     --ckpt checkpoints/final.pt --tokenizer tok.json

Running bare ``llm`` in a terminal opens the interactive menu instead (also ``llm tui``).
Every training command accepts ``--config FILE`` and repeatable ``--set section.field=value``
overrides, e.g. ``--set train.steps=2000 --set model.n_embed=256``.
"""

from __future__ import annotations

import argparse
import sys

from textllm.config import Config, apply_overrides


def _config_from_args(args) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    overrides = {}
    for item in getattr(args, "set", []) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects section.field=value, got '{item}'")
        key, value = item.split("=", 1)
        overrides[key] = value
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def _add_config_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to a JSON config; defaults are used if omitted")
    p.add_argument("--set", action="append", default=[], metavar="section.field=value")


def _add_sampling_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)


# -- command handlers -------------------------------------------------------------


def _cmd_tokenizer(args) -> None:
    from textllm.chat_template import SPECIAL_TOKENS
    from textllm.tokenizer import Tokenizer

    text = open(args.input, encoding="utf-8").read()
    tok = Tokenizer()
    tok.train(text, vocab_size=args.vocab_size, verbose=True)
    tok.add_special(SPECIAL_TOKENS)
    tok.save(args.out)
    print(f"trained tokenizer with {tok.vocab_size} tokens -> {args.out}")


def _cmd_data(args) -> None:
    from textllm.chat_template import EOT
    from textllm.data import build_bin
    from textllm.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.tokenizer)
    text = open(args.input, encoding="utf-8").read()

    if args.doc_sep:
        # split into documents so each ends on EOT and train/val cuts at boundaries
        docs = [d for d in text.split(args.doc_sep) if d.strip()]
        n_val = int(len(docs) * args.val_fraction)
        train_docs = docs[: len(docs) - n_val]
        val_docs = docs[len(docs) - n_val :] if n_val else []
    else:
        split = int(len(text) * (1 - args.val_fraction))
        train_docs = [text[:split]]
        val_docs = [text[split:]] if args.val_fraction > 0 else []

    train_n = build_bin(tokenizer, train_docs, f"{args.out_dir}/train.bin", eot=EOT)
    print(f"train.bin: {train_n} tokens ({len(train_docs)} documents)")
    if val_docs:
        val_n = build_bin(tokenizer, val_docs, f"{args.out_dir}/val.bin", eot=EOT)
        print(f"val.bin: {val_n} tokens ({len(val_docs)} documents)")


def _cmd_pretrain(args) -> None:
    from textllm.train.pretrain import run_pretrain

    run_pretrain(_config_from_args(args), resume=args.resume)


def _cmd_sft(args) -> None:
    from textllm.train.sft import run_sft

    run_sft(_config_from_args(args), args.base, args.tokenizer, args.data)


def _cmd_reward(args) -> None:
    from textllm.train.reward import run_reward

    run_reward(_config_from_args(args), args.base, args.tokenizer, args.data)


def _cmd_dpo(args) -> None:
    from textllm.train.dpo import run_dpo

    run_dpo(_config_from_args(args), args.base, args.tokenizer, args.data, beta=args.beta)


def _cmd_grpo(args) -> None:
    from textllm.train.grpo import run_grpo
    from textllm.train.rewards import contains

    prompts = [ln for ln in open(args.prompts, encoding="utf-8").read().splitlines() if ln.strip()]
    run_grpo(_config_from_args(args), args.base, args.tokenizer, prompts, contains(args.contains))


def _cmd_generate(args) -> None:
    from textllm.infer.chat import complete

    text = complete(
        args.ckpt, args.tokenizer, args.prompt, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    )
    print(text)


def _cmd_chat(args) -> None:
    from textllm.infer.chat import chat_repl

    chat_repl(
        args.ckpt, args.tokenizer, system=args.system,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    )


def _cmd_serve(args) -> None:
    from textllm.infer.serve import serve

    serve(args.ckpt, args.tokenizer, host=args.host, port=args.port)


def _cmd_eval(args) -> None:
    if args.metric == "ppl":
        from textllm.eval import perplexity

        print(f"perplexity: {perplexity(args.ckpt, args.data):.3f}")
    else:
        if not args.tokenizer:
            raise SystemExit("llm eval acc requires --tokenizer")
        from textllm.eval import accuracy

        print(f"accuracy: {accuracy(args.ckpt, args.tokenizer, args.data):.3f}")


def _cmd_tui(args) -> None:
    from textllm.tui import run

    raise SystemExit(run())


# -- parser -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm", description="Train a language model from scratch.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tokenizer", help="train a byte-level BPE tokenizer")
    tsub = p.add_subparsers(dest="action", required=True)
    pt = tsub.add_parser("train")
    pt.add_argument("--input", required=True)
    pt.add_argument("--vocab-size", type=int, default=8192)
    pt.add_argument("--out", default="tokenizer.json")
    pt.set_defaults(func=_cmd_tokenizer)

    p = sub.add_parser("data", help="tokenize text into train/val .bin files")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--doc-sep", default=None, help="split the input into documents on this string, e.g. '<|endoftext|>'")
    p.set_defaults(func=_cmd_data)

    p = sub.add_parser("pretrain", help="pretrain the base model")
    _add_config_flags(p)
    p.add_argument("--resume")
    p.set_defaults(func=_cmd_pretrain)

    p = sub.add_parser("sft", help="supervised fine-tuning")
    _add_config_flags(p)
    p.add_argument("--base", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True)
    p.set_defaults(func=_cmd_sft)

    p = sub.add_parser("reward", help="train a Bradley-Terry reward model")
    _add_config_flags(p)
    p.add_argument("--base", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True)
    p.set_defaults(func=_cmd_reward)

    p = sub.add_parser("dpo", help="direct preference optimization")
    _add_config_flags(p)
    p.add_argument("--base", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.set_defaults(func=_cmd_dpo)

    p = sub.add_parser("grpo", help="GRPO / RLVR with a verifiable reward")
    _add_config_flags(p)
    p.add_argument("--base", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompts", required=True, help="text file, one prompt per line")
    p.add_argument("--contains", required=True, help="reward completions containing this string")
    p.set_defaults(func=_cmd_grpo)

    p = sub.add_parser("generate", help="complete a raw prompt")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    _add_sampling_flags(p)
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("chat", help="interactive chat")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--system", default=None)
    _add_sampling_flags(p)
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("serve", help="serve the model over HTTP")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("eval", help="evaluate a checkpoint")
    p.add_argument("metric", choices=["ppl", "acc"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--data", required=True)
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("tui", help="interactive menu (same as running bare `llm`)")
    p.set_defaults(func=_cmd_tui)

    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        # bare `llm` opens the interactive menu
        from textllm.tui import run

        raise SystemExit(run())
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
