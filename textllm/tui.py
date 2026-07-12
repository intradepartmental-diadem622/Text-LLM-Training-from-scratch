"""An arrow-key menu for the whole pipeline — run `llm` with no arguments to get it.

The TUI is a thin front-end over the CLI: each screen collects answers, shows the exact
`llm ...` command it built, then runs it through the same code path, so the two
interfaces cannot drift apart.

Everything here is standard library. Keys are read in raw mode (termios on Unix, msvcrt
on Windows) and drawn with ANSI escapes. Menu and text-field state live in small pure
classes so the behavior is testable without a terminal.
"""

from __future__ import annotations

import codecs
import os
import shlex
import shutil
import sys
from pathlib import Path

# -- key parsing --------------------------------------------------------------------

_ANSI_KEYS = {
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    b"\x1b[H": "home",
    b"\x1b[F": "end",
    b"\x1b[3~": "delete",
}

_WIN_KEYS = {"H": "up", "P": "down", "M": "right", "K": "left", "G": "home", "O": "end", "S": "delete"}


def parse_key(seq: bytes) -> str:
    """Map a raw byte sequence to a key name, or the literal character typed."""
    if seq in _ANSI_KEYS:
        return _ANSI_KEYS[seq]
    if seq in (b"\r", b"\n"):
        return "enter"
    if seq in (b"\x7f", b"\x08"):
        return "backspace"
    if seq == b"\x1b":
        return "esc"
    if seq == b"\x03":
        return "ctrl-c"
    try:
        ch = seq.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    return ch if ch.isprintable() else ""


def parse_windows_key(prefix: str, code: str | None = None) -> str:
    """Map msvcrt-style keys: arrows arrive as a two-character sequence."""
    if prefix in ("\xe0", "\x00"):
        return _WIN_KEYS.get(code or "", "")
    if prefix == "\r":
        return "enter"
    if prefix == "\x08":
        return "backspace"
    if prefix == "\x1b":
        return "esc"
    if prefix == "\x03":
        return "ctrl-c"
    return prefix if prefix.isprintable() else ""


# -- interaction state (pure, no terminal) -------------------------------------------


class MenuState:
    """Cursor over a list of options; wraps at both ends."""

    def __init__(self, n_options: int) -> None:
        if n_options < 1:
            raise ValueError("a menu needs at least one option")
        self.n = n_options
        self.index = 0

    def handle(self, key: str) -> int | None:
        """Move the cursor. Returns the chosen index on enter, -1 on esc, else None."""
        if key == "up":
            self.index = (self.index - 1) % self.n
        elif key == "down":
            self.index = (self.index + 1) % self.n
        elif key == "enter":
            return self.index
        elif key in ("esc", "ctrl-c"):
            return -1
        return None


class InputState:
    """A single-line text field with cursor movement and mid-line editing."""

    def __init__(self, initial: str = "") -> None:
        self.chars = list(initial)
        self.cursor = len(self.chars)

    @property
    def text(self) -> str:
        return "".join(self.chars)

    def handle(self, key: str) -> str | None:
        """Edit the field. Returns the text on enter, None otherwise; esc returns "\\0"."""
        if key == "enter":
            return self.text
        if key in ("esc", "ctrl-c"):
            return "\0"
        if key == "left":
            self.cursor = max(0, self.cursor - 1)
        elif key == "right":
            self.cursor = min(len(self.chars), self.cursor + 1)
        elif key == "home":
            self.cursor = 0
        elif key == "end":
            self.cursor = len(self.chars)
        elif key == "backspace":
            if self.cursor > 0:
                self.chars.pop(self.cursor - 1)
                self.cursor -= 1
        elif key == "delete":
            if self.cursor < len(self.chars):
                self.chars.pop(self.cursor)
        elif len(key) == 1:
            self.chars.insert(self.cursor, key)
            self.cursor += 1
        return None


# -- terminal I/O ---------------------------------------------------------------------


class _Terminal:
    """Raw-mode keyboard reading plus a handful of ANSI drawing helpers."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._saved = None

    def __enter__(self) -> "_Terminal":
        self._raw_on()
        return self

    def __exit__(self, *exc) -> None:
        self._raw_off()
        self._write("\x1b[?25h")  # never leave the cursor hidden

    def _raw_on(self) -> None:
        if os.name == "nt":
            return  # msvcrt reads keys directly; no mode switch needed
        import termios
        import tty

        self._saved = termios.tcgetattr(self._fd)
        # TCSANOW: FLUSH drops typed-ahead keys, DRAIN can block if the reader stalls
        tty.setcbreak(self._fd, termios.TCSANOW)

    def _raw_off(self) -> None:
        if self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSANOW, self._saved)
            self._saved = None

    def read_key(self) -> str:
        if os.name == "nt":
            import msvcrt

            ch = msvcrt.getwch()
            if ch in ("\xe0", "\x00"):
                return parse_windows_key(ch, msvcrt.getwch())
            return parse_windows_key(ch)

        import select

        def pending() -> bool:
            return bool(select.select([self._fd], [], [], 0.02)[0])

        seq = os.read(self._fd, 1)
        if seq == b"\x1b":
            # read exactly one CSI sequence so pasted keystrokes don't glue together
            for _ in range(2):
                if not pending():
                    break
                seq += os.read(self._fd, 1)
            if len(seq) == 3 and seq[2:3].isdigit() and pending():
                seq += os.read(self._fd, 1)  # the trailing '~' of keys like delete
        elif seq and seq[0] >= 0x80:
            # Multi-byte UTF-8 character: keep reading until it decodes.
            decoder = codecs.getincrementaldecoder("utf-8")()
            if not decoder.decode(seq):
                while pending():
                    seq += os.read(self._fd, 1)
                    if codecs.getincrementaldecoder("utf-8")().decode(seq, final=False):
                        break
        return parse_key(seq)

    # drawing

    def _write(self, s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()

    def clear(self) -> None:
        self._write("\x1b[2J\x1b[H")

    def menu(self, title: str, options: list[tuple[str, str]], footer: str = "") -> int:
        """Draw a menu until the user picks (returns index) or backs out (returns -1)."""
        state = MenuState(len(options))
        label_w = max(len(label) for label, _ in options)
        self._write("\x1b[?25l")
        try:
            while True:
                self.clear()
                self._write(f"\x1b[1m  {title}\x1b[0m\r\n\r\n")
                for i, (label, hint) in enumerate(options):
                    marker = "\x1b[7m" if i == state.index else ""
                    pad = label.ljust(label_w)
                    hint_txt = f"  \x1b[2m{hint}\x1b[0m" if hint else ""
                    self._write(f"  {marker}  {pad}  \x1b[0m{hint_txt}\r\n")
                self._write(f"\r\n\x1b[2m  ↑/↓ move · enter select · esc {footer or 'back'}\x1b[0m\r\n")
                result = state.handle(self.read_key())
                if result is not None:
                    return result
        finally:
            self._write("\x1b[?25h")

    def ask(self, prompt: str, default: str = "", validate=None) -> str | None:
        """A grey input box. Returns the text, or None if the user pressed esc."""
        state = InputState(default)
        width = max(30, min(60, shutil.get_terminal_size().columns - len(prompt) - 8))
        error = ""
        while True:
            self.clear()
            self._write(f"\x1b[1m  {prompt}\x1b[0m\r\n\r\n")
            shown = state.text[:width].ljust(width)
            self._write(f"  \x1b[48;5;237m\x1b[97m {shown} \x1b[0m\r\n")
            if error:
                self._write(f"\r\n  \x1b[31m{error}\x1b[0m\r\n")
            self._write("\r\n\x1b[2m  enter confirm · esc back\x1b[0m")
            # Park the real (blinking) cursor inside the box, right at the edit point.
            col = 4 + min(state.cursor, width)
            self._write(f"\x1b[{4 if error else 3}A\r\x1b[{col}C" if error else f"\x1b[3A\r\x1b[{col}C")
            self._write("\x1b[s")  # save, so we can restore before redrawing
            result = state.handle(self.read_key())
            self._write("\x1b[u")
            if result == "\0":
                return None
            if result is not None:
                if validate:
                    error = validate(result) or ""
                    if error:
                        continue
                return result

    def say(self, text: str) -> None:
        self.clear()
        self._write(f"  {text}\r\n\r\n\x1b[2m  press any key\x1b[0m\r\n")
        self.read_key()


# -- validators and defaults ----------------------------------------------------------


def _must_exist(path: str) -> str | None:
    return None if Path(path).exists() else f"not found: {path}"


def _must_be_int(value: str) -> str | None:
    return None if value.isdigit() else "enter a whole number"


def _default(path: str) -> str:
    """Offer a path as the prefilled answer only when it actually exists."""
    return path if Path(path).exists() else ""


def _config_options() -> list[str]:
    found = sorted(str(p) for p in Path("configs").glob("*.json"))
    return found or ["configs/tiny.json"]


# -- flows: each one collects answers and returns an `llm ...` argv -------------------


def flow_tokenizer(term: _Terminal) -> list[str] | None:
    corpus = term.ask("Text file to learn the vocabulary from", validate=_must_exist)
    if corpus is None:
        return None
    vocab = term.ask("Vocabulary size", "8192", validate=_must_be_int)
    if vocab is None:
        return None
    out = term.ask("Save tokenizer as", "tok.json")
    if out is None:
        return None
    return ["tokenizer", "train", "--input", corpus, "--vocab-size", vocab, "--out", out]


def flow_data(term: _Terminal) -> list[str] | None:
    tok = term.ask("Tokenizer file", _default("tok.json"), validate=_must_exist)
    if tok is None:
        return None
    corpus = term.ask("Text file to tokenize", validate=_must_exist)
    if corpus is None:
        return None
    sep = term.ask("Document separator, if the file has one (blank for none)", "")
    if sep is None:
        return None
    argv = ["data", "--tokenizer", tok, "--input", corpus, "--out-dir", "data"]
    if sep:
        argv += ["--doc-sep", sep]
    return argv


def flow_pretrain(term: _Terminal) -> list[str] | None:
    configs = _config_options()
    choice = term.menu("Pick a model size", [(c, "") for c in configs])
    if choice < 0:
        return None
    if _must_exist("data/train.bin"):
        term.say("data/train.bin is missing — run “Prepare training data” first.")
        return None
    argv = ["pretrain", "--config", configs[choice]]
    resume = term.ask("Resume from a checkpoint? (blank to start fresh)", "")
    if resume is None:
        return None
    if resume:
        argv += ["--resume", resume]
    return argv


def flow_sft(term: _Terminal) -> list[str] | None:
    base = term.ask("Base model checkpoint", _default("checkpoints/final.pt"), validate=_must_exist)
    if base is None:
        return None
    tok = term.ask("Tokenizer file", _default("tok.json"), validate=_must_exist)
    if tok is None:
        return None
    data = term.ask("Chat data (.jsonl with {\"messages\": [...]})", _default("examples/sft.jsonl"), validate=_must_exist)
    if data is None:
        return None
    configs = _config_options()
    choice = term.menu("Training settings", [(c, "") for c in configs])
    if choice < 0:
        return None
    return ["sft", "--config", configs[choice], "--base", base, "--tokenizer", tok,
            "--data", data, "--set", "train.out_dir=sft_ckpt"]


def flow_dpo(term: _Terminal) -> list[str] | None:
    base = term.ask("Model checkpoint to tune", _default("sft_ckpt/final.pt") or _default("checkpoints/final.pt"), validate=_must_exist)
    if base is None:
        return None
    tok = term.ask("Tokenizer file", _default("tok.json"), validate=_must_exist)
    if tok is None:
        return None
    data = term.ask("Preference data (.jsonl with prompt/chosen/rejected)", _default("examples/prefs.jsonl"), validate=_must_exist)
    if data is None:
        return None
    return ["dpo", "--base", base, "--tokenizer", tok, "--data", data,
            "--set", "train.out_dir=dpo_ckpt"]


def flow_generate(term: _Terminal) -> list[str] | None:
    ckpt = term.ask("Model checkpoint", _default("checkpoints/final.pt"), validate=_must_exist)
    if ckpt is None:
        return None
    tok = term.ask("Tokenizer file", _default("tok.json"), validate=_must_exist)
    if tok is None:
        return None
    prompt = term.ask("Prompt", "Once upon a time")
    if prompt is None:
        return None
    return ["generate", "--ckpt", ckpt, "--tokenizer", tok, "--prompt", prompt,
            "--max-new-tokens", "150", "--temperature", "0.8", "--top-p", "0.9"]


def flow_chat(term: _Terminal) -> list[str] | None:
    ckpt = term.ask("Model checkpoint", _default("sft_ckpt/final.pt") or _default("checkpoints/final.pt"), validate=_must_exist)
    if ckpt is None:
        return None
    tok = term.ask("Tokenizer file", _default("tok.json"), validate=_must_exist)
    if tok is None:
        return None
    return ["chat", "--ckpt", ckpt, "--tokenizer", tok]


def flow_eval(term: _Terminal) -> list[str] | None:
    which = term.menu("What to measure", [
        ("Perplexity", "how well the model predicts held-out text (lower is better)"),
        ("Accuracy", "question answering against a .jsonl of questions/answers"),
    ])
    if which < 0:
        return None
    ckpt = term.ask("Model checkpoint", _default("checkpoints/final.pt"), validate=_must_exist)
    if ckpt is None:
        return None
    if which == 0:
        data = term.ask("Token file to test on", _default("data/val.bin"), validate=_must_exist)
        if data is None:
            return None
        return ["eval", "ppl", "--ckpt", ckpt, "--data", data]
    tok = term.ask("Tokenizer file", _default("tok.json"), validate=_must_exist)
    if tok is None:
        return None
    data = term.ask("Questions file (.jsonl)", _default("examples/questions.jsonl"), validate=_must_exist)
    if data is None:
        return None
    return ["eval", "acc", "--ckpt", ckpt, "--tokenizer", tok, "--data", data]


MAIN_MENU = [
    ("Train a tokenizer", "learn a vocabulary from your text file", flow_tokenizer),
    ("Prepare training data", "turn text into token files the trainer reads", flow_data),
    ("Pretrain a model", "the main event — train on your data", flow_pretrain),
    ("Fine-tune for chat (SFT)", "teach a trained model to follow a chat format", flow_sft),
    ("Tune with preferences (DPO)", "push it toward responses people prefer", flow_dpo),
    ("Generate text", "give a prompt, watch it continue", flow_generate),
    ("Chat", "talk to a fine-tuned model", flow_chat),
    ("Evaluate", "measure perplexity or accuracy", flow_eval),
    ("Quit", "", None),
]


def run() -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("the interactive menu needs a real terminal — run `llm --help` for the commands")
        return 1

    from textllm.cli import main as run_cli

    while True:
        with _Terminal() as term:
            choice = term.menu(
                "textllm — train a language model from scratch",
                [(label, hint) for label, hint, _ in MAIN_MENU],
                footer="quit",
            )
            if choice < 0 or MAIN_MENU[choice][2] is None:
                term.clear()
                return 0
            argv = MAIN_MENU[choice][2](term)
            term.clear()

        if argv is None:
            continue
        # out of raw mode: print the equivalent command, then run it
        print(f"$ llm {' '.join(shlex.quote(a) for a in argv)}\n", flush=True)
        try:
            run_cli(argv)
        except KeyboardInterrupt:
            print("\ninterrupted — back to the menu")
        except SystemExit as e:
            if e.code not in (0, None):
                print(f"\ncommand failed: {e.code}")
        except Exception as e:  # keep the menu alive whatever a stage throws
            print(f"\nerror: {e}")
        try:
            input("\npress enter to return to the menu…")
        except (EOFError, KeyboardInterrupt):
            return 0
