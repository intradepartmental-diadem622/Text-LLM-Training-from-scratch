"""The TUI's interaction logic is pure state — test it without a terminal."""

from textllm.tui import InputState, MenuState, parse_key, parse_windows_key


def test_parse_key_covers_the_keys_we_use():
    assert parse_key(b"\x1b[A") == "up"
    assert parse_key(b"\x1b[B") == "down"
    assert parse_key(b"\x1b[D") == "left"
    assert parse_key(b"\x1b[C") == "right"
    assert parse_key(b"\r") == "enter"
    assert parse_key(b"\n") == "enter"
    assert parse_key(b"\x7f") == "backspace"
    assert parse_key(b"\x1b") == "esc"
    assert parse_key(b"\x03") == "ctrl-c"
    assert parse_key(b"a") == "a"
    assert parse_key("é".encode()) == "é"
    assert parse_key(b"\x00") == ""  # unprintable control bytes are ignored


def test_parse_windows_arrow_sequences():
    assert parse_windows_key("\xe0", "H") == "up"
    assert parse_windows_key("\xe0", "P") == "down"
    assert parse_windows_key("\r") == "enter"
    assert parse_windows_key("x") == "x"


def test_menu_wraps_and_selects():
    m = MenuState(3)
    assert m.handle("up") is None and m.index == 2      # wraps from top to bottom
    assert m.handle("down") is None and m.index == 0    # and back
    m.handle("down")
    assert m.handle("enter") == 1
    assert m.handle("esc") == -1


def test_input_editing_mid_line():
    s = InputState("helo")
    s.handle("left")                      # cursor before the 'o'
    s.handle("l")
    assert s.text == "hello"
    assert s.handle("enter") == "hello"


def test_input_backspace_delete_home_end():
    s = InputState("abc")
    s.handle("home")
    s.handle("delete")
    assert s.text == "bc"
    s.handle("end")
    s.handle("backspace")
    assert s.text == "b"
    assert s.handle("esc") == "\0"        # esc cancels with a sentinel


def test_input_prefilled_default():
    s = InputState("tok.json")
    assert s.handle("enter") == "tok.json"


class _ScriptedTerminal:
    """Stands in for _Terminal: answers ask() and menu() from a queue."""

    def __init__(self, answers):
        self.answers = list(answers)

    def ask(self, prompt, default="", validate=None):
        value = self.answers.pop(0)
        if value == "<default>":
            value = default
        return value

    def menu(self, title, options, footer=""):
        return self.answers.pop(0)

    def say(self, text):
        self.said = text


def test_tokenizer_flow_builds_the_right_command(tmp_path):
    from textllm.tui import flow_tokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello")
    term = _ScriptedTerminal([str(corpus), "4096", "tok.json"])
    argv = flow_tokenizer(term)
    assert argv == ["tokenizer", "train", "--input", str(corpus), "--vocab-size", "4096", "--out", "tok.json"]


def test_flows_bail_out_on_esc(tmp_path):
    from textllm.tui import flow_generate, flow_tokenizer

    assert flow_tokenizer(_ScriptedTerminal([None])) is None
    assert flow_generate(_ScriptedTerminal([None])) is None


def test_data_flow_includes_separator_only_when_given(tmp_path, monkeypatch):
    from textllm.tui import flow_data

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tok.json").write_text("{}")
    (tmp_path / "corpus.txt").write_text("hello")

    argv = flow_data(_ScriptedTerminal(["tok.json", "corpus.txt", ""]))
    assert "--doc-sep" not in argv
    argv = flow_data(_ScriptedTerminal(["tok.json", "corpus.txt", "<|endoftext|>"]))
    assert argv[-2:] == ["--doc-sep", "<|endoftext|>"]


def test_pretrain_flow_warns_when_data_missing(tmp_path, monkeypatch):
    from textllm.tui import flow_pretrain

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "tiny.json").write_text("{}")
    term = _ScriptedTerminal([0])
    assert flow_pretrain(term) is None
    assert "Prepare training data" in term.said


def test_bare_llm_without_a_tty_prints_help_not_crash(capsys):
    import pytest

    from textllm.cli import main

    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 1  # no TTY in pytest: friendly message, nonzero exit
    assert "terminal" in capsys.readouterr().out
