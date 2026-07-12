# Contributing

Contributions are welcome. This document describes how to set up a development
environment, run the checks, and submit changes.

## Development setup

```bash
git clone https://github.com/Y0oshi/Text-LLM-Training-from-scratch.git
cd Text-LLM-Training-from-scratch
pip install -e ".[dev]"
```

The `dev` extra installs pytest and ruff.

## Running the checks

Before opening a pull request, make sure both of these pass:

```bash
pytest
ruff check textllm tests
```

The test suite runs on CPU in a few seconds. Continuous integration runs the same two
checks on every push and pull request.

## Code style

The project favors readability. A few conventions keep the codebase consistent:

- Use PyTorch directly rather than adding higher-level training dependencies.
- Keep device and precision handling in `textllm/device.py`. No other module should
  reference `torch.cuda` or hardcode a device.
- Write comments only where they explain a constraint or a decision that the code cannot
  express on its own.
- Add or update a test when you change behavior. The suite is the specification for how
  each stage is expected to work.

## Submitting changes

1. Create a branch for your change.
2. Make the change, with tests, and confirm `pytest` and `ruff check` pass.
3. Open a pull request describing what changed and why.

## Reporting issues

Open an issue with a clear description, the command or code that triggers the problem, the
expected result, and the actual result. Include the Python and PyTorch versions and the
device (CPU, MPS, or CUDA) when relevant.
