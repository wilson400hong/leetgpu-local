# Runtime Details

## How PyTorch Code Runs

When you click **Run** or **Submit**, the browser posts the current editor contents to `POST /api/judge`.

The server does not execute user code inside the web server process. Instead, `app.py` writes a temporary request file and starts `judge_worker.py` in a separate subprocess using the selected Python executable:

1. `app.py` finds the challenge directory, saves the latest draft, and chooses a PyTorch-capable Python.
2. The worker imports the challenge's `challenge.py` and instantiates `Challenge(device=...)`.
3. The worker writes the submitted code to a temporary `solution.py`, executes it, and looks for a callable `solve` function.
4. The worker generates tests from the challenge:
   - **Run** uses `generate_example_test()`.
   - **Submit** uses `generate_example_test()` plus `generate_functional_test()`.
5. For each test case, the worker clones inputs into two copies:
   - one copy is passed to `challenge.reference_impl(...)`
   - one copy is passed to the submitted `solve(...)`
6. Output tensors are selected from `get_solve_signature()` entries marked `out` or `inout`.
7. The worker compares submitted outputs against reference outputs with the challenge's `atol` and `rtol`.
8. The JSON result is returned to the browser and recorded in SQLite.

The worker also accepts solutions that return the first output tensor instead of mutating it, which is useful for a few PyTorch starters.

By default, the app uses CUDA if `torch.cuda.is_available()` is true, otherwise CPU. The device dropdown in the IDE can force `CUDA` or `CPU` for a run.

This is a local development sandbox, not a hardened security boundary. The subprocess has a timeout and optional resource limits, but submitted Python code still runs on the same machine as the server.
