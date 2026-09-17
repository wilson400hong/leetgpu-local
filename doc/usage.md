# Usage

## Prerequisites

- Python 3.10+ for the web server.
- A Python environment with PyTorch for judging submitted solutions.
- The `leetgpu-challenges` directory available next to `app.py`, or configured with `challenges_dir`.

## Start the Server

Fetch the challenge dataset submodule once after cloning:

```bash
git submodule update --init --recursive
```

Then start the app:

```bash
python3 app.py
```

Open the printed URL in a browser. By default this is:

```text
http://127.0.0.1:8000
```

The server binds to `127.0.0.1` and picks the next open port if `8000` is busy.

## Config File

Local settings live in `leetgpu.config.json`:

```json
{
  "storage_dir": "~/.local/share/leetgpu-local",
  "challenges_dir": "leetgpu-challenges",
  "judge_python": "",
  "host": "127.0.0.1",
  "port": 8000,
  "run_timeout_seconds": 20,
  "submit_timeout_seconds": 90,
  "max_test_bytes": 1073741824,
  "memory_mb": 0,
  "cpu_seconds": 0
}
```

Relative paths in the config are resolved from the folder containing `leetgpu.config.json`.

To change where user history is stored:

```json
{
  "storage_dir": "/path/to/leetgpu-user-data"
}
```

The app stores history at:

```text
<storage_dir>/leetgpu.sqlite3
```

To force a specific PyTorch runtime:

```json
{
  "judge_python": "/path/to/python-with-torch"
}
```

To use a challenge dataset in another location:

```json
{
  "challenges_dir": "/path/to/leetgpu-challenges"
}
```

## Environment Overrides

Environment variables override `leetgpu.config.json` for one-off runs.

Use a different config file:

```bash
LEETGPU_CONFIG=/path/to/leetgpu.config.json python3 app.py
```

Use a different storage folder:

```bash
LEETGPU_DATA_DIR=/path/to/app-data python3 app.py
```

Point directly at a database file:

```bash
LEETGPU_DB=/path/to/leetgpu.sqlite3 python3 app.py
```

Force a specific PyTorch runtime:

```bash
LEETGPU_PYTHON=/path/to/python-with-torch python3 app.py
```

Use a different challenge dataset:

```bash
LEETGPU_CHALLENGES_DIR=/path/to/leetgpu-challenges python3 app.py
```

## Remote Access

If the server is running on a remote dev machine, use SSH port forwarding from your laptop:

```bash
ssh -N -L 18000:127.0.0.1:8000 user@devserver
```

Then open:

```text
http://127.0.0.1:18000
```

## History Storage

Progress, saved drafts, and submissions are stored in the history database printed at startup.

Default history path:

```text
~/.local/share/leetgpu-local/leetgpu.sqlite3
```

If an old project-local `data/leetgpu.sqlite3` exists and the new default database does not, the app copies it to the external history path on startup. The old file is left in place.

The project-local `data/` directory is ignored by Git because it may contain personal solve history.
