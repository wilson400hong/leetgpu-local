#!/usr/bin/env python3
from __future__ import annotations

import ast
import datetime as dt
import html
import json
import mimetypes
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
APP_NAME = "leetgpu-local"
LEGACY_DATA_ROOT = ROOT / "data"
CONFIG_PATH = Path(os.environ.get("LEETGPU_CONFIG", ROOT / "leetgpu.config.json")).expanduser()
JUDGE_WORKER = ROOT / "judge_worker.py"

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
META_KEYS = {"name", "atol", "rtol", "num_gpus", "access_tier"}
CHALLENGE_ID_RE = re.compile(r"^(?P<number>\d+)_")


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid config file {CONFIG_PATH}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Config file {CONFIG_PATH} must contain a JSON object")
    return parsed


CONFIG = load_config()


def config_value(key: str, env_name: str, default: Any = None) -> Any:
    if env_name in os.environ and os.environ[env_name] != "":
        return os.environ[env_name]
    value = CONFIG.get(key)
    if value == "":
        return default
    return default if value is None else value


def config_int(key: str, env_name: str, default: int) -> int:
    value = config_value(key, env_name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Config value {key!r} must be an integer, got {value!r}")


def config_path(key: str, env_name: str, default: str | Path) -> Path:
    value = config_value(key, env_name, default)
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    return (CONFIG_PATH.parent / path).resolve()


def default_data_root() -> Path:
    configured = config_value("storage_dir", "LEETGPU_DATA_DIR")
    if configured:
        return config_path("storage_dir", "LEETGPU_DATA_DIR", configured)
    legacy_configured = CONFIG.get("data_dir")
    if legacy_configured:
        return config_path("data_dir", "LEETGPU_DATA_DIR", legacy_configured)
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


CHALLENGES_REPO = config_path("challenges_dir", "LEETGPU_CHALLENGES_DIR", ROOT / "leetgpu-challenges")
CHALLENGES_ROOT = CHALLENGES_REPO / "challenges"
DATA_ROOT = default_data_root()
DB_PATH = config_path("db_path", "LEETGPU_DB", DATA_ROOT / "leetgpu.sqlite3")
LEGACY_DB_PATH = LEGACY_DATA_ROOT / "leetgpu.sqlite3"


class FirstParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_first_p = False
        self.done = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "p" and not self.done and not self.in_first_p:
            self.in_first_p = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "p" and self.in_first_p:
            self.in_first_p = False
            self.done = True

    def handle_data(self, data: str) -> None:
        if self.in_first_p:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


@dataclass(frozen=True)
class ChallengeInfo:
    id: str
    number: int
    slug: str
    difficulty: str
    title: str
    description: str
    path: Path
    access_tier: str
    atol: float | None
    rtol: float | None
    num_gpus: int | None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def migrate_legacy_db() -> None:
    if "LEETGPU_DB" in os.environ:
        return
    if DB_PATH.exists() or not LEGACY_DB_PATH.exists():
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_DB_PATH, DB_PATH)
    print(f"Migrated history database: {LEGACY_DB_PATH} -> {DB_PATH}")


def init_db() -> None:
    migrate_legacy_db()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
                challenge_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'untried',
                attempts INTEGER NOT NULL DEFAULT 0,
                solved_at TEXT,
                last_attempt_at TEXT,
                last_code TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge_id TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                passed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                skipped INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                message TEXT NOT NULL,
                output TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def db_rows(query: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return list(conn.execute(query, args))


def db_execute(query: str, args: tuple[Any, ...] = ()) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(query, args)
        conn.commit()


def get_progress_map() -> dict[str, dict[str, Any]]:
    rows = db_rows("SELECT * FROM progress")
    return {row["challenge_id"]: dict(row) for row in rows}


def get_saved_code(challenge_id: str) -> str | None:
    rows = db_rows("SELECT last_code FROM progress WHERE challenge_id = ?", (challenge_id,))
    if not rows:
        return None
    return rows[0]["last_code"]


def save_code(challenge_id: str, code: str) -> None:
    now = utc_now()
    db_execute(
        """
        INSERT INTO progress (challenge_id, status, attempts, last_code, updated_at)
        VALUES (?, 'untried', 0, ?, ?)
        ON CONFLICT(challenge_id) DO UPDATE SET
            last_code = excluded.last_code,
            updated_at = excluded.updated_at
        """,
        (challenge_id, code, now),
    )


def record_submission(
    challenge_id: str,
    action: str,
    code: str,
    result: dict[str, Any],
) -> None:
    now = utc_now()
    success = bool(result.get("success"))
    status = "solved" if success and action == "submit" else "tried"
    summary = result.get("summary") or {}
    passed = int(summary.get("passed") or 0)
    failed = int(summary.get("failed") or 0)
    skipped = int(summary.get("skipped") or 0)
    duration_ms = int(result.get("durationMs") or 0)
    message = str(result.get("message") or result.get("status") or "")
    output = json.dumps(result, ensure_ascii=True)

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT status, attempts FROM progress WHERE challenge_id = ?",
            (challenge_id,),
        ).fetchone()
        previous_status = row[0] if row else "untried"
        attempts = int(row[1] if row else 0) + 1
        next_status = "solved" if previous_status == "solved" else status
        solved_at = now if next_status == "solved" and previous_status != "solved" else None
        if previous_status == "solved":
            solved_at = conn.execute(
                "SELECT solved_at FROM progress WHERE challenge_id = ?",
                (challenge_id,),
            ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO progress (
                challenge_id, status, attempts, solved_at, last_attempt_at, last_code, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(challenge_id) DO UPDATE SET
                status = excluded.status,
                attempts = excluded.attempts,
                solved_at = excluded.solved_at,
                last_attempt_at = excluded.last_attempt_at,
                last_code = excluded.last_code,
                updated_at = excluded.updated_at
            """,
            (challenge_id, next_status, attempts, solved_at, now, code, now),
        )
        conn.execute(
            """
            INSERT INTO submissions (
                challenge_id, action, status, passed, failed, skipped, duration_ms,
                message, output, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                challenge_id,
                action,
                str(result.get("status") or "error"),
                passed,
                failed,
                skipped,
                duration_ms,
                message,
                output,
                now,
            ),
        )
        conn.commit()


def parse_challenge_metadata(challenge_py: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(challenge_py.read_text(encoding="utf-8"), filename=str(challenge_py))
    except SyntaxError:
        return {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Challenge":
            continue
        result: dict[str, Any] = {}
        for child in node.body:
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(child, ast.Assign):
                targets = list(child.targets)
                value = child.value
            elif isinstance(child, ast.AnnAssign):
                targets = [child.target]
                value = child.value
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in META_KEYS:
                    try:
                        result[target.id] = ast.literal_eval(value)
                    except Exception:
                        pass
        return result
    return {}


def extract_first_paragraph(fragment: str) -> str:
    parser = FirstParagraphParser()
    try:
        parser.feed(fragment)
    except Exception:
        return ""
    return html.unescape(parser.text())


def title_from_slug(slug: str) -> str:
    without_number = CHALLENGE_ID_RE.sub("", slug)
    return without_number.replace("_", " ").title()


def challenge_number(slug: str) -> int:
    match = CHALLENGE_ID_RE.match(slug)
    return int(match.group("number")) if match else 999999


def scan_challenges() -> list[ChallengeInfo]:
    challenges: list[ChallengeInfo] = []
    for difficulty in ("easy", "medium", "hard"):
        difficulty_dir = CHALLENGES_ROOT / difficulty
        if not difficulty_dir.is_dir():
            continue
        for path in difficulty_dir.iterdir():
            if not path.is_dir():
                continue
            challenge_py = path / "challenge.py"
            challenge_html = path / "challenge.html"
            if not challenge_py.is_file() or not challenge_html.is_file():
                continue
            metadata = parse_challenge_metadata(challenge_py)
            html_fragment = challenge_html.read_text(encoding="utf-8")
            slug = path.name
            challenges.append(
                ChallengeInfo(
                    id=f"{difficulty}/{slug}",
                    number=challenge_number(slug),
                    slug=slug,
                    difficulty=difficulty,
                    title=str(metadata.get("name") or title_from_slug(slug)),
                    description=extract_first_paragraph(html_fragment),
                    path=path,
                    access_tier=str(metadata.get("access_tier") or "free"),
                    atol=metadata.get("atol"),
                    rtol=metadata.get("rtol"),
                    num_gpus=metadata.get("num_gpus"),
                )
            )
    return sorted(challenges, key=lambda item: (DIFFICULTY_ORDER[item.difficulty], item.number))


def challenge_to_json(challenge: ChallengeInfo, progress: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": challenge.id,
        "number": challenge.number,
        "slug": challenge.slug,
        "difficulty": challenge.difficulty,
        "title": challenge.title,
        "description": challenge.description,
        "accessTier": challenge.access_tier,
        "atol": challenge.atol,
        "rtol": challenge.rtol,
        "numGpus": challenge.num_gpus,
        "status": (progress or {}).get("status", "untried"),
        "attempts": (progress or {}).get("attempts", 0),
        "solvedAt": (progress or {}).get("solved_at"),
        "lastAttemptAt": (progress or {}).get("last_attempt_at"),
    }


def find_challenge(challenge_id: str) -> ChallengeInfo | None:
    for challenge in scan_challenges():
        if challenge.id == challenge_id:
            return challenge
    return None


def get_starter_code(challenge: ChallengeInfo) -> str:
    starter = challenge.path / "starter" / "starter.pytorch.py"
    if starter.is_file():
        return starter.read_text(encoding="utf-8")
    return "import torch\n\n\ndef solve(*args):\n    pass\n"


def get_challenge_detail(challenge_id: str) -> dict[str, Any] | None:
    challenge = find_challenge(challenge_id)
    if challenge is None:
        return None
    progress = get_progress_map().get(challenge.id)
    saved_code = get_saved_code(challenge.id)
    return {
        **challenge_to_json(challenge, progress),
        "html": (challenge.path / "challenge.html").read_text(encoding="utf-8"),
        "starter": get_starter_code(challenge),
        "code": saved_code or get_starter_code(challenge),
        "submissions": get_submissions(challenge.id),
    }


def get_submissions(challenge_id: str | None = None) -> list[dict[str, Any]]:
    if challenge_id:
        rows = db_rows(
            """
            SELECT id, challenge_id, action, status, passed, failed, skipped,
                   duration_ms, message, created_at
            FROM submissions
            WHERE challenge_id = ?
            ORDER BY id DESC
            LIMIT 50
            """,
            (challenge_id,),
        )
    else:
        rows = db_rows(
            """
            SELECT id, challenge_id, action, status, passed, failed, skipped,
                   duration_ms, message, created_at
            FROM submissions
            ORDER BY id DESC
            LIMIT 100
            """
        )
    return [dict(row) for row in rows]


def candidate_python_paths() -> list[Path]:
    candidates: list[Path] = []
    env_python = os.environ.get("LEETGPU_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    config_python = CONFIG.get("judge_python")
    if config_python:
        candidates.append(config_path("judge_python", "LEETGPU_PYTHON", config_python))
    candidates.append(Path(sys.executable))
    for path in (
        Path.home() / ".conda/envs/cosmos/bin/python",
        Path.home() / ".conda/envs/hulk_ranker/bin/python",
        Path("/usr/bin/python3"),
        Path("/usr/local/bin/python3"),
    ):
        candidates.append(path)
    which_python = shutil.which("python3")
    if which_python:
        candidates.append(Path(which_python))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen and candidate.exists():
            unique.append(candidate)
            seen.add(key)
    return unique


def probe_python(path: Path) -> dict[str, Any]:
    script = (
        "import json, sys\n"
        "try:\n"
        "    import torch\n"
        "    data = {'ok': True, 'python': sys.executable, 'torchVersion': torch.__version__, "
        "'cuda': bool(torch.cuda.is_available()), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
        "except Exception as exc:\n"
        "    data = {'ok': False, 'python': sys.executable, 'error': str(exc)}\n"
        "print(json.dumps(data))\n"
    )
    try:
        completed = subprocess.run(
            [str(path), "-c", script],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "python": str(path), "error": str(exc)}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "ok": False,
            "python": str(path),
            "error": (completed.stderr or completed.stdout or "probe failed").strip(),
        }


def runtime_info() -> dict[str, Any]:
    probes = []
    for candidate in candidate_python_paths():
        probe = probe_python(candidate)
        probes.append(probe)
        if probe.get("ok"):
            return {
                **probe,
                "candidates": probes,
                "configPath": str(CONFIG_PATH),
                "storageDir": str(DATA_ROOT),
                "dbPath": str(DB_PATH),
                "challengesDir": str(CHALLENGES_REPO),
            }
    fallback = probes[0] if probes else {"python": sys.executable, "ok": False}
    return {
        **fallback,
        "candidates": probes,
        "configPath": str(CONFIG_PATH),
        "storageDir": str(DATA_ROOT),
        "dbPath": str(DB_PATH),
        "challengesDir": str(CHALLENGES_REPO),
    }


def run_judge(challenge: ChallengeInfo, code: str, action: str, device: str) -> dict[str, Any]:
    selected_runtime = runtime_info()
    python_path = Path(str(selected_runtime.get("python") or sys.executable))
    timeout_key = "run_timeout_seconds" if action == "run" else "submit_timeout_seconds"
    timeout_default = 20 if action == "run" else 90
    timeout = config_int(timeout_key, "LEETGPU_TIMEOUT_SECONDS", timeout_default)
    request = {
        "challengeDir": str(challenge.path),
        "challengesRoot": str(CHALLENGES_ROOT),
        "code": code,
        "action": action,
        "device": device,
        "maxTestBytes": config_int(
            "max_test_bytes", "LEETGPU_MAX_TEST_BYTES", 1024 * 1024 * 1024
        ),
    }

    with tempfile.TemporaryDirectory(prefix="leetgpu-judge-") as temp_dir:
        temp_path = Path(temp_dir)
        request_path = temp_path / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(CHALLENGES_ROOT)
            if not existing_pythonpath
            else str(CHALLENGES_ROOT) + os.pathsep + existing_pythonpath
        )
        if "LEETGPU_MEMORY_MB" not in env and "memory_mb" in CONFIG:
            env["LEETGPU_MEMORY_MB"] = str(CONFIG["memory_mb"])
        if "LEETGPU_CPU_SECONDS" not in env and "cpu_seconds" in CONFIG:
            env["LEETGPU_CPU_SECONDS"] = str(CONFIG["cpu_seconds"])
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            [str(python_path), str(JUDGE_WORKER), str(request_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=temp_dir,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            stdout, stderr = process.communicate()
            return {
                "success": False,
                "status": "timeout",
                "message": f"Judge timed out after {timeout}s",
                "summary": {"passed": 0, "failed": 1, "skipped": 0},
                "tests": [],
                "durationMs": timeout * 1000,
                "output": (stdout + "\n" + stderr).strip()[-4000:],
                "runtime": selected_runtime,
            }

    payload = parse_worker_json(stdout)
    if payload is None:
        return {
            "success": False,
            "status": "error",
            "message": "Judge worker did not return valid JSON",
            "summary": {"passed": 0, "failed": 1, "skipped": 0},
            "tests": [],
            "durationMs": 0,
            "output": (stdout + "\n" + stderr).strip()[-4000:],
            "runtime": selected_runtime,
        }
    if stderr.strip():
        payload["workerStderr"] = stderr.strip()[-4000:]
    payload["runtime"] = selected_runtime
    return payload


def parse_worker_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class LeetGPUHandler(BaseHTTPRequestHandler):
    server_version = "LeetGPULocal/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/challenges":
                self.handle_challenges()
                return
            if parsed.path == "/api/challenge":
                challenge_id = parse_qs(parsed.query).get("id", [""])[0]
                self.handle_challenge_detail(unquote(challenge_id))
                return
            if parsed.path == "/api/submissions":
                challenge_id = parse_qs(parsed.query).get("id", [None])[0]
                self.send_json({"submissions": get_submissions(challenge_id)})
                return
            if parsed.path == "/api/runtime":
                self.send_json(runtime_info())
                return
            if parsed.path == "/" or parsed.path == "":
                self.serve_file(STATIC_ROOT / "index.html")
                return
            if parsed.path.startswith("/static/"):
                relative = parsed.path.removeprefix("/static/")
                self.serve_static(relative)
                return
            self.not_found("Not found")
        except Exception as exc:
            self.server_error(exc)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            body = self.read_json()
            if parsed.path == "/api/save":
                challenge_id = str(body.get("challengeId") or "")
                code = str(body.get("code") or "")
                if not find_challenge(challenge_id):
                    self.not_found("Unknown challenge")
                    return
                save_code(challenge_id, code)
                self.send_json({"ok": True})
                return
            if parsed.path == "/api/judge":
                self.handle_judge(body)
                return
            if parsed.path == "/api/reset":
                self.handle_reset(body)
                return
            self.not_found("Not found")
        except Exception as exc:
            self.server_error(exc)

    def handle_challenges(self) -> None:
        progress = get_progress_map()
        items = [challenge_to_json(item, progress.get(item.id)) for item in scan_challenges()]
        counts: dict[str, dict[str, int]] = {}
        for item in items:
            difficulty = item["difficulty"]
            counts.setdefault(difficulty, {"total": 0, "solved": 0, "tried": 0})
            counts[difficulty]["total"] += 1
            if item["status"] == "solved":
                counts[difficulty]["solved"] += 1
            elif item["status"] == "tried":
                counts[difficulty]["tried"] += 1
        self.send_json({"challenges": items, "counts": counts})

    def handle_challenge_detail(self, challenge_id: str) -> None:
        detail = get_challenge_detail(challenge_id)
        if detail is None:
            self.not_found("Unknown challenge")
            return
        self.send_json(detail)

    def handle_judge(self, body: dict[str, Any]) -> None:
        challenge_id = str(body.get("challengeId") or "")
        code = str(body.get("code") or "")
        action = str(body.get("action") or "run")
        device = str(body.get("device") or "auto")
        if action not in {"run", "submit"}:
            self.bad_request("Invalid action")
            return
        if device not in {"auto", "cuda", "cpu"}:
            self.bad_request("Invalid device")
            return
        challenge = find_challenge(challenge_id)
        if challenge is None:
            self.not_found("Unknown challenge")
            return
        save_code(challenge_id, code)
        result = run_judge(challenge, code, action, device)
        record_submission(challenge_id, action, code, result)
        self.send_json(result)

    def handle_reset(self, body: dict[str, Any]) -> None:
        challenge_id = str(body.get("challengeId") or "")
        if not challenge_id:
            self.bad_request("Missing challengeId")
            return
        db_execute("DELETE FROM progress WHERE challenge_id = ?", (challenge_id,))
        db_execute("DELETE FROM submissions WHERE challenge_id = ?", (challenge_id,))
        self.send_json({"ok": True})

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON body")
        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object")
        return parsed

    def serve_static(self, relative: str) -> None:
        candidate = (STATIC_ROOT / relative).resolve()
        try:
            candidate.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.not_found("Not found")
            return
        self.serve_file(candidate)

    def serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.not_found("Not found")
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif path.suffix in {".html", ".css"}:
            content_type = content_type + "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def bad_request(self, message: str) -> None:
        self.send_json({"error": message}, status=400)

    def not_found(self, message: str) -> None:
        self.send_json({"error": message}, status=404)

    def server_error(self, exc: Exception) -> None:
        self.send_json({"error": str(exc)}, status=500)


def choose_port(host: str, requested_port: int) -> int:
    for port in range(requested_port, requested_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found from {requested_port} to {requested_port + 49}")


def main() -> int:
    init_db()
    if not CHALLENGES_ROOT.is_dir():
        print(f"Challenge directory not found: {CHALLENGES_ROOT}", file=sys.stderr)
        return 1
    host = str(config_value("host", "LEETGPU_HOST", "127.0.0.1"))
    requested_port = config_int("port", "LEETGPU_PORT", 8000)
    port = choose_port(host, requested_port)
    server = ThreadingHTTPServer((host, port), LeetGPUHandler)
    runtime = runtime_info()
    print(f"LeetGPU local server: http://{host}:{port}")
    print(f"Config file: {CONFIG_PATH}")
    print(f"History database: {DB_PATH}")
    if runtime.get("ok"):
        device = runtime.get("device", "CPU")
        print(
            "Judge runtime: "
            f"{runtime.get('python')} | torch {runtime.get('torchVersion')} | {device}"
        )
    else:
        print(
            "Judge runtime: PyTorch not found. Set LEETGPU_PYTHON to a Python with torch.",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
