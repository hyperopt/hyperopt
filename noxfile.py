"""Nox sessions derived from .github/workflows/test.yml (single source of truth)."""

import shlex
from pathlib import Path

import nox
import yaml

PROJECT_DIR = Path(__file__).parent
TEST_WORKFLOW = PROJECT_DIR / ".github" / "workflows" / "test.yml"

_workflow = yaml.safe_load(TEST_WORKFLOW.read_text())
_job = _workflow["jobs"]["test"]
_python_versions = [str(v) for v in _job["strategy"]["matrix"]["python-version"]]

nox.options.default_venv_backend = "uv"


def _parse_env_cmd(line: str) -> tuple[list[str], dict[str, str]]:
    """Split 'KEY=val cmd args' into (cmd_parts, {key: val})."""
    parts = shlex.split(line.strip())
    env: dict[str, str] = {}
    while parts and "=" in parts[0] and parts[0][0].isalpha():
        k, v = parts.pop(0).split("=", 1)
        env[k] = v
    return parts, env


@nox.session(venv_backend="none", default=False)
def tests_parallel(session: nox.Session) -> None:
    """Run all Python version test sessions concurrently (use instead of `nox` for speed)."""
    # adopt this once https://github.com/wntrblm/nox/issues/198 is implemented

    import asyncio

    async def _run(version: str) -> tuple[str, int, str]:
        print(f"→ starting tests-{version}")
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "nox",
            "-s",
            f"tests-{version}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return version, proc.returncode, stdout.decode()

    async def _run_all() -> list[str]:
        failed = []
        for coro in asyncio.as_completed([_run(v) for v in _python_versions]):
            version, returncode, output = await coro
            label = "PASSED" if returncode == 0 else "FAILED"
            print(f"\n--- tests-{version} {label} ---\n{output}")
            if returncode != 0:
                failed.append(version)
        return failed

    failed = asyncio.run(_run_all())
    if failed:
        session.error(f"Failed sessions: {', '.join(f'tests-{v}' for v in failed)}")


@nox.session(python=_python_versions)
def tests(session: nox.Session) -> None:
    """Run the CI test steps for the given Python version."""
    # Sync the lockfile into this session's isolated uv-managed venv
    session.run(
        "uv",
        "sync",
        "--group",
        "dev",
        env={"VIRTUAL_ENV": str(session.virtualenv.location)},
        external=True,
    )
    for step in _job["steps"]:
        run = step.get("run", "").strip()
        if not run or "uv sync" in run:
            continue
        for line in run.splitlines():
            line = line.strip()
            if not line:
                continue
            cmd, env = _parse_env_cmd(line)
            # Strip 'uv run' – nox's uv backend puts the venv on PATH
            if cmd[:2] == ["uv", "run"]:
                cmd = cmd[2:]
            session.run(*cmd, env=env)
