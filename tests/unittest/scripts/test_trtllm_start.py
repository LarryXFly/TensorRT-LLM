import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "scripts" / "trtllm-start"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TRTLLM_START_SKIP_ENV_CHECKS"] = "1"
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_local_dry_run_constructs_uv_and_trtllm_serve_command(tmp_path):
    result = run_script(
        "Qwen/Qwen3-8B",
        "--dry-run",
        "--venv-dir",
        str(tmp_path / ".venv"),
        "--port",
        "9000",
    )

    assert result.returncode == 0, result.stderr
    assert "uv venv" in result.stdout
    assert "uv pip install" in result.stdout
    assert "trtllm-serve Qwen/Qwen3-8B --port 9000" in result.stdout


def test_local_dry_run_forwards_user_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("max_batch_size: 1\n", encoding="utf-8")

    result = run_script(
        "Qwen/Qwen3-8B",
        "--dry-run",
        "--config",
        str(config_path),
    )

    assert result.returncode == 0, result.stderr
    assert f"--config {config_path}" in result.stdout


def test_docker_dry_run_constructs_docker_command():
    result = run_script(
        "Qwen/Qwen3-8B",
        "--docker",
        "--dry-run",
        "--port",
        "9001",
    )

    assert result.returncode == 0, result.stderr
    assert "docker run" in result.stdout
    assert "--gpus all" in result.stdout
    assert "-p 9001:9001" in result.stdout
    assert "trtllm-serve Qwen/Qwen3-8B --port 9001" in result.stdout
    assert "--host 0.0.0.0" in result.stdout


def test_missing_model_prints_usage():
    result = run_script("--dry-run")

    assert result.returncode == 2
    assert "Usage:" in result.stderr
