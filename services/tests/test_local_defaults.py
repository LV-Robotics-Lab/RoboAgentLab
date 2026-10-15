"""Machine-local defaults of the scripts a newcomer runs (finetuned/, robots/maniskill/): derived
from the checkout (<checkout>/.local) or an environment variable, never a lab machine's path."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from pi_embodied_services.finetuned import download
from pi_embodied_services.utils import config

PKG = Path(config.__file__).resolve().parents[1]
SERVICES = PKG.parent
SCRIPTS = [
    PKG / "finetuned" / "serve.sh",
    PKG / "finetuned" / "train.sh",
    PKG / "finetuned" / "episode.sh",
    PKG / "finetuned" / "setup_llamafactory.sh",
    PKG / "finetuned" / "download.py",
    PKG / "finetuned" / "download_dataset.py",
    PKG / "robots" / "maniskill" / "fetch_real2sim.sh",
    PKG / "robots" / "maniskill" / "scenes.py",
    SERVICES / "setup.sh",
]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_lab_paths(path):
    assert "/root/autodl-tmp" not in path.read_text()
    assert "/root/miniconda" not in path.read_text()


def test_local_dir_is_the_checkout_of_the_services(monkeypatch, tmp_path):
    monkeypatch.delenv("PI_EMBODIED_SERVICES", raising=False)
    assert config.get_local_dir() == SERVICES.parent / ".local"
    monkeypatch.setenv("PI_EMBODIED_SERVICES", str(tmp_path / "co" / "services"))
    assert config.get_local_dir() == (tmp_path / "co" / ".local").resolve()


def test_weights_default_follows_setup_sh(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_EMBODIED_SERVICES", str(tmp_path / "co" / "services"))
    monkeypatch.delenv("PI_EMBODIED_WEIGHTS", raising=False)
    want = (tmp_path / "co" / ".local" / "checkpoints").resolve()
    assert config.get_weights_dir() == want
    assert download.default_dest() == want
    monkeypatch.setenv("PI_EMBODIED_WEIGHTS", str(tmp_path / "w"))
    assert config.get_weights_dir() == tmp_path / "w"
    assert download.default_dest() == tmp_path / "w"


def _checkout(tmp_path: Path) -> Path:
    """A checkout's .local with the base model, the released adapter and a vLLM venv."""
    local = tmp_path / ".local"
    (local / "checkpoints" / "Qwen3.5-2B").mkdir(parents=True)
    (local / "checkpoints" / "Qwen3.5-2B" / "config.json").write_text("{}")
    adapter = local / "checkpoints" / "Show-Harness-VLMs" / "qwen3_5_2b_sim"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_text("x")
    (local / "venvs" / "vllm" / "bin").mkdir(parents=True)
    (local / "venvs" / "vllm" / "bin" / "activate").write_text("")
    (tmp_path / "services").mkdir()
    return local


def _serve(env: dict) -> subprocess.CompletedProcess:
    base = {
        k: v for k, v in os.environ.items() if not k.startswith(("PI_EMBODIED", "VLLM"))
    }
    return subprocess.run(
        ["bash", str(PKG / "finetuned" / "serve.sh")],
        env={**base, "DRY_RUN": "1", "PORT": "1", **env},
        capture_output=True,
        text=True,
    )


def test_serve_sh_defaults_to_the_checkouts_local_dir(tmp_path):
    local = _checkout(tmp_path)
    proc = _serve({"PI_EMBODIED_SERVICES": str(tmp_path / "services")})
    assert proc.returncode == 0, proc.stderr
    assert f"vllm serve {local.resolve()}/checkpoints/Qwen3.5-2B " in proc.stdout
    assert (
        f"qwen3_5_2b_showharness_sim={local.resolve()}/checkpoints/Show-Harness-VLMs/qwen3_5_2b_sim"
        in proc.stdout
    )
    # Explicit overrides still win.
    model = tmp_path / "elsewhere"
    model.mkdir()
    (model / "config.json").write_text("{}")
    proc = _serve(
        {"PI_EMBODIED_SERVICES": str(tmp_path / "services"), "MODEL": str(model)}
    )
    assert proc.returncode == 0, proc.stderr
    assert f"vllm serve {model} " in proc.stdout


def test_serve_sh_names_a_missing_vllm_venv(tmp_path):
    _checkout(tmp_path)
    proc = _serve(
        {
            "PI_EMBODIED_SERVICES": str(tmp_path / "services"),
            "VLLM_VENV": str(tmp_path / "novenv"),
        }
    )
    assert proc.returncode == 1
    assert f"no venv at {tmp_path / 'novenv'}: set VLLM_VENV" in proc.stderr
