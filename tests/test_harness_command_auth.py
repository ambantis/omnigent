"""A configured ``harness.<id>.command`` reaches every invocation, and
``auth: command`` makes readiness trust it."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.harness_startup_config import (
    configured_harness_command,
    harness_command_owns_auth,
    resolve_harness_config,
)


def _wrapper(tmp_path: Path, name: str) -> Path:
    """An executable standing in for a credential-injecting wrapper."""
    path = tmp_path / "wrappers" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def write_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the effective config at a temporary user config file."""
    home = tmp_path / "config-home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("OMNIGENT_CONFIG_HOME", str(home))
    monkeypatch.chdir(project)
    for var in ("OMNIGENT_CLAUDE_PATH", "OMNIGENT_CODEX_PATH", "OMNIGENT_PI_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("OMNIGENT_OPENCODE_PATH", raising=False)
    # No stored Codex login, so only ``auth: command`` can satisfy the gate.
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    def write(text: str) -> None:
        (home / "config.yaml").write_text(text)

    return write


# ── config parsing ───────────────────────────────────────────────────


def test_auth_command_is_parsed() -> None:
    cfg = {"harness": {"codex-native": {"command": "/w/codex", "auth": "command"}}}
    _, overrides = resolve_harness_config(cfg)
    assert overrides["codex-native"] == {"command": "/w/codex", "auth": "command"}
    assert harness_command_owns_auth("codex-native", cfg)
    assert configured_harness_command("codex-native", cfg) == "/w/codex"


def test_auth_without_command_is_not_trusted() -> None:
    cfg = {"harness": {"codex-native": {"auth": "command"}}}
    assert not harness_command_owns_auth("codex-native", cfg)


def test_unknown_auth_value_warns_and_is_ignored(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = {"harness": {"codex-native": {"command": "/w/codex", "auth": "sometimes"}}}
    _, overrides = resolve_harness_config(cfg)
    assert overrides["codex-native"] == {"command": "/w/codex"}
    assert not harness_command_owns_auth("codex-native", cfg)
    assert "harness.codex-native.auth" in capsys.readouterr().err


# ── the command reaches non-terminal invocations ─────────────────────


def test_claude_probes_use_the_configured_command_and_args(write_config) -> None:
    from omnigent.claude_launcher import resolve_claude_cli_launch

    write_config(
        "harness:\n  claude-native:\n    command: /w/claude\n    args: ['--']\n",
    )
    assert resolve_claude_cli_launch(["-p", "hi"]) == ("/w/claude", ["--", "-p", "hi"])


def test_claude_probes_default_to_claude(write_config) -> None:
    from omnigent.claude_launcher import resolve_claude_cli_launch

    write_config("{}\n")
    assert resolve_claude_cli_launch(["-p"]) == ("claude", ["-p"])


def test_native_codex_app_server_uses_the_configured_command(write_config, tmp_path) -> None:
    from omnigent.inner.codex_executor import _find_native_codex_cli

    wrapper = _wrapper(tmp_path, "codex")
    write_config(f"harness:\n  codex-native:\n    command: {wrapper}\n")
    assert _find_native_codex_cli() == str(wrapper)


def test_native_codex_env_override_still_wins(write_config, tmp_path, monkeypatch) -> None:
    from omnigent.inner.codex_executor import _find_native_codex_cli

    env_codex = _wrapper(tmp_path, "env-codex")
    write_config(f"harness:\n  codex-native:\n    command: {_wrapper(tmp_path, 'codex')}\n")
    monkeypatch.setenv("OMNIGENT_CODEX_PATH", str(env_codex))
    assert _find_native_codex_cli() == str(env_codex)


def test_opencode_server_uses_the_configured_command(write_config, tmp_path) -> None:
    from omnigent.harnesses.opencode_native.app_server import find_opencode_cli

    wrapper = _wrapper(tmp_path, "opencode")
    write_config(f"harness:\n  opencode-native:\n    command: {wrapper}\n")
    assert find_opencode_cli() == str(wrapper)


def test_pi_native_runner_uses_the_configured_command(write_config) -> None:
    from omnigent.harnesses.pi_native.main import _configured_pi_command

    write_config("harness:\n  pi-native:\n    command: /w/pi\n")
    assert _configured_pi_command({}) == "/w/pi"


def test_pi_native_env_override_still_wins(write_config) -> None:
    from omnigent.harnesses.pi_native.main import _configured_pi_command

    write_config("harness:\n  pi-native:\n    command: /w/pi\n")
    assert _configured_pi_command({"OMNIGENT_PI_PATH": "/env/pi"}) == "/env/pi"


# ── auth: command ────────────────────────────────────────────────────


@pytest.mark.parametrize("harness", ["codex-native", "opencode-native", "pi-native"])
def test_readiness_trusts_a_self_authenticating_command(write_config, tmp_path, harness) -> None:
    from omnigent.onboarding.harness_readiness import _harness_availability

    wrapper = _wrapper(tmp_path, harness)
    write_config(f"harness:\n  {harness}:\n    command: {wrapper}\n    auth: command\n")
    assert _harness_availability(harness) is True


@pytest.mark.parametrize(
    ("spelling", "entry", "expected"),
    [
        ("codex-native", "codex-native", True),
        ("native-codex", "codex-native", True),
        ("codex", "codex", True),
        # An entry for one id never makes the other ready: each launch reads its own.
        ("codex", "codex-native", False),
        ("codex-native", "codex", False),
    ],
)
def test_codex_readiness_follows_the_entry_its_launch_reads(
    write_config, tmp_path, spelling, entry, expected
) -> None:
    from omnigent.onboarding.harness_readiness import _command_auth_availability

    write_config(
        f"harness:\n  {entry}:\n    command: {_wrapper(tmp_path, 'codex')}\n    auth: command\n"
    )
    assert (_command_auth_availability(spelling) is True) is expected


def test_readiness_map_reports_codex_ready(write_config, tmp_path) -> None:
    from omnigent.onboarding.harness_readiness import configured_harness_map

    write_config(
        f"harness:\n  codex-native:\n    command: {_wrapper(tmp_path, 'codex')}\n"
        "    auth: command\n"
    )
    readiness = configured_harness_map()
    assert readiness["codex-native"] is True
    assert readiness["native-codex"] is True


def test_readiness_reports_a_missing_self_authenticating_command(write_config, tmp_path) -> None:
    from omnigent.onboarding.harness_readiness import (
        HARNESS_BINARY_MISSING,
        _harness_availability,
    )

    write_config(
        f"harness:\n  codex-native:\n    command: {tmp_path / 'absent'}\n    auth: command\n"
    )
    assert _harness_availability("codex-native") == HARNESS_BINARY_MISSING


def test_codex_launch_does_not_require_a_login(write_config, tmp_path) -> None:
    from omnigent.harnesses.codex_native.app_server import _codex_login_usable

    write_config(
        f"harness:\n  codex-native:\n    command: {_wrapper(tmp_path, 'codex')}\n"
        "    auth: command\n"
    )
    assert _codex_login_usable()


def test_codex_launch_still_requires_a_login_without_auth_command(write_config, tmp_path) -> None:
    from omnigent.harnesses.codex_native.app_server import _codex_login_usable

    write_config(f"harness:\n  codex-native:\n    command: {_wrapper(tmp_path, 'codex')}\n")
    assert not _codex_login_usable()


def test_codex_app_server_overrides_precede_the_subcommand() -> None:
    # Codex drops root-level ``-c`` flags when the subcommand gets its own, so
    # a wrapper's overrides only survive if every override is root-level.
    from omnigent.harnesses.codex_native.app_server import _build_native_codex_app_server_argv

    argv = _build_native_codex_app_server_argv(
        tagged_argv0="codex",
        listen_url="ws://127.0.0.1:1",
        config_overrides=['model="m"'],
    )
    assert argv == ["codex", "-c", 'model="m"', "app-server", "--listen", "ws://127.0.0.1:1"]


# ── SDK harnesses read their own entry ───────────────────────────────


def test_codex_sdk_and_native_read_separate_entries(write_config, tmp_path) -> None:
    from omnigent.inner.codex_executor import _find_codex_cli, _find_native_codex_cli

    sdk, native = _wrapper(tmp_path, "sdk-codex"), _wrapper(tmp_path, "native-codex")
    write_config(
        f"harness:\n  codex:\n    command: {sdk}\n  codex-native:\n    command: {native}\n"
    )
    assert _find_codex_cli() == str(sdk)
    assert _find_native_codex_cli() == str(native)


def test_codex_sdk_ignores_the_native_entry(write_config, tmp_path, monkeypatch) -> None:
    from omnigent.inner import codex_executor

    write_config(f"harness:\n  codex-native:\n    command: {_wrapper(tmp_path, 'codex')}\n")
    monkeypatch.setattr(codex_executor, "resolve_cli_binary", lambda name, **_: f"/path/{name}")
    assert codex_executor._find_codex_cli() == "/path/codex"


def test_codex_sdk_env_override_still_wins(write_config, tmp_path, monkeypatch) -> None:
    from omnigent.inner.codex_executor import _find_codex_cli

    env_codex = _wrapper(tmp_path, "env-codex")
    write_config(f"harness:\n  codex:\n    command: {_wrapper(tmp_path, 'codex')}\n")
    monkeypatch.setenv("OMNIGENT_CODEX_PATH", str(env_codex))
    assert _find_codex_cli() == str(env_codex)


def test_claude_sdk_uses_the_configured_command(write_config, tmp_path) -> None:
    from omnigent.inner.claude_sdk_executor import _find_system_claude

    wrapper = _wrapper(tmp_path, "claude")
    write_config(f"harness:\n  claude-sdk:\n    command: {wrapper}\n")
    assert _find_system_claude() == str(wrapper)


def test_pi_sdk_uses_the_configured_command(write_config, tmp_path) -> None:
    from omnigent.inner.pi_executor import _find_pi_cli

    wrapper = _wrapper(tmp_path, "pi")
    write_config(f"harness:\n  pi:\n    command: {wrapper}\n")
    assert _find_pi_cli() == str(wrapper)


# ── Settings account probes ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("account_key", "harness"),
    [
        ("codex", "codex"),
        ("codex", "codex-native"),
        ("claude", "claude-native"),
        ("claude", "claude-sdk"),
        ("pi", "pi-native"),
        ("opencode", "opencode-native"),
    ],
)
def test_account_probe_skips_the_vendor_cli_when_the_command_owns_auth(
    write_config, tmp_path, monkeypatch, account_key, harness
) -> None:
    from omnigent.onboarding import harness_accounts

    write_config(
        f"harness:\n  {harness}:\n    command: {_wrapper(tmp_path, harness)}\n    auth: command\n"
    )

    def forbidden(*_a, **_k):
        raise AssertionError("the vendor CLI must not be probed")

    monkeypatch.setattr(harness_accounts.shutil, "which", forbidden)
    monkeypatch.setattr(harness_accounts.subprocess, "Popen", forbidden)
    monkeypatch.setattr(harness_accounts.subprocess, "run", forbidden)
    account = harness_accounts.probe_harness_account(account_key)
    assert account is not None
    assert account["kind"] == "providers"
    assert account["providers"] == [
        {"id": "command", "label": "Configured command", "auth_type": "wrapper"}
    ]


def test_account_probe_still_probes_without_auth_command(
    write_config, tmp_path, monkeypatch
) -> None:
    from omnigent.onboarding import harness_accounts

    write_config(f"harness:\n  codex-native:\n    command: {_wrapper(tmp_path, 'codex')}\n")
    monkeypatch.setattr(harness_accounts.shutil, "which", lambda _name: None)
    assert harness_accounts.probe_harness_account("codex") is None
