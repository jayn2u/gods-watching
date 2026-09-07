from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from typer.testing import CliRunner

from gods_watching.auth.credentials_command import (
    CredentialCommandSettings,
    credentials_app,
)
from gods_watching.auth.types import (
    PasswordReplacement,
    SessionRevocation,
    SessionRevocationReason,
)
from gods_watching.contracts.identifiers import LoginSessionId

_RUNNER = CliRunner()


def test_credentials_help_exposes_set_and_password_file_option() -> None:
    # Given
    # When
    result = _RUNNER.invoke(credentials_app, ["--help"])
    set_help = _RUNNER.invoke(credentials_app, ["set", "--help"])

    # Then
    assert result.exit_code == 0
    assert "set" in result.stdout
    assert set_help.exit_code == 0
    assert "--password-file" in set_help.stdout


def test_oversized_password_file_is_rejected_before_database_configuration(
    tmp_path: Path,
) -> None:
    # Given
    password_file = tmp_path / "oversized.secret"
    _ = password_file.write_bytes(b"p" * 515)
    _ = password_file.chmod(0o600)

    # When
    result = _RUNNER.invoke(
        credentials_app,
        ["set", "--password-file", str(password_file)],
    )

    # Then
    assert result.exit_code == 2
    assert "password file rejected" in result.output
    assert "database configuration" not in result.output


def test_settings_require_explicit_database_url_without_a_test_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.delenv("GW_DATABASE_URL", raising=False)

    # When / Then
    with pytest.raises(ValidationError):
        _ = CredentialCommandSettings()


@pytest.mark.parametrize(
    ("changed", "revoked_sessions"),
    [(True, 2), (False, 0)],
)
def test_set_reports_only_non_secret_replacement_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed: bool,
    revoked_sessions: int,
) -> None:
    # Given
    password_file = tmp_path / "operator.secret"
    _ = password_file.write_text("correct horse battery staple", encoding="utf-8")
    _ = password_file.chmod(0o600)
    database_url = "postgresql+asyncpg://operator:secret@db/gw"
    monkeypatch.setenv("GW_DATABASE_URL", database_url)
    event = SessionRevocation(
        session_id=LoginSessionId(UUID(int=1)),
        reason=SessionRevocationReason.CREDENTIAL_REPLACED,
        revoked_at=datetime(2026, 9, 7, tzinfo=UTC),
    )

    async def replace_password_file(
        _path: Path,
        _settings: CredentialCommandSettings,
    ) -> PasswordReplacement:
        revoked = (event, event) if revoked_sessions == 2 else ()
        return PasswordReplacement(changed=changed, revoked=revoked)

    monkeypatch.setattr(
        "gods_watching.auth.credentials_command._replace_password_file",
        replace_password_file,
    )

    # When
    result = _RUNNER.invoke(
        credentials_app,
        ["set", "--password-file", str(password_file)],
    )

    # Then
    assert result.exit_code == 0
    assert result.output.strip() == (
        f'{{"changed":{str(changed).lower()},"revoked_sessions":{revoked_sessions}}}'
    )
    assert database_url not in result.output
    assert "correct horse battery staple" not in result.output


def test_database_failure_is_generic_and_does_not_echo_error_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    password_file = tmp_path / "operator.secret"
    _ = password_file.write_text("correct horse battery staple", encoding="utf-8")
    _ = password_file.chmod(0o600)
    monkeypatch.setenv("GW_DATABASE_URL", "postgresql+asyncpg://operator:secret@db/gw")

    async def fail_replacement(
        _path: Path,
        _settings: CredentialCommandSettings,
    ) -> CredentialCommandSettings:
        failure_detail = "password=correct horse battery staple"
        raise SQLAlchemyError(failure_detail)

    monkeypatch.setattr(
        "gods_watching.auth.credentials_command._replace_password_file",
        fail_replacement,
    )

    # When
    result = _RUNNER.invoke(
        credentials_app,
        ["set", "--password-file", str(password_file)],
    )

    # Then
    assert result.exit_code == 1
    assert "credential update failed" in result.output
    assert "correct horse battery staple" not in result.output
    assert "postgresql+asyncpg://operator:secret@db/gw" not in result.output


def test_unexpected_database_failure_is_generic_and_does_not_render_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    password_file = tmp_path / "operator.secret"
    _ = password_file.write_text("correct horse battery staple", encoding="utf-8")
    _ = password_file.chmod(0o600)
    monkeypatch.setenv("GW_DATABASE_URL", "postgresql+asyncpg://operator:secret@db/gw")

    async def fail_replacement(
        _path: Path,
        _settings: CredentialCommandSettings,
    ) -> CredentialCommandSettings:
        failure_detail = "password=correct horse battery staple"
        raise RuntimeError(failure_detail)

    monkeypatch.setattr(
        "gods_watching.auth.credentials_command._replace_password_file",
        fail_replacement,
    )

    # When
    result = _RUNNER.invoke(
        credentials_app,
        ["set", "--password-file", str(password_file)],
    )

    # Then
    assert result.exit_code == 1
    assert result.output.strip() == "credential update failed"
    assert "correct horse battery staple" not in result.output
    assert "Traceback" not in result.output
