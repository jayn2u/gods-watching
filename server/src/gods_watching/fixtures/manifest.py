"""Load the fixture manifest at its filesystem trust boundary."""

from pathlib import Path

from pydantic import ValidationError

from .errors import FixturePreparationError
from .models import FixtureManifest


def load_fixture_manifest(manifest_path: Path) -> FixtureManifest:
    """Parse an attributed stream manifest or raise one typed preparation error."""
    try:
        payload = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise FixturePreparationError(code="missing_manifest", detail=str(manifest_path)) from error
    try:
        return FixtureManifest.model_validate_json(payload)
    except ValidationError as error:
        raise FixturePreparationError(code="invalid_manifest", detail=str(error)) from error
