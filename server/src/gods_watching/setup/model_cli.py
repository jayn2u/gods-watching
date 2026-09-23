"""Command boundary for pinned model asset preparation and validation."""

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from typer.models import OptionInfo

from gods_watching.model_selection.registry import ClipModelRegistry

from .model_preparation import PreparationPaths, prepare_model_assets
from .models import (
    AssetValidationError,
    LockRegistryMismatchError,
    load_models_lock,
    validate_model_assets,
    validate_registry_agreement,
)

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Manage pinned model assets."""


@app.command()
def validate(
    lock_path: Annotated[Path, OptionInfo(default=..., param_decls=("--lock",))],
    assets: Annotated[Path, OptionInfo(default=..., param_decls=("--assets",))],
) -> None:
    """Validate an existing model cache without network access."""
    try:
        lock = load_models_lock(lock_path)
        validate_registry_agreement(lock, ClipModelRegistry())
        validated = validate_model_assets(lock, assets)
    except AssetValidationError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        typer.echo('{"code":"invalid_lock"}')
        raise typer.Exit(code=2) from error
    except LockRegistryMismatchError as error:
        typer.echo(
            json.dumps(
                {"code": "lock_registry_mismatch", "mismatches": error.mismatches},
                separators=(",", ":"),
            )
        )
        raise typer.Exit(code=2) from error
    except OSError as error:
        typer.echo('{"code":"lock_unreadable"}')
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"code": "assets_valid", "files": len(validated)}, separators=(",", ":")))


@app.command()
def prepare(
    repository_root: Annotated[Path, OptionInfo(default=..., param_decls=("--repository-root",))],
    lock_path: Annotated[Path, OptionInfo(default=..., param_decls=("--lock",))],
    assets: Annotated[Path, OptionInfo(default=..., param_decls=("--assets",))],
) -> None:
    """Build the pinned image, download assets, and emit GPU proof."""
    manifest = prepare_model_assets(
        PreparationPaths(
            repository_root=repository_root.resolve(),
            lock_path=lock_path.resolve(),
            assets_root=assets.resolve(),
        )
    )
    typer.echo(manifest.model_dump_json())


if __name__ == "__main__":
    app()
