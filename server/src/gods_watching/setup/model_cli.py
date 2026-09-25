"""Command boundary for pinned model asset preparation and validation."""

# ruff: noqa: TRY003, EM101, PLR0913

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from typer.models import OptionInfo

from gods_watching.model_selection.importer import ClipPackageImportError, import_clip_package
from gods_watching.model_selection.registry import ClipModelRegistry, load_clip_registry

from .model_preparation import PreparationPaths, prepare_model_assets
from .models import (
    AssetValidationError,
    LockRegistryMismatchError,
    load_models_lock,
    validate_model_assets,
    validate_registry_agreement,
)

app = typer.Typer(add_completion=False)


@app.command("benchmark-embedding")
def benchmark_switch(
    model_id: Annotated[str, OptionInfo(default=..., param_decls=("--model-id",))],
    assets: Annotated[Path, OptionInfo(default=..., param_decls=("--assets",))],
    crops: Annotated[Path, OptionInfo(default=..., param_decls=("--crops",))],
    sample_keys_file: Annotated[Path, OptionInfo(default=..., param_decls=("--sample-keys",))],
    detector: Annotated[Path, OptionInfo(default=..., param_decls=("--detector",))],
) -> None:
    """Record diagnostic image throughput; this does not authorize a switch."""
    from gods_watching.model_selection.benchmark import benchmark_package  # noqa: PLC0415

    package = load_clip_registry(assets).get(model_id)
    if package is None:
        raise typer.BadParameter("unknown model", param_hint="--model-id")
    keys = [line.strip() for line in sample_keys_file.read_text().splitlines() if line.strip()]
    try:
        result = benchmark_package(
            package,
            assets_root=assets,
            crops_root=crops,
            sample_keys=keys,
            detector_path=detector,
        )
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(json.dumps({"code": "benchmark_failed", "message": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(str(result))


@app.callback()
def main() -> None:
    """Manage pinned model assets."""


@app.command("import")
def import_package(
    source: Annotated[Path, OptionInfo(default=..., param_decls=("--source",))],
    assets: Annotated[Path, OptionInfo(default=..., param_decls=("--assets",))],
) -> None:
    """Import one immutable local CLIP package."""
    try:
        manifest = import_clip_package(source, assets)
    except ClipPackageImportError as error:
        typer.echo(json.dumps({"code": error.code}, separators=(",", ":")), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "model_id": manifest.model_id,
                "revision": manifest.revision,
                "package_sha256": manifest.package_sha256,
            },
            separators=(",", ":"),
        )
    )


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


@app.command("rehearse-switch")
def rehearse_switch_command(
    database_dump: Annotated[Path, OptionInfo(default=..., param_decls=("--database-dump",))],
    crop_snapshot: Annotated[Path, OptionInfo(default=..., param_decls=("--crop-snapshot",))],
    assets: Annotated[Path, OptionInfo(default=..., param_decls=("--assets",))],
    target_model_id: Annotated[str, OptionInfo(default=..., param_decls=("--target-model-id",))],
    source_model_id: Annotated[str, OptionInfo(default=..., param_decls=("--source-model-id",))],
    source_revision: Annotated[str, OptionInfo(default=..., param_decls=("--source-revision",))],
    source_dimension: Annotated[int, OptionInfo(default=..., param_decls=("--source-dimension",))],
    gpu_uuid: Annotated[str, OptionInfo(default=..., param_decls=("--gpu-uuid",))],
    app_image: Annotated[str, OptionInfo(default=..., param_decls=("--app-image",))],
    camera_cipher_key_file: Annotated[
        Path, OptionInfo(default=..., param_decls=("--camera-cipher-key-file",))
    ],
    model_lock: Annotated[Path, OptionInfo(default=..., param_decls=("--lock",))],
) -> None:
    """Measure one full switch on disposable offline PostgreSQL and Triton."""
    import anyio  # noqa: PLC0415

    from gods_watching.model_selection.rehearsal import (  # noqa: PLC0415
        rehearse_switch,
    )
    from gods_watching.model_selection.rehearsal_stack import (  # noqa: PLC0415
        RehearsalInputs,
    )

    inputs = RehearsalInputs(
        database_dump=database_dump,
        crop_snapshot=crop_snapshot,
        assets=assets,
        gpu_uuid=gpu_uuid,
        target_model_id=target_model_id,
        model_lock=model_lock,
    )
    try:
        path = anyio.run(
            lambda: rehearse_switch(
                inputs,
                source_model_id=source_model_id,
                source_revision=source_revision,
                source_dimension=source_dimension,
                app_image=app_image,
                camera_cipher_key_file=camera_cipher_key_file,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(json.dumps({"code": "rehearsal_failed", "message": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(str(path))


@app.command("rehearse-inner", hidden=True)
def rehearse_inner_command(
    database_url: Annotated[str, OptionInfo(default=..., param_decls=("--database-url",))],
    triton_url: Annotated[str, OptionInfo(default=..., param_decls=("--triton-url",))],
    crops: Annotated[Path, OptionInfo(default=..., param_decls=("--crops",))],
    assets: Annotated[Path, OptionInfo(default=..., param_decls=("--assets",))],
    model_lock: Annotated[Path, OptionInfo(default=..., param_decls=("--lock",))],
    target_model_id: Annotated[str, OptionInfo(default=..., param_decls=("--target-model-id",))],
    source_model_id: Annotated[str, OptionInfo(default=..., param_decls=("--source-model-id",))],
    source_revision: Annotated[str, OptionInfo(default=..., param_decls=("--source-revision",))],
    source_dimension: Annotated[int, OptionInfo(default=..., param_decls=("--source-dimension",))],
    output: Annotated[Path, OptionInfo(default=..., param_decls=("--output",))],
    camera_cipher_key_file: Annotated[
        Path, OptionInfo(default=..., param_decls=("--camera-cipher-key-file",))
    ],
) -> None:
    """Run only inside the network created by rehearse-switch."""
    import anyio  # noqa: PLC0415

    from gods_watching.model_selection.rehearsal import InnerInputs, run_inner  # noqa: PLC0415

    try:
        _ = anyio.run(
            run_inner,
            InnerInputs(
                database_url,
                triton_url,
                crops,
                assets,
                model_lock,
                target_model_id,
                source_model_id,
                source_revision,
                source_dimension,
                output,
                camera_cipher_key_file,
            ),
        )
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(json.dumps({"code": "inner_rehearsal_failed", "message": str(error)}), err=True)
        raise typer.Exit(code=2) from error


if __name__ == "__main__":
    app()
