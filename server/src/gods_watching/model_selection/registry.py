"""Immutable registry of locally prepared CLIP model packages."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

B16_REVISION: Final = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
B32_REVISION: Final = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
L14_REVISION: Final = "32bd64288804d66eefd0ccbe215aa642df71cc41"

DEFAULT_CLIP_MODEL_ID: Final = "openai/clip-vit-base-patch16"


@dataclass(frozen=True, slots=True)
class ClipModelPackage:
    """Describe one prepared model snapshot and the runtime that serves it.

    The package contains no mutable preparation state.  A later model family
    can register the same metadata shape with a different processor or
    runtime adapter without changing inference callers.
    """

    model_id: str
    revision: str
    snapshot_path: Path
    dimension: int
    processor: str
    runtime: str
    display_name: str | None = None

    def __post_init__(self) -> None:
        """Normalize paths and reject metadata that cannot be served safely."""
        if not self.model_id.strip() or not self.revision.strip():
            error_code = "clip_model_identity_required"
            raise _metadata_value_error(error_code)
        if not self.processor.strip() or not self.runtime.strip():
            error_code = "clip_model_runtime_metadata_required"
            raise _metadata_value_error(error_code)
        if self.dimension < 1:
            error_code = "clip_model_dimension_positive"
            raise _metadata_value_error(error_code)
        path = Path(self.snapshot_path)
        if not path.is_absolute():
            error_code = "clip_model_snapshot_path_absolute"
            raise _metadata_value_error(error_code)
        object.__setattr__(self, "snapshot_path", path)
        if self.display_name is None:
            object.__setattr__(self, "display_name", self.model_id)
        elif not self.display_name.strip():
            error_code = "clip_model_display_name_required"
            raise _metadata_value_error(error_code)

    @property
    def key(self) -> str:
        """Return the canonical registry key."""
        return self.model_id

    @property
    def path(self) -> Path:
        """Return the immutable local snapshot path."""
        return self.snapshot_path

    @property
    def embedding_dimension(self) -> int:
        """Return the vector dimension expected from image and text models."""
        return self.dimension

    def runtime_parameters(self) -> dict[str, dict[str, str]]:
        """Return the Triton string parameters for this exact package."""
        return {
            "model_id": {"string_value": self.model_id},
            "snapshot_path": {"string_value": str(self.snapshot_path)},
            "model_revision": {"string_value": self.revision},
            "embedding_dimension": {"string_value": str(self.dimension)},
            "processor": {"string_value": self.processor},
            "runtime": {"string_value": self.runtime},
        }


class UnknownClipModelError(LookupError):
    """Report a model identifier absent from the immutable registry."""

    model_id: str

    def __init__(self, model_id: str) -> None:
        """Capture the untrusted identifier without exposing filesystem data."""
        self.model_id = model_id
        super().__init__(f"unknown_clip_model: {model_id}")


_DEFAULT_PACKAGES: Final = (
    ClipModelPackage(
        model_id=DEFAULT_CLIP_MODEL_ID,
        revision=B16_REVISION,
        snapshot_path=Path("/models/clip"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
        display_name="OpenAI CLIP ViT-B/16",
    ),
    ClipModelPackage(
        model_id="openai/clip-vit-base-patch32",
        revision=B32_REVISION,
        snapshot_path=Path("/models/clip-vit-base-patch32"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
        display_name="OpenAI CLIP ViT-B/32",
    ),
    ClipModelPackage(
        model_id="openai/clip-vit-large-patch14",
        revision=L14_REVISION,
        snapshot_path=Path("/models/clip-vit-large-patch14"),
        dimension=768,
        processor="CLIPProcessor",
        runtime="transformers",
        display_name="OpenAI CLIP ViT-L/14",
    ),
)

BUILTIN_CLIP_MODELS: Final[tuple[ClipModelPackage, ...]] = _DEFAULT_PACKAGES
DEFAULT_CLIP_MODEL: Final[ClipModelPackage] = _DEFAULT_PACKAGES[0]


@final
class ClipModelRegistry:
    """Expose a deterministic, immutable-by-use model package catalog."""

    def __init__(
        self,
        packages: Iterable[ClipModelPackage] = _DEFAULT_PACKAGES,
        *,
        default_model_id: str = DEFAULT_CLIP_MODEL_ID,
    ) -> None:
        """Create a registry from complete package values."""
        package_tuple = tuple(packages)
        by_id: dict[str, ClipModelPackage] = {}
        for package in package_tuple:
            if package.model_id in by_id:
                message = f"duplicate_clip_model: {package.model_id}"
                raise ValueError(message)
            by_id[package.model_id] = package
        if default_model_id not in by_id:
            raise UnknownClipModelError(default_model_id)
        self._packages: tuple[ClipModelPackage, ...] = package_tuple
        self._by_id: dict[str, ClipModelPackage] = by_id
        self._default_model_id: str = default_model_id

    @property
    def default(self) -> ClipModelPackage:
        """Return the default B/16 package."""
        return self._by_id[self._default_model_id]

    @property
    def packages(self) -> tuple[ClipModelPackage, ...]:
        """Return packages in stable display order."""
        return self._packages

    def get(self, model_id: str) -> ClipModelPackage | None:
        """Return a package by exact canonical model identifier."""
        return self._by_id.get(model_id)

    def require(self, model_id: str) -> ClipModelPackage:
        """Return a package or fail closed for an unsupported identifier."""
        package = self.get(model_id)
        if package is None:
            raise UnknownClipModelError(model_id)
        return package

    def __len__(self) -> int:
        """Return the number of registered packages."""
        return len(self._packages)


_REGISTRY: Final = ClipModelRegistry()


def get_clip_model(model_id: str) -> ClipModelPackage:
    """Resolve one built-in package for consumers that need a singleton catalog."""
    return _REGISTRY.require(model_id)


def load_clip_registry(assets_root: Path) -> ClipModelRegistry:
    """Discover installed, content-addressed packages from the shared asset mount."""
    from .imported_manifest import ClipPackageImportError  # noqa: PLC0415

    imported_root = Path(assets_root) / "imported"
    packages = list(BUILTIN_CLIP_MODELS)
    if imported_root.exists():
        for directory in sorted(imported_root.iterdir()):
            if directory.name.startswith("."):
                continue
            if not directory.is_dir() or directory.is_symlink():
                code = "invalid_installed_package"
                raise ClipPackageImportError(code)
            manifest = _load_installed_manifest(directory)
            packages.append(
                ClipModelPackage(
                    model_id=str(manifest["model_id"]),
                    revision=str(manifest["revision"]),
                    snapshot_path=Path("/models/imported") / str(manifest["package_sha256"]),
                    dimension=int(manifest["dimension"]),
                    processor="CLIPProcessor",
                    runtime="transformers",
                    display_name=str(manifest["display_name"]),
                )
            )
    return ClipModelRegistry(packages)


def _load_installed_manifest(directory: Path) -> dict[str, object]:  # noqa: C901, PLR0912
    """Validate an installed manifest and every byte it binds."""
    from .imported_manifest import ClipPackageImportError  # noqa: PLC0415

    def invalid() -> None:
        code = "invalid_installed_package"
        raise ClipPackageImportError(code)

    try:
        if (directory / "manifest.json").is_symlink():
            invalid()
        value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {
            "model_id",
            "revision",
            "display_name",
            "base_model_id",
            "dimension",
            "files",
            "package_sha256",
            "cuhk_report",
        }:
            invalid()
        digest = value["package_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != len(hashlib.sha256().hexdigest())
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            invalid()
        if directory.name != digest or value["revision"] != digest:
            invalid()
        if (
            value["base_model_id"] != DEFAULT_CLIP_MODEL_ID
            or value["dimension"] != DEFAULT_CLIP_MODEL.dimension
        ):
            invalid()
        if not all(
            isinstance(value[k], str) and value[k]
            for k in ("model_id", "display_name", "cuhk_report")
        ):
            invalid()
        files = value["files"]
        if not isinstance(files, list) or not files:
            invalid()
        names = set()
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                invalid()
            name = item["path"]
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or "\\" in name
                or name in names
            ):
                invalid()
            names.add(name)
            target = directory / name
            if (
                target.is_symlink()
                or not target.is_file()
                or any(
                    parent.is_symlink()
                    for parent in target.parents
                    if parent != directory and directory in parent.parents
                )
            ):
                invalid()
            checksum = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            if target.stat().st_size != item["size"] or checksum.hexdigest() != item["sha256"]:
                invalid()
        if {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()} != names | {
            "manifest.json"
        }:
            invalid()
        canonical = json.dumps(
            sorted(files, key=lambda f: f["path"]), sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != digest:
            invalid()
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        code = "invalid_installed_package"
        raise ClipPackageImportError(code) from error
    return value


def _metadata_value_error(code: str) -> ValueError:
    """Build a stable metadata validation error."""
    return ValueError(code)


__all__ = [
    "B16_REVISION",
    "B32_REVISION",
    "BUILTIN_CLIP_MODELS",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_CLIP_MODEL_ID",
    "L14_REVISION",
    "ClipModelPackage",
    "ClipModelRegistry",
    "UnknownClipModelError",
    "get_clip_model",
    "load_clip_registry",
]
