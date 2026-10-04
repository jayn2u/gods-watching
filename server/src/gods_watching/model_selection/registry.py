"""Immutable registry of locally prepared CLIP model packages."""

import hashlib
import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, final

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
        assets_root: Path | None = None,
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
        self._refresh_lock: threading.RLock = threading.RLock()
        self._assets_root: Path | None = None
        self._metadata_signature: tuple[object, ...] | None = None
        self._rejected_signature: tuple[object, ...] | None = None
        self._package_signatures: dict[str, tuple[object, ...]] = {}
        if assets_root is not None:
            self._remember_assets_root(assets_root)

    @property
    def default(self) -> ClipModelPackage:
        """Return the default B/16 package."""
        with self._refresh_lock:
            return self._by_id[self._default_model_id]

    @property
    def packages(self) -> tuple[ClipModelPackage, ...]:
        """Return packages in stable display order."""
        with self._refresh_lock:
            return self._packages

    def get(self, model_id: str) -> ClipModelPackage | None:
        """Return a package by exact canonical model identifier."""
        with self._refresh_lock:
            return self._by_id.get(model_id)

    def require(self, model_id: str) -> ClipModelPackage:
        """Return a package or fail closed for an unsupported identifier."""
        package = self.get(model_id)
        if package is None:
            raise UnknownClipModelError(model_id)
        return package

    def __len__(self) -> int:
        """Return the number of registered packages."""
        with self._refresh_lock:
            return len(self._packages)

    def refresh_from(self, assets_root: Path) -> bool:  # noqa: C901
        """Add newly validated imports without replacing the last valid snapshot.

        File metadata is inspected on every call so package changes are noticed
        without hashing large weight files. Only new or metadata-changed
        package trees are strictly revalidated. A malformed tree rejects this
        refresh while preserving the registry already shared by API, worker,
        and preparation catalog.
        """
        from .imported_manifest import ClipPackageImportError  # noqa: PLC0415

        def reject(code: str) -> NoReturn:
            raise ClipPackageImportError(code)

        root = Path(assets_root)
        imported_root = root / "imported"
        with self._refresh_lock:
            try:
                signature = _registry_metadata_signature(imported_root)
            except OSError:
                return False
            if self._assets_root == root and self._metadata_signature == signature:
                return False
            if self._assets_root == root and self._rejected_signature == signature:
                return False

            packages = list(self._packages)
            by_id = {package.model_id: package for package in packages}
            by_revision = {
                package.revision: package
                for package in packages
                if "imported" in package.snapshot_path.parts
            }
            package_signatures = dict(self._package_signatures)
            try:
                if imported_root.is_symlink():
                    reject("invalid_installed_package")
                if imported_root.exists():
                    for directory in sorted(imported_root.iterdir()):
                        if directory.name.startswith("."):
                            continue
                        if directory.is_symlink() or not directory.is_dir():
                            reject("invalid_installed_package")
                        tree_signature = _tree_metadata_signature(directory)
                        existing = by_revision.get(directory.name)
                        if (
                            existing is not None
                            and package_signatures.get(directory.name) == tree_signature
                        ):
                            continue

                        manifest = _load_installed_manifest(directory)
                        package = _package_from_manifest(manifest)
                        prior = by_id.get(package.model_id)
                        if prior is not None and (
                            prior.revision != package.revision
                            or prior.snapshot_path != package.snapshot_path
                        ):
                            reject("duplicate_clip_model")
                        if prior is None:
                            packages.append(package)
                            by_id[package.model_id] = package
                            by_revision[package.revision] = package
                        package_signatures[directory.name] = tree_signature
            except (ClipPackageImportError, OSError, ValueError, TypeError):
                self._assets_root = root
                self._rejected_signature = signature
                return False

            self._packages = tuple(packages)
            self._by_id = by_id
            self._assets_root = root
            self._metadata_signature = signature
            self._rejected_signature = None
            self._package_signatures = package_signatures
            return True

    def _remember_assets_root(self, assets_root: Path) -> None:
        """Seed metadata signatures after the initial strict package scan."""
        root = Path(assets_root)
        imported_root = root / "imported"
        with self._refresh_lock:
            try:
                signature = _registry_metadata_signature(imported_root)
                package_signatures = {
                    directory.name: _tree_metadata_signature(directory)
                    for directory in imported_root.iterdir()
                    if not directory.name.startswith(".") and directory.is_dir()
                }
            except OSError:
                signature = None
                package_signatures = {}
            self._assets_root = root
            self._metadata_signature = signature
            self._rejected_signature = None
            self._package_signatures = package_signatures


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
            packages.append(_package_from_manifest(manifest))
    return ClipModelRegistry(packages, assets_root=Path(assets_root))


def _package_from_manifest(manifest: dict[str, object]) -> ClipModelPackage:
    """Narrow a strictly validated installed record into a registry package."""
    from .imported_manifest import ClipPackageImportError  # noqa: PLC0415

    model_id = manifest.get("model_id")
    revision = manifest.get("revision")
    display_name = manifest.get("display_name")
    package_sha256 = manifest.get("package_sha256")
    dimension = manifest.get("dimension")
    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(revision, str)
        or not revision
        or not isinstance(display_name, str)
        or not display_name
        or not isinstance(package_sha256, str)
        or not package_sha256
        or type(dimension) is not int
    ):
        code = "invalid_installed_package"
        raise ClipPackageImportError(code)
    return ClipModelPackage(
        model_id=model_id,
        revision=revision,
        snapshot_path=Path("/models/imported") / package_sha256,
        dimension=dimension,
        processor="CLIPProcessor",
        runtime="transformers",
        display_name=display_name,
    )


def _tree_metadata_signature(root: Path) -> tuple[object, ...]:
    """Describe a tree by no-follow stat metadata and its complete path listing."""
    def record(path: Path, relative: str) -> tuple[object, ...]:
        item = path.lstat()
        return (
            relative,
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    rows = [record(root, ".")]
    rows.extend(
        record(path, path.relative_to(root).as_posix())
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
    )
    return tuple(rows)


def _registry_metadata_signature(imported_root: Path) -> tuple[object, ...]:
    """Return a cheap metadata signature even when the import directory is absent."""
    try:
        return _tree_metadata_signature(imported_root)
    except FileNotFoundError:
        return (("missing", imported_root.as_posix()),)


def _load_installed_manifest(directory: Path) -> dict[str, object]:  # noqa: C901, PLR0912
    """Validate an installed manifest and every byte it binds."""
    from .imported_manifest import (  # noqa: PLC0415
        ClipPackageImportError,
        ImportedClipManifest,
        ImportedFile,
        parse_source_manifest,
    )

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
        installed = ImportedClipManifest(
            model_id=value["model_id"],
            revision=value["revision"],
            display_name=value["display_name"],
            base_model_id=value["base_model_id"],
            dimension=value["dimension"],
            files=tuple(ImportedFile(**item) for item in files),
            package_sha256=value["package_sha256"],
            cuhk_report=value["cuhk_report"],
        )
        if parse_source_manifest(directory, installed_manifest=installed) != installed:
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
