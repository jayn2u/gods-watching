"""Typed CLIP inference and explicit Triton model lifecycle boundaries."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn, Protocol, cast, final, override

import numpy as np
import tritonclient.grpc.aio as grpc_aio
from pydantic import ValidationError
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.utils import InferenceServerException

from gods_watching.contracts.search import TextSearchRequest

if TYPE_CHECKING:
    from types import TracebackType

    from numpy.typing import NDArray

    from gods_watching.model_selection.registry import ClipModelPackage

type ClipEmbedding = tuple[float, ...]
_DEFAULT_EMBEDDING_DIMENSION: Final = 512
_UNIT_NORM_TOLERANCE: Final = 1e-3
_CLIP_MODEL_NAMES: Final[tuple[str, str]] = ("clip_image", "clip_text")
_IMAGE_MODEL_NAME: Final = "clip_image"
_TEXT_MODEL_NAME: Final = "clip_text"
_CLEANUP_TIMEOUT_SECONDS: Final = 5.0


@dataclass(frozen=True, slots=True)
class ClipInputError(ValueError):
    """Reject a request before it reaches inference."""

    code: str

    @override
    def __str__(self) -> str:
        return self.code


class ClipImageDecodeError(ClipInputError):
    """Report malformed encoded image bytes separately from inference failures."""


@dataclass(frozen=True, slots=True)
class ClipInferenceError(RuntimeError):
    """Report a failed or malformed Triton response."""

    code: str

    @override
    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class ClipRuntimeError(RuntimeError):
    """Report a failed explicit Triton model-management operation."""

    code: str

    @override
    def __str__(self) -> str:
        return self.code


class ClipRuntimeIdentityError(ClipRuntimeError):
    """Report a loaded Triton model that differs from the requested package."""


@dataclass(frozen=True, slots=True)
class ClipRuntimeIdentity:
    """Describe the identity Triton reports for its resident CLIP pair."""

    model_id: str
    revision: str
    snapshot_path: Path
    dimension: int
    processor: str
    runtime: str

    @classmethod
    def from_package(cls, package: ClipModelPackage) -> ClipRuntimeIdentity:
        """Convert immutable package metadata into an observed identity value."""
        return cls(
            model_id=package.model_id,
            revision=package.revision,
            snapshot_path=package.snapshot_path,
            dimension=package.dimension,
            processor=package.processor,
            runtime=package.runtime,
        )


@dataclass(frozen=True, slots=True)
class _CleanupOutcome:
    """Capture whether every explicit unload completed successfully."""

    errors: tuple[BaseException, ...] = ()
    timed_out: bool = False

    @property
    def successful(self) -> bool:
        """Return whether both modality names were attempted without failure."""
        return not self.errors and not self.timed_out

    @property
    def cancellation(self) -> asyncio.CancelledError | None:
        """Return the first cancellation raised by a client unload call."""
        for error in self.errors:
            if isinstance(error, asyncio.CancelledError):
                return error
        return None


class ClipTransport(Protocol):
    """Provide modality-specific normalized embedding calls."""

    async def embed_image(self, image: bytes) -> Sequence[float]:
        """Return one normalized image embedding."""
        ...

    async def embed_text(self, text: str) -> Sequence[float]:
        """Return one normalized text embedding."""
        ...


class TritonModelControlClient(Protocol):
    """Subset of the asynchronous Triton client used by the runtime manager."""

    async def load_model(self, model_name: str, *, config: str) -> None:
        """Load one model using an explicit JSON model-config override."""
        ...

    async def unload_model(self, model_name: str) -> None:
        """Unload one explicitly controlled model."""
        ...

    async def get_model_config(
        self, model_name: str, *, as_json: bool
    ) -> Mapping[str, object]:
        """Read the server's effective model configuration."""
        ...


@final
class ClipRuntimeManager:
    """Load and unload one exact CLIP package through Triton's gRPC API.

    The manager deliberately has no Docker or filesystem mutation capability.
    Model packages are expected to have been prepared and validated before a
    caller requests activation. Every successful load is followed by an
    identity readback for both modalities; a mismatch unloads all models that
    this operation loaded and leaves the manager without an active identity.
    """

    def __init__(
        self,
        client: TritonModelControlClient | str,
        *,
        model_control_mode: str = "explicit",
        model_names: tuple[str, str] = _CLIP_MODEL_NAMES,
        cleanup_timeout_seconds: float = _CLEANUP_TIMEOUT_SECONDS,
    ) -> None:
        """Create a manager around a client or a private gRPC URL.

        Triton must be started with explicit model control. Requiring that mode
        here prevents an accidental call from pretending to own lifecycle when
        Triton is configured to load every repository model automatically.
        """
        if model_control_mode != "explicit":
            error_code = "clip_runtime_explicit_mode_required"
            raise _runtime_value_error(error_code)
        if (
            len(model_names) != len(_CLIP_MODEL_NAMES)
            or set(model_names) != set(_CLIP_MODEL_NAMES)
        ):
            error_code = "clip_runtime_modalities_required"
            raise _runtime_value_error(error_code)
        if cleanup_timeout_seconds <= 0:
            error_code = "clip_runtime_cleanup_timeout_positive"
            raise _runtime_value_error(error_code)
        if isinstance(client, str):
            self._client: TritonModelControlClient = cast(
                "TritonModelControlClient",
                cast("object", grpc_aio.InferenceServerClient(client)),
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        self._model_names = model_names
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._pending_cleanup_task: asyncio.Task[_CleanupOutcome] | None = None
        self._active_identity: ClipRuntimeIdentity | None = None

    @property
    def active_identity(self) -> ClipRuntimeIdentity | None:
        """Return the last verified active identity, if one is loaded."""
        return self._active_identity

    async def __aenter__(self) -> ClipRuntimeManager:
        """Retain the manager for a scoped runtime lifecycle."""
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close an internally owned gRPC channel."""
        del exception_type, exception, traceback
        _ = await self._drain_pending_cleanup(swallow_cancellation=True)
        pending = self._pending_cleanup_task
        if pending is not None:
            _ = pending.cancel()
            with suppress(BaseException):
                _ = await asyncio.wait_for(
                    asyncio.shield(pending), timeout=self._cleanup_timeout_seconds
                )
            if pending.done():
                self._pending_cleanup_task = None
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if close is not None:
                await close()

    async def load_model(self, package: ClipModelPackage) -> ClipRuntimeIdentity:  # noqa: C901
        """Load and verify the requested image/text package pair."""
        _ = await self._drain_pending_cleanup(swallow_cancellation=False)
        expected = ClipRuntimeIdentity.from_package(package)
        if self._active_identity is not None:
            try:
                observed = await self.inspect_identity()
            except ClipRuntimeIdentityError:
                self._active_identity = None
            else:
                if observed == expected:
                    self._active_identity = observed
                    return observed
                self._active_identity = None
        # A newly constructed manager cannot trust the resident server state.
        # Clear both names before loading so an old pair cannot survive a
        # restart and be mistaken for the requested package.
        cleanup = await self._cleanup_all_modalities(swallow_cancellation=False)
        if not cleanup.successful:
            detail = "timeout" if cleanup.timed_out else str(cleanup.errors[0])
            raise ClipRuntimeError(code=f"clip_runtime_preload_cleanup_failed: {detail}")

        try:
            for model_name in self._model_names:
                config = _triton_model_config(model_name, package)
                await self._client.load_model(
                    model_name,
                    config=json.dumps(config, sort_keys=True, separators=(",", ":")),
                )
            observed = await self.inspect_identity()
            if observed != expected:
                raise _identity_mismatch_error(expected, observed)  # noqa: TRY301
        except ClipRuntimeIdentityError:
            _ = await self._cleanup_all_modalities(swallow_cancellation=True)
            self._active_identity = None
            raise
        except BaseException as error:
            _ = await self._cleanup_all_modalities(swallow_cancellation=True)
            self._active_identity = None
            if isinstance(error, asyncio.CancelledError):
                raise
            if not isinstance(error, Exception):
                raise
            raise ClipRuntimeError(code=f"clip_runtime_load_failed: {error}") from error
        self._active_identity = observed
        return observed

    async def load(self, package: ClipModelPackage) -> ClipRuntimeIdentity:
        """Alias for callers that treat runtime selection as an activation."""
        return await self.load_model(package)

    async def unload_model(self) -> None:
        """Unload both CLIP modalities and forget any previously verified identity."""
        self._active_identity = None
        _ = await self._drain_pending_cleanup(swallow_cancellation=False)
        cleanup = await self._cleanup_all_modalities(swallow_cancellation=False)
        if cleanup.cancellation is not None:
            raise cleanup.cancellation
        if not cleanup.successful:
            detail = "timeout" if cleanup.timed_out else str(cleanup.errors[0])
            raise ClipRuntimeError(code=f"clip_runtime_unload_failed: {detail}")

    async def unload(self) -> None:
        """Alias for callers that use symmetric load/unload operations."""
        await self.unload_model()

    async def inspect_identity(self) -> ClipRuntimeIdentity:
        """Read and cross-check actual identity metadata from both models."""
        identities: list[ClipRuntimeIdentity] = []
        try:
            for model_name in self._model_names:
                response = await self._client.get_model_config(model_name, as_json=True)
                identities.append(_identity_from_model_config(response))
        except ClipRuntimeIdentityError:
            raise
        except Exception as error:
            raise ClipRuntimeIdentityError(
                code=f"clip_runtime_identity_unavailable: {error}"
            ) from error
        if not identities or any(identity != identities[0] for identity in identities[1:]):
            raise ClipRuntimeIdentityError(
                code=f"clip_runtime_identity_mismatch: modalities={identities!r}"
            )
        return identities[0]

    async def _cleanup_all_modalities(self, *, swallow_cancellation: bool) -> _CleanupOutcome:
        """Attempt both unloads under a bounded shielded cleanup task."""
        cleanup_task = asyncio.create_task(self._unload_all_modalities())
        try:
            return await asyncio.wait_for(
                asyncio.shield(cleanup_task), timeout=self._cleanup_timeout_seconds
            )
        except TimeoutError:
            _ = cleanup_task.cancel()
            self._pending_cleanup_task = cleanup_task
            return _CleanupOutcome(timed_out=True)
        except asyncio.CancelledError:
            try:
                _ = await asyncio.wait_for(
                    asyncio.shield(cleanup_task), timeout=self._cleanup_timeout_seconds
                )
            except TimeoutError:
                _ = cleanup_task.cancel()
                self._pending_cleanup_task = cleanup_task
                cleanup = _CleanupOutcome(timed_out=True)
            except asyncio.CancelledError:
                _ = cleanup_task.cancel()
                self._pending_cleanup_task = cleanup_task
                cleanup = _CleanupOutcome(timed_out=True)
            else:
                cleanup = cleanup_task.result()
            if not swallow_cancellation:
                raise
            return cleanup

    async def _drain_pending_cleanup(
        self, *, swallow_cancellation: bool
    ) -> _CleanupOutcome | None:
        """Wait for an earlier timed-out cleanup before any new RPCs."""
        pending = self._pending_cleanup_task
        if pending is None:
            return None
        try:
            outcome = await asyncio.wait_for(
                asyncio.shield(pending), timeout=self._cleanup_timeout_seconds
            )
        except TimeoutError as error:
            if not swallow_cancellation:
                raise ClipRuntimeError(code="clip_runtime_cleanup_pending") from error
            return _CleanupOutcome(timed_out=True)
        except asyncio.CancelledError:
            if not swallow_cancellation:
                raise
            return _CleanupOutcome(timed_out=True)
        self._pending_cleanup_task = None
        return outcome

    async def _unload_all_modalities(self) -> _CleanupOutcome:
        """Unload every modality, even when a previous RPC had unknown outcome."""
        errors: list[BaseException] = []
        for model_name in self._model_names:
            try:
                await self._client.unload_model(model_name)
            except BaseException as error:  # noqa: BLE001 - collect every unload outcome
                errors.append(error)
        return _CleanupOutcome(errors=tuple(errors))


TritonClipRuntime = ClipRuntimeManager


@final
class TritonClipTransport:
    """Call the two resident CLIP models through Triton gRPC."""

    def __init__(
        self,
        url: str,
        package: ClipModelPackage | None = None,
        *,
        dimension: int | None = None,
    ) -> None:
        """Create a client for a private Triton gRPC endpoint."""
        self._client = grpc_aio.InferenceServerClient(url=url)
        selected_dimension = package.dimension if package is not None else dimension
        if selected_dimension is None:
            selected_dimension = _DEFAULT_EMBEDDING_DIMENSION
        if selected_dimension < 1:
            error_code = "clip_embedding_dimension_positive"
            raise _runtime_value_error(error_code)
        self.dimension = selected_dimension
        self.model_id = package.model_id if package is not None else None
        self.model_revision = package.revision if package is not None else None

    async def __aenter__(self) -> TritonClipTransport:
        """Retain the transport until its context exits."""
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the owned gRPC channel."""
        del exception_type, exception, traceback
        await self._client.close()

    async def embed_image(self, image: bytes) -> ClipEmbedding:
        """Submit one encoded image to the resident image model."""
        return await self._infer(model_name=_IMAGE_MODEL_NAME, input_name="IMAGE", value=image)

    async def embed_text(self, text: str) -> ClipEmbedding:
        """Submit one normalized query to the resident text model."""
        return await self._infer(
            model_name=_TEXT_MODEL_NAME, input_name="TEXT", value=text.encode("utf-8")
        )

    async def _infer(self, *, model_name: str, input_name: str, value: bytes) -> ClipEmbedding:
        infer_input = InferInput(input_name, [1, 1], "BYTES")
        _ = infer_input.set_data_from_numpy(np.asarray([[value]], dtype=np.object_))
        try:
            result = await self._client.infer(
                model_name=model_name,
                inputs=[infer_input],
                outputs=[InferRequestedOutput("EMBEDDING")],
            )
        except InferenceServerException as error:
            _raise_transport_error(error)
        raw: NDArray[np.float32] | None = result.as_numpy("EMBEDDING")
        if raw is None:
            raise ClipInferenceError(code="clip_embedding_missing")
        return tuple(float(value) for value in raw.reshape(-1))


@final
class ClipAdapter:
    """Validate CLIP requests and enforce the selected vector contract."""

    def __init__(
        self,
        transport: ClipTransport,
        dimension: int = _DEFAULT_EMBEDDING_DIMENSION,
        *,
        package: ClipModelPackage | None = None,
    ) -> None:
        """Wrap one typed modality transport with an explicit dimension."""
        selected_dimension = package.dimension if package is not None else dimension
        if selected_dimension < 1:
            error_code = "clip_embedding_dimension_positive"
            raise _runtime_value_error(error_code)
        self._transport = transport
        self.dimension = selected_dimension
        self.model_id = package.model_id if package is not None else None
        self.model_revision = package.revision if package is not None else None

    async def embed_image(self, image: bytes) -> ClipEmbedding:
        """Validate encoded bytes and return a unit image embedding."""
        if not image:
            raise ClipInputError(code="clip_image_empty")
        return self._parse_embedding(await self._transport.embed_image(image))

    async def embed_text(self, text: str) -> ClipEmbedding:
        """Normalize in-scope text and return a unit text embedding."""
        try:
            request = TextSearchRequest(mode="text", query=text)
        except ValidationError as error:
            raise ClipInputError(code="clip_text_invalid") from error
        return self._parse_embedding(await self._transport.embed_text(request.query))

    def _parse_embedding(self, values: Sequence[float]) -> ClipEmbedding:
        embedding = tuple(float(value) for value in values)
        norm = math.sqrt(sum(value * value for value in embedding))
        if (
            len(embedding) != self.dimension
            or not all(math.isfinite(value) for value in embedding)
            or not math.isfinite(norm)
            or abs(norm - 1.0) > _UNIT_NORM_TOLERANCE
        ):
            raise ClipInferenceError(code="clip_embedding_invalid")
        return embedding


def _triton_model_config(model_name: str, package: ClipModelPackage) -> dict[str, object]:
    """Build a complete explicit Triton config for one CLIP modality."""
    input_name = "IMAGE" if model_name == _IMAGE_MODEL_NAME else "TEXT"
    return {
        "name": model_name,
        "backend": "python",
        "max_batch_size": 8,
        "input": [{"name": input_name, "data_type": "TYPE_STRING", "dims": [1]}],
        "output": [
            {"name": "EMBEDDING", "data_type": "TYPE_FP32", "dims": [package.dimension]}
        ],
        "instance_group": [{"count": 1, "kind": "KIND_GPU", "gpus": [0]}],
        "dynamic_batching": {"max_queue_delay_microseconds": 10000},
        "parameters": package.runtime_parameters(),
    }


def _identity_from_model_config(response: Mapping[str, object]) -> ClipRuntimeIdentity:
    """Parse the effective Triton JSON response into a typed identity."""
    raw_config_value = response.get("config", response)
    if not isinstance(raw_config_value, Mapping):
        raise ClipRuntimeIdentityError(code="clip_runtime_identity_invalid")
    raw_config: Mapping[str, object] = cast(
        "Mapping[str, object]", cast("object", raw_config_value)
    )
    raw_parameters_value = raw_config.get("parameters")
    if not isinstance(raw_parameters_value, Mapping):
        raise ClipRuntimeIdentityError(code="clip_runtime_identity_invalid")
    raw_parameters: Mapping[str, object] = cast(
        "Mapping[str, object]", cast("object", raw_parameters_value)
    )

    def parameter(name: str) -> str:
        raw_value = raw_parameters.get(name)
        if not isinstance(raw_value, Mapping):
            raise ClipRuntimeIdentityError(code=f"clip_runtime_parameter_missing: {name}")
        parameter_value: Mapping[str, object] = cast(
            "Mapping[str, object]", cast("object", raw_value)
        )
        value = parameter_value.get("string_value")
        if not isinstance(value, str) or not value:
            raise ClipRuntimeIdentityError(code=f"clip_runtime_parameter_invalid: {name}")
        return value

    dimension_text = parameter("embedding_dimension")
    try:
        dimension = int(dimension_text)
    except ValueError as error:
        raise ClipRuntimeIdentityError(code="clip_runtime_dimension_invalid") from error
    if dimension < 1:
        raise ClipRuntimeIdentityError(code="clip_runtime_dimension_invalid")
    return ClipRuntimeIdentity(
        model_id=parameter("model_id"),
        revision=parameter("model_revision"),
        snapshot_path=Path(parameter("snapshot_path")),
        dimension=dimension,
        processor=parameter("processor"),
        runtime=parameter("runtime"),
    )


def _raise_transport_error(error: InferenceServerException) -> NoReturn:
    """Map backend diagnostics to stable client-side error classes."""
    message = str(error)
    lowered = message.lower()
    if "clip_image_decode_failed" in lowered:
        raise ClipImageDecodeError(code="clip_image_decode_failed") from error
    if "clip_gpu_oom" in lowered or "out of memory" in lowered:
        raise ClipInferenceError(code="clip_gpu_oom") from error
    if "clip_image_inference_failed" in lowered:
        raise ClipInferenceError(code="clip_image_inference_failed") from error
    raise ClipInferenceError(code=f"clip_inference_failed: {message}") from error


def _runtime_value_error(code: str) -> ValueError:
    """Build a stable configuration error without exposing client internals."""
    return ValueError(code)


def _identity_mismatch_error(
    expected: ClipRuntimeIdentity, observed: ClipRuntimeIdentity
) -> ClipRuntimeIdentityError:
    """Build a diagnostic identity mismatch after a load readback."""
    return ClipRuntimeIdentityError(
        code=(
            "clip_runtime_identity_mismatch: "
            f"expected={expected!r} observed={observed!r}"
        )
    )


__all__ = [
    "ClipAdapter",
    "ClipEmbedding",
    "ClipImageDecodeError",
    "ClipInferenceError",
    "ClipInputError",
    "ClipRuntimeError",
    "ClipRuntimeIdentity",
    "ClipRuntimeIdentityError",
    "ClipRuntimeManager",
    "ClipTransport",
    "TritonClipRuntime",
    "TritonClipTransport",
    "TritonModelControlClient",
]
