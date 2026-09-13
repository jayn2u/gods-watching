"""Camera/session-scoped lifecycle state around the real ByteTrack adapter."""

from collections.abc import Iterable
from dataclasses import dataclass

from .adapter import ByteTrackAdapter, observations_for_frame
from .models import (
    TRACK_EXPIRY_SECONDS,
    ClosedScopeError,
    DetectionFrame,
    LifecycleKind,
    LocalTrackId,
    ResetReason,
    StaleFrameError,
    TrackingParameters,
    TrackingScope,
    TrackKey,
    TrackLifecycle,
    TrackObservation,
)

_MAX_PENDING_CANDIDATES = 256
_MIN_PENDING_IOU = 0.1


@dataclass(slots=True)
class _ActiveTrack:
    key: TrackKey
    first_candidate: TrackObservation
    observation: TrackObservation


class ByteTrackScope:
    """Own one bounded ByteTrack state machine for one camera source generation."""

    def __init__(self, *, scope: TrackingScope, threshold: float) -> None:
        """Create a real per-scope tracker with the approved settings."""
        self._scope: TrackingScope = scope
        self._parameters: TrackingParameters = TrackingParameters.for_camera_threshold(threshold)
        self._adapter: ByteTrackAdapter = ByteTrackAdapter(parameters=self._parameters)
        self._active: dict[LocalTrackId, _ActiveTrack] = {}
        self._pending: list[TrackObservation] = []
        self._library_to_local: dict[int, LocalTrackId] = {}
        self._next_local_track_id: int = 1
        self._epoch: int = 0
        self._sequence: int = 0
        self._last_frame_monotonic: float | None = None
        self._closed: bool = False

    @property
    def scope(self) -> TrackingScope:
        """Return the current camera/session/source-generation identity."""
        return self._scope

    @property
    def parameters(self) -> TrackingParameters:
        """Return immutable parameters for this scope's real tracker."""
        return self._parameters

    @property
    def library_max_time_lost(self) -> int:
        """Return supervision's sampled-frame loss budget."""
        return self._parameters.library_max_time_lost

    @property
    def active_track_count(self) -> int:
        """Return the number of open lifecycle tracks."""
        return len(self._active)

    @property
    def pending_candidate_count(self) -> int:
        """Return the bounded number of waiting first candidates."""
        return len(self._pending)

    def feed(self, frame: DetectionFrame) -> tuple[TrackLifecycle, ...]:
        """Process one sampled detector result and emit ordered lifecycle events."""
        self._validate_frame(frame)
        events = list(self._expire_at(frame))
        self._last_frame_monotonic = frame.ingress_monotonic
        if not self._scope.detection_enabled:
            self._discard_expired_candidates(frame.ingress_monotonic)
            return tuple(events)

        observations = observations_for_frame(frame)
        tracked = self._adapter.update(frame)
        current_indices = set(range(len(observations)))
        for result in sorted(tracked, key=lambda item: int(item.local_track_id)):
            current_index = _best_observation_index(result.observation, observations)
            if current_index is not None:
                current_indices.discard(current_index)
            local_id = self._global_local_id(int(result.local_track_id))
            active = self._active.get(local_id)
            if active is None:
                candidate = self._take_pending(result.observation) or result.observation
                key = TrackKey(
                    camera_id=self._scope.camera_id,
                    session_id=self._scope.session_id,
                    source_generation_id=self._scope.source_generation_id,
                    local_track_id=local_id,
                )
                active = _ActiveTrack(
                    key=key,
                    first_candidate=candidate,
                    observation=result.observation,
                )
                self._active[local_id] = active
                events.append(self._event(LifecycleKind.START, active))
            else:
                active.observation = result.observation
                events.append(self._event(LifecycleKind.UPDATE, active))

        self._remember_candidates(
            observations[index]
            for index in sorted(current_indices)
            if observations[index].confidence >= self._parameters.camera_threshold
        )
        self._discard_expired_candidates(frame.ingress_monotonic)
        return tuple(events)

    def expire(self, boundary: DetectionFrame) -> tuple[TrackLifecycle, ...]:
        """Advance the monotonic fence and close tracks stale for more than two seconds."""
        self._validate_frame(boundary, allow_equal=True)
        self._last_frame_monotonic = boundary.ingress_monotonic
        return self._expire_at(boundary)

    def reset(
        self,
        *,
        reason: ResetReason,
        boundary: DetectionFrame,
        replacement_scope: TrackingScope | None = None,
        threshold: float | None = None,
    ) -> tuple[TrackLifecycle, ...]:
        """End current tracks and install a fresh source/session/detection namespace."""
        self._ensure_open()
        self._validate_boundary(boundary)
        events = [
            self._event(LifecycleKind.END, active, boundary=boundary, reason=reason)
            for active in sorted(
                self._active.values(), key=lambda value: int(value.key.local_track_id)
            )
        ]
        self._active.clear()
        self._pending.clear()
        self._adapter.reset()
        self._library_to_local.clear()
        self._last_frame_monotonic = boundary.ingress_monotonic
        self._epoch += 1
        if threshold is not None:
            self._parameters = TrackingParameters.for_camera_threshold(threshold)
            self._adapter = ByteTrackAdapter(parameters=self._parameters)
        if replacement_scope is not None:
            if replacement_scope.camera_id != self._scope.camera_id:
                raise StaleFrameError(detail="replacement scope camera does not match active scope")
            self._scope = replacement_scope
        return tuple(events)

    def close(self, boundary: DetectionFrame) -> tuple[TrackLifecycle, ...]:
        """Close the scope and reject all subsequent detector results."""
        replacement_scope = TrackingScope(
            camera_id=self._scope.camera_id,
            session_id=self._scope.session_id,
            source_generation_id=self._scope.source_generation_id,
            detection_enabled=False,
        )
        events = self.reset(
            reason=ResetReason.CLOSED,
            boundary=boundary,
            replacement_scope=replacement_scope,
        )
        self._closed = True
        return events

    def _validate_frame(self, frame: DetectionFrame, *, allow_equal: bool = False) -> None:
        self._ensure_open()
        if not self._same_identity(frame):
            raise StaleFrameError(detail="frame identity does not match active tracking scope")
        if frame.detection_enabled != self._scope.detection_enabled:
            raise StaleFrameError(detail="frame detection mode does not match active scope")
        if self._last_frame_monotonic is None:
            return
        if allow_equal:
            is_stale = frame.ingress_monotonic < self._last_frame_monotonic
        else:
            is_stale = frame.ingress_monotonic <= self._last_frame_monotonic
        if is_stale:
            raise StaleFrameError(detail="frame ingress monotonic timestamp crossed the fence")

    def _validate_boundary(self, boundary: DetectionFrame) -> None:
        self._ensure_open()
        if not self._same_identity(boundary):
            raise StaleFrameError(detail="reset boundary identity does not match active scope")
        if self._last_frame_monotonic is not None and (
            boundary.ingress_monotonic < self._last_frame_monotonic
        ):
            raise StaleFrameError(detail="reset boundary crossed the ingress monotonic fence")

    def _same_identity(self, frame: DetectionFrame) -> bool:
        return (
            frame.camera_id == self._scope.camera_id
            and frame.session_id == self._scope.session_id
            and frame.source_generation_id == self._scope.source_generation_id
        )

    def _expire_at(self, boundary: DetectionFrame) -> tuple[TrackLifecycle, ...]:
        stale = [
            active
            for active in self._active.values()
            if boundary.ingress_monotonic - active.observation.ingress_monotonic
            > TRACK_EXPIRY_SECONDS
        ]
        events = [
            self._event(LifecycleKind.END, active, boundary=boundary, reason=ResetReason.EXPLICIT)
            for active in sorted(stale, key=lambda value: int(value.key.local_track_id))
        ]
        for active in stale:
            _ = self._active.pop(active.key.local_track_id, None)
            self._library_to_local = {
                library_id: local_id
                for library_id, local_id in self._library_to_local.items()
                if local_id != active.key.local_track_id
            }
        if not self._active and stale:
            self._adapter.reset()
            self._library_to_local.clear()
        self._discard_expired_candidates(boundary.ingress_monotonic)
        return tuple(events)

    def _remember_candidates(self, candidates: Iterable[TrackObservation]) -> None:
        for candidate in candidates:
            self._pending.append(candidate)
        if len(self._pending) > _MAX_PENDING_CANDIDATES:
            self._pending = self._pending[-_MAX_PENDING_CANDIDATES:]

    def _discard_expired_candidates(self, now_monotonic: float) -> None:
        self._pending = [
            candidate
            for candidate in self._pending
            if now_monotonic - candidate.ingress_monotonic <= TRACK_EXPIRY_SECONDS
        ]

    def _take_pending(self, observation: TrackObservation) -> TrackObservation | None:
        best_index: int | None = None
        best_iou = _MIN_PENDING_IOU
        for index, candidate in enumerate(self._pending):
            iou = _iou(observation.bounding_box, candidate.bounding_box)
            if iou > best_iou:
                best_index = index
                best_iou = iou
        if best_index is None:
            return None
        return self._pending.pop(best_index)

    def _global_local_id(self, library_id: int) -> LocalTrackId:
        local_id = self._library_to_local.get(library_id)
        if local_id is None:
            local_id = LocalTrackId(self._next_local_track_id)
            self._next_local_track_id += 1
            self._library_to_local[library_id] = local_id
        return local_id

    def _event(
        self,
        kind: LifecycleKind,
        active: _ActiveTrack,
        *,
        boundary: DetectionFrame | None = None,
        reason: ResetReason | None = None,
    ) -> TrackLifecycle:
        self._sequence += 1
        return TrackLifecycle(
            kind=kind,
            track_key=active.key,
            source_generation_id=active.key.source_generation_id,
            scope_epoch=self._epoch,
            sequence=self._sequence,
            observation=active.observation,
            first_candidate=active.first_candidate,
            first_seen=active.first_candidate.ingress_utc,
            last_seen=active.observation.ingress_utc,
            t_detect_monotonic=active.first_candidate.detector_result_monotonic,
            ended_at=None if boundary is None else boundary.ingress_utc,
            ended_monotonic=None if boundary is None else boundary.ingress_monotonic,
            end_reason=reason,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedScopeError


def _best_observation_index(
    target: TrackObservation,
    observations: tuple[TrackObservation, ...],
) -> int | None:
    if not observations:
        return None
    best_index: int | None = None
    best_iou = 0.0
    for index, observation in enumerate(observations):
        iou = _iou(target.bounding_box, observation.bounding_box)
        if iou > best_iou:
            best_index = index
            best_iou = iou
    return best_index


def _iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = width * height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0
