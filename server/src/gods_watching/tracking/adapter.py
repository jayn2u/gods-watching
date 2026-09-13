"""Strict adapter around the pinned supervision ByteTrack implementation."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import numpy.typing as npt
import supervision as sv

from gods_watching.inference.detector import Detection

from .models import (
    DetectionFrame,
    LocalTrackId,
    TrackingParameters,
    TrackObservation,
)

type FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """Pair a supervision local ID with its original detector handoff."""

    local_track_id: LocalTrackId
    observation: TrackObservation


class ByteTrackAdapter:
    """Adapt typed detector frames to one real supervision ByteTrack instance."""

    def __init__(self, *, parameters: TrackingParameters) -> None:
        """Create a fresh tracker with the fixed per-camera parameters."""
        self._parameters: TrackingParameters = parameters
        self._tracker: sv.ByteTrack = sv.ByteTrack(
            track_activation_threshold=parameters.track_activation_threshold,
            lost_track_buffer=parameters.lost_track_buffer,
            minimum_matching_threshold=parameters.minimum_matching_threshold,
            frame_rate=parameters.frame_rate,
            minimum_consecutive_frames=parameters.minimum_consecutive_frames,
        )

    @property
    def parameters(self) -> TrackingParameters:
        """Return the immutable configuration used by the real tracker."""
        return self._parameters

    def update(self, frame: DetectionFrame) -> tuple[TrackedDetection, ...]:
        """Run one detector result through supervision and retain source metadata."""
        observations = observations_for_frame(frame)
        detections = sv.Detections(
            xyxy=_boxes(observations),
            confidence=_scores(observations),
        )
        tracked = self._tracker.update_with_detections(detections)
        if tracked.tracker_id is None or tracked.confidence is None:
            return ()
        tracker_ids: npt.NDArray[np.int64] = tracked.tracker_id
        tracked_boxes: npt.NDArray[np.float32] = tracked.xyxy
        tracked_confidence: npt.NDArray[np.float32] = tracked.confidence
        results: list[TrackedDetection] = []
        for index in range(len(tracker_ids)):
            track_id = tracker_ids.item(index)
            if track_id < 0:
                continue
            tracked_box = _box_at(tracked_boxes, index)
            source_index = _best_source_index(
                tracked_box,
                observations,
            )
            if source_index is None:
                continue
            source = observations[source_index]
            confidence = float(tracked_confidence.item(index))
            results.append(
                TrackedDetection(
                    local_track_id=LocalTrackId(track_id),
                    observation=TrackObservation(
                        bounding_box=tracked_box,
                        confidence=confidence,
                        detector_input_reference=source.detector_input_reference,
                        ingress_utc=source.ingress_utc,
                        ingress_monotonic=source.ingress_monotonic,
                        detector_result_monotonic=source.detector_result_monotonic,
                    ),
                )
            )
        return tuple(results)

    def reset(self) -> None:
        """Clear the real library state at a source/session boundary."""
        self._tracker.reset()


def _valid_detection(frame: DetectionFrame, detection: Detection) -> bool:
    values = detection.xyxy
    return (
        detection.class_id == 0
        and isfinite(detection.confidence)
        and 0.0 <= detection.confidence <= 1.0
        and all(isfinite(value) for value in values)
        and 0.0 <= values[0] < values[2] <= frame.source_width
        and 0.0 <= values[1] < values[3] <= frame.source_height
    )


def observations_for_frame(frame: DetectionFrame) -> tuple[TrackObservation, ...]:
    """Return valid original-frame observations accepted by the tracker boundary."""
    return tuple(
        _observation(frame, detection)
        for detection in frame.detections
        if _valid_detection(frame, detection)
    )


def _observation(frame: DetectionFrame, detection: Detection) -> TrackObservation:
    return TrackObservation(
        bounding_box=detection.xyxy,
        confidence=detection.confidence,
        detector_input_reference=frame.detector_input_reference,
        ingress_utc=frame.ingress_utc,
        ingress_monotonic=frame.ingress_monotonic,
        detector_result_monotonic=frame.detector_result_monotonic,
    )


def _boxes(observations: tuple[TrackObservation, ...]) -> FloatArray:
    if not observations:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray([observation.bounding_box for observation in observations], dtype=np.float32)


def _scores(observations: tuple[TrackObservation, ...]) -> FloatArray:
    if not observations:
        return np.empty((0,), dtype=np.float32)
    return np.asarray([observation.confidence for observation in observations], dtype=np.float32)


def _best_source_index(
    tracked_box: tuple[float, float, float, float],
    observations: tuple[TrackObservation, ...],
) -> int | None:
    if not observations:
        return None
    best_index: int | None = None
    best_iou = 0.0
    for index, observation in enumerate(observations):
        iou = _iou(tracked_box, observation.bounding_box)
        if iou > best_iou:
            best_index = index
            best_iou = iou
    return best_index


def _box_at(boxes: npt.NDArray[np.float32], index: int) -> tuple[float, float, float, float]:
    """Read one fixed-width box without exposing NumPy's untyped row view."""
    return (
        boxes.item(index, 0),
        boxes.item(index, 1),
        boxes.item(index, 2),
        boxes.item(index, 3),
    )


def _iou(
    left: npt.NDArray[np.floating] | tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = (float(value) for value in left)
    right_x1, right_y1, right_x2, right_y2 = right
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0
