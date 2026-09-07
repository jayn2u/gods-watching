import numpy as np
import numpy.typing as npt

type BoxArray = npt.NDArray[np.float32]
type ScoreArray = npt.NDArray[np.float32]
type TrackIdArray = npt.NDArray[np.int64]

class Detections:
    xyxy: BoxArray
    confidence: ScoreArray | None
    class_id: npt.NDArray[np.int64] | None
    tracker_id: TrackIdArray | None

    def __init__(
        self,
        *,
        xyxy: BoxArray,
        mask: npt.NDArray[np.bool_] | None = None,
        confidence: ScoreArray | None = None,
        class_id: npt.NDArray[np.int64] | None = None,
        tracker_id: TrackIdArray | None = None,
        data: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None: ...

class ByteTrack:
    def __init__(
        self,
        *,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ) -> None: ...
    def update_with_detections(self, detections: Detections) -> Detections: ...
    def reset(self) -> None: ...
