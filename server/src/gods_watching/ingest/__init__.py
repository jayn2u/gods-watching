"""Recording-free PyAV ingest and fair detector scheduling."""

from .crop import CropExtractionError, crop_for_observation, extract_rgb_crop
from .decoder import (
    DecodedFrameSink,
    DecoderConfiguration,
    DecoderConfigurationError,
    DecoderOptions,
    DecoderStatusSink,
    GenerationReconnect,
    PyAvRtspDecoder,
    RtspDecodeError,
)
from .models import (
    DecodedFrame,
    DecoderInputError,
    IngestStats,
    IngestStatsSnapshot,
    LatestFrameSlot,
)
from .scheduler import FairRoundRobinScheduler, SchedulerConfigurationError, UnknownCameraError
from .worker import (
    DetectorPort,
    IngestCoordinator,
    IngestWorker,
    IngestWorkerConfiguration,
    PipelineHandoffConsumer,
    StaleGenerationError,
    WorkerConfigurationError,
)

__all__ = [
    "CropExtractionError",
    "DecodedFrame",
    "DecodedFrameSink",
    "DecoderConfiguration",
    "DecoderConfigurationError",
    "DecoderInputError",
    "DecoderOptions",
    "DecoderStatusSink",
    "DetectorPort",
    "FairRoundRobinScheduler",
    "GenerationReconnect",
    "IngestCoordinator",
    "IngestStats",
    "IngestStatsSnapshot",
    "IngestWorker",
    "IngestWorkerConfiguration",
    "LatestFrameSlot",
    "PipelineHandoffConsumer",
    "PyAvRtspDecoder",
    "RtspDecodeError",
    "SchedulerConfigurationError",
    "StaleGenerationError",
    "UnknownCameraError",
    "WorkerConfigurationError",
    "crop_for_observation",
    "extract_rgb_crop",
]
