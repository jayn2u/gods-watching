"""Boundary models for attributed prepared test streams."""

from typing import Annotated, ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

NonEmptyText = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_REQUIRED_FIXTURE_COUNT: Final = 4
_MIN_CONTINUOUS_PERSON_SECONDS: Final = 20.0


class FixtureModel(BaseModel):
    """Reject unknown manifest fields and mutation after parsing."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ContinuousPersonInterval(FixtureModel):
    """Record a manually inspected same-person visibility interval."""

    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(gt=0)]
    description: NonEmptyText

    @model_validator(mode="after")
    def require_twenty_seconds(self) -> Self:
        """Reject an interval too short for active-indexing verification."""
        if self.duration_seconds < _MIN_CONTINUOUS_PERSON_SECONDS:
            error_code = "short_person_interval"
            error_message = "continuous person intervals must be at least 20 seconds"
            raise PydanticCustomError(error_code, error_message)
        return self

    @property
    def duration_seconds(self) -> float:
        """Return the inspected interval length in seconds."""
        return self.end_seconds - self.start_seconds


class ReplacedCandidate(FixtureModel):
    """Retain why a planned R7 source was not accepted."""

    source_page: HttpUrl
    reason: NonEmptyText


class FixtureStream(FixtureModel):
    """Bind source attribution and measured prepared media properties."""

    stream_id: Annotated[str, StringConstraints(pattern=r"^camera-[1-4]$")]
    prepared_path: NonEmptyText
    source_page: HttpUrl
    direct_url: HttpUrl
    author: NonEmptyText
    license_name: NonEmptyText
    license_url: HttpUrl
    sha256: Sha256Hex
    duration_seconds: Annotated[float, Field(gt=0)]
    fps: NonEmptyText
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    codec: Annotated[str, StringConstraints(pattern="^h264$")]
    scene: NonEmptyText
    adjustments: tuple[NonEmptyText, ...]
    visual_inspection: NonEmptyText
    continuous_person_intervals: tuple[ContinuousPersonInterval, ...] = ()
    replaces: ReplacedCandidate


class FixtureManifest(FixtureModel):
    """Require the complete four-scene attributed fixture set."""

    schema_version: Annotated[int, Field(ge=1)]
    prepared_inputs_redistributable: bool
    streams: tuple[FixtureStream, ...]

    @model_validator(mode="after")
    def require_complete_distinct_set(self) -> Self:
        """Reject incomplete, duplicate, or redistribution-marked fixture sets."""
        if len(self.streams) != _REQUIRED_FIXTURE_COUNT:
            error_code = "fixture_count"
            error_message = "exactly four fixture streams are required"
            raise PydanticCustomError(error_code, error_message)
        distinct_fields = (
            tuple(stream.stream_id for stream in self.streams),
            tuple(str(stream.source_page) for stream in self.streams),
            tuple(str(stream.direct_url) for stream in self.streams),
            tuple(stream.scene for stream in self.streams),
            tuple(stream.sha256 for stream in self.streams),
        )
        if any(len(set(values)) != _REQUIRED_FIXTURE_COUNT for values in distinct_fields):
            error_code = "duplicate_fixture"
            error_message = "stream IDs, sources, scenes, and inputs must be distinct"
            raise PydanticCustomError(error_code, error_message)
        if not any(
            interval.duration_seconds >= _MIN_CONTINUOUS_PERSON_SECONDS
            for stream in self.streams
            for interval in stream.continuous_person_intervals
        ):
            error_code = "missing_continuous_person"
            error_message = (
                "at least one inspected same-person interval must be at least 20 seconds"
            )
            raise PydanticCustomError(error_code, error_message)
        if self.prepared_inputs_redistributable:
            error_code = "fixture_redistribution"
            error_message = "prepared inputs are local test assets only"
            raise PydanticCustomError(error_code, error_message)
        return self


class FixtureProbe(FixtureModel):
    """Expose the measured video properties used for binary verification."""

    stream_id: str
    path: str
    sha256: Sha256Hex
    duration_seconds: float
    fps: str
    width: int
    height: int
    codec: str
    size_bytes: int


class FFProbeStream(FixtureModel):
    """Parse the selected ffprobe video stream."""

    codec_name: str
    width: int
    height: int
    r_frame_rate: str


class FFProbeFormat(FixtureModel):
    """Parse ffprobe container properties."""

    duration: str
    size: str


class FFProbeProgram(FixtureModel):
    """Reject unexpected nonempty program records in prepared MP4 inputs."""


class FFProbeDocument(FixtureModel):
    """Parse the bounded ffprobe JSON result."""

    streams: tuple[FFProbeStream, ...]
    format: FFProbeFormat
    programs: tuple[FFProbeProgram, ...] = ()
