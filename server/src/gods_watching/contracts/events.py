"""Typed event export filters and rows."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


class InvalidEventExportFilterError(ValueError):
    """Describe a rejected export filter without echoing its supplied value."""

    field_name: str
    problem: str

    def __init__(self, field_name: str, problem: str) -> None:
        """Store the field and issue without retaining rejected values."""
        self.field_name = field_name
        self.problem = problem
        message = f"{field_name} {problem}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class EventExportFilters:
    """Bound camera and time filters for an event export."""

    camera_id: UUID | None = None
    since: datetime | None = None
    until: datetime | None = None

    def __post_init__(self) -> None:
        """Require explicit offsets and a non-inverted half-open time range."""
        for field_name, value in (("since", self.since), ("until", self.until)):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                problem = "requires a timezone offset"
                raise InvalidEventExportFilterError(field_name, problem)
        if self.since is not None and self.until is not None and self.since > self.until:
            field_name = "since"
            problem = "must not be later than until"
            raise InvalidEventExportFilterError(field_name, problem)


@dataclass(frozen=True, slots=True)
class EventExportRow:
    """The five approved, non-secret camera event export values."""

    id: int
    occurred_at: datetime
    event_type: str
    camera_id: UUID
    camera_name: str
