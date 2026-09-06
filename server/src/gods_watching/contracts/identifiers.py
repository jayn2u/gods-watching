"""Distinct identifiers used across public contracts."""

from typing import NewType
from uuid import UUID

AppearanceId = NewType("AppearanceId", UUID)
CameraId = NewType("CameraId", UUID)
CameraSessionId = NewType("CameraSessionId", UUID)
LoginSessionId = NewType("LoginSessionId", UUID)
