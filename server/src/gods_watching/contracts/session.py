"""Single-operator session boundary contracts."""

from typing import Annotated

from pydantic import Field, SecretStr

from .base import ContractModel
from .primitives import UtcDatetime


class LoginRequest(ContractModel):
    """Parse the operator identifier and password without serializing the secret."""

    username: Annotated[str, Field(default="admin", min_length=1, max_length=64)]
    password: Annotated[SecretStr, Field(min_length=4, max_length=128)]


class SessionResponse(ContractModel):
    """Expose session expiry state without an opaque token."""

    authenticated: bool
    idle_expires_at: UtcDatetime | None
    absolute_expires_at: UtcDatetime | None
