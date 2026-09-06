"""Machine-readable API error contracts."""

from .base import ContractModel


class ErrorDetail(ContractModel):
    """Describe a stable error code and an English operator message."""

    code: str
    message: str


class ErrorResponse(ContractModel):
    """Wrap one boundary error."""

    error: ErrorDetail
