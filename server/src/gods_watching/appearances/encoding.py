"""Encode the bounded original RGB crop and calculate its sharpness metric."""

from enum import StrEnum
from io import BytesIO
from math import isfinite
from typing import override

import numpy as np
from PIL import Image

from gods_watching.contracts.pipeline import RgbCrop


class EncodingErrorCode(StrEnum):
    """Name invalid encoder outputs."""

    JPEG_EMPTY = "jpeg_empty"
    LAPLACIAN_INVALID = "laplacian_invalid"


class EncodingError(ValueError):
    """Describe a crop encoding failure at the media boundary."""

    def __init__(self, code: EncodingErrorCode) -> None:
        """Retain the typed encoding failure code."""
        self.code: EncodingErrorCode = code
        super().__init__(code)

    @override
    def __str__(self) -> str:
        return self.code.value


def encode_rgb_crop(crop: RgbCrop) -> bytes:
    """Encode original RGB pixels as a quality-90 JPEG without frame expansion."""
    image = Image.frombytes("RGB", (crop.width, crop.height), crop.data)
    output = BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=False)
    encoded = output.getvalue()
    if not encoded:
        raise EncodingError(EncodingErrorCode.JPEG_EMPTY)
    return encoded


def laplacian_variance(crop: RgbCrop) -> float:
    """Return the variance of a three-by-three grayscale Laplacian."""
    pixels = np.frombuffer(crop.data, dtype=np.uint8).reshape(crop.height, crop.width, 3)
    rgb = pixels.astype(dtype=np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
    )
    variance = float(np.var(laplacian))
    if not isfinite(variance) or variance < 0.0:
        raise EncodingError(EncodingErrorCode.LAPLACIAN_INVALID)
    return variance
