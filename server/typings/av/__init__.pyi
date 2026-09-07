from collections.abc import Iterable
from fractions import Fraction
from types import TracebackType
from typing import Literal, Self

import numpy as np
import numpy.typing as npt

class Packet:
    def __bytes__(self) -> bytes: ...

class VideoFrame:
    width: int
    height: int

    def to_ndarray(self, **kwargs: str) -> npt.NDArray[np.uint8]: ...
    def reformat(self, **kwargs: str) -> VideoFrame: ...

class Container:
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def decode(self, *, video: int) -> Iterable[VideoFrame]: ...

def open(file: str, mode: Literal["r"], *, options: dict[str, str]) -> Container: ...  # noqa: A001

class CodecContext:
    width: int
    height: int
    pix_fmt: str
    time_base: Fraction

    @classmethod
    def create(cls, codec_name: str, mode: str) -> CodecContext: ...
    def encode(self, frame: VideoFrame | None = None) -> list[Packet]: ...
