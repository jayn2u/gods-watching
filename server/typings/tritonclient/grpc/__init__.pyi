from collections.abc import Sequence
from typing import Literal, Self, overload

import numpy as np
import numpy.typing as npt

type TritonTensor = npt.NDArray[np.float32] | npt.NDArray[np.int32] | npt.NDArray[np.object_]

class InferInput:
    def __init__(self, name: str, shape: Sequence[int], datatype: str) -> None: ...
    def set_data_from_numpy(self, input_tensor: TritonTensor) -> Self: ...

class InferRequestedOutput:
    def __init__(self, name: str, class_count: int = 0) -> None: ...

class InferResult:
    @overload
    def as_numpy(self, name: Literal["COUNT"]) -> npt.NDArray[np.int32] | None: ...
    @overload
    def as_numpy(self, name: Literal["BOXES", "EMBEDDING"]) -> npt.NDArray[np.float32] | None: ...
    @overload
    def as_numpy(self, name: str) -> npt.NDArray[np.generic] | None: ...
