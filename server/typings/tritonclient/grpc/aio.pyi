from collections.abc import Sequence

from tritonclient.grpc import InferInput, InferRequestedOutput, InferResult

class InferenceServerClient:
    def __init__(self, url: str, verbose: bool = False) -> None: ...
    async def infer(
        self,
        model_name: str,
        inputs: Sequence[InferInput],
        model_version: str = "",
        outputs: Sequence[InferRequestedOutput] | None = None,
        request_id: str = "",
        client_timeout: float | None = None,
    ) -> InferResult: ...
    async def is_server_ready(self) -> bool: ...
    async def is_model_ready(self, model_name: str, model_version: str = "") -> bool: ...
    async def close(self) -> None: ...
