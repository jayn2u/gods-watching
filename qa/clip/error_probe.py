# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.2.6", "tritonclient[grpc]==2.61.0"]
# ///
# ─── How to run ───
# python /qa/error_probe.py /evidence/task-8-errors.json

"""Prove malformed CLIP requests fail through the real Triton surface."""

import json
import sys
from pathlib import Path

import numpy as np
from tritonclient import grpc
from tritonclient.utils import InferenceServerException


def _rejected(client: grpc.InferenceServerClient, model: str, name: str, value: bytes) -> str:
    tensor = grpc.InferInput(name, [1, 1], "BYTES")
    tensor.set_data_from_numpy(np.asarray([[value]], dtype=np.object_))
    try:
        client.infer(model, [tensor], outputs=[grpc.InferRequestedOutput("EMBEDDING")])
    except InferenceServerException as error:
        return str(error)
    return ""


def _main() -> int:
    client = grpc.InferenceServerClient(url="localhost:8001")
    errors = {
        "bad_image": _rejected(client, "clip_image", "IMAGE", b"not-an-image"),
        "empty_text": _rejected(client, "clip_text", "TEXT", b""),
        "unsupported_text": _rejected(client, "clip_text", "TEXT", "person 🎒".encode()),
        "over_token_limit": _rejected(client, "clip_text", "TEXT", ("a " * 76).encode()),
    }
    Path(sys.argv[1]).write_text(
        json.dumps(errors, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    passed = all(errors.values()) and "clip_text_too_many_tokens" in errors["over_token_limit"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
