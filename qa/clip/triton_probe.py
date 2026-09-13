# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.2.6",
#   "pillow==11.3.0",
#   "torch==2.7.1",
#   "transformers==4.56.1",
#   "tritonclient[grpc]==2.61.0",
# ]
# ///
# ─── How to run ───
# python /qa/triton_probe.py /evidence/clip-input.jpg /evidence/task-8-clip.json

"""Compare real Triton CLIP outputs with direct pinned GPU inference."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from tritonclient import grpc

NORM_TOLERANCE = 1e-5
REFERENCE_COSINE = 0.999


def _infer(
    client: grpc.InferenceServerClient, model: str, name: str, values: list[bytes]
) -> np.ndarray:
    tensor = grpc.InferInput(name, [len(values), 1], "BYTES")
    tensor.set_data_from_numpy(np.asarray(values, dtype=np.object_).reshape(-1, 1))
    response = client.infer(model, [tensor], outputs=[grpc.InferRequestedOutput("EMBEDDING")])
    result = response.as_numpy("EMBEDDING")
    if result is None:
        message = "probe_embedding_missing"
        raise RuntimeError(message)
    return np.asarray(result, dtype=np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _main() -> int:
    image_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    image_bytes = image_path.read_bytes()
    texts = [b"a person with a red bag", b"a pedestrian walking on a city street"]
    client = grpc.InferenceServerClient(url="localhost:8001")
    text_batch = _infer(client, "clip_text", "TEXT", texts)
    image_batch = _infer(client, "clip_image", "IMAGE", [image_bytes, image_bytes])
    processor = AutoProcessor.from_pretrained("/models/clip", local_files_only=True)
    model = CLIPModel.from_pretrained("/models/clip", local_files_only=True).to("cuda").eval()
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
    text_inputs = processor(
        text=[item.decode() for item in texts], return_tensors="pt", padding=True
    )
    image_inputs = processor(images=[rgb, rgb], return_tensors="pt")
    with torch.inference_mode():
        direct_text = model.get_text_features(
            **{name: value.to("cuda") for name, value in text_inputs.items()}
        )
        direct_image = model.get_image_features(
            **{name: value.to("cuda") for name, value in image_inputs.items()}
        )
    direct_text = torch.nn.functional.normalize(direct_text, dim=-1).cpu().float().numpy()
    direct_image = torch.nn.functional.normalize(direct_image, dim=-1).cpu().float().numpy()
    record = {
        "cuda_device": torch.cuda.get_device_name(0),
        "model_revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
        "text": {
            "shape": list(text_batch.shape),
            "finite": bool(np.isfinite(text_batch).all()),
            "norms": [float(np.linalg.norm(row)) for row in text_batch],
            "reference_cosines": [
                _cosine(row, direct) for row, direct in zip(text_batch, direct_text, strict=True)
            ],
        },
        "image": {
            "shape": list(image_batch.shape),
            "finite": bool(np.isfinite(image_batch).all()),
            "norms": [float(np.linalg.norm(row)) for row in image_batch],
            "reference_cosines": [
                _cosine(row, direct) for row, direct in zip(image_batch, direct_image, strict=True)
            ],
        },
        "batches_match": bool(
            math.isclose(_cosine(image_batch[0], image_batch[1]), 1.0, abs_tol=1e-6)
        ),
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    passed = (
        record["text"]["shape"] == [2, 512]
        and record["image"]["shape"] == [2, 512]
        and record["text"]["finite"]
        and record["image"]["finite"]
        and all(abs(value - 1.0) <= NORM_TOLERANCE for value in record["text"]["norms"])
        and all(abs(value - 1.0) <= NORM_TOLERANCE for value in record["image"]["norms"])
        and min(record["text"]["reference_cosines"]) >= REFERENCE_COSINE
        and min(record["image"]["reference_cosines"]) >= REFERENCE_COSINE
        and record["batches_match"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
