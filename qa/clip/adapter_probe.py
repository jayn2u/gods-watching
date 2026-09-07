# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "anyio==4.10.0",
#   "numpy==2.2.6",
#   "pydantic==2.11.7",
#   "tritonclient[grpc]==2.61.0",
# ]
# ///
# ─── How to run ───
# PYTHONPATH=/app python /qa/adapter_probe.py /evidence/clip-input.jpg /evidence/adapter.json

"""Drive the production typed CLIP adapter against real Triton gRPC."""

import json
import sys
from pathlib import Path

import anyio

from gods_watching.inference.clip import ClipAdapter, TritonClipTransport


async def _run(image_path: Path, output_path: Path) -> None:
    async with TritonClipTransport("localhost:8001") as transport:
        adapter = ClipAdapter(transport)
        text = await adapter.embed_text("  a person   walking on a city street  ")
        image = await adapter.embed_image(image_path.read_bytes())
    output_path.write_text(
        json.dumps(
            {
                "text_dimension": len(text),
                "image_dimension": len(image),
                "text_norm": sum(value * value for value in text) ** 0.5,
                "image_norm": sum(value * value for value in image) ** 0.5,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _main() -> int:
    anyio.run(_run, Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
