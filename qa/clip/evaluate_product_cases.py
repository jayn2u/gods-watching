"""Evaluate fixed real-crop retrieval cases against built-in and imported CLIP."""

# ruff: noqa: INP001, C901, PLR0912, PLR0915, TRY003, EM101, EM102, T201, PLR2004

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases(path: Path, expected_hash: str) -> dict:
    """Validate every declared case before loading either model."""
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError("fixed product cases missing or hash mismatch")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, dict) or set(cases) != {
        "appearances",
        "text_queries",
        "image_queries",
    }:
        raise ValueError("invalid product case structure")
    appearances = cases["appearances"]
    text_queries = cases["text_queries"]
    image_queries = cases["image_queries"]
    if not all(isinstance(rows, list) for rows in (appearances, text_queries, image_queries)):
        raise ValueError("invalid product case rows")
    if len(appearances) < 40 or len(text_queries) < 20 or len(image_queries) < 20:
        raise ValueError("insufficient real-crop cases")
    appearance_ids: set[str] = set()
    gallery_hashes: set[str] = set()
    scenes: set[str] = set()
    for row in appearances:
        if not isinstance(row, dict) or set(row) != {"id", "crop", "scene", "sha256"}:
            raise ValueError("invalid appearance case")
        if not all(isinstance(row[key], str) and row[key] for key in row):
            raise ValueError("invalid appearance metadata")
        if row["id"] in appearance_ids:
            raise ValueError("duplicate appearance ID")
        crop = Path(row["crop"])
        if crop.is_absolute() or ".." in crop.parts or not (path.parent / crop).is_file():
            raise ValueError("real crop missing")
        if _sha256(path.parent / crop) != row["sha256"]:
            raise ValueError("real crop hash mismatch")
        appearance_ids.add(row["id"])
        gallery_hashes.add(row["sha256"])
        scenes.add(row["scene"])
    if len(scenes) < 2:
        raise ValueError("at least two real scenes required")
    for kind, rows, required in (
        ("text", text_queries, {"id", "text", "relevant_ids"}),
        ("image", image_queries, {"id", "crop", "sha256", "relevant_ids"}),
    ):
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError(f"invalid {kind} query")
            if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
                raise ValueError(f"duplicate or missing {kind} query ID")
            seen.add(row["id"])
            relevant = row["relevant_ids"]
            if (
                not isinstance(relevant, list)
                or not relevant
                or any(not isinstance(item, str) for item in relevant)
                or len(relevant) != len(set(relevant))
                or not set(relevant) <= appearance_ids
            ):
                raise ValueError("invalid predeclared relevance set")
            if kind == "image":
                if not isinstance(row["crop"], str) or not isinstance(row["sha256"], str):
                    raise ValueError("invalid held-out image query metadata")
                crop = Path(row["crop"])
                if crop.is_absolute() or ".." in crop.parts or not (path.parent / crop).is_file():
                    raise ValueError("held-out image query crop missing")
                if _sha256(path.parent / crop) != row["sha256"]:
                    raise ValueError("held-out image query crop hash mismatch")
                if row["sha256"] in gallery_hashes:
                    raise ValueError("image query crop must be held out from gallery")
            if kind == "text" and (not isinstance(row["text"], str) or not row["text"].strip()):
                raise ValueError("missing English query")
    return cases


def _evaluate(model_dir: Path, cases: dict, case_dir: Path) -> tuple[float, float]:
    """Run both modalities through one locally pinned CLIP pair."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for product evaluation")
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    processor = CLIPProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = model.eval().cuda()
    gallery = cases["appearances"]
    ids = [row["id"] for row in gallery]
    with torch.inference_mode():
        images = [Image.open(case_dir / row["crop"]).convert("RGB") for row in gallery]
        encoded = processor(images=images, return_tensors="pt", padding=True)
        gallery_vectors = model.get_image_features(**{k: v.cuda() for k, v in encoded.items()})
        gallery_vectors = torch.nn.functional.normalize(gallery_vectors, dim=-1)
        text_scores = []
        for row in cases["text_queries"]:
            encoded = processor(text=[row["text"]], return_tensors="pt", padding=True)
            query = model.get_text_features(**{k: v.cuda() for k, v in encoded.items()})
            query = torch.nn.functional.normalize(query, dim=-1)
            ranked = torch.topk(query @ gallery_vectors.T, 5).indices[0].tolist()
            text_scores.append(
                len({ids[index] for index in ranked} & set(row["relevant_ids"]))
                / len(row["relevant_ids"])
            )
        image_scores = []
        for row in cases["image_queries"]:
            with Image.open(case_dir / row["crop"]) as query_image:
                encoded = processor(images=[query_image.convert("RGB")], return_tensors="pt")
            query = model.get_image_features(**{k: v.cuda() for k, v in encoded.items()})
            query = torch.nn.functional.normalize(query, dim=-1)
            scores = (query @ gallery_vectors.T)[0]
            ranked = torch.topk(scores, 5).indices.tolist()
            image_scores.append(
                len({ids[index] for index in ranked} & set(row["relevant_ids"]))
                / len(row["relevant_ids"])
            )
    return sum(text_scores) / len(text_scores), sum(image_scores) / len(image_scores)


def main() -> int:
    """Run both pinned models and write a content-bound evidence record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("qa/retrieval-cases.json"))
    parser.add_argument("--policy", type=Path, default=Path("assets/retrieval-quality-policy.json"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from gods_watching.model_selection.quality import load_quality_policy  # noqa: PLC0415

    policy = load_quality_policy(args.policy)
    if policy is None:
        parser.error("trusted quality policy missing or invalid")
    if _sha256(Path(__file__)) != policy.evaluator_revision:
        parser.error("evaluator revision does not match trusted policy")
    if args.output.resolve().is_relative_to(args.manifest.parent.resolve()):
        parser.error("quality evidence must be outside immutable model package")
    if args.candidate.resolve() != args.manifest.parent.resolve():
        parser.error("candidate model path must match installed manifest directory")
    try:
        cases = _cases(args.cases, policy.product_cases_sha256)
        from gods_watching.model_selection.registry import _load_installed_manifest  # noqa: PLC0415

        manifest = _load_installed_manifest(args.manifest.parent)
        if args.output.name != f"{manifest['package_sha256']}.json":
            parser.error("evidence filename must match candidate package hash")
        report = (args.manifest.parent / manifest["cuhk_report"]).read_bytes().decode("utf-8")
        baseline_text, baseline_image = _evaluate(args.baseline, cases, args.cases.parent)
        candidate_text, candidate_image = _evaluate(args.candidate, cases, args.cases.parent)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    evidence = {
        "package_sha256": manifest["package_sha256"],
        "baseline_model_id": policy.baseline_model_id,
        "baseline_revision": policy.baseline_revision,
        "cuhk_report_json": report,
        "product_cases_sha256": policy.product_cases_sha256,
        "text_baseline_recall_at_5": baseline_text,
        "text_candidate_recall_at_5": candidate_text,
        "image_baseline_recall_at_5": baseline_image,
        "image_candidate_recall_at_5": candidate_image,
        "evaluator_revision": policy.evaluator_revision,
        "created_at": datetime.now(UTC).isoformat(),
    }
    from gods_watching.model_selection.imported_manifest import (  # noqa: PLC0415
        ImportedClipManifest,
        ImportedFile,
    )
    from gods_watching.model_selection.quality import (  # noqa: PLC0415
        QualityEvidence,
        assess_quality,
    )

    imported = ImportedClipManifest(
        model_id=manifest["model_id"],
        revision=manifest["revision"],
        display_name=manifest["display_name"],
        base_model_id=manifest["base_model_id"],
        dimension=manifest["dimension"],
        files=tuple(ImportedFile(**item) for item in manifest["files"]),
        package_sha256=manifest["package_sha256"],
        cuhk_report=manifest["cuhk_report"],
    )
    status = assess_quality(imported, QualityEvidence.parse(evidence), policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        f"CUHK: {json.loads(report)['baseline_score']} -> {json.loads(report)['candidate_score']}"
    )
    print(f"Product text Recall@5: {baseline_text:.3f} -> {candidate_text:.3f}")
    print(f"Product image Recall@5: {baseline_image:.3f} -> {candidate_image:.3f}")
    print(f"Quality eligibility: {'passed' if status.passed else status.reason}")
    return 0 if status.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
