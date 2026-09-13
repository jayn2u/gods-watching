"""Typed evidence emitted by the Task 14 search API driver."""

from typing import Literal

from gods_watching.verification.models import VerificationModel


class SearchEvidence(VerificationModel):
    """Record authenticated search behavior over real pipeline-published appearances."""

    unauthenticated_status: int
    cross_origin_status: int
    browse_results: int
    browse_camera_filter_only: bool
    browse_time_filter_only: bool
    text_status: int
    text_results: int
    text_ranked_desc: bool
    text_stable: bool
    similar_status: int
    similar_results: int
    similar_excludes_seed: bool
    similar_ranked_desc: bool
    similar_stable: bool
    detail_status: int
    crop_status: int
    crop_jpeg: bool
    crop_no_store: bool


class SearchErrorsEvidence(VerificationModel):
    """Record rejected input, expired references, and inference outage isolation."""

    blank_text_status: int
    unknown_camera_status: int
    expired_seed_status: int
    expired_detail_status: int
    expired_crop_status: int
    outage_text_status: int
    outage_browse_status: int
    outage_similar_status: int


class Task14DriverEvidence(VerificationModel):
    """Combine the observations for one Task 14 scenario mode."""

    mode: Literal["search", "search-errors"]
    search: SearchEvidence | None = None
    errors: SearchErrorsEvidence | None = None


__all__ = ["SearchErrorsEvidence", "SearchEvidence", "Task14DriverEvidence"]
