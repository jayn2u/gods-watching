"""Build binary checks from typed Task 14 driver evidence."""

from typing import Final, Literal

from gods_watching.verification.models import Check

from .task14_models import SearchErrorsEvidence, SearchEvidence, Task14DriverEvidence

_MISSING: Final = "driver produced no observations for this mode"
_OK: Final = 200
_UNAUTHORIZED: Final = 401
_FORBIDDEN: Final = 403
_NOT_FOUND: Final = 404
_UNPROCESSABLE: Final = 422
_UNAVAILABLE: Final = 503


def build_checks(
    mode: Literal["search", "search-errors"],
    evidence: Task14DriverEvidence,
    *,
    cleanup_succeeded: bool,
) -> tuple[Check, ...]:
    """Require every search or error-path observable plus exact cleanup."""
    behavior = (
        _search_checks(evidence.search) if mode == "search" else _error_checks(evidence.errors)
    )
    return (
        *behavior,
        Check(
            name="owned-resource-cleanup",
            passed=cleanup_succeeded,
            detail=f"cleanup_succeeded={cleanup_succeeded}",
        ),
    )


def _search_checks(search: SearchEvidence | None) -> tuple[Check, ...]:
    names = (
        "authorization-required",
        "cross-origin-search-rejected",
        "filtered-browse-eligible-only",
        "text-search-ranked-and-stable",
        "similar-search-ranked-stable-excludes-seed",
        "detail-and-private-crop",
    )
    if search is None:
        return tuple(Check(name=name, passed=False, detail=_MISSING) for name in names)
    observed = (
        (
            search.unauthenticated_status == _UNAUTHORIZED,
            f"unauthenticated_status={search.unauthenticated_status}",
        ),
        (
            search.cross_origin_status == _FORBIDDEN,
            f"cross_origin_status={search.cross_origin_status}",
        ),
        (
            search.browse_results > 0
            and search.browse_camera_filter_only
            and search.browse_time_filter_only,
            (
                f"results={search.browse_results}; "
                f"camera_only={search.browse_camera_filter_only}; "
                f"time_only={search.browse_time_filter_only}"
            ),
        ),
        (
            search.text_status == _OK
            and search.text_results > 0
            and search.text_ranked_desc
            and search.text_stable,
            (
                f"status={search.text_status}; results={search.text_results}; "
                f"ranked_desc={search.text_ranked_desc}; stable={search.text_stable}"
            ),
        ),
        (
            search.similar_status == _OK
            and search.similar_results > 0
            and search.similar_excludes_seed
            and search.similar_ranked_desc
            and search.similar_stable,
            (
                f"status={search.similar_status}; results={search.similar_results}; "
                f"excludes_seed={search.similar_excludes_seed}; "
                f"ranked_desc={search.similar_ranked_desc}; stable={search.similar_stable}"
            ),
        ),
        (
            search.detail_status == _OK
            and search.crop_status == _OK
            and search.crop_jpeg
            and search.crop_no_store,
            (
                f"detail={search.detail_status}; crop={search.crop_status}; "
                f"jpeg={search.crop_jpeg}; no_store={search.crop_no_store}"
            ),
        ),
    )
    return tuple(
        Check(name=name, passed=passed, detail=detail)
        for name, (passed, detail) in zip(names, observed, strict=True)
    )


def _error_checks(errors: SearchErrorsEvidence | None) -> tuple[Check, ...]:
    names = (
        "invalid-input-rejected",
        "expired-seed-and-crop-not-served",
        "text-inference-outage-reported",
        "vector-search-survives-inference-outage",
    )
    if errors is None:
        return tuple(Check(name=name, passed=False, detail=_MISSING) for name in names)
    observed = (
        (
            errors.blank_text_status == _UNPROCESSABLE
            and errors.unknown_camera_status == _UNPROCESSABLE,
            (
                f"blank_text={errors.blank_text_status}; "
                f"unknown_camera={errors.unknown_camera_status}"
            ),
        ),
        (
            errors.expired_seed_status == _NOT_FOUND
            and errors.expired_detail_status == _NOT_FOUND
            and errors.expired_crop_status == _NOT_FOUND,
            (
                f"seed={errors.expired_seed_status}; detail={errors.expired_detail_status}; "
                f"crop={errors.expired_crop_status}"
            ),
        ),
        (
            errors.outage_text_status == _UNAVAILABLE,
            f"outage_text={errors.outage_text_status}",
        ),
        (
            errors.outage_browse_status == _OK and errors.outage_similar_status == _OK,
            (
                f"outage_browse={errors.outage_browse_status}; "
                f"outage_similar={errors.outage_similar_status}"
            ),
        ),
    )
    return tuple(
        Check(name=name, passed=passed, detail=detail)
        for name, (passed, detail) in zip(names, observed, strict=True)
    )


__all__ = ["build_checks"]
