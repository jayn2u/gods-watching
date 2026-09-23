from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import sqrt
from typing import TYPE_CHECKING, Final, final, override
from uuid import UUID, uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import delete

from gods_watching.contracts.appearances import AppearancePublication, BoundingBox
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.contracts.search import (
    BrowseSearchRequest,
    SimilarSearchRequest,
    TextSearchRequest,
)
from gods_watching.search import (
    AppearanceLookupService,
    CropChangedDuringReadError,
    CropUnavailableError,
    SearchInferenceUnavailableError,
    SearchRepository,
    SearchSeedNotFoundError,
    SearchService,
    UnknownCameraError,
)
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository
from gods_watching.storage.models import Appearance, Camera, CameraSession, CropGarbage

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import AnyUrl
    from sqlalchemy.ext.asyncio import AsyncSession


_MODEL_REVISION: Final = "fixture-search-revision"
_OBSERVED_AT: Final = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def _source(name: str) -> AnyUrl:
    return CameraCreateRequest.model_validate(
        {"name": name, "source_url": f"rtsp://fixture:8554/{name.lower()}"}
    ).source_url


def _vector(*coordinates: tuple[int, float]) -> tuple[float, ...]:
    values = [0.0] * 512
    norm = sqrt(sum(value * value for _, value in coordinates))
    for index, value in coordinates:
        values[index] = value / norm
    return tuple(values)


def _publication(  # noqa: PLR0913
    *,
    appearance_id: UUID,
    camera_id: UUID,
    session_id: UUID,
    track_id: int,
    first_seen: datetime,
    embedding: tuple[float, ...],
    model_revision: str = _MODEL_REVISION,
    crop_object_key: str | None = None,
    representative_version: int = 1,
) -> AppearancePublication:
    return AppearancePublication(
        appearance_id=AppearanceId(appearance_id),
        camera_id=CameraId(camera_id),
        session_id=CameraSessionId(session_id),
        track_id=track_id,
        first_seen=first_seen,
        last_seen=first_seen + timedelta(minutes=5),
        ended_at=first_seen + timedelta(minutes=5),
        representative_version=representative_version,
        crop_object_key=crop_object_key or f"aa/bb/{appearance_id}.jpg",
        bounding_box=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.91,
        crop_quality=42.5,
        byte_size=13,
        embedded_at=first_seen,
        model_id="fixture/clip",
        model_revision=model_revision,
        embedding=embedding,
    )


@pytest.mark.anyio
async def test_postgres_search_returns_only_visible_rows_with_filtered_cosine_ranking(
    session: AsyncSession,
) -> None:
    # Given: committed vectors spanning active, archived, vectorless, tombstoned, and old revisions
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    primary = await storage.add_camera(
        session,
        name="Search primary",
        source_url=_source("search-primary"),
    )
    archived = await storage.add_camera(
        session,
        name="Search archived",
        source_url=_source("search-archived"),
    )
    archived.deleted_at = _OBSERVED_AT + timedelta(days=1)
    primary_session = await storage.start_camera_session(session, primary.id, cause="fixture")
    archived_session = await storage.start_camera_session(session, archived.id, cause="fixture")

    seed_id = UUID("11111111-1111-4111-8111-111111111111")
    nearest_id = UUID("22222222-2222-4222-8222-222222222222")
    tie_id = UUID("33333333-3333-4333-8333-333333333333")
    vectorless_id = UUID("44444444-4444-4444-8444-444444444444")
    tombstoned_id = UUID("55555555-5555-4555-8555-555555555555")
    archived_id = UUID("66666666-6666-4666-8666-666666666666")
    old_revision_id = UUID("77777777-7777-4777-8777-777777777777")

    seed = await storage.publish_appearance(
        session,
        _publication(
            appearance_id=seed_id,
            camera_id=primary.id,
            session_id=primary_session.id,
            track_id=1,
            first_seen=_OBSERVED_AT,
            embedding=_vector((0, 1.0)),
        ),
    )
    nearest = await storage.publish_appearance(
        session,
        _publication(
            appearance_id=nearest_id,
            camera_id=primary.id,
            session_id=primary_session.id,
            track_id=2,
            first_seen=_OBSERVED_AT + timedelta(hours=1),
            embedding=_vector((0, 0.99), (1, 0.14)),
        ),
    )
    tie = await storage.publish_appearance(
        session,
        _publication(
            appearance_id=tie_id,
            camera_id=primary.id,
            session_id=primary_session.id,
            track_id=3,
            first_seen=_OBSERVED_AT + timedelta(hours=2),
            embedding=_vector((0, 0.99), (1, 0.14)),
        ),
    )
    vectorless = await storage.publish_appearance(
        session,
        _publication(
            appearance_id=vectorless_id,
            camera_id=primary.id,
            session_id=primary_session.id,
            track_id=4,
            first_seen=_OBSERVED_AT + timedelta(hours=3),
            embedding=_vector((0, 1.0)),
        ),
    )
    vectorless.embedding = None
    tombstoned = await storage.publish_appearance(
        session,
        _publication(
            appearance_id=tombstoned_id,
            camera_id=primary.id,
            session_id=primary_session.id,
            track_id=5,
            first_seen=_OBSERVED_AT + timedelta(hours=4),
            embedding=_vector((0, 1.0)),
        ),
    )
    await storage.tombstone_appearance(session, tombstoned.id)
    archived_appearance = await storage.publish_appearance(
        session,
        _publication(
            appearance_id=archived_id,
            camera_id=archived.id,
            session_id=archived_session.id,
            track_id=1,
            first_seen=_OBSERVED_AT + timedelta(hours=5),
            embedding=_vector((0, 0.2), (1, 0.98)),
        ),
    )
    old_revision = await storage.publish_appearance(
        session,
        _publication(
            appearance_id=old_revision_id,
            camera_id=primary.id,
            session_id=primary_session.id,
            track_id=6,
            first_seen=_OBSERVED_AT + timedelta(hours=6),
            embedding=_vector((0, 1.0)),
            model_revision="old-fixture-revision",
        ),
    )
    await session.flush()

    service = SearchService(
        SearchRepository(),
        model_id="fixture/clip",
        model_revision=_MODEL_REVISION,
    )

    # When: browse is filtered to the primary camera and an inclusive interval
    browse = await service.search(
        session,
        BrowseSearchRequest.model_validate(
            {
                "mode": "browse",
                "camera_ids": (CameraId(primary.id),),
                "from": _OBSERVED_AT + timedelta(hours=1, minutes=5),
                "to": _OBSERVED_AT + timedelta(hours=3),
                "limit": 10,
            }
        ),
    )

    # Then: newest-first browse excludes vectorless and tombstoned representatives
    assert [result.appearance_id for result in browse.results] == [
        AppearanceId(tie.id),
        AppearanceId(nearest.id),
    ]
    assert all(result.similarity is None for result in browse.results)

    archived_browse = await service.search(
        session,
        BrowseSearchRequest(mode="browse", limit=10),
    )
    assert AppearanceId(archived_appearance.id) in {
        result.appearance_id for result in archived_browse.results
    }

    # When: similarity uses the stored seed vector and has no text inference dependency
    similar = await service.search(
        session,
        SimilarSearchRequest(
            mode="similar",
            appearance_id=AppearanceId(seed.id),
            camera_ids=(CameraId(primary.id),),
            limit=10,
        ),
    )

    # Then: same-revision vectors rank by cosine, break equal scores by UUID, and exclude the seed
    assert [result.appearance_id for result in similar.results] == [
        AppearanceId(nearest.id),
        AppearanceId(tie.id),
    ]
    assert all(result.appearance_id != AppearanceId(seed.id) for result in similar.results)
    assert all(result.model_revision == _MODEL_REVISION for result in similar.results)
    assert all(result.similarity is not None for result in similar.results)

    # When: text retrieval is supplied the same fixture vector
    class _FixtureTextTransport:
        async def embed_text(self, text: str) -> tuple[float, ...]:
            del text
            return _vector((0, 1.0))

    text_service = SearchService(
        SearchRepository(),
        _FixtureTextTransport(),
        model_id="fixture/clip",
        model_revision=_MODEL_REVISION,
    )
    text_results = await text_service.search(
        session,
        TextSearchRequest(mode="text", query="person near entrance", limit=2),
    )

    # Then: text ranking follows the same exact filtered result set and omits stale revisions
    assert [result.appearance_id for result in text_results.results] == [
        AppearanceId(seed.id),
        AppearanceId(nearest.id),
    ]
    assert AppearanceId(archived_appearance.id) not in {
        result.appearance_id for result in text_results.results
    }
    assert AppearanceId(old_revision.id) not in {
        result.appearance_id for result in text_results.results
    }

    # Then: typed failure boundaries remain distinct from an empty result set
    with pytest.raises(UnknownCameraError):
        _ = await service.search(
            session,
            BrowseSearchRequest(
                mode="browse",
                camera_ids=(CameraId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")),),
            ),
        )
    for missing_seed in (
        vectorless.id,
        tombstoned.id,
        UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
    ):
        with pytest.raises(SearchSeedNotFoundError):
            _ = await service.search(
                session,
                SimilarSearchRequest(
                    mode="similar",
                    appearance_id=AppearanceId(missing_seed),
                ),
            )
    with pytest.raises(SearchInferenceUnavailableError):
        _ = await service.search(
            session,
            TextSearchRequest(mode="text", query="person"),
        )


@pytest.mark.anyio
async def test_crop_revalidation_rejects_upgrade_committed_by_independent_session(
    database_url: str,
    tmp_path: Path,
) -> None:
    # Given: a committed version-one crop and a read session that has loaded its ORM identity
    database = Database.connect(database_url)
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    crop_store = CropObjectStore(tmp_path / "crops")
    appearance_id = uuid4()
    first_crop = crop_store.write(b"first")
    second_crop = crop_store.write(b"second")
    camera_id: UUID | None = None
    camera_session_id: UUID | None = None

    try:
        async with database.transaction() as setup:
            camera = await storage.add_camera(
                setup,
                name="Search crop race",
                source_url=_source("search-crop-race"),
            )
            camera_session = await storage.start_camera_session(
                setup,
                camera.id,
                cause="fixture",
            )
            camera_id = camera.id
            camera_session_id = camera_session.id
            _ = await storage.publish_appearance(
                setup,
                _publication(
                    appearance_id=appearance_id,
                    camera_id=camera.id,
                    session_id=camera_session.id,
                    track_id=1,
                    first_seen=_OBSERVED_AT,
                    embedding=_vector((0, 1.0)),
                    crop_object_key=first_crop.object_key,
                ),
            )
        assert camera_id is not None
        assert camera_session_id is not None

        @final
        class RaceRepository(SearchRepository):
            calls: int = 0

            def __init__(self) -> None:
                super().__init__()
                object.__setattr__(self, "calls", 0)

            @override
            async def get_visible(
                self,
                session: AsyncSession,
                appearance_id: UUID,
            ) -> tuple[Appearance, Camera] | None:
                object.__setattr__(self, "calls", self.calls + 1)
                if self.calls == 2:
                    async with database.transaction() as upgrade:
                        _ = await storage.publish_appearance(
                            upgrade,
                            _publication(
                                appearance_id=appearance_id,
                                camera_id=camera_id,
                                session_id=camera_session_id,
                                track_id=1,
                                first_seen=_OBSERVED_AT,
                                embedding=_vector((0, 1.0)),
                                crop_object_key=second_crop.object_key,
                                representative_version=2,
                            ),
                        )
                return await super().get_visible(session, appearance_id)

        service = AppearanceLookupService(RaceRepository(), crop_store)

        # When: the independent session commits version two before the recheck
        with pytest.raises(CropChangedDuringReadError) as raised:
            async with database.transaction() as read:
                _ = await service.get_crop(read, appearance_id)

        # Then: no obsolete bytes cross the version fence
        assert raised.value.representative_version == 1
    finally:
        async with database.transaction() as cleanup:
            _ = await cleanup.execute(
                delete(CropGarbage).where(
                    CropGarbage.object_key.in_((first_crop.object_key, second_crop.object_key))
                )
            )
            _ = await cleanup.execute(delete(Appearance).where(Appearance.id == appearance_id))
            if camera_session_id is not None:
                _ = await cleanup.execute(
                    delete(CameraSession).where(CameraSession.id == camera_session_id)
                )
            if camera_id is not None:
                _ = await cleanup.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_search_error_crosses_database_transaction_as_typed_error(
    database_url: str,
    tmp_path: Path,
) -> None:
    # Given: a lookup for an appearance absent from durable history
    database = Database.connect(database_url)
    try:
        # When: the typed service failure leaves the transaction context
        with pytest.raises(CropUnavailableError):
            async with database.transaction() as session:
                _ = await AppearanceLookupService(
                    SearchRepository(), CropObjectStore(tmp_path / "crops")
                ).get_crop(session, UUID("99999999-9999-4999-8999-999999999999"))

        # Then: transaction unwinding preserves the declared error type
    finally:
        await database.close()
