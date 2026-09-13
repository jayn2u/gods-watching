"""Read the committed camera sessions the pipeline worker should be running."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.cameras import CameraGenerationId, CameraRepository
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.storage import Camera, CameraSession

from .reconcile import DesiredCamera


async def load_desired_cameras(
    session: AsyncSession,
    repository: CameraRepository,
) -> dict[CameraId, DesiredCamera]:
    """Return undeleted detection-enabled cameras with their newest active session."""
    statement = (
        select(Camera, CameraSession)
        .join(CameraSession, CameraSession.camera_id == Camera.id)
        .where(
            Camera.deleted_at.is_(None),
            Camera.detection_enabled.is_(True),
            CameraSession.ended_at.is_(None),
        )
        .order_by(Camera.id, CameraSession.started_at.desc(), CameraSession.id.desc())
    )
    desired: dict[CameraId, DesiredCamera] = {}
    for camera, camera_session in (await session.execute(statement)).tuples().all():
        camera_id = CameraId(camera.id)
        if camera_id in desired:
            continue
        desired[camera_id] = DesiredCamera(
            camera_id=camera_id,
            version=camera.version,
            session_id=CameraSessionId(camera_session.id),
            generation_id=CameraGenerationId(camera_session.generation_id),
            source_url=await repository.source(session, camera),
            threshold=camera.detection_threshold,
        )
    return desired


__all__ = ["load_desired_cameras"]
