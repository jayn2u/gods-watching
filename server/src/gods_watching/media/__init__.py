"""Recording-free RTSP control and browser WHEP transport."""

from .control import MediaControlAdapter, MediaControlGateway, camera_path
from .errors import InvalidGatewayLocationError
from .generation import SourceGenerationCoordinator
from .http_gateway import (
    HttpMediaControlGateway,
    HttpWhepGateway,
    MediaControlRequestError,
    MediaGatewayConnection,
)
from .models import (
    AuthorizedMediaSession,
    GatewayResponse,
    GenerationEvent,
    GenerationEventKind,
    MediaPath,
    MediaSessionId,
    ProxyResponse,
    SourceGenerationId,
    WhepResourceId,
)
from .proxy import WhepGateway, WhepProxyService
from .routes import DenyAllMediaSessionAuthorizer, MediaSessionAuthorizer, build_whep_router

__all__ = [
    "AuthorizedMediaSession",
    "DenyAllMediaSessionAuthorizer",
    "GatewayResponse",
    "GenerationEvent",
    "GenerationEventKind",
    "HttpMediaControlGateway",
    "HttpWhepGateway",
    "InvalidGatewayLocationError",
    "MediaControlAdapter",
    "MediaControlGateway",
    "MediaControlRequestError",
    "MediaGatewayConnection",
    "MediaPath",
    "MediaSessionAuthorizer",
    "MediaSessionId",
    "ProxyResponse",
    "SourceGenerationCoordinator",
    "SourceGenerationId",
    "WhepGateway",
    "WhepProxyService",
    "WhepResourceId",
    "build_whep_router",
    "camera_path",
]
