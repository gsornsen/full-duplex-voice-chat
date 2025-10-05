"""LiveKit room management for orchestrator sessions.

Handles room creation, JWT token generation, and lifecycle management
for LiveKit-based WebRTC sessions.
"""

import secrets
import time
from datetime import timedelta

import aiohttp
from livekit import api
from livekit.api import AccessToken, VideoGrants
from livekit.api.room_service import RoomService

from src.orchestrator.config import LiveKitConfig


class LiveKitRoomManager:
    """Manages LiveKit room lifecycle and authentication.

    Creates rooms with auto-generated names, generates JWT tokens for client
    authentication, and handles room cleanup on session termination.
    """

    def __init__(self, config: LiveKitConfig) -> None:
        """Initialize room manager with LiveKit configuration.

        Args:
            config: LiveKit server configuration
        """
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._room_service: RoomService | None = None

    async def _ensure_session(self) -> RoomService:
        """Ensure aiohttp session and room service are initialized.

        Returns:
            RoomService instance
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        if self._room_service is None:
            self._room_service = RoomService(
                self._session,
                self.config.url,
                self.config.api_key,
                self.config.api_secret,
            )

        return self._room_service

    async def close(self) -> None:
        """Close aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
            self._room_service = None

    def generate_room_name(self) -> str:
        """Generate unique room name with timestamp and random suffix.

        Returns:
            Room name in format: {prefix}-{timestamp}-{random}
        """
        timestamp = int(time.time())
        random_suffix = secrets.token_hex(4)
        return f"{self.config.room_prefix}-{timestamp}-{random_suffix}"

    def create_access_token(
        self, room_name: str, participant_identity: str, ttl_hours: int = 24
    ) -> str:
        """Generate JWT access token for client authentication.

        Args:
            room_name: Name of the room to grant access to
            participant_identity: Unique identifier for the participant
            ttl_hours: Token validity duration in hours

        Returns:
            JWT token string for client authentication
        """
        token = AccessToken(self.config.api_key, self.config.api_secret)
        token.with_identity(participant_identity)
        token.with_name(participant_identity)
        token.with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        # Set token expiration
        token.with_ttl(timedelta(hours=ttl_hours))

        return token.to_jwt()

    async def create_room(
        self, room_name: str | None = None, empty_timeout_seconds: int = 300
    ) -> str:
        """Create a new LiveKit room.

        Args:
            room_name: Optional room name (auto-generated if not provided)
            empty_timeout_seconds: Seconds before empty room is automatically closed

        Returns:
            Name of the created room

        Raises:
            RuntimeError: If room creation fails
        """
        if room_name is None:
            room_name = self.generate_room_name()

        try:
            service = await self._ensure_session()
            await service.create_room(
                api.CreateRoomRequest(
                    name=room_name,
                    empty_timeout=empty_timeout_seconds,
                )
            )
            return room_name

        except Exception as e:
            raise RuntimeError(f"Failed to create LiveKit room '{room_name}': {e}") from e

    async def delete_room(self, room_name: str) -> None:
        """Delete a LiveKit room.

        Args:
            room_name: Name of the room to delete

        Raises:
            RuntimeError: If room deletion fails
        """
        try:
            service = await self._ensure_session()
            await service.delete_room(
                api.DeleteRoomRequest(room=room_name)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to delete LiveKit room '{room_name}': {e}") from e

    async def list_rooms(self) -> list[api.Room]:
        """List all active rooms.

        Returns:
            List of active room objects

        Raises:
            RuntimeError: If listing rooms fails
        """
        try:
            service = await self._ensure_session()
            response = await service.list_rooms(api.ListRoomsRequest())
            return list(response.rooms)
        except Exception as e:
            raise RuntimeError(f"Failed to list LiveKit rooms: {e}") from e

    async def get_room(self, room_name: str) -> api.Room | None:
        """Get room information.

        Args:
            room_name: Name of the room

        Returns:
            Room object if exists, None otherwise
        """
        try:
            rooms = await self.list_rooms()
            for room in rooms:
                if room.name == room_name:
                    return room
            return None
        except Exception:
            return None
