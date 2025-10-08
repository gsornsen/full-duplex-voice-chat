"""LiveKit transport implementation for WebRTC connections.

Provides LiveKit/WebRTC-based client connections for browser clients,
with audio track publishing and data channel for text messaging.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

import numpy as np
from livekit import rtc

from src.orchestrator.audio.packetizer import (
    SAMPLE_RATE_HZ,
    validate_frame_size,
)
from src.orchestrator.config import LiveKitConfig
from src.orchestrator.livekit_utils.room_manager import LiveKitRoomManager
from src.orchestrator.transport.base import Transport, TransportSession

logger = logging.getLogger(__name__)


class LiveKitSession(TransportSession):
    """LiveKit-based transport session.

    Implements the TransportSession interface for LiveKit/WebRTC connections,
    handling audio track publishing and data channel text messaging.
    """

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant,
        session_id: str,
        room_name: str,
    ) -> None:
        """Initialize LiveKit session.

        Args:
            room: LiveKit room instance
            participant: Remote participant object
            session_id: Unique session identifier
            room_name: LiveKit room name
        """
        self._room = room
        self._participant = participant
        self._session_id = session_id
        self._room_name = room_name
        self._connected = True

        # Audio track setup
        self._audio_source: rtc.AudioSource | None = None
        self._audio_track: rtc.LocalAudioTrack | None = None
        self._sequence_number = 0

        # Text message queue from data channel
        self._text_queue: asyncio.Queue[str] = asyncio.Queue()

        # Track disconnection
        self._disconnect_event = asyncio.Event()

        logger.info(
            "LiveKit session initialized",
            extra={
                "session_id": session_id,
                "room": room_name,
                "participant": participant.identity,
            },
        )

    @property
    def session_id(self) -> str:
        """Get unique session identifier."""
        return self._session_id

    @property
    def is_connected(self) -> bool:
        """Check if the session connection is still active."""
        return (
            self._connected
            and self._room.connection_state == rtc.ConnectionState.CONN_CONNECTED
        )

    async def initialize_audio_track(self) -> None:
        """Initialize audio source and track for publishing.

        Must be called before sending audio frames.
        """
        # Create audio source: 48kHz mono
        self._audio_source = rtc.AudioSource(SAMPLE_RATE_HZ, num_channels=1)

        # Create local audio track
        self._audio_track = rtc.LocalAudioTrack.create_audio_track(
            "orchestrator-audio",
            self._audio_source,
        )

        # Publish track to room
        await self._room.local_participant.publish_track(
            self._audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        logger.info(
            "Audio track published",
            extra={"session_id": self._session_id, "room": self._room_name},
        )

    async def send_audio_frame(self, frame: bytes) -> None:
        """Send a 20ms PCM audio frame to the client.

        Args:
            frame: PCM audio data (960 samples @ 48kHz, 16-bit LE)
                  Expected size: 1920 bytes (960 samples * 2 bytes)

        Raises:
            ConnectionError: If the connection is closed or broken
            ValueError: If frame size is invalid
            RuntimeError: If audio track not initialized
        """
        if not self.is_connected:
            raise ConnectionError("LiveKit connection is closed")

        if self._audio_source is None:
            raise RuntimeError("Audio track not initialized. Call initialize_audio_track() first.")

        # Validate frame size
        validate_frame_size(frame)

        try:
            # Convert bytes to AudioFrame
            # Frame format: 960 samples, 48kHz, mono, 16-bit LE
            num_samples = len(frame) // 2  # 2 bytes per sample

            # Create rtc.AudioFrame
            audio_frame = rtc.AudioFrame.create(
                sample_rate=SAMPLE_RATE_HZ,
                num_channels=1,
                samples_per_channel=num_samples,
            )

            # Convert PCM bytes to int16 numpy array
            pcm_data = np.frombuffer(frame, dtype=np.int16)

            # Copy data to audio frame
            # AudioFrame.data is a memoryview that we can write to
            np.copyto(np.asarray(audio_frame.data), pcm_data)

            # Capture frame to audio source
            await self._audio_source.capture_frame(audio_frame)

            self._sequence_number += 1

            logger.debug(
                "Audio frame sent",
                extra={
                    "session_id": self._session_id,
                    "sequence": self._sequence_number,
                    "size": len(frame),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to send audio frame",
                extra={"session_id": self._session_id, "error": str(e)},
            )
            raise

    async def receive_text(self) -> AsyncIterator[str]:
        """Receive text chunks from the client via data channel.

        Yields text chunks as they arrive from the client data channel.

        Yields:
            str: Text chunk from client

        Raises:
            ConnectionError: If the connection is closed or broken
        """
        try:
            while self.is_connected:
                try:
                    # Wait for text with timeout to check connection periodically
                    text = await asyncio.wait_for(
                        self._text_queue.get(),
                        timeout=1.0,
                    )
                    logger.debug(
                        "Text received from data channel",
                        extra={
                            "session_id": self._session_id,
                            "text_length": len(text),
                        },
                    )
                    yield text

                except TimeoutError:
                    # Check if still connected
                    if not self.is_connected:
                        break
                    continue

        except Exception as e:
            logger.error(
                "Error in receive_text",
                extra={"session_id": self._session_id, "error": str(e)},
            )
            raise ConnectionError(f"LiveKit receive error: {e}") from e
        finally:
            logger.info(
                "LiveKit text reception ended",
                extra={"session_id": self._session_id},
            )

    def on_data_received(self, data: bytes, participant: rtc.RemoteParticipant) -> None:
        """Handle data channel message from participant.

        Called by LiveKit when data is received on the data channel.

        Args:
            data: Raw message data
            participant: Participant who sent the message
        """
        try:
            # Decode text message
            text = data.decode("utf-8")

            # Queue for receive_text generator
            self._text_queue.put_nowait(text)

            logger.debug(
                "Data received",
                extra={
                    "session_id": self._session_id,
                    "participant": participant.identity,
                    "size": len(data),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to process data channel message",
                extra={"session_id": self._session_id, "error": str(e)},
            )

    async def close(self) -> None:
        """Clean session shutdown.

        Unpublishes audio track, disconnects from room, and releases resources.
        """
        if not self._connected:
            return

        logger.info(
            "Closing LiveKit session",
            extra={"session_id": self._session_id, "room": self._room_name},
        )

        try:
            # Unpublish audio track
            if self._audio_track is not None:
                await self._room.local_participant.unpublish_track(
                    self._audio_track.sid
                )

            # Disconnect from room
            await self._room.disconnect()

        except Exception as e:
            logger.warning(
                "Error during session close",
                extra={"session_id": self._session_id, "error": str(e)},
            )
        finally:
            self._connected = False
            self._disconnect_event.set()


class LiveKitTransport(Transport):
    """LiveKit transport server.

    Manages LiveKit room lifecycle and creates LiveKitSession instances
    for incoming participant connections.
    """

    def __init__(self, config: LiveKitConfig) -> None:
        """Initialize LiveKit transport.

        Args:
            config: LiveKit server configuration
        """
        self._config = config
        self._room_manager = LiveKitRoomManager(config)
        self._running = False

        # Session queue for accept_session()
        self._session_queue: asyncio.Queue[LiveKitSession] = asyncio.Queue()

        # Active rooms and sessions
        self._active_rooms: dict[str, rtc.Room] = {}
        self._active_sessions: dict[str, LiveKitSession] = {}
        
        # Room monitoring
        self._monitor_task: asyncio.Task | None = None
        self._monitored_rooms: set[str] = set()

        logger.info(
            "LiveKit transport initialized",
            extra={"url": config.url, "room_prefix": config.room_prefix},
        )

    @property
    def transport_type(self) -> str:
        """Transport type identifier."""
        return "livekit"

    @property
    def is_running(self) -> bool:
        """Check if the transport server is currently running."""
        return self._running

    async def start(self) -> None:
        """Start the LiveKit transport.

        Prepares the transport to accept connections and starts monitoring
        for new rooms to join as an agent.

        Raises:
            RuntimeError: If the transport fails to start
        """
        if self._running:
            raise RuntimeError("LiveKit transport is already running")

        logger.info(
            "Starting LiveKit transport",
            extra={"url": self._config.url},
        )

        try:
            # Verify LiveKit server is accessible by listing rooms
            await self._room_manager.list_rooms()

            self._running = True

            # Start room monitoring task
            self._monitor_task = asyncio.create_task(self._monitor_rooms())

            logger.info(
                "LiveKit transport started",
                extra={"url": self._config.url},
            )

        except Exception as e:
            logger.error(
                "Failed to start LiveKit transport",
                extra={"error": str(e)},
            )
            raise RuntimeError(f"Failed to start LiveKit transport: {e}") from e

    async def stop(self) -> None:
        """Stop the LiveKit transport.

        Closes all active sessions, disconnects from rooms, and releases resources.
        """
        if not self._running:
            return

        logger.info("Stopping LiveKit transport")

        self._running = False

        # Close all active sessions
        for session in list(self._active_sessions.values()):
            try:
                await session.close()
            except Exception as e:
                logger.warning(
                    "Error closing session during shutdown",
                    extra={"session_id": session.session_id, "error": str(e)},
                )

        # Clean up rooms
        for room_name in list(self._active_rooms.keys()):
            try:
                await self._room_manager.delete_room(room_name)
            except Exception as e:
                logger.warning(
                    "Error deleting room during shutdown",
                    extra={"room": room_name, "error": str(e)},
                )

        # Stop room monitoring
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Close room manager
        await self._room_manager.close()

        self._active_rooms.clear()
        self._active_sessions.clear()
        self._monitored_rooms.clear()

        logger.info("LiveKit transport stopped")

    async def accept_session(self) -> TransportSession:
        """Accept a new client session.

        Creates a new room and waits for a participant to join, then returns
        a LiveKitSession for that participant.

        Returns:
            TransportSession: New client session

        Raises:
            RuntimeError: If the transport is not running
            ConnectionError: If session acceptance fails
        """
        if not self._running:
            raise RuntimeError("LiveKit transport is not running")

        # Wait for next session from the queue
        session = await self._session_queue.get()
        return session

    async def _monitor_rooms(self) -> None:
        """Monitor for new rooms and join them as an agent.
        
        Continuously polls for new rooms and joins them as an agent when
        participants are present.
        """
        logger.info("Starting room monitoring")
        
        while self._running:
            try:
                # List all active rooms
                rooms = await self._room_manager.list_rooms()
                logger.info(f"Found {len(rooms)} rooms during monitoring")
                
                for room in rooms:
                    room_name = room.name
                    logger.info(f"Checking room: {room_name}, participants: {room.num_participants}")
                    
                    # Skip if we're already monitoring this room
                    if room_name in self._monitored_rooms:
                        logger.info(f"Already monitoring room: {room_name}")
                        continue
                    
                    # Skip if room has no participants (except our agent)
                    if room.num_participants == 0:
                        logger.info(f"Room {room_name} has no participants, skipping")
                        continue
                    
                    # Check if room matches our prefix pattern
                    if not room_name.startswith(self._config.room_prefix):
                        logger.info(f"Room {room_name} doesn't match prefix {self._config.room_prefix}")
                        continue
                    
                    logger.info(
                        "Found new room with participants",
                        extra={"room": room_name, "participants": room.num_participants}
                    )
                    
                    # Mark as monitored
                    self._monitored_rooms.add(room_name)
                    
                    # Join room as agent
                    asyncio.create_task(self._join_room_as_agent(room_name))
                
                # Wait before next poll
                await asyncio.sleep(2.0)
                
            except asyncio.CancelledError:
                logger.info("Room monitoring cancelled")
                break
            except Exception as e:
                logger.error("Error in room monitoring", extra={"error": str(e)})
                await asyncio.sleep(5.0)  # Wait longer on error
        
        logger.info("Room monitoring stopped")

    async def _join_room_as_agent(self, room_name: str) -> None:
        """Join a room as an agent and wait for participants.
        
        Args:
            room_name: Name of the room to join
        """
        try:
            logger.info("Joining room as agent", extra={"room": room_name})
            
            # Create room instance
            room = rtc.Room()
            
            # Set up event handlers BEFORE connecting
            participant_event = asyncio.Event()
            participants: list[rtc.RemoteParticipant] = []
            
            def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
                if participant.identity != "agent":  # Don't count our own agent
                    participants.append(participant)
                    participant_event.set()
                    logger.info(
                        "Participant joined room",
                        extra={"room": room_name, "participant": participant.identity}
                    )
            
            # Set up event handler BEFORE connecting
            room.on("participant_connected", on_participant_connected)
            
            # Generate agent token
            agent_token = self._room_manager.create_access_token(
                room_name=room_name,
                participant_identity="agent",
                ttl_hours=1
            )
            
            # Connect to room
            await room.connect(self._config.url, agent_token)
            logger.info("Agent connected to room", extra={"room": room_name})
            
            # Store active room
            self._active_rooms[room_name] = room
            
            # Check if there are already participants in the room
            existing_participants = [p for p in room.remote_participants.values() if p.identity != "agent"]
            if existing_participants:
                logger.info(
                    "Found existing participants in room",
                    extra={"room": room_name, "count": len(existing_participants)}
                )
                participants.extend(existing_participants)
                participant_event.set()
            else:
                # Wait for participants to join with timeout
                try:
                    await asyncio.wait_for(participant_event.wait(), timeout=30.0)
                except TimeoutError:
                    logger.warning("No participants joined room within timeout", extra={"room": room_name})
                    await room.disconnect()
                    return
            
            # Create session for each participant
            for participant in participants:
                session_id = f"lk-{uuid.uuid4().hex[:12]}"
                session = LiveKitSession(room, participant, session_id, room_name)
                
                # Set up data channel handler
                def make_data_handler(sess, part):
                    return lambda data, p: sess.on_data_received(data, p) if p.identity == part.identity else None
                
                room.on("data_received", make_data_handler(session, participant))
                
                # Initialize audio track
                await session.initialize_audio_track()
                
                # Store session
                self._active_sessions[session_id] = session
                
                # Queue session for accept_session()
                await self._session_queue.put(session)
                
                logger.info(
                    "Created session for participant",
                    extra={"session_id": session_id, "participant": participant.identity}
                )
            
        except Exception as e:
            logger.error("Error joining room as agent", extra={"room": room_name, "error": str(e)})
            logger.exception("Full exception details for room join error")
            # Clean up on error
            if room_name in self._active_rooms:
                del self._active_rooms[room_name]
            if room_name in self._monitored_rooms:
                self._monitored_rooms.remove(room_name)

    async def create_room_and_wait_for_participant(
        self, timeout_seconds: float = 300.0
    ) -> LiveKitSession:
        """Create a room and wait for the first participant to join.

        This is a helper method that can be called externally to create
        a session with a specific workflow.

        Args:
            timeout_seconds: Maximum time to wait for participant

        Returns:
            LiveKitSession for the connected participant

        Raises:
            TimeoutError: If no participant joins within timeout
            RuntimeError: If room creation or connection fails
        """
        if not self._running:
            raise RuntimeError("LiveKit transport is not running")

        # Create room
        room_name = await self._room_manager.create_room()

        logger.info(
            "Room created, waiting for participant",
            extra={"room": room_name},
        )

        # Create room instance and connect as server
        room = rtc.Room()

        # Generate token for local participant (orchestrator)
        token = self._room_manager.create_access_token(
            room_name,
            participant_identity=f"orchestrator-{uuid.uuid4().hex[:8]}",
        )

        try:
            # Connect to room
            await room.connect(self._config.url, token)

            # Store active room
            self._active_rooms[room_name] = room

            # Wait for participant to join
            participant_event = asyncio.Event()
            first_participant: rtc.RemoteParticipant | None = None

            def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
                nonlocal first_participant
                if first_participant is None:
                    first_participant = participant
                    participant_event.set()

            # Set up event handler
            room.on("participant_connected", on_participant_connected)

            # Wait for participant with timeout
            try:
                await asyncio.wait_for(
                    participant_event.wait(),
                    timeout=timeout_seconds,
                )
            except TimeoutError as e:
                await room.disconnect()
                await self._room_manager.delete_room(room_name)
                raise TimeoutError(
                    f"No participant joined room '{room_name}' within {timeout_seconds}s"
                ) from e

            if first_participant is None:
                raise RuntimeError("Participant event set but no participant found")

            # Create session
            session_id = f"lk-{uuid.uuid4().hex[:12]}"
            session = LiveKitSession(room, first_participant, session_id, room_name)

            # Set up data channel handler
            def make_data_handler(sess):
                return lambda data, participant: sess.on_data_received(data, participant)
            
            room.on("data_received", make_data_handler(session))

            # Initialize audio track
            await session.initialize_audio_track()

            # Store session
            self._active_sessions[session_id] = session

            # Queue session for accept_session()
            await self._session_queue.put(session)

            logger.info(
                "Participant connected to room",
                extra={
                    "room": room_name,
                    "session_id": session_id,
                    "participant": first_participant.identity,
                },
            )

            return session

        except Exception as e:
            # Cleanup on error
            try:
                await room.disconnect()
            except Exception as cleanup_err:
                logger.debug("Error during cleanup", extra={"error": str(cleanup_err)})

            try:
                await self._room_manager.delete_room(room_name)
            except Exception as cleanup_err:
                logger.debug("Error during cleanup", extra={"error": str(cleanup_err)})

            raise RuntimeError(f"Failed to create session: {e}") from e
