"""Unit tests for gRPC protobuf message serialization/deserialization.

This module tests the roundtrip serialization and deserialization of all
protobuf messages defined in src/rpc/tts.proto to ensure data integrity
and proper field encoding.
"""


from rpc.generated import tts_pb2


class TestTextChunk:
    """Tests for TextChunk message serialization/deserialization."""

    def test_text_chunk_roundtrip(self) -> None:
        """Test basic TextChunk roundtrip with all fields populated."""
        # Create message
        chunk = tts_pb2.TextChunk(
            session_id="test-session-123",
            text="Hello world",
            is_final=False,
            sequence_number=1,
        )

        # Serialize
        data = chunk.SerializeToString()

        # Deserialize
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        # Verify all fields
        assert chunk2.session_id == "test-session-123"
        assert chunk2.text == "Hello world"
        assert chunk2.is_final is False
        assert chunk2.sequence_number == 1

    def test_text_chunk_empty_values(self) -> None:
        """Test TextChunk with empty/default values."""
        chunk = tts_pb2.TextChunk(
            session_id="",
            text="",
            is_final=False,
            sequence_number=0,
        )

        data = chunk.SerializeToString()
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        assert chunk2.session_id == ""
        assert chunk2.text == ""
        assert chunk2.is_final is False
        assert chunk2.sequence_number == 0

    def test_text_chunk_final_flag(self) -> None:
        """Test TextChunk with is_final flag set to True."""
        chunk = tts_pb2.TextChunk(
            session_id="session-final",
            text="Final chunk",
            is_final=True,
            sequence_number=999,
        )

        data = chunk.SerializeToString()
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        assert chunk2.is_final is True
        assert chunk2.sequence_number == 999

    def test_text_chunk_large_sequence_number(self) -> None:
        """Test TextChunk with max int64 sequence number."""
        max_int64 = 9223372036854775807
        chunk = tts_pb2.TextChunk(
            session_id="session-large",
            text="Large seq",
            is_final=False,
            sequence_number=max_int64,
        )

        data = chunk.SerializeToString()
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        assert chunk2.sequence_number == max_int64

    def test_text_chunk_unicode_text(self) -> None:
        """Test TextChunk with Unicode characters."""
        chunk = tts_pb2.TextChunk(
            session_id="unicode-session",
            text="Hello 世界 🌍 Привет",
            is_final=False,
            sequence_number=42,
        )

        data = chunk.SerializeToString()
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        assert chunk2.text == "Hello 世界 🌍 Привет"


class TestAudioFrame:
    """Tests for AudioFrame message serialization/deserialization."""

    def test_audio_frame_roundtrip(self) -> None:
        """Test basic AudioFrame roundtrip with all fields populated."""
        # Create 20ms of 48kHz mono PCM (960 samples * 2 bytes = 1920 bytes)
        audio_data = b"\x00\x01" * 960

        frame = tts_pb2.AudioFrame(
            session_id="audio-session-456",
            audio_data=audio_data,
            sample_rate=48000,
            frame_duration_ms=20,
            sequence_number=10,
            is_final=False,
        )

        # Serialize
        data = frame.SerializeToString()

        # Deserialize
        frame2 = tts_pb2.AudioFrame()
        frame2.ParseFromString(data)

        # Verify all fields
        assert frame2.session_id == "audio-session-456"
        assert frame2.audio_data == audio_data
        assert frame2.sample_rate == 48000
        assert frame2.frame_duration_ms == 20
        assert frame2.sequence_number == 10
        assert frame2.is_final is False

    def test_audio_frame_empty_data(self) -> None:
        """Test AudioFrame with empty audio data."""
        frame = tts_pb2.AudioFrame(
            session_id="empty-audio",
            audio_data=b"",
            sample_rate=48000,
            frame_duration_ms=20,
            sequence_number=0,
            is_final=False,
        )

        data = frame.SerializeToString()
        frame2 = tts_pb2.AudioFrame()
        frame2.ParseFromString(data)

        assert frame2.audio_data == b""
        assert len(frame2.audio_data) == 0

    def test_audio_frame_final_flag(self) -> None:
        """Test AudioFrame with is_final flag set to True."""
        frame = tts_pb2.AudioFrame(
            session_id="final-audio",
            audio_data=b"\x00\x00\x00\x00",
            sample_rate=48000,
            frame_duration_ms=20,
            sequence_number=100,
            is_final=True,
        )

        data = frame.SerializeToString()
        frame2 = tts_pb2.AudioFrame()
        frame2.ParseFromString(data)

        assert frame2.is_final is True

    def test_audio_frame_large_binary_data(self) -> None:
        """Test AudioFrame with large binary audio data."""
        # Create 1 second of audio data (48000 samples * 2 bytes)
        large_audio = b"\xFF\x7F" * 48000  # Max positive int16

        frame = tts_pb2.AudioFrame(
            session_id="large-frame",
            audio_data=large_audio,
            sample_rate=48000,
            frame_duration_ms=1000,
            sequence_number=1,
            is_final=False,
        )

        data = frame.SerializeToString()
        frame2 = tts_pb2.AudioFrame()
        frame2.ParseFromString(data)

        assert frame2.audio_data == large_audio
        assert len(frame2.audio_data) == 96000

    def test_audio_frame_different_sample_rates(self) -> None:
        """Test AudioFrame with various sample rates."""
        for sample_rate in [8000, 16000, 24000, 48000]:
            frame = tts_pb2.AudioFrame(
                session_id=f"sr-{sample_rate}",
                audio_data=b"\x00\x00",
                sample_rate=sample_rate,
                frame_duration_ms=20,
                sequence_number=1,
                is_final=False,
            )

            data = frame.SerializeToString()
            frame2 = tts_pb2.AudioFrame()
            frame2.ParseFromString(data)

            assert frame2.sample_rate == sample_rate


class TestControlMessages:
    """Tests for Control request/response messages."""

    def test_control_request_pause(self) -> None:
        """Test ControlRequest with PAUSE command."""
        request = tts_pb2.ControlRequest(
            session_id="control-session-1",
            command=tts_pb2.ControlCommand.PAUSE,
        )

        data = request.SerializeToString()
        request2 = tts_pb2.ControlRequest()
        request2.ParseFromString(data)

        assert request2.session_id == "control-session-1"
        assert request2.command == tts_pb2.ControlCommand.PAUSE
        assert request2.command == 0

    def test_control_request_resume(self) -> None:
        """Test ControlRequest with RESUME command."""
        request = tts_pb2.ControlRequest(
            session_id="control-session-2",
            command=tts_pb2.ControlCommand.RESUME,
        )

        data = request.SerializeToString()
        request2 = tts_pb2.ControlRequest()
        request2.ParseFromString(data)

        assert request2.command == tts_pb2.ControlCommand.RESUME
        assert request2.command == 1

    def test_control_request_stop(self) -> None:
        """Test ControlRequest with STOP command."""
        request = tts_pb2.ControlRequest(
            session_id="control-session-3",
            command=tts_pb2.ControlCommand.STOP,
        )

        data = request.SerializeToString()
        request2 = tts_pb2.ControlRequest()
        request2.ParseFromString(data)

        assert request2.command == tts_pb2.ControlCommand.STOP
        assert request2.command == 2

    def test_control_request_reload(self) -> None:
        """Test ControlRequest with RELOAD command."""
        request = tts_pb2.ControlRequest(
            session_id="control-session-4",
            command=tts_pb2.ControlCommand.RELOAD,
        )

        data = request.SerializeToString()
        request2 = tts_pb2.ControlRequest()
        request2.ParseFromString(data)

        assert request2.command == tts_pb2.ControlCommand.RELOAD
        assert request2.command == 3

    def test_control_response_success(self) -> None:
        """Test ControlResponse with success status."""
        response = tts_pb2.ControlResponse(
            success=True,
            message="Command executed successfully",
            timestamp_ms=1234567890123,
        )

        data = response.SerializeToString()
        response2 = tts_pb2.ControlResponse()
        response2.ParseFromString(data)

        assert response2.success is True
        assert response2.message == "Command executed successfully"
        assert response2.timestamp_ms == 1234567890123

    def test_control_response_failure(self) -> None:
        """Test ControlResponse with failure status."""
        response = tts_pb2.ControlResponse(
            success=False,
            message="Command failed: session not found",
            timestamp_ms=1234567890456,
        )

        data = response.SerializeToString()
        response2 = tts_pb2.ControlResponse()
        response2.ParseFromString(data)

        assert response2.success is False
        assert response2.message == "Command failed: session not found"

    def test_control_command_enum_values(self) -> None:
        """Test all ControlCommand enum values match expected integers."""
        assert tts_pb2.ControlCommand.PAUSE == 0
        assert tts_pb2.ControlCommand.RESUME == 1
        assert tts_pb2.ControlCommand.STOP == 2
        assert tts_pb2.ControlCommand.RELOAD == 3


class TestSessionManagement:
    """Tests for session lifecycle messages."""

    def test_start_session_request_basic(self) -> None:
        """Test StartSessionRequest with basic fields."""
        request = tts_pb2.StartSessionRequest(
            session_id="session-start-1",
            model_id="cosyvoice2-en-base",
        )

        data = request.SerializeToString()
        request2 = tts_pb2.StartSessionRequest()
        request2.ParseFromString(data)

        assert request2.session_id == "session-start-1"
        assert request2.model_id == "cosyvoice2-en-base"
        assert len(request2.options) == 0

    def test_start_session_request_with_options(self) -> None:
        """Test StartSessionRequest with options map."""
        request = tts_pb2.StartSessionRequest(
            session_id="session-opts",
            model_id="xtts-v2",
            options={
                "speaker": "female-1",
                "speed": "1.2",
                "emotion": "neutral",
            },
        )

        data = request.SerializeToString()
        request2 = tts_pb2.StartSessionRequest()
        request2.ParseFromString(data)

        assert request2.session_id == "session-opts"
        assert request2.model_id == "xtts-v2"
        assert len(request2.options) == 3
        assert request2.options["speaker"] == "female-1"
        assert request2.options["speed"] == "1.2"
        assert request2.options["emotion"] == "neutral"

    def test_start_session_request_empty_options(self) -> None:
        """Test StartSessionRequest with explicitly empty options."""
        request = tts_pb2.StartSessionRequest(
            session_id="session-no-opts",
            model_id="piper-en",
            options={},
        )

        data = request.SerializeToString()
        request2 = tts_pb2.StartSessionRequest()
        request2.ParseFromString(data)

        assert len(request2.options) == 0

    def test_start_session_response_success(self) -> None:
        """Test StartSessionResponse with success status."""
        response = tts_pb2.StartSessionResponse(
            success=True,
            message="Session started successfully",
        )

        data = response.SerializeToString()
        response2 = tts_pb2.StartSessionResponse()
        response2.ParseFromString(data)

        assert response2.success is True
        assert response2.message == "Session started successfully"

    def test_start_session_response_failure(self) -> None:
        """Test StartSessionResponse with failure status."""
        response = tts_pb2.StartSessionResponse(
            success=False,
            message="Failed to start session: model not found",
        )

        data = response.SerializeToString()
        response2 = tts_pb2.StartSessionResponse()
        response2.ParseFromString(data)

        assert response2.success is False
        assert response2.message == "Failed to start session: model not found"

    def test_end_session_request(self) -> None:
        """Test EndSessionRequest roundtrip."""
        request = tts_pb2.EndSessionRequest(session_id="session-end-123")

        data = request.SerializeToString()
        request2 = tts_pb2.EndSessionRequest()
        request2.ParseFromString(data)

        assert request2.session_id == "session-end-123"

    def test_end_session_response_success(self) -> None:
        """Test EndSessionResponse with success status."""
        response = tts_pb2.EndSessionResponse(success=True)

        data = response.SerializeToString()
        response2 = tts_pb2.EndSessionResponse()
        response2.ParseFromString(data)

        assert response2.success is True

    def test_end_session_response_failure(self) -> None:
        """Test EndSessionResponse with failure status."""
        response = tts_pb2.EndSessionResponse(success=False)

        data = response.SerializeToString()
        response2 = tts_pb2.EndSessionResponse()
        response2.ParseFromString(data)

        assert response2.success is False


class TestModelManagement:
    """Tests for model management messages."""

    def test_list_models_request(self) -> None:
        """Test ListModelsRequest (empty message)."""
        request = tts_pb2.ListModelsRequest()

        data = request.SerializeToString()
        request2 = tts_pb2.ListModelsRequest()
        request2.ParseFromString(data)

        # Empty message should serialize/deserialize without error
        assert request2 is not None

    def test_model_info_complete(self) -> None:
        """Test ModelInfo with all fields populated."""
        model_info = tts_pb2.ModelInfo(
            model_id="cosyvoice2-en-base",
            family="cosyvoice2",
            is_loaded=True,
            languages=["en", "zh"],
            metadata={
                "domain": "general",
                "expressive": "true",
                "version": "2.0",
            },
        )

        data = model_info.SerializeToString()
        model_info2 = tts_pb2.ModelInfo()
        model_info2.ParseFromString(data)

        assert model_info2.model_id == "cosyvoice2-en-base"
        assert model_info2.family == "cosyvoice2"
        assert model_info2.is_loaded is True
        assert list(model_info2.languages) == ["en", "zh"]
        assert model_info2.metadata["domain"] == "general"
        assert model_info2.metadata["expressive"] == "true"
        assert model_info2.metadata["version"] == "2.0"

    def test_model_info_minimal(self) -> None:
        """Test ModelInfo with minimal fields."""
        model_info = tts_pb2.ModelInfo(
            model_id="piper-en-us",
            family="piper",
            is_loaded=False,
        )

        data = model_info.SerializeToString()
        model_info2 = tts_pb2.ModelInfo()
        model_info2.ParseFromString(data)

        assert model_info2.model_id == "piper-en-us"
        assert model_info2.family == "piper"
        assert model_info2.is_loaded is False
        assert len(model_info2.languages) == 0
        assert len(model_info2.metadata) == 0

    def test_list_models_response(self) -> None:
        """Test ListModelsResponse with multiple models."""
        response = tts_pb2.ListModelsResponse(
            models=[
                tts_pb2.ModelInfo(
                    model_id="model-1",
                    family="family-a",
                    is_loaded=True,
                    languages=["en"],
                ),
                tts_pb2.ModelInfo(
                    model_id="model-2",
                    family="family-b",
                    is_loaded=False,
                    languages=["zh", "ja"],
                ),
            ]
        )

        data = response.SerializeToString()
        response2 = tts_pb2.ListModelsResponse()
        response2.ParseFromString(data)

        assert len(response2.models) == 2
        assert response2.models[0].model_id == "model-1"
        assert response2.models[0].is_loaded is True
        assert response2.models[1].model_id == "model-2"
        assert list(response2.models[1].languages) == ["zh", "ja"]

    def test_load_model_request(self) -> None:
        """Test LoadModelRequest with preload flag."""
        request = tts_pb2.LoadModelRequest(
            model_id="xtts-v2-en",
            preload_only=True,
        )

        data = request.SerializeToString()
        request2 = tts_pb2.LoadModelRequest()
        request2.ParseFromString(data)

        assert request2.model_id == "xtts-v2-en"
        assert request2.preload_only is True

    def test_load_model_response(self) -> None:
        """Test LoadModelResponse with load duration."""
        response = tts_pb2.LoadModelResponse(
            success=True,
            message="Model loaded successfully",
            load_duration_ms=2500,
        )

        data = response.SerializeToString()
        response2 = tts_pb2.LoadModelResponse()
        response2.ParseFromString(data)

        assert response2.success is True
        assert response2.message == "Model loaded successfully"
        assert response2.load_duration_ms == 2500

    def test_unload_model_request(self) -> None:
        """Test UnloadModelRequest roundtrip."""
        request = tts_pb2.UnloadModelRequest(model_id="sesame-lora-1")

        data = request.SerializeToString()
        request2 = tts_pb2.UnloadModelRequest()
        request2.ParseFromString(data)

        assert request2.model_id == "sesame-lora-1"

    def test_unload_model_response(self) -> None:
        """Test UnloadModelResponse roundtrip."""
        response = tts_pb2.UnloadModelResponse(
            success=True,
            message="Model unloaded",
        )

        data = response.SerializeToString()
        response2 = tts_pb2.UnloadModelResponse()
        response2.ParseFromString(data)

        assert response2.success is True
        assert response2.message == "Model unloaded"


class TestCapabilities:
    """Tests for capabilities messages."""

    def test_get_capabilities_request(self) -> None:
        """Test GetCapabilitiesRequest (empty message)."""
        request = tts_pb2.GetCapabilitiesRequest()

        data = request.SerializeToString()
        request2 = tts_pb2.GetCapabilitiesRequest()
        request2.ParseFromString(data)

        assert request2 is not None

    def test_capabilities_complete(self) -> None:
        """Test Capabilities message with all fields."""
        capabilities = tts_pb2.Capabilities(
            streaming=True,
            zero_shot=True,
            lora=False,
            cpu_ok=False,
            languages=["en", "zh", "ja", "es"],
            emotive_zero_prompt=True,
            max_concurrent_sessions=5,
        )

        data = capabilities.SerializeToString()
        capabilities2 = tts_pb2.Capabilities()
        capabilities2.ParseFromString(data)

        assert capabilities2.streaming is True
        assert capabilities2.zero_shot is True
        assert capabilities2.lora is False
        assert capabilities2.cpu_ok is False
        assert list(capabilities2.languages) == ["en", "zh", "ja", "es"]
        assert capabilities2.emotive_zero_prompt is True
        assert capabilities2.max_concurrent_sessions == 5

    def test_capabilities_cpu_only(self) -> None:
        """Test Capabilities for CPU-only adapter (e.g., Piper)."""
        capabilities = tts_pb2.Capabilities(
            streaming=True,
            zero_shot=False,
            lora=False,
            cpu_ok=True,
            languages=["en"],
            emotive_zero_prompt=False,
            max_concurrent_sessions=10,
        )

        data = capabilities.SerializeToString()
        capabilities2 = tts_pb2.Capabilities()
        capabilities2.ParseFromString(data)

        assert capabilities2.cpu_ok is True
        assert capabilities2.zero_shot is False
        assert capabilities2.max_concurrent_sessions == 10

    def test_get_capabilities_response(self) -> None:
        """Test GetCapabilitiesResponse with all fields."""
        response = tts_pb2.GetCapabilitiesResponse(
            capabilities=tts_pb2.Capabilities(
                streaming=True,
                zero_shot=True,
                lora=False,
                cpu_ok=False,
                languages=["en"],
                emotive_zero_prompt=True,
                max_concurrent_sessions=3,
            ),
            resident_models=["cosyvoice2-en-base", "xtts-v2-demo"],
            metrics={
                "rtf": 0.2,
                "p50_latency_ms": 250.5,
                "p95_latency_ms": 350.8,
                "queue_depth": 0.0,
            },
        )

        data = response.SerializeToString()
        response2 = tts_pb2.GetCapabilitiesResponse()
        response2.ParseFromString(data)

        assert response2.capabilities.streaming is True
        assert response2.capabilities.max_concurrent_sessions == 3
        assert list(response2.resident_models) == [
            "cosyvoice2-en-base",
            "xtts-v2-demo",
        ]
        assert response2.metrics["rtf"] == 0.2
        assert response2.metrics["p50_latency_ms"] == 250.5
        assert response2.metrics["queue_depth"] == 0.0

    def test_get_capabilities_response_empty_metrics(self) -> None:
        """Test GetCapabilitiesResponse with no metrics."""
        response = tts_pb2.GetCapabilitiesResponse(
            capabilities=tts_pb2.Capabilities(streaming=True),
            resident_models=[],
            metrics={},
        )

        data = response.SerializeToString()
        response2 = tts_pb2.GetCapabilitiesResponse()
        response2.ParseFromString(data)

        assert len(response2.resident_models) == 0
        assert len(response2.metrics) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_session_id(self) -> None:
        """Test messages with empty session IDs."""
        chunk = tts_pb2.TextChunk(
            session_id="",
            text="Some text",
            is_final=False,
            sequence_number=1,
        )

        data = chunk.SerializeToString()
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        assert chunk2.session_id == ""

    def test_very_long_session_id(self) -> None:
        """Test message with very long session ID."""
        long_id = "x" * 1000
        request = tts_pb2.StartSessionRequest(
            session_id=long_id,
            model_id="test-model",
        )

        data = request.SerializeToString()
        request2 = tts_pb2.StartSessionRequest()
        request2.ParseFromString(data)

        assert request2.session_id == long_id
        assert len(request2.session_id) == 1000

    def test_very_long_text(self) -> None:
        """Test TextChunk with very long text (simulating paragraph)."""
        long_text = "Lorem ipsum dolor sit amet. " * 200  # ~5600 chars
        chunk = tts_pb2.TextChunk(
            session_id="long-text-session",
            text=long_text,
            is_final=False,
            sequence_number=1,
        )

        data = chunk.SerializeToString()
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        assert chunk2.text == long_text
        assert len(chunk2.text) > 5000

    def test_zero_values(self) -> None:
        """Test messages with all zero/false values."""
        frame = tts_pb2.AudioFrame(
            session_id="",
            audio_data=b"",
            sample_rate=0,
            frame_duration_ms=0,
            sequence_number=0,
            is_final=False,
        )

        data = frame.SerializeToString()
        frame2 = tts_pb2.AudioFrame()
        frame2.ParseFromString(data)

        assert frame2.sample_rate == 0
        assert frame2.frame_duration_ms == 0
        assert frame2.sequence_number == 0

    def test_negative_sequence_number(self) -> None:
        """Test with negative sequence number (valid in int64)."""
        chunk = tts_pb2.TextChunk(
            session_id="neg-seq",
            text="Test",
            is_final=False,
            sequence_number=-1,
        )

        data = chunk.SerializeToString()
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        assert chunk2.sequence_number == -1

    def test_max_int32_sample_rate(self) -> None:
        """Test AudioFrame with max int32 sample rate."""
        max_int32 = 2147483647
        frame = tts_pb2.AudioFrame(
            session_id="max-sr",
            audio_data=b"\x00\x00",
            sample_rate=max_int32,
            frame_duration_ms=20,
            sequence_number=1,
            is_final=False,
        )

        data = frame.SerializeToString()
        frame2 = tts_pb2.AudioFrame()
        frame2.ParseFromString(data)

        assert frame2.sample_rate == max_int32

    def test_many_options(self) -> None:
        """Test StartSessionRequest with many options."""
        many_options = {f"option_{i}": f"value_{i}" for i in range(100)}
        request = tts_pb2.StartSessionRequest(
            session_id="many-opts",
            model_id="test-model",
            options=many_options,
        )

        data = request.SerializeToString()
        request2 = tts_pb2.StartSessionRequest()
        request2.ParseFromString(data)

        assert len(request2.options) == 100
        assert request2.options["option_0"] == "value_0"
        assert request2.options["option_99"] == "value_99"

    def test_many_models_in_list(self) -> None:
        """Test ListModelsResponse with many models."""
        models = [
            tts_pb2.ModelInfo(
                model_id=f"model-{i}",
                family=f"family-{i % 5}",
                is_loaded=(i % 2 == 0),
            )
            for i in range(50)
        ]

        response = tts_pb2.ListModelsResponse(models=models)

        data = response.SerializeToString()
        response2 = tts_pb2.ListModelsResponse()
        response2.ParseFromString(data)

        assert len(response2.models) == 50
        assert response2.models[0].model_id == "model-0"
        assert response2.models[49].model_id == "model-49"

    def test_special_characters_in_strings(self) -> None:
        """Test messages with special characters in string fields."""
        special_chars = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`\"\\\n\t\r"
        chunk = tts_pb2.TextChunk(
            session_id=special_chars,
            text=special_chars,
            is_final=False,
            sequence_number=1,
        )

        data = chunk.SerializeToString()
        chunk2 = tts_pb2.TextChunk()
        chunk2.ParseFromString(data)

        assert chunk2.session_id == special_chars
        assert chunk2.text == special_chars
