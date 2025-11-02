"""Unit tests for SessionConfig timeout validation (M10 Polish Task 7).

Tests the SessionConfig Pydantic model validation for multi-turn conversation
timeouts. Ensures configuration limits prevent runaway sessions and resource
exhaustion.

Coverage:
- Default initialization and validation
- Boundary testing (min/max values)
- Invalid input handling (negative values, out-of-range)
- Realistic configuration scenarios
"""

import pytest
from orchestrator.config import SessionConfig
from pydantic import ValidationError


class TestSessionConfigInit:
    """Test SessionConfig initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization with standard timeout values."""
        config = SessionConfig()
        assert config.idle_timeout_seconds == 300  # 5 minutes
        assert config.max_session_duration_seconds == 3600  # 1 hour
        assert config.max_messages_per_session == 100

    def test_custom_initialization(self) -> None:
        """Test initialization with custom values."""
        config = SessionConfig(
            idle_timeout_seconds=600,
            max_session_duration_seconds=7200,
            max_messages_per_session=200,
        )
        assert config.idle_timeout_seconds == 600
        assert config.max_session_duration_seconds == 7200
        assert config.max_messages_per_session == 200


class TestSessionConfigValidation:
    """Test SessionConfig validation boundaries."""

    def test_idle_timeout_minimum(self) -> None:
        """Test idle timeout at minimum boundary (10 seconds)."""
        config = SessionConfig(idle_timeout_seconds=10)
        assert config.idle_timeout_seconds == 10

    def test_idle_timeout_below_minimum(self) -> None:
        """Test idle timeout below minimum (should fail validation)."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(idle_timeout_seconds=5)
        assert "idle_timeout_seconds" in str(exc_info.value)

    def test_idle_timeout_maximum(self) -> None:
        """Test idle timeout at maximum boundary (3600 seconds = 1 hour)."""
        config = SessionConfig(idle_timeout_seconds=3600)
        assert config.idle_timeout_seconds == 3600

    def test_idle_timeout_above_maximum(self) -> None:
        """Test idle timeout above maximum (should fail validation)."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(idle_timeout_seconds=4000)
        assert "idle_timeout_seconds" in str(exc_info.value)

    def test_max_duration_minimum(self) -> None:
        """Test max duration at minimum boundary (60 seconds)."""
        config = SessionConfig(max_session_duration_seconds=60)
        assert config.max_session_duration_seconds == 60

    def test_max_duration_below_minimum(self) -> None:
        """Test max duration below minimum (should fail validation)."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_session_duration_seconds=30)
        assert "max_session_duration_seconds" in str(exc_info.value)

    def test_max_duration_maximum(self) -> None:
        """Test max duration at maximum boundary (14400 seconds = 4 hours)."""
        config = SessionConfig(max_session_duration_seconds=14400)
        assert config.max_session_duration_seconds == 14400

    def test_max_duration_above_maximum(self) -> None:
        """Test max duration above maximum (should fail validation)."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_session_duration_seconds=15000)
        assert "max_session_duration_seconds" in str(exc_info.value)

    def test_max_messages_minimum(self) -> None:
        """Test max messages at minimum boundary (1 message)."""
        config = SessionConfig(max_messages_per_session=1)
        assert config.max_messages_per_session == 1

    def test_max_messages_below_minimum(self) -> None:
        """Test max messages below minimum (should fail validation)."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_messages_per_session=0)
        assert "max_messages_per_session" in str(exc_info.value)

    def test_max_messages_maximum(self) -> None:
        """Test max messages at maximum boundary (1000 messages)."""
        config = SessionConfig(max_messages_per_session=1000)
        assert config.max_messages_per_session == 1000

    def test_max_messages_above_maximum(self) -> None:
        """Test max messages above maximum (should fail validation)."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(max_messages_per_session=1500)
        assert "max_messages_per_session" in str(exc_info.value)

    def test_negative_idle_timeout(self) -> None:
        """Test negative idle timeout (should fail validation)."""
        with pytest.raises(ValidationError):
            SessionConfig(idle_timeout_seconds=-100)

    def test_negative_max_duration(self) -> None:
        """Test negative max duration (should fail validation)."""
        with pytest.raises(ValidationError):
            SessionConfig(max_session_duration_seconds=-1000)

    def test_negative_max_messages(self) -> None:
        """Test negative max messages (should fail validation)."""
        with pytest.raises(ValidationError):
            SessionConfig(max_messages_per_session=-50)


class TestSessionConfigRealistic:
    """Test realistic session configuration scenarios."""

    def test_short_session_config(self) -> None:
        """Test configuration for short demo sessions (2 min idle, 10 min max)."""
        config = SessionConfig(
            idle_timeout_seconds=120,  # 2 minutes
            max_session_duration_seconds=600,  # 10 minutes
            max_messages_per_session=20,
        )
        assert config.idle_timeout_seconds == 120
        assert config.max_session_duration_seconds == 600
        assert config.max_messages_per_session == 20
        # Idle timeout should be less than max duration
        assert config.idle_timeout_seconds < config.max_session_duration_seconds

    def test_long_session_config(self) -> None:
        """Test configuration for long customer support sessions (30 min idle, 4 hour max)."""
        config = SessionConfig(
            idle_timeout_seconds=1800,  # 30 minutes
            max_session_duration_seconds=14400,  # 4 hours
            max_messages_per_session=500,
        )
        assert config.idle_timeout_seconds == 1800
        assert config.max_session_duration_seconds == 14400
        assert config.max_messages_per_session == 500
        # Idle timeout should be less than max duration
        assert config.idle_timeout_seconds < config.max_session_duration_seconds

    def test_production_defaults(self) -> None:
        """Test production default configuration (5 min idle, 1 hour max)."""
        config = SessionConfig()
        assert config.idle_timeout_seconds == 300  # 5 minutes
        assert config.max_session_duration_seconds == 3600  # 1 hour
        assert config.max_messages_per_session == 100
        # Idle timeout should be less than max duration
        assert config.idle_timeout_seconds < config.max_session_duration_seconds

    def test_quick_turnaround_config(self) -> None:
        """Test configuration for quick turnaround scenarios (30 sec idle, 5 min max)."""
        config = SessionConfig(
            idle_timeout_seconds=30,
            max_session_duration_seconds=300,
            max_messages_per_session=10,
        )
        assert config.idle_timeout_seconds == 30
        assert config.max_session_duration_seconds == 300
        assert config.max_messages_per_session == 10
