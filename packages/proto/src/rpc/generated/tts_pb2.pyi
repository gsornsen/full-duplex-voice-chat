from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ControlCommand(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PAUSE: _ClassVar[ControlCommand]
    RESUME: _ClassVar[ControlCommand]
    STOP: _ClassVar[ControlCommand]
    RELOAD: _ClassVar[ControlCommand]
PAUSE: ControlCommand
RESUME: ControlCommand
STOP: ControlCommand
RELOAD: ControlCommand

class StartSessionRequest(_message.Message):
    __slots__ = ("session_id", "model_id", "options")
    class OptionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    model_id: str
    options: _containers.ScalarMap[str, str]
    def __init__(self, session_id: _Optional[str] = ..., model_id: _Optional[str] = ..., options: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StartSessionResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class EndSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class EndSessionResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class TextChunk(_message.Message):
    __slots__ = ("session_id", "text", "is_final", "sequence_number")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    text: str
    is_final: bool
    sequence_number: int
    def __init__(self, session_id: _Optional[str] = ..., text: _Optional[str] = ..., is_final: bool = ..., sequence_number: _Optional[int] = ...) -> None: ...

class AudioFrame(_message.Message):
    __slots__ = ("session_id", "audio_data", "sample_rate", "frame_duration_ms", "sequence_number", "is_final")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    FRAME_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    audio_data: bytes
    sample_rate: int
    frame_duration_ms: int
    sequence_number: int
    is_final: bool
    def __init__(self, session_id: _Optional[str] = ..., audio_data: _Optional[bytes] = ..., sample_rate: _Optional[int] = ..., frame_duration_ms: _Optional[int] = ..., sequence_number: _Optional[int] = ..., is_final: bool = ...) -> None: ...

class ControlRequest(_message.Message):
    __slots__ = ("session_id", "command")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    command: ControlCommand
    def __init__(self, session_id: _Optional[str] = ..., command: _Optional[_Union[ControlCommand, str]] = ...) -> None: ...

class ControlResponse(_message.Message):
    __slots__ = ("success", "message", "timestamp_ms")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    timestamp_ms: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("model_id", "family", "is_loaded", "languages", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    FAMILY_FIELD_NUMBER: _ClassVar[int]
    IS_LOADED_FIELD_NUMBER: _ClassVar[int]
    LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    family: str
    is_loaded: bool
    languages: _containers.RepeatedScalarFieldContainer[str]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, model_id: _Optional[str] = ..., family: _Optional[str] = ..., is_loaded: bool = ..., languages: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class LoadModelRequest(_message.Message):
    __slots__ = ("model_id", "preload_only")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PRELOAD_ONLY_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    preload_only: bool
    def __init__(self, model_id: _Optional[str] = ..., preload_only: bool = ...) -> None: ...

class LoadModelResponse(_message.Message):
    __slots__ = ("success", "message", "load_duration_ms")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LOAD_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    load_duration_ms: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., load_duration_ms: _Optional[int] = ...) -> None: ...

class UnloadModelRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class UnloadModelResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class GetCapabilitiesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Capabilities(_message.Message):
    __slots__ = ("streaming", "zero_shot", "lora", "cpu_ok", "languages", "emotive_zero_prompt", "max_concurrent_sessions")
    STREAMING_FIELD_NUMBER: _ClassVar[int]
    ZERO_SHOT_FIELD_NUMBER: _ClassVar[int]
    LORA_FIELD_NUMBER: _ClassVar[int]
    CPU_OK_FIELD_NUMBER: _ClassVar[int]
    LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    EMOTIVE_ZERO_PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENT_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    streaming: bool
    zero_shot: bool
    lora: bool
    cpu_ok: bool
    languages: _containers.RepeatedScalarFieldContainer[str]
    emotive_zero_prompt: bool
    max_concurrent_sessions: int
    def __init__(self, streaming: bool = ..., zero_shot: bool = ..., lora: bool = ..., cpu_ok: bool = ..., languages: _Optional[_Iterable[str]] = ..., emotive_zero_prompt: bool = ..., max_concurrent_sessions: _Optional[int] = ...) -> None: ...

class GetCapabilitiesResponse(_message.Message):
    __slots__ = ("capabilities", "resident_models", "metrics")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    RESIDENT_MODELS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    capabilities: Capabilities
    resident_models: _containers.RepeatedScalarFieldContainer[str]
    metrics: _containers.ScalarMap[str, float]
    def __init__(self, capabilities: _Optional[_Union[Capabilities, _Mapping]] = ..., resident_models: _Optional[_Iterable[str]] = ..., metrics: _Optional[_Mapping[str, float]] = ...) -> None: ...
