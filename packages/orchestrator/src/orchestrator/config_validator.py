"""Configuration validation for orchestrator startup."""

import logging
import os
import os.path

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigValidator:
    """Validates orchestrator configuration at startup."""

    @staticmethod
    def validate_tts_configuration() -> tuple[bool, list[str]]:
        """
        Validate TTS worker configuration matches orchestrator expectations.

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Get configuration
        default_model = os.getenv("DEFAULT_MODEL_ID", os.getenv("DEFAULT_MODEL"))
        adapter_type = os.getenv("ADAPTER_TYPE")

        if not default_model:
            warnings.append("DEFAULT_MODEL or DEFAULT_MODEL_ID not set - using system defaults")
            return True, warnings

        # Validate model ID matches expected adapter
        if default_model.startswith("cosyvoice2-"):
            # CosyVoice model
            if adapter_type and adapter_type != "cosyvoice2":
                warnings.append(
                    f"Model '{default_model}' requires ADAPTER_TYPE=cosyvoice2, "
                    f"but ADAPTER_TYPE={adapter_type} is set"
                )

            # Check if voicepack exists
            # Note: CosyVoice models are auto-downloaded if not found, so this is just a warning
            voice_name = default_model.replace("cosyvoice2-", "")
            voicepack_path = f"voicepacks/cosyvoice/{voice_name}"
            # Also check for auto-download directory (CosyVoice downloads to voicepacks/cosyvoice/)
            auto_download_path = "voicepacks/cosyvoice"

            if not os.path.exists(voicepack_path) and not os.path.exists(auto_download_path):
                warnings.append(
                    f"Voicepack not found: {voicepack_path} - "
                    f"model '{default_model}' will be auto-downloaded on first use"
                )

        elif default_model.startswith("piper-"):
            # Piper model
            if adapter_type and adapter_type != "piper":
                warnings.append(
                    f"Model '{default_model}' requires ADAPTER_TYPE=piper, "
                    f"but ADAPTER_TYPE={adapter_type} is set"
                )

        elif default_model == "mock":
            # Mock adapter
            if adapter_type and adapter_type != "mock":
                warnings.append(
                    f"Model '{default_model}' requires ADAPTER_TYPE=mock, "
                    f"but ADAPTER_TYPE={adapter_type} is set"
                )

        else:
            warnings.append(
                f"Unknown model ID format: '{default_model}' - "
                f"expected cosyvoice2-*, piper-*, or 'mock'"
            )

        return len(warnings) == 0, warnings

    @staticmethod
    def validate_asr_configuration() -> tuple[bool, list[str]]:
        """
        Validate ASR configuration.

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        asr_device = os.getenv("ASR_DEVICE", "cpu")
        asr_compute_type = os.getenv("ASR_COMPUTE_TYPE", "default")

        # Check if GPU requested but not available
        if asr_device in ("cuda", "auto"):
            try:
                import torch

                if not torch.cuda.is_available():
                    warnings.append(
                        f"ASR_DEVICE={asr_device} but CUDA not available - "
                        "will fall back to CPU (expect slower performance)"
                    )
            except ImportError:
                warnings.append("PyTorch not installed - cannot validate CUDA availability")

        # Validate compute type matches device
        if asr_device == "cpu" and asr_compute_type == "float16":
            warnings.append(
                "ASR_COMPUTE_TYPE=float16 not supported on CPU - will auto-select int8"
            )

        return len(warnings) == 0, warnings

    @staticmethod
    def validate_all(strict: bool = False) -> None:
        """
        Run all configuration validations.

        Args:
            strict: If True, raise ConfigurationError on any warnings.
                   If False, only log warnings.

        Raises:
            ConfigurationError: If strict=True and validation fails.
        """
        all_warnings = []

        # Validate TTS configuration
        tts_valid, tts_warnings = ConfigValidator.validate_tts_configuration()
        all_warnings.extend(tts_warnings)

        # Validate ASR configuration
        asr_valid, asr_warnings = ConfigValidator.validate_asr_configuration()
        all_warnings.extend(asr_warnings)

        # Log all warnings
        if all_warnings:
            logger.warning("Configuration validation found issues:")
            for i, warning in enumerate(all_warnings, 1):
                logger.warning(f"  {i}. {warning}")

            if strict:
                raise ConfigurationError(
                    f"Configuration validation failed with {len(all_warnings)} warnings. "
                    "Fix configuration or set strict=False to continue."
                )
        else:
            logger.info("Configuration validation passed ✓")
