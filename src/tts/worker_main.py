"""TTS worker entry point for running standalone worker server.

This module provides a simple entry point for starting a TTS worker server
from the command line or in a subprocess. Used for testing and development.

This delegates to the __main__ module which provides full CLI argument parsing.
"""

from src.tts.__main__ import main

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
