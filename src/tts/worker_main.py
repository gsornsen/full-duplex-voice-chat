"""TTS worker entry point for running standalone worker server.

This module provides a simple entry point for starting a TTS worker server
from the command line or in a subprocess. Used for testing and development.
"""

import asyncio
import logging

from src.tts.worker import start_worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point for TTS worker.

    Starts the worker server on the default port (7001) and runs until
    interrupted with Ctrl+C or SIGTERM.
    """
    config = {"port": 7001}
    logger.info("Starting TTS worker server", extra={"config": config})

    try:
        await start_worker(config)
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.exception("Worker failed with error", extra={"error": str(e)})
        raise


if __name__ == "__main__":
    asyncio.run(main())
