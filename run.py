import uvicorn
from backend.utils.logging import setup_logging

logger = setup_logging()
logger.info("=" * 60)
logger.info("RUNG Application Starting")
logger.info("=" * 60)

if __name__ == "__main__":
    try:
        uvicorn.run(
            "backend.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="warning"
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error("Application crashed: %s", e, exc_info=True)
    finally:
        logger.info("=" * 60)
        logger.info("RUNG Application Shutdown")
        logger.info("=" * 60)