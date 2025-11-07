import os
import sys
import logging
from loguru import logger
from pathlib import Path

from ..config import settings

def setup_logging():
    """Configure logging with loguru."""
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=settings.DEBUG,
    )
    
    # Add file logger if in production
    if not settings.DEBUG:
        log_file = Path("logs/aegisclaim.log")
        log_file.parent.mkdir(exist_ok=True)
        
        logger.add(
            log_file,
            rotation="100 MB",
            retention="30 days",
            level=settings.LOG_LEVEL,
            format=settings.LOG_FORMAT,
            colorize=False,
            backtrace=True,
            diagnose=settings.DEBUG,
            enqueue=True,  # For async support
        )
    
    # Configure standard library logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where the logged message originated
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Configure uvicorn logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set log levels for specific loggers
    for logger_name in ["uvicorn", "uvicorn.error", "fastapi"]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
    
    # Disable noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logger.info("Logging configured successfully")
