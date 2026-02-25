"""
logger_config.py — Centralized logging configuration for VerifAI.

Import ``setup_logging`` once at app startup (streamlit.py) and use
``get_logger(__name__)`` everywhere else.
"""

import logging
import os
import sys


# ── Log format constants ──
_LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)-20s │ %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str = "INFO"):
    """Configure the root logger and silence noisy third-party libraries.

    Call this **once** at app startup (e.g. in ``streamlit.py``).

    Args:
        level: Log level for VerifAI modules. Third-party libs are
               always set to WARNING or higher.
    """
    import warnings

    log_level = getattr(logging, level.upper(), logging.INFO)

    # ── Root logger — set to DEBUG so nothing is filtered at root ──
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers (avoids duplicates on Streamlit reruns)
    root.handlers.clear()

    # Console handler with clean formatting
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    root.addHandler(handler)

    # ── Capture Python warnings (DeprecationWarning, etc.) via logging ──
    logging.captureWarnings(True)
    warnings.simplefilter("default")  # Show deprecation warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)  # Suppress unclosed sqlite3 from chromadb

    # ── Silence noisy third-party loggers ──
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
        "openai",
        "litellm",
        "crewai",
        "crewai.agent",
        "crewai.crew",
        "crewai.task",
        "crewai.tools",
        "crewai.utilities",
        "crewai_tools",
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_google_genai",
        "chromadb",
        "chromadb.config",
        "chromadb.segment",
        "chromadb.api",
        "google",
        "google.generativeai",
        "PIL",
        "matplotlib",
        "fsevents",
        "watchdog",
        "streamlit",
        "tornado",
        "praw",
        "prawcore",
        "google_genai",
        "google_genai._api_client",
        "asyncio",
        "sentence_transformers",
        "matplotlib.transforms",
        "py.warnings",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

    # ── VerifAI modules get the user-configured level ──
    verifai_modules = [
        "streamlit",   # our streamlit.py (shares name but we handle it)
        "app",
        "agents",
        "tasks",
        "reddit",
        "setup",
        "save_report",
        "cache_service",
        "trends_service",
        "models",
        "__main__",
    ]
    for name in verifai_modules:
        logging.getLogger(name).setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Usage::

        from logger_config import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    return logging.getLogger(name)
