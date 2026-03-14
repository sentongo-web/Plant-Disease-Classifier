"""
src/utils/logger.py — Centralised Logging Setup
================================================

Logging is how your code talks to you while it runs.  Using Python's built-in
`logging` module (rather than scattered `print()` calls) gives you:

  - Timestamps on every message so you know exactly when something happened.
  - Severity levels (DEBUG, INFO, WARNING, ERROR) so you can filter noise.
  - Simultaneous output to the terminal AND a log file with one setup call.
  - The ability to silence noisy third-party library logs without touching
    your own log statements.

We also use the `rich` library here to add colour to terminal output.
Colour-coded logs look far more professional and make errors jump out
immediately instead of getting lost in a wall of text.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with consistent formatting.

    Python's logging system keeps a global registry of loggers by name.
    Calling get_logger("trainer") twice returns the same logger object,
    so we never accidentally add duplicate handlers — a common bug.

    Parameters
    ----------
    name : str
        A descriptive name for this logger, e.g. "trainer", "data_loader".
        Appears in every log message so you can see which module produced it.
    log_file : str, optional
        If provided, log messages are also written to this file path.
        Useful for keeping a permanent record of training runs.
    level : int
        Minimum severity level to record.  logging.INFO ignores DEBUG
        messages; logging.DEBUG records everything.

    Returns
    -------
    logging.Logger
        A configured logger ready to use.

    Example
    -------
    >>> logger = get_logger("trainer", log_file="logs/train.log")
    >>> logger.info("Starting epoch 1")
    >>> logger.warning("Learning rate is very high")
    >>> logger.error("CUDA out of memory!")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this function is called again
    # with the same name (which happens in Jupyter notebooks).
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Format ──────────────────────────────────────────────────────────────
    # %(asctime)s   → human-readable timestamp
    # %(name)s      → the logger name we set above
    # %(levelname)s → INFO / WARNING / ERROR
    # %(message)s   → the actual log text
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ─────────────────────────────────────────────────────
    # Writes to stdout so the messages appear in the terminal.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── File handler (optional) ─────────────────────────────────────────────
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Prevent messages from bubbling up to the root logger and being printed
    # a second time.
    logger.propagate = False

    return logger
