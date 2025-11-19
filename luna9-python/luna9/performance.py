"""
Performance tracking for Luna Nine operations.

Tracks timing and metadata for embeddings, rebuilds, queries.
Logs to CSV for analysis with pandas.
"""

import time
import csv
import json
from pathlib import Path
from functools import wraps
from typing import Any, Dict, Optional, Callable
from datetime import datetime

# Performance tracking flag - set to True to enable logging
ENABLE_PERFORMANCE_TRACKING = True

# Log file location
PERFORMANCE_LOG_FILE = Path("performance_log.csv")


class PerformanceTracker:
    """
    Simple performance tracker for Luna Nine operations.

    Logs operation timing and metadata to CSV.
    """

    def __init__(self, log_file: Path = PERFORMANCE_LOG_FILE):
        self.log_file = log_file
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Create log file with headers if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'operation',
                    'duration_ms',
                    'metadata'
                ])

    def log(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a performance measurement.

        Args:
            operation: Operation name (embed/rebuild/query/etc)
            duration_ms: Duration in milliseconds
            metadata: Optional dict with operation-specific data
        """
        if not ENABLE_PERFORMANCE_TRACKING:
            return

        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                operation,
                f"{duration_ms:.2f}",
                metadata_json
            ])

    def track(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator to track function performance.

        Usage:
            tracker = PerformanceTracker()

            @tracker.track("embed_messages", metadata={"batch_size": 10})
            def embed(messages):
                ...

        Args:
            operation: Operation name
            metadata: Static metadata (dynamic metadata should be added in function)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not ENABLE_PERFORMANCE_TRACKING:
                    return func(*args, **kwargs)

                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()

                duration_ms = (end - start) * 1000
                self.log(operation, duration_ms, metadata)

                return result
            return wrapper
        return decorator


# Global tracker instance
_tracker = PerformanceTracker()


def track_performance(operation: str, **metadata) -> Callable:
    """
    Convenience decorator for tracking performance.

    Usage:
        @track_performance("embed_messages", batch_size=10)
        def my_function():
            ...

    Args:
        operation: Operation name
        **metadata: Static metadata as kwargs
    """
    return _tracker.track(operation, metadata or None)


def log_performance(operation: str, duration_ms: float, **metadata):
    """
    Manually log a performance measurement.

    Usage:
        start = time.perf_counter()
        # ... do work ...
        duration = (time.perf_counter() - start) * 1000
        log_performance("my_operation", duration, size=100)

    Args:
        operation: Operation name
        duration_ms: Duration in milliseconds
        **metadata: Metadata as kwargs
    """
    _tracker.log(operation, duration_ms, metadata or None)
