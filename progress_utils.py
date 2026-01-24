"""
Progress helpers with Rich when available.
"""

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TaskProgressColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class NoProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *args, **kwargs):
        return 0

    def update(self, *args, **kwargs):
        return None

    def advance(self, *args, **kwargs):
        return None


def build_progress(enabled=True, transient=False):
    if not enabled or not RICH_AVAILABLE:
        return NoProgress(), False

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[detail]}"),
        refresh_per_second=10,
        transient=transient,
    )
    return progress, True


def progress_stride(total, target_updates=1000):
    if total <= 0:
        return 1
    return max(1, total // target_updates)
