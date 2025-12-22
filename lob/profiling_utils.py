import time
from contextlib import contextmanager
from collections import defaultdict


@contextmanager
def record_time(name, metrics_dict):
    """Context manager to record execution time of a code block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name in metrics_dict:
        metrics_dict[name].append(elapsed)
    else:
        metrics_dict[name] = [elapsed]


class GoodputMonitor:
    """Monitor and track data loading vs compute time."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record(self, name):
        """Returns a context manager for recording time."""
        return record_time(name, self.metrics)

    def get_stats(self, name):
        """Get statistics for a metric."""
        if name not in self.metrics:
            return None
        values = self.metrics[name]
        return {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }

    def print_summary(self):
        """Print summary of all metrics."""
        for name in sorted(self.metrics.keys()):
            stats = self.get_stats(name)
            print(f"{name:20s}: mean={stats['mean']:.4f}s, min={stats['min']:.4f}s, max={stats['max']:.4f}s, count={stats['count']}")
