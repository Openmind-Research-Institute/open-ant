import csv
import signal
import socket
from datetime import datetime


class StepStatsLogger:
    """Collects per-step timing rows in memory and flushes them to CSV on ctrl-c."""

    def __init__(self, prefix="step_stats"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()
        self.path = f"{prefix}_{hostname}_{timestamp}.csv"
        self.rows = []
        self._fieldnames = []
        self._prev_handler = signal.signal(signal.SIGINT, self._on_sigint)

    def record(self, **row):
        if not self._fieldnames:
            self._fieldnames = list(row.keys())
        self.rows.append(row)

    def flush(self):
        if not self.rows:
            return
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def _on_sigint(self, signum, frame):
        self.flush()
        signal.signal(signal.SIGINT, self._prev_handler)
        raise KeyboardInterrupt

    def close(self):
        signal.signal(signal.SIGINT, self._prev_handler)
        self.flush()
