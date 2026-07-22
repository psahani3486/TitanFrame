from titanframe.telemetry.tracker import TelemetryTracker, tracker

class MemoryTracker:

    def __init__(self):
        self._tracker = tracker

    def get_stats(self):
        return self._tracker.get_snapshot()
__all__ = ['TelemetryTracker', 'tracker', 'MemoryTracker']
