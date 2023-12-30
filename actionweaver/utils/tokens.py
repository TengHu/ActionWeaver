import collections
from typing import Dict


class TokenUsageTrackerException(Exception):
    pass


class TokenUsageTracker:
    def __init__(self, budget=None):
        self.tracker = collections.Counter()
        self.budget = budget

    def clear(self):
        self.tracker = collections.Counter()
        return self

    def track_usage(self, usage: Dict):
        self.tracker = self.tracker + collections.Counter(usage)

        if self.budget is not None and self.tracker["total_tokens"] > self.budget:
            raise TokenUsageTrackerException(
                f"Token budget exceeded. Budget: {self.budget}, Usage: {dict(self.tracker)}"
            )
        return self.tracker
