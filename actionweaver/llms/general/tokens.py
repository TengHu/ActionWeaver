import collections
import logging
import time
from typing import Dict


class TokenUsageTrackerException(Exception):
    pass


class TokenUsageTracker:
    def __init__(self, budget=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.tracker = collections.Counter()
        self.budget = budget

    def clear(self):
        self.tracker = collections.Counter()
        return self

    def track_usage(self, usage: Dict):
        self.tracker = self.tracker + collections.Counter(usage)

        self.logger.debug(
            {
                "message": "token usage updated",
                "usage": usage,
                "total_usage": dict(self.tracker),
                "timestamp": time.time(),
                "budget": self.budget,
            },
        )
        if self.budget is not None and self.tracker["total_tokens"] > self.budget:
            self.logger.error(
                {
                    "message": "Token budget exceeded",
                    "usage": usage,
                    "total_usage": dict(self.tracker),
                    "budget": self.budget,
                },
                exc_info=True,
            )
            raise TokenUsageTrackerException(
                f"Token budget exceeded. Budget: {self.budget}, Usage: {dict(self.tracker)}"
            )
        return self.tracker
