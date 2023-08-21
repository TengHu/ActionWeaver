import collections
import logging
from typing import Dict


class TokenUsageTrackerException(Exception):
    pass


class TokenUsageTracker:
    def __init__(self, budget=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.tracker = collections.Counter()
        self.budget = budget

    def track_usage(self, usage: Dict):
        self.tracker = self.tracker + collections.Counter(usage)

        self.logger.debug(
            f"token used: {usage}, total: {dict(self.tracker)}, budget: {self.budget}"
        )

        if self.budget is not None and self.tracker["total_tokens"] > self.budget:
            self.logger.error(
                f"Token budget exceeded. Usage: {dict(self.tracker)}, Budget: {self.budget}"
            )
            raise TokenUsageTrackerException(
                f"Token budget exceeded. Budget: {self.budget}, Usage: {dict(self.tracker)}"
            )

        return self.tracker
