import collections
import time

from openai.types import CompletionUsage


class TokenUsageTrackerException(Exception):
    pass


class TokenUsageTracker:
    def __init__(self, budget=None, logger=None):
        self.logger = logger
        self.tracker = CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )
        self.budget = budget

    def clear(self):
        self.tracker = CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )
        return self

    def track_usage(self, usage: CompletionUsage):
        self.tracker = CompletionUsage(
            completion_tokens=self.tracker.completion_tokens + usage.completion_tokens,
            prompt_tokens=self.tracker.prompt_tokens + usage.prompt_tokens,
            total_tokens=self.tracker.total_tokens + usage.total_tokens,
        )

        if self.logger:
            self.logger.debug(
                {
                    "message": "token usage updated",
                    "usage": usage,
                    "total_usage": dict(self.tracker),
                    "timestamp": time.time(),
                    "budget": self.budget,
                },
            )

        if self.budget is not None and self.tracker.total_tokens > self.budget:
            if self.logger:
                self.logger.error(
                    {
                        "message": "Token budget exceeded",
                        "usage": usage,
                        "total_usage": self.tracker,
                        "budget": self.budget,
                    },
                    exc_info=True,
                )
            raise TokenUsageTrackerException(
                f"Token budget exceeded. Budget: {self.budget}, Usage: {self.tracker}"
            )
        return self.tracker
