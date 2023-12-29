import collections
import time

from openai.types import CompletionUsage


class TokenUsageTrackerException(Exception):
    pass


class TokenUsageTracker:
    def __init__(self, budget=None):
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

        if self.budget is not None and self.tracker.total_tokens > self.budget:
            raise TokenUsageTrackerException(
                f"Token budget exceeded. Budget: {self.budget}, Usage: {self.tracker}"
            )
        return self.tracker
