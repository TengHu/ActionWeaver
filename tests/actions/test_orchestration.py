from __future__ import annotations

import unittest

from actionweaver.actions import Orchestration


class OrchestrationTestCase(unittest.TestCase):
    def test_orchestration1(self):
        orch = Orchestration()

        orch["a"] = 1

        assert orch["a"] == 1
        assert len(orch) == 1


if __name__ == "__main__":
    unittest.main()
