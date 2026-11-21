import importlib.util
from pathlib import Path
import unittest

from scripts import budget_run


class Budget(unittest.TestCase):
    def test_conservative_time_and_rejected_limits(self):
        seconds = budget_run.allowance(1.98, 1)
        self.assertLessEqual((seconds + 120) / 3600 * 1.98, 1)
        for price, budget in ((0, 1), (3, 1), (1.98, 50), (1.98, -1)):
            with self.assertRaises(ValueError):
                budget_run.allowance(price, budget)
