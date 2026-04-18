import unittest

from module_2.week_2 import other_module
from module_2.week_2 import error_module


class TestBalanceFunction(unittest.TestCase):
    def test_balance_function(self):
        with self.assertRaises(error_module.BalanceError):
            other_module.func(10, 20)


if __name__ == "__main__":
    unittest.main()
