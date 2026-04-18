import unittest

from module_2.week_2 import other_module
from module_2.week_2 import error_module

from module_2.week_1 import tasks


class TestBalanceFunction(unittest.TestCase):
    def test_balance_function(self):
        with self.assertRaises(error_module.BalanceError):
            other_module.func(10, 20)


class TestFirstWeekFunctions(unittest.TestCase):
    def test_function(self):
        self.assertEqual(tasks.caesar_cipher("HELLO"), "TQXXA")


if __name__ == "__main__":
    unittest.main()
