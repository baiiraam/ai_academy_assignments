from module_2.week_2.error_module import BalanceError


def func(b, w):
    """
    Raises custom exception when withdraw > balance.

    Parameters:
    b: balance
    w: withdraw amount"""
    if w > b:
        raise BalanceError("oops")


if __name__ == "__main__":
    balance = 10
    withdraw = 100
    func(10, 5)
    func(10, 50)
