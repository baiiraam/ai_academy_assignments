from module_2.week_2.error_module import BalanceError

balance = 10
withdraw = 100


def func(b, w):
    if w > b:
        raise BalanceError("oops")


func(10, 5)
func(10, 50)
