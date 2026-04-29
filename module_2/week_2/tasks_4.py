# tasks about threading and multiprocessing

import concurrent.futures
from time import sleep, perf_counter
import requests


def send_email():
    print("send_email start")
    sleep(2)
    print("send_email finish")


def download_email():
    print("download_email start")
    sleep(2)
    print("download_email finish")


def save_log():
    print("save_log start")
    sleep(2)
    print("save_log finish")


if __name__ == "__main__":
    t1 = perf_counter()
    functions = [send_email, download_email, save_log]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda f: f(), functions)
    t2 = perf_counter()
    print(f"{t2 - t1} seconds passed")


# 10 different urls
urls = [
    "https://Google.com",
    "https://Facebook.com",
    "https://YouTube.com",
    "https://Amazon.com",
    "https://Wikipedia.org",
    "https://X.com",
    "https://Instagram.com",
    "https://LinkedIn.com",
    "https://Netflix.com",
    "https://Microsoft.com",
]


def get_url_and_print_status_code(url):
    res = requests.get(url)
    print(f"{url} - status code: {res.status_code}")
    return res.status_code


if __name__ == "__main__":
    t1 = perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(get_url_and_print_status_code, urls))
    t2 = perf_counter()
    print(f"{t2 - t1} seconds passed")


# Lock
# def increasing_1(num_list):
#     for _ in range(100_000):
#         num_list[0] += 1

# def increasing_2(num_list):
#     for _ in range(100_000):
#         num_list[0] += 1

# if __name__ == "__main__":
#     t1 = perf_counter()
#     counterr = [0]
#     functions = [increasing_1, increasing_2]
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = executor.map(lambda f: f(counterr), functions)

#     for result in results:
#         pass
#     t2 = perf_counter()
#     print(f"{t2-t1} seconds passed")
#     print(f"final value of counter: {counterr[0]}")


# 4 processes
def square_num(num):
    return num * num


if __name__ == "__main__":
    nums = [num for num in range(1_000)]
    t1 = perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(square_num, nums)

    for result in results:
        pass
    t2 = perf_counter()
    print(f"{t2 - t1} seconds passed")
    print("let's try sequentially")
    t1 = perf_counter()
    for num in nums:
        square_num(num)
    t2 = perf_counter()
    print(f"{t2 - t1} seconds passed")


# interesting thing I got. I will try factorial
# import math

# def calculate_factorial(num):
#     return math.factorial(num)

# if __name__ == "__main__":
#     nums = [num for num in range(1, 101)]
#     t1 = perf_counter()
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = executor.map(calculate_factorial, nums)
#         factorial_results = list(results)
#     t2 = perf_counter()
#     print(f"Multiprocessing took {t2-t1} seconds")
#     t1 = perf_counter()
#     seq_res = [calculate_factorial(num) for num in nums]
#     t2 = perf_counter()
#     print(f"sequential took {t2-t1} seconds")


# sequential is still faster ;)
# I will try with big numbers this time

# def square_num(n):
#     return n*n

# def cube_num(n):
#     return n*n*n

# if __name__ == "__main__":
#     num_ranges = [100, 300, 500, 700]
#     for num_range in num_ranges:
#         nums = [num for num in range(num_range + 1)]

#         t1 = perf_counter()
#         seq_res = [calculate_factorial(n) for n in nums]
#         square_seq_res = [square_num(n) for n in seq_res]
#         cube_seq_res = [cube_num(n) for n in square_seq_res]
#         t2 = perf_counter()

#         print(f"Sequential processing for {num_range} numbers took {t2-t1} seconds")

#         t1 = perf_counter()
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             mp_results = executor.map(calculate_factorial, nums)
#             mp_square_res = [n*n for n in mp_results]
#             mp_cube_res = [n*n*n for n in mp_square_res]

#         t2 = perf_counter()
#         print(f"Multi-processing for {num_range} numbers took {t2-t1} seconds")

# Sequential is still faster for the above case.

# Why the below thing works?
# def sum_of_squares(n):
#     return sum(n*n for n in range(n))

# if __name__ == "__main__":
#     nums = [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000]
#     t1 = perf_counter()
#     seq_res = [sum_of_squares(i) for i in nums]
#     t2 = perf_counter()
#     print(f"Sequential took {t2-t1} seconds")

#     t1 = perf_counter()
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         mp_res = list(executor.map(sum_of_squares, nums))
#     t2 = perf_counter()
#     print(f"Multi took {t2-t1} seconds")


# Fifth exercise
def return_square(num):
    return num * num


if __name__ == "__main__":
    nums = [2 * i for i in range(1, 7)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        mp_res = list(executor.map(return_square, nums))
    print(mp_res)
