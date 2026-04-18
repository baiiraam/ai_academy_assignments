"""
Tasks and my implementations for week 1 of module_2
"""

import math


def calculate_gravitational_force(
    m1: float = 1, m2: float = 1, r: float = 1, G: float = 6.67430e-11
):
    """
    Returns the gravitational force between  given masses:
    F = G * m1 * m2 / r^2

    Parameters:
    m1: mass of the first object,
    m2: mass of the second object,
    r: distance between the centers of the two objects,
    G: gravitational constant (default value for Earth is 6.67430e-11)
    """
    if r <= 0:
        raise ValueError("Distance cannot be zero or negative")
    F = G * m1 * m2 / r**2
    rounded_F = round(F, 3)
    return rounded_F


def ideal_gas_law(P=None, V=None, T=None, n=6.022e23, R=8.314):
    """
    Returns the missing variable in the ideal gas equation:
    PV = nRT

    Parameters:
    P: pressure,
    V: volume,
    T: temperature,
    n: number of moles (default value is Avogadro),
    R: ideal gas constant (default value is 8.314)
    """
    # P is None
    if P is None:
        if V is None or T is None:
            raise ValueError("Only one variable can be None")
        return n * R * T / V
    # V is None
    elif V is None:
        if P is None or T is None:
            raise ValueError("Only one variable can be None")
        return n * R * T / P
    # T is None
    if P is None or V is None:
        raise ValueError("Only one variable can be None")
    return P * V / (n * R)


# All numbers could have been rounded, but I think it is ok for now


# math.pi = pi_value, math.exp(1) = e_value
def calculate_pdf(mu=0, sigma=1, x=0):
    """
    Returns the pdf value for a gaussian:
    1 over sigma times sqrt of 2 pi times e power -1/2 times ((x-mu) over sigma) squared

    Parameters:
    mu: mean,
    sigma: standard deviation,
    x: value to calculate pdf
    """
    if sigma <= 0:
        raise ValueError("Standard deviation cannot be negative or zero")
    coeff = 1 / sigma * math.sqrt(2 * math.pi)
    exponent = math.exp(-0.5 * math.pow((x - mu) / sigma, 2))
    return coeff * exponent


def credit_card_application():
    """
    Returns a decision (Approve, Deny) based on the following criteria:
    ...

    Parameters:
    age: age,
    credit_rating_is_excellent: credit rating is excellent or not (True/False)
    is_student: is a student or not (True/False)
    """
    age = int(input("Enter age: "))
    if age < 24:
        is_student = input("Are you a student? (y/n):").lower() == "y"
        if is_student:
            return "Approve"
        return "Deny"
    elif 24 <= age < 60:
        return "Approve"

    credit_rating_is_excellent = (
        input("Is your credit rating excellent? (y/n): ").lower() == "y"
    )
    if credit_rating_is_excellent:
        return "Approve"
    return "Deny"


def func():
    """
    Answer the questions with (y/n)
    """
    print("Answer the following questions with (y/n):")
    if input("Does the thing work?:\n").lower() == "y":
        print("Don't mess with it!")
        print("No problem")
    else:
        if input("Did you break it?:\n").lower() == "y":
            print("Did you break it?")
            if input("Does anyone know?:\n").lower() == "y":
                if input("Can you blame someone else?:\n").lower() == "y":
                    print("No problem")
                else:
                    print("Sorry to hear that!")
            else:
                print("Hide it!")
                print("No problem")
        else:
            if input("Will you be in trouble?:\n").lower() == "y":
                if input("Can you blame someone else?:\n").lower() == "y":
                    print("No problem")
                else:
                    print("Sorry to hear that!")
            else:
                print("Throw it away!")
                print("No problem")


def caesar_cipher(textt="ABCDE", shiftt=12):
    """
    Returns the caesar cipher of the given text and shift.

    Parameters:
    textt: text,
    shiftt: shift
    """
    res = []
    for char in textt:
        if char.isalpha() and char.isupper():
            encrypted_char = ord(char) + shiftt
            if encrypted_char > ord("Z"):
                encrypted_char -= 26
            res.append(chr(encrypted_char))
    return "".join(res)


def multiples_of_n_1():
    """
    Returns a list of multiples of n less than 100 for n in [2, 3, 7, 9]
    """
    for n in [2, 3, 7, 9]:
        ll = [n * i for i in range(51) if n * i < 100]
        print(f"n={n} --> {ll}")


def terms_of_a_series(N=5):
    """
    Returns the first N terms of defined series and cumulative sum.

    Parameters:
    N: number of items
    """
    res = []
    cumSum = 0
    for i in range(N + 1):
        term = (3 + 2 * i) / math.pow(2, i)
        res.append(term)
        cumSum += term
    return f"{res} \n {cumSum}"


def multiples_of_n(n, a=1, b=10):
    """
    Returns multiples of n between a and b.

    Parameters:
    n: number,
    a: range start,
    b: range end
    """
    res = []
    loww = a // n
    highh = b // n
    for i in range(loww, highh + 1):
        if a < n * i <= b:
            res.append(n * i)
    return res


def main():
    m1 = 1.989e30
    m2 = 5.972e24
    r = 14959787e04
    print(
        f"F for m1={m1}, m2={m2}, and r={r} is:\n {calculate_gravitational_force(m1, m2)}"
    )

    V = 0.25
    T = 300
    print(f"P for V={V} and T={T} is:\n{ideal_gas_law(V=V, T=T)}")

    print(credit_card_application())
    func()
    print(caesar_cipher("HELLO", 21))
    multiples_of_n_1()
    print(terms_of_a_series())
    print(multiples_of_n(3, 1, 30))


if __name__ == "__main__":
    main()