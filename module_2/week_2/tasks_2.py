class InsufficientBalanceError(Exception):
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(
            f"Insufficient balance! Available: {balance}, Requested: {amount}"
        )


class Person:
    def __init__(self, name, surname, age, address):
        self.__name = name
        self.__surname = surname
        self.__age = age
        self.__address = address

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, age):
        if age > 0:
            self.__age = age
        else:
            print("Age must be greater than 0")

    def info(self):
        print(
            f"{self.__name} {self.__surname} is {self.__age} years old, lives at {self.__address}"
        )


class Employee:
    def __init__(self, name, salary, mkr_score):
        self.name = name
        self.__mkr_score = mkr_score
        self.salary = salary

    @property
    def salary(self):
        return self.__salary

    @salary.setter
    def salary(self, salary):
        if 1000 <= salary <= 2000:
            self.__salary = salary - (salary * 0.4)
        elif 2000 < salary <= 5000:
            self.__salary = salary - (salary * 0.25)
        elif salary > 5000:
            self.__salary = salary - (salary * 0.15)
        else:
            print("Invalid salary amount")
            self.__salary = 0


class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance

    @property
    def balance(self):
        return self.__balance

    @balance.setter
    def balance(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Balance must be int or float")
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self.__balance = value
        if self.__balance > 10000:
            print("VIP account")

    @property
    def owner(self):
        return self.__owner

    @owner.setter
    def owner(self, name):
        if not name.isalpha():
            raise ValueError("Owner name must contain only letters")
        if len(name) < 3:
            raise ValueError("Owner name must be at least 3 characters")
        self.__owner = name

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.__balance += amount
        if self.__balance > 10000:
            print("VIP account")

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.__balance:
            raise InsufficientBalanceError(self.__balance, amount)
        self.__balance -= amount


class Student:
    def __init__(self, name, age, grades):
        self.name = name
        self.age = age
        self.grades = grades

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self.__name = name

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, age):
        if not 16 <= age <= 30:
            raise ValueError("Age must be between 16 and 30")
        self.__age = age

    @property
    def grades(self):
        return self.__grades

    @grades.setter
    def grades(self, grades):
        if not isinstance(grades, list):
            raise TypeError("Grades must be a list")
        for grade in grades:
            if not (0 <= grade <= 100):
                raise ValueError("Each grade must be between 0 and 100")
        self.__grades = grades

    @property
    def average(self):
        if not self.__grades:
            return 0
        return sum(self.__grades) / len(self.__grades)

    @property
    def status(self):
        avg = self.average
        if avg >= 90:
            return "Excellent"
        elif avg >= 75:
            return "Good"
        elif avg >= 60:
            return "Normal"
        else:
            return "Fail"