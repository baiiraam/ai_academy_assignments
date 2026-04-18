from typing import Any


class Car:
    def __init__(self, name: str = "default_name", model: str = "default_model"):
        self.name = name
        self.model = model


car1 = Car()
print(car1.name)


class Person:
    def __init__(self, name: str = "default_name", age: str | int = "default_age"):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Name: {self.name}, age: {self.age}"

    def __del__(self):
        return f"instance is deleted: {self.name}"


person1 = Person()
print(person1.name)
person1.location = "some_location"  # bad practice
print(person1.location)
print(person1)


class LivingThing:
    pass


class Animal(LivingThing):
    def __init__(self, name="animal", age=2):
        self.name = name
        self.age = age


class Dog(Animal):
    def __init__(self, name, age=2):
        super().__init__(name, age)

    def bark(self):
        return f"Woof!"

    def __str__(self):
        return f"Dog: {self.name} {self.age}"


class Alabay(Dog):
    def __init__(self, age=2, nickname="Toplan"):
        super().__init__(age)
        self.nickname = nickname


alabay1 = Alabay()
print(alabay1.mro())

# Overloading
from multipledispatch import dispatch


@dispatch(str)
def print_data(data):
    print(f"str data {data}")


@dispatch(list)
def print_data(data):
    print(f"list data {data}")


@dispatch(int)
def print_data(data):
    print(f"int data {data}")


print_data("str")
print_data([1, 2, 3])


def print_data_obj(data):
    if isinstance(data, Animal):
        print(data.name, data.age)
    elif isinstance(data, Dog):
        print("dog instance")
    elif isinstance(data, Alabay):
        print("Alabay")
    else:
        print("probably living thing")


dog = Dog("shepherd", 11)
print_data_obj(dog)

alabayy = Alabay()
print_data_obj(alabayy)
