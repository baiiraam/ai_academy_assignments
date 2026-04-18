class Car:
    def __init__(self, name='default_name', model='default_model'):
        self.name=name
        self.model=model

car1=Car()
print(car1.name)


class Person:
    def __init__(self, name='default_name', age='default_age'):
        self.name=name
        self.age=age

person1=Person()

print(person1.name)