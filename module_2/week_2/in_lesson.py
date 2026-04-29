from typing import Union

# Maybe you need to add argparse.


class Employee:
    def __init__(
        self,
        name: str | None = None,
        experience: int | None = None,
        salary: int | None = None,
    ) -> None:
        self.__salary = salary
        self.name = name
        self.experience = experience

    def __get_bonus(self) -> Union[float, None]:
        if self.__salary is not None and self.experience is not None:
            self.__bonus = self.__salary * self.experience * 0.5
        else:
            return None
        return self.__bonus

    @staticmethod
    def calculate_bonus(
        experience: int | None, salary: int | None
    ) -> Union[float, None]:
        if experience and salary:
            bonus = salary * experience * 0.5
        else:
            return None
        return bonus

    def get_info(self) -> None:
        print(f"Name: {self.name}")
        print(f"Salary: {self.__salary}")
        print(f"Bonus: {self.__get_bonus()}")


def main():
    e = Employee(name="Enforcer", experience=25, salary=10_000)
    e.get_info()
    Employee.calculate_bonus(5, 4500)


if __name__ == "__main__":
    main()
