import sys
import math
from math import sqrt
import random
from typing import Self


class Vec:
    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = []
        else:
            elements = list(src)
            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")
            self.elements = elements


    def __add__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t.elements):
            raise TypeError("Type error - vectors must be of same dimensions")
        return Vec([x + y for x, y in zip(self.elements, t.elements)])

    def __sub__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t.elements):
            raise TypeError("Type error - vectors must be of same dimensions")
        return Vec([x - y for x, y in zip(self.elements, t.elements)])


    def __rmul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        return Vec([x * scalar for x in self.elements])

    def __mul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        return Vec([scalar * x for x in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        for i, val in enumerate(self.elements):
            self.elements[i] = val * scalar
        return self


    def __repr__(self) -> str:
        return repr(self.elements)


    def __len__(self) -> int:
      return len(self.elements)


    def __neg__(self) -> Self:
        return Vec([-x for x in self.elements])

    def __radd__(self, other) -> Self:
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")
        return other + self


    def __iadd__(self, other) -> Self:
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")
        if len(self) != len(other):
            raise TypeError("Type error - vectors must be of same dimensions")
        for i in range(len(self)):
            self.elements[i] = self.elements[i] + other.elements[i]
        return self


    @staticmethod
    def zeros(n: int) -> Self:
        if not isinstance(n, int):
            raise TypeError("The value of n should be of type integer")
        if n <= 0:
            raise ValueError("The n value must be greater than 0")
        return Vec([0] * n)

    @staticmethod
    def ones(n: int) -> Self:
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n <= 0:
            raise ValueError("n must be greater than 0")
        return Vec([1] * n)

    @staticmethod
    def uniform(n: int) -> Self:
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n <= 0:
            raise ValueError("n must be greater than 0")
        return Vec([random.uniform(0, 1) for _ in range(n)])


    def norm(self) -> float:
        sum = 0
        for i in range(len(self.elements)):
            sum = sum + self.elements[i] * self.elements[i]
        return round(math.sqrt(sum), 3)

if sys.version_info < (3, 8):
    sys.exit("Error: This script requires Python 3.8 or higher.")


if __name__ == "__main__":

    print("v1 vector is initialised here")
    v1 = Vec([2, -3, 4.5])
    print("v1 =", v1)

    v9 = v1 * 3
    print("v9 = ", v9)


    print("\nScalar multiplication of the vector v1 with scalar 3")
    v3 = 3 * v1
    print("3 * v1 =", v3)

    print("\nScalar multiplication of the vector v1 with scalar 2 (in-place multication)")
    v3 *= -2
    print("v3 *= -2 =", v3)

    print("\nAddition of two vectors v1 and v3 and is stored in v2")
    v2 = v1 + v3
    print("v1 + v3 =", v2)

    print("\n Subtraction of two vectors v3 and v1 and is stored in v4")
    v4 = v3 - v1
    print("v3 - v1 =", v4)

    print("\n Negation of vector v1 is stored in v5")
    v5 = -v1
    print("-v1 =", v5)


    print("\nAddition of two vectors v6 and v7 and is stored in v6 (in-place addition)")
    v6 = Vec([2, 4, 6])
    v7 = Vec([-1, 3, 5])
    print("v6 =", v6)
    print("v7 =", v7)

    v6 += v7

    print("After v6 += v7:")
    print("v6 =", v6)

    print("\n Returns a vector with zeroes")
    z = Vec.zeros(4)
    print("Vec.zeros(4) =", z)

    print("\n Returns a vector with ones")
    o = Vec.ones(6)
    print("Vec.ones(6) =", o)

    print("\n Returns a vector uniform random numbers")
    u = Vec.uniform(3)
    print("Vec.uniform(3) =", u)

    print("\n length of vector v1 len(v1) =", len(v1))

    print("\n Norm form")

    v8 = Vec([1, 2, 2])
    print("v8 =", v8)
    print("norm(v8) =", v8.norm())