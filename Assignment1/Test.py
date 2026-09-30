#This is the same test case as in vec.py code.
import sys;
import platform
sys.path.append("../")
from ALA_Lab.vector import Vec;

print("v1 vector is initialised here")
v1 = Vec([2, -3, 4.5])
print("v1 =", v1)


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