"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

def mul(a: float, b: float) -> float:
    """
    Multiplies two floats and returns the result.
    """

    return a * b

def id(x: any):
    """
    Returns the value passed to the function.
    """

    return x

def add(a: float, b: float) -> float:
    """
    Adds two floats and returns the result.
    """

    return a + b

def neg(x: float) -> float:
    """
    Returns the negation of a float passed to it.
    """

    return -x

def lt(a: float, b: float) -> float:
    """
    Checks if one number is less than another.
    """

    return 1.0 if a < b else 0.0

def eq(a: float, b: float) -> float:
    """
    Checks if two numbers are equal to each other.
    """

    return 1.0 if a == b else 0.0

def max(a: float, b: float) -> float:
    """
    Returns the max of two numbers.
    """

    return a if a > b else b

def is_close(a: float, b: float) -> float:
    """
    Checks if two numbers are close in value
    """

    return 1.0 if abs(a - b) < 1e-2 else 0.0

def sigmoid(x: float) -> float:
    """
    Calculates the sigmoid function
    """

    if x >= 0:
        return (1 / (1 + math.exp(-x)))
    else:
        exp_x = math.exp(x)
        return (exp_x / (1 + exp_x))

def relu(x: float) -> float:
    """
    Applies the ReLU activation function
    """

    return x if x >= 0.0 else 0.0

def log(x: float) -> float:
    """
    Calculates the natural logarithm
    """

    return math.log(x)

def exp(x: float) -> float:
    """
    Calculates the exponential function
    """

    return math.exp(x)

def inv(x: float) -> float:
    """
    Calculates the reciprocal
    """

    return 1 / x

def log_back(a: float, b: float) -> float:
    """
    Computes the derivative of log times a second argument.
    
    The derivative of ln(x) is 1/x.
    """

    return inv(a) * b

def inv_back(a: float, b: float) -> float:
    """
    Computes the derivative of inv times a second argument.

    The derivative of 1/x is -1/x^2.
    """

    return (-1 / a ** 2) * b

def relu_back(a: float, b: float) -> float:
    """
    Computes the derivative of ReLU times a second argument.

    The derivative of ReLU is 0 if x < 0 and 1 if x >= 0.
    """

    return b if a >= 0 else 0

def sigmoid_back(a: float, b: float) -> float:
    sig_a = sigmoid(a)
    return (sig_a * (1 - sig_a)) * b

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(map_func: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    mapped_ls = []

    for x in ls:
        mapped_ls.append(map_func(x))

    return mapped_ls

def zipWith(zip_func: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    end_ls = []
    list_len = len(ls1) if len(ls1) <= len(ls2) else len(ls2)

    for i in range(list_len):
        end_ls.append(zip_func(ls1[i], ls2[i]))

    return end_ls

def reduce(reduce_func: Callable[[float, float], float], ls: Iterable[float]) -> float:
    if len(ls) == 1:
        return ls[0]
    elif len(ls) == 0:
        return 0

    func_value = reduce_func(ls[0], ls[1])

    for i in range(2, len(ls)):
        func_value = reduce_func(func_value, ls[i])

    return func_value

def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg, ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    return reduce(add, ls)

def prod(ls: Iterable[float]) -> float:
    return reduce(mul, ls)
