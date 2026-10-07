from typing import Callable as _Callable

# --- Utils ---
type MathFunction = _Callable[[float], float]


def _check_equality(a: list[float], b: list[float]):
    if len(a) != len(b):
        raise Exception("two arrays are not the same length")


# --- Linear Algebra ---
def dot(arr1: list[float], arr2: list[float]) -> float:
    _check_equality(arr1, arr2)
    return sum(x * y for x, y in zip(arr1, arr2))


# --- Simple math expressions ---
def add(a: list[float], b: list[float]) -> list[float]:
    _check_equality(a, b)
    return [x + y for x, y in zip(a, b)]


def subtract(a: list[float], b: list[float]) -> list[float]:
    _check_equality(a, b)
    return [x - y for x, y in zip(a, b)]


def multiply(arr: list[float], scalar: float) -> list[float]:
    return [x * scalar for x in arr]


# --- Statistics ---
def mean(arr: list[float]) -> float:
    if not arr:
        raise ValueError("cannot calculate mean of empty list")
    return sum(arr) / len(arr)


# --- Calculus ---
def derivative(fn: MathFunction, x: float, h: float = 0.000001):
    return (fn(x + h) - fn(x)) / h


def integral(
    function: MathFunction,
    start: float,
    end: float,
    steps: int = 100000,
) -> float:
    dx = (end - start) / steps

    return sum(function(start + i * dx) * dx for i in range(steps))
