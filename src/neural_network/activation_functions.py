import math
from typing import Final
from neural_network.math import MathFunction

type ActivationFunction = MathFunction
# Linear
Linear: Final[ActivationFunction] = lambda x: x

# Non-Linear
ReLU: Final[ActivationFunction] = lambda x: max(0, x)
LeakyReLU: Final[ActivationFunction] = lambda x: x if x > 0 else x * 0.00001
TanH: Final[ActivationFunction] = lambda x: 2 / (1 + math.exp(-2 * x)) - 1
SoftPlus: Final[ActivationFunction] = lambda x: math.log(1 + math.exp(x))

# Exponential Linear
ELU: Final[ActivationFunction] = lambda x: x if x > 0 else (math.exp(x) - 1)
SELU: Final[ActivationFunction] = lambda x: (
    1.05 * x if x > 0 else 1.67 * (math.exp(x) - 1)
)

# Output Layer Functions
Sigmoid: Final[ActivationFunction] = lambda x: (
    1 / (1 + math.exp(-x)) if x > 0 else 1 / (1 + math.exp(x))
)
