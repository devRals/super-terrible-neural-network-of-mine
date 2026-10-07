import random
from typing import List

from neural_network.activation_functions import Linear, ActivationFunction
from neural_network.math import dot

type Values = List[float]
type Weights = List[float]


class Network:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, input: Values) -> Values:
        for layer in self.layers:
            input = layer.forward(input)
        return input


class Layer:
    def __init__(
        self,
        input_count: int,
        neuron_count: int,
        activation_function: ActivationFunction = Linear,
    ):
        self.neurons = [
            Neuron(
                [random.uniform(-1, 1) for _ in range(input_count)],
                random.uniform(-1, 1),
                activation_function,
            )
            for _ in range(neuron_count)
        ]

    def forward(self, input: Values) -> Values:
        return [n.forward(input) for n in self.neurons]


class Neuron:
    def __init__(self, weights: Weights, bias: float, activation: ActivationFunction):
        self.weights = weights
        self.bias = bias
        self.activation = activation

        self.last_input: Values = []

    def forward(self, input: Values) -> float:
        self.last_input = input
        self.last_z = dot(self.weights, input) + self.bias
        return self.activation(self.last_z)
