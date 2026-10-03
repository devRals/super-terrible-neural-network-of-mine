import random

type Values = list[float]


class ActivationFunction:
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, value: float) -> float:
        return self.fn(value)


ReLU = ActivationFunction(lambda x: max(0.0, x))
Linear = ActivationFunction(lambda x: x)


def linear(value: float) -> float:
    return value


class Neuron:

    def __init__(
        self, weights: Values, bias: float, activation: ActivationFunction = ReLU
    ):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, inputs: Values) -> float:
        value = dot(inputs, self.weights) + self.bias
        return self.activation(value)


class Layer:
    def __init__(self, neurons: list[Neuron]) -> None:
        self.neurons = neurons

    @classmethod
    def random(
        cls, n_inputs: int, n_neurons: int, activation: ActivationFunction = ReLU
    ) -> Layer:
        return cls(
            [
                Neuron(
                    [random.uniform(-1, 1) for _ in range(n_inputs)],
                    random.uniform(-1, 1),
                    activation,
                )
                for _ in range(n_neurons)
            ]
        )

    def forward(self, inputs: Values) -> Values:
        return [n.forward(inputs) for n in self.neurons]


class Network:
    def __init__(self, layers: list[Layer], output_layer: Layer) -> None:
        self.hidden_layers: list[Layer] = layers
        self.output_layer = output_layer

    def hidden_forward(self, inputs: Values) -> Values:
        for l in self.hidden_layers:
            inputs = l.forward(inputs)
        return inputs

    def forward(self, inputs: Values) -> Values:
        return self.output_layer.forward(self.hidden_forward(inputs))


def dot(arr1: list[float], arr2: list[float]) -> float:
    n = len(arr1)
    if n != len(arr2):
        raise Exception(f"mismatched arr lengths for dot product {n} != {len(arr2)}")
    total = 0.0
    for i in range(n):
        total += arr1[i] * arr2[i]
    return total
