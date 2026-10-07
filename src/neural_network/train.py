import random
from typing import List

from neural_network import Layer, Network, Values
from neural_network.math import derivative


class TrainableNetwork(Network):
    def __init__(self, layers: List[Layer]):
        super().__init__(layers)

    def train_once(self, data: tuple[Values, Values], learning_rate=0.01) -> float:
        input, expected = data

        prediction = self.forward(input)

        loss = sum((p - e) ** 2 for p, e in zip(prediction, expected)) / len(prediction)

        # 1. Initial errors for the output layer
        errors = [
            p - e for e, p in zip(expected, prediction)
        ]  # Note: (prediction - expected) or (expected - prediction) depending on loss convention

        # Keep track of errors to pass backward to previous layers
        next_layer_weights = None

        for layer in reversed(self.layers):
            new_errors = [0.0] * len(
                layer.neurons[0].weights
            )  # To accumulate errors for the previous layer

            for i, neuron in enumerate(layer.neurons):
                # Fill in the first blank: evaluate derivative using last_z
                deriv = derivative(neuron.activation, neuron.last_z)
                delta = deriv * errors[i]

                # Update weights and bias
                for j in range(len(neuron.weights)):
                    # Accumulate error to pass to the previous layer before updating weight
                    if next_layer_weights and j < len(next_layer_weights):
                        new_errors[j] += delta * next_layer_weights[j][i]

                    neuron.weights[j] -= learning_rate * delta * neuron.last_input[j]

                neuron.bias -= learning_rate * delta

            # Save current weights for the next backward step, and update errors for the previous layer
            next_layer_weights = [n.weights for n in layer.neurons]
            errors = new_errors
        return loss

    def train(
        self,
        train_time: int,
        data: list[tuple[Values, Values]],
        learning_rate=0.01,
    ):
        for epoch in range(train_time):
            random.shuffle(data)
            loss = 0.0
            first = True
            for d in data:
                loss = self.train_once(d, learning_rate)
                if first and epoch % 100 == 0:
                    print(f"{epoch}. step: Loss -> {loss}")
                    first = False
