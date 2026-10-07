from typing import Final

from neural_network import Layer, Network, Values
from neural_network.activation_functions import Linear, Sigmoid
from neural_network.train import TrainableNetwork

PROMPT: Final = ">> "


def fn(x: float) -> float:
    return x * 2 + 1


def repl(nn: Network):
    while True:
        try:
            user_input = input(PROMPT)
            if user_input.lower() == "q":
                break
            value = float(user_input)
            expected = fn(value)
            prediction = nn.forward([value])[0]

            print(f"Expected: {expected}, Prediction: {prediction}")
        except ValueError:
            print("Not a number")


def main() -> int | None:
    layers: list[Layer] = [
        Layer(input_count=1, neuron_count=8, activation_function=Sigmoid),
        Layer(input_count=8, neuron_count=1, activation_function=Linear),
    ]
    tnn = TrainableNetwork(layers)

    input: list[Values] = [[x] for x in [30.0, 5.0, 12.0, -23.0, 0.0, 4.3, 1.0]]
    expected: list[Values] = [[fn(x[0])] for x in input]

    data = [(i, e) for i, e in zip(input, expected)]

    tnn.train(train_time=5_000, data=data, learning_rate=0.001)
    repl(tnn)
    return None


if __name__ == "__main__":
    exit(main())
