from neural_network import Layer, Network, Neuron, Values, Linear, ReLU
import random


def main():
    dataset = [([x], [2 * x + 1]) for x in [0.0, 1.0, 2.0, 3.0, 4.0]]
    nn = Network(
        layers=[Layer.random(1, 10, ReLU), Layer.random(10, 10, ReLU)],
        output_layer=Layer.random(10, 1, Linear),
    )
    train(nn, dataset, 2000)
    for x in [0.5, 2.5, 10.0]:
        print(x, nn.forward([x]), 2 * x + 1)


def activation_grad(neuron: Neuron, out: float) -> float:
    if neuron.activation is ReLU:
        return 1.0 if out > 0 else 0.0
    return 1.0  # linear


def train(
    nn: Network,
    dataset: list[tuple[Values, Values]],
    train_time: int,
    lr: float = 0.001,
):
    for epoch in range(train_time):
        random.shuffle(dataset)  # avoids learning the order of examples
        total = sum(train_step(nn, x, y, lr) for x, y in dataset)
        if epoch % 100 == 0:
            print(f"epoch {epoch}: loss {total / len(dataset):.4f}")


def train_step(nn: Network, inputs: Values, expected: Values, lr: float) -> float:
    layers = nn.hidden_layers + [nn.output_layer]

    layer_inputs, layer_outputs = [], []
    x = inputs
    for layer in layers:
        layer_inputs.append(x)
        x = layer.forward(x)
        layer_outputs.append(x)

    loss = sum((o - e) ** 2 for o, e in zip(x, expected))
    grads = [2 * (o - e) for o, e in zip(x, expected)]

    r_layers, r_in_layers, r_out_layers = (
        reversed(layers),
        reversed(layer_inputs),
        reversed(layer_outputs),
    )

    for layer, xs, outs in zip(r_layers, r_in_layers, r_out_layers):
        prev_grads = [0.0] * len(xs)
        for neuron, out, g in zip(layer.neurons, outs, grads):
            g *= activation_grad(neuron, out)
            for i, xi in enumerate(xs):
                prev_grads[i] += g * neuron.weights[i]
                neuron.weights[i] -= lr * g * xi
            neuron.bias -= lr * g
        grads = prev_grads

    return loss


if __name__ == "__main__":
    main()
