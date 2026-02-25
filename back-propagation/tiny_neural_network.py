import random

import numpy as np

from autograd_tools import Tensor, add, mul, relu, matmul2x2_bias
from generate_dataset import generate_circles_with_band_gap
from visualization_utilities import plot_results, animate_training, animate_decision_boundaries


def forward_network(x1_val, x2_val, W1, b1, W2, b2):
    x = [Tensor(float(x1_val)), Tensor(float(x2_val))]

    # First layer gets the z1 
    z1 = matmul2x2_bias(W1, x, b1)
    # Then we apply ReLU in the second layer to get our h 
    h0 = relu(z1[0])
    h1 = relu(z1[1])

    # We use our h to get the final output score y_pred
    w2h0 = mul(W2[0], h0)
    w2h1 = mul(W2[1], h1)

    sum = add(w2h0, w2h1)
    y_pred= add(sum, b2) 

    z1[0].forward()
    z1[1].forward()
    h0.forward()
    h1.forward()
    w2h0.forward()
    w2h1.forward()
    sum.forward()
    y_pred.forward()

    return y_pred


def zero_grads(W1, b1, W2, b2):
    for row in W1:
        for param in row:
            param.grad = 0.0
    for param in b1:
        param.grad = 0.0
    for param in W2:
        param.grad = 0.0
    b2.grad = 0.0


def sgd_train(X, y, epochs=200, eta=0.01, seed=0):
    rng = random.Random(seed)

    W1 = [
        [Tensor(rng.uniform(-1.0, 1.0)), Tensor(rng.uniform(-1.0, 1.0))],
        [Tensor(rng.uniform(-1.0, 1.0)), Tensor(rng.uniform(-1.0, 1.0))],
    ]
    b1 = [Tensor(0.0), Tensor(0.0)]

    W2 = [Tensor(rng.uniform(-1.0, 1.0)), Tensor(rng.uniform(-1.0, 1.0))]
    b2 = Tensor(0.0)

    history = []
    w_trace = []

    def snapshot_params():
        w1_np = np.array(
            [
                [W1[0][0].data, W1[0][1].data],
                [W1[1][0].data, W1[1][1].data],
            ],
            dtype=float,
        )
        b1_np = np.array([b1[0].data, b1[1].data], dtype=float)
        w2_np = np.array([W2[0].data, W2[1].data], dtype=float)
        b2_val = float(b2.data)
        return (w1_np, b1_np, w2_np, b2_val)

    w_trace.append(snapshot_params())

    for _ in range(epochs):
        indices = list(range(len(X)))
        rng.shuffle(indices)

        epoch_loss = 0.0
        for i in indices:
            x1_val, x2_val = X[i]
            y_val = y[i]

            zero_grads(W1, b1, W2, b2)

            y_pred = forward_network(x1_val, x2_val, W1, b1, W2, b2)

            y_tensor = Tensor(-y_val)

            margin = mul(y_tensor, y_pred)
            bias = Tensor(1.0)
            margin_shifted = add(margin, bias)
            loss = relu(margin_shifted)
            
            margin.forward()
            margin_shifted.forward()
            loss.forward()

            loss.backward()

            for row in W1:
                for param in row:
                    param.data -= eta * param.grad
            for param in b1:
                param.data -= eta * param.grad
            for param in W2:
                param.data -= eta * param.grad
            b2.data -= eta * b2.grad

            epoch_loss += loss.data

        print("Epoch loss - ", epoch_loss) 

        history.append(epoch_loss / len(X))
        w_trace.append(snapshot_params())

    return W1, b1, W2, b2, history, w_trace


def main():

    # The data is sampled between [-2,2]^2 , don't increase radius beyond that.
    X, y = generate_circles_with_band_gap(n=400, radius=1.0, noise=0.05, seed=0)
    W1, b1, W2, b2, history, w_trace = sgd_train(X, y, epochs=100, eta=0.005, seed=0)

    y_pred = forward_network(X[0, 0], X[0, 1], W1, b1, W2, b2)
    print("Final loss:", history[-1])
    print("Sample prediction:", y_pred.data, "ground truth - ", y[0])

    plot_results(X, y, None, None, None, None, history)

    def nn_score(point, params):
        w1_np, b1_np, w2_np, b2_val = params
        z1 = w1_np @ point + b1_np
        h = np.maximum(0.0, z1)
        sum_h = float(np.dot(w2_np, h))
        return sum_h + b2_val

    animate_decision_boundaries(
        X,
        y,
        w_trace,
        score_fn=nn_score,
        output_path="Figures/nn_decision_boundaries.mp4",
        grid_size=160,
    )



if __name__ == "__main__":
    main()
