import random
import numpy as np

from visualization_utilities import plot_results, animate_training
from generate_dataset import generate_linearly_separable_data
from autograd_tools import Tensor, add, mul, relu


def approx_perceptron_algorithm(X, y, epochs=100, eta=0.005, seed=1):
    rng = random.Random(seed)

    w1 = Tensor(rng.uniform(-10, -8))
    w2 = Tensor(rng.uniform(10, 12))
    b = Tensor(rng.uniform(-1, 1))

    history = []
    param_trace = []

    for _ in range(epochs):
        indices = list(range(len(X)))
        rng.shuffle(indices)

        epoch_loss = 0.0
        for i in indices:
            '''Copy training script from Assignment Step 2-b of Section 1.2 here. ''' 
            x1_val, x2_val = X[i]
            y_val = y[i]
            
            x1 = Tensor(float(x1_val))
            x2 = Tensor(float(x2_val))
            y_tensor = Tensor(-float(y_val))
            
            wx1 = mul(w1, x1)
            wx2 = mul(w2, x2)
            
            sum = add(wx1, wx2)
            score = add(sum, b)
            
            margin = mul(y_tensor, score)
            
            loss = relu(margin)
            
            w1.grad = 0.0
            w2.grad = 0.0
            b.grad = 0.0
            
            wx1.forward()
            wx2.forward()
            
            sum.forward()
            
            score.forward()
            margin.forward()
            
            loss.forward()
            loss.backward()
            
            w1.data -= eta * w1.grad
            w2.data -= eta * w2.grad
            b.data -= eta * b.grad
            
            epoch_loss += loss.data   
            
            pass 
            

        history.append(epoch_loss / len(X))
        param_trace.append((w1.data, w2.data, b.data, history[-1]))

    return w1.data, w2.data, b.data, history, param_trace



def main():
    X, y, w_true, b_true = generate_linearly_separable_data()
    w1, w2, b, history, param_trace = approx_perceptron_algorithm(X, y)
    plot_results(X, y, w_true, b_true, (w1, w2), b, history)
    animate_training(X, y, w_true, b_true, param_trace)


if __name__ == "__main__":
    main()
