'''
 Author: Dr. Souradeep Dutta
 
 This code is provided for demonstration and instructional purposes
 for CPEN 355 at the University of British Columbia (UBC).
 
 Course website:
 https://souradeep-dutta-01.github.io/ubc-cpen-355-website
 
'''

import math

class Tensor:
    def __init__(self, data=None, parents=(), forward_fn=None, backward_fn=None):
        self.data = data
        self.grad = 0.0

        self.parents = parents
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self):
        """
        Compute the forward value using the forward_fn.
        """
        if self.forward_fn is not None:
            self.data = self.forward_fn()

        return self.data

    def backward(self, grad=1.0):
        """
        Backpropagate gradients.
        """
        self.grad += grad

        if self.backward_fn is not None:
            self.backward_fn(grad)


def mul(a: Tensor, b: Tensor):
    def forward_fn():
        return a.data * b.data

    out = Tensor(parents=(a, b), forward_fn=forward_fn)

    def backward_fn(grad_out):
        a.backward(grad_out * b.data)  # ∂(ab)/∂a = b
        b.backward(grad_out * a.data)  # ∂(ab)/∂b = a

    out.backward_fn = backward_fn
    return out

def sigmoid(x: Tensor):
    def forward_fn():
        return 1.0 / (1.0 + math.exp(-x.data))

    out = Tensor(parents=(x,), forward_fn=forward_fn)

    def backward_fn(grad_out):
        sig = out.data
        local_grad = sig * (1.0 - sig)
        x.backward(grad_out * local_grad)

    out.backward_fn = backward_fn
    return out

def squared_error(y_pred: Tensor, y_true: float):
    def forward_fn():
        diff = y_pred.data - y_true
        return 0.5 * diff * diff

    out = Tensor(parents=(y_pred,), forward_fn=forward_fn)

    def backward_fn(grad_out):
        diff = y_pred.data - y_true
        y_pred.backward(grad_out * diff)

    out.backward_fn = backward_fn
    return out

def main():
    # Inputs and parameters
    x = Tensor(2.0)
    w1 = Tensor(0.5)
    w2 = Tensor(1.5)
    y = 1.0

    # Build computation graph
    z1 = mul(w1, x)
    a1 = sigmoid(z1)
    y_pred = mul(w2, a1)
    loss = squared_error(y_pred, y)

    # Forward pass (manual, explicit)
    z1.forward()
    a1.forward()
    y_pred.forward()
    loss.forward()

    # Backward pass
    loss.backward()

    print("Loss:", loss.data)
    print("dL/dw1:", w1.grad)
    print("dL/dw2:", w2.grad)
    print("dL/dx:", x.grad)

if __name__ == "__main__":
    main()
