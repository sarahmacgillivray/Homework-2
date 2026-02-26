'''
 Author: Sarah MacGillivray
    Date: 2024-06-01
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

def add(a, b):
    def forward_fn():
        return a.data + b.data

    def backward_fn(grad_out):
        a.backward(grad_out)
        b.backward(grad_out)

    return Tensor(parents=(a, b), forward_fn=forward_fn, backward_fn=backward_fn)

def relu(x: Tensor):
    def forward_fn():
        return max(0.0, x.data)

    # setup the tensor to send out 
    out = Tensor(parents=(x,), forward_fn=forward_fn)

    # Relu just uses above 0 as a grad to be pass, else it sends no extra grad
    def backward_fn(grad_out):
        local_grad = 1.0 if x.data > 0 else 0.0 
        x.backward(grad_out * local_grad)

    out.backward_fn = backward_fn
    return out


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

def matmul2x2_bias(W, x, b):
    ''' Helper function for 2x2 matrix multiplication with bias.
        W: 2x2 weight matrix (list of lists)
        x: 2D input vector (list)
        b: bias vector (list)
        return 2D output vector (list) 
        '''
    # Since add and mul already have forwards and backwards defined, we can just use those to build this function 
    t00 = mul(W[0][0], x[0])
    t01 = mul(W[0][1], x[1])
    t10 = mul(W[1][0], x[0])
    t11 = mul(W[1][1], x[1])
    
    # We need to call forward to make sure data becomes chained together for backpropagation 
    # because we made mul objects, they need to do their calculations
    t00.forward()
    t01.forward()
    t10.forward()
    t11.forward()
    
    # Now we add the products together and add the bias to get the final output of this layer
    y0 = add(t00, t01)
    y1 = add(t10, t11)

    # Must send the output of the matrix multiplication through forward to chain the data together for backpropagation
    y0.forward()
    y1.forward()
    
    # Finally we get our output tensors for this layer by adding the bias and calling forward again to chain the data together for backpropagation
    z0 = add(y0, b[0])
    z1 = add(y1, b[1])
    z0.forward()
    z1.forward()
    return z0, z1

def main():

    '''Once the right functions are here this should work.'''
    
    # Sample input (2D) and label
    x1 = Tensor(2.0)
    x2 = Tensor(-1.0)
    y_true = -1.0

    # Parameters
    w1 = Tensor(0.5)
    w2 = Tensor(-0.25)
    b = Tensor(0.1)

    # Linear score and prediction
    wx1 = mul(w1, x1)
    wx2 = mul(w2, x2)
    sum = add(wx1, wx2)
    score = add(sum, b)
    y_pred = sigmoid(score)

    # Hinge loss: max(0, -y * score)
    y = Tensor(-y_true)
    margin = mul(y, score)
    loss = relu(margin)

    # Forward pass
    wx1.forward()
    wx2.forward()
    sum.forward()
    score.forward()
    y_pred.forward()
    margin.forward()
    loss.forward()

    # Backward pass
    loss.backward()

    print("Score:", score.data)
    print("Prediction:", y_pred.data)
    print("Loss:", loss.data)
    print("dL/dw1:", w1.grad)
    print("dL/dw2:", w2.grad)
    print("dL/db:", b.grad)
    print("dL/dx1:", x1.grad)
    print("dL/dx2:", x2.grad)

if __name__ == "__main__":
    main()
