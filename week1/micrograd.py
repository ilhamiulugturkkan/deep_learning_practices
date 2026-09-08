import random


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda : None

    def __repr__(self):
        return f'{self.data}'

    def __add__(self, other):
        if isinstance(other, Value) == False:
            other = Value(other)

        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Value) == False:
            other = Value(other)

        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * -1
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):

        if isinstance(other, Value) == False:
            other = Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(0, (self,), 'relu')
        if self.data > 0:
            out.data = self.data

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out
        

    def backward(self):
        topo = []
        visited = set()
        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    topo_sort(c)
                topo.append(v)
        topo_sort(self)

        self.grad = 1.0

        for n in reversed(topo):
            n._backward()



a = Value(5)
b = Value(6)
c = a * b
d = a + b
e = c * d
f = e.relu()

topo = f.backward()

class Neuron:

    def __init__(self, nin = 0):
        self.w = [Value(random.uniform(0,3)) for _ in range(nin)]
        self.b = Value(random.uniform(0,1))
        self.out = 0

    def __repr__(self):
        return f'{self.w}, {self.b}'

    def __call__(self, x):
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out.relu()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, size):
        self.neurons = [Neuron(nin) for _ in range(size)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def __repr__(self):
        return f'{[n for n in self.neurons]}'

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, input_size, hl_sizes = [], class_count = 0):
        iter = hl_sizes + [class_count,]
        self.layers = [Layer(input_size if i == 0 else iter[i-1], size) for i, size in enumerate(iter)]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# INIT MLP and our target
mlp = MLP(4, [4,4,4], 1)
target = Value(700)



# LET'S TRAIN MLP
learning_rate = 0.00000001
for i in range(100):
    y_pred = mlp([4,1,7,8]) # FEED FORWARD
    print(f'Epoch : {i}, Y_pred = {y_pred}')

    cost = (target - y_pred) * (target - y_pred) # WEIRD cost function

    for p in mlp.parameters(): # ZERO_GRAD
        p.grad = 0

    cost.backward() # BACK PROPOGATION

    # UPDATE WEIGHTS
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad

    



