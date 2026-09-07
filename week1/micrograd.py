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

        print(topo)
        for ps in topo:
            print(ps.grad)



a = Value(5)
b = Value(6)
c = a * b
d = a + b
e = c * d

topo = e.backward()
