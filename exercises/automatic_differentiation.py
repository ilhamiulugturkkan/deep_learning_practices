import torch

## A simple nn
input = torch.ones(5)
output = torch.ones(3)
w = torch.rand(5,3, requires_grad=True)
b = torch.rand(3, requires_grad=True)
z = torch.matmul(input,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,output)

print(input.requires_grad)
print(z.requires_grad)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# To compute derivatives we call backward
print(w.grad)
print(b.grad)
loss.backward()
print(w.grad)
print(b.grad)

# If we only want to do forward computations through the network. We can stop tracking computations by surrounding our computation code with torch.no_grad() block: or use detach

with torch.no_grad():
    z = torch.matmul(input, w)+b

w.detach_()

print(z.requires_grad)
print(w.requires_grad)