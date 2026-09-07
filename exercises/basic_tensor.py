import torch
import numpy as np

# Tensors are just like numpy arrays except they can be processed by GPUs and numpy arrays cannot. Syntax and implementations are pretty much the same.

data = [[1,2],[2,3]]

x_data = torch.tensor(data)

print(x_data)
print(x_data.device)

np_data = np.array(data)

x_np = torch.from_numpy(np_data)

print(x_np)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    print('GPU is available')
    tensor = tensor.to(torch.accelerator.current_accelerator())
else:
    print('GPU is not available')

print(f"Device tensor is stored on: {tensor.device}")

t1 = torch.cat([tensor, tensor, tensor], dim=1)

t2 = torch.stack([tensor, tensor, tensor], dim=0)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Convert one element tensor to item
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# If _ is added to the end of the function it means the function is a in-place operation. It directly stores the result to operand
print(f"{tensor} \n")
tensor2 = torch.ones_like(tensor)
tensor2.copy_(tensor)
tensor2.add(5)
tensor.add_(5)
print(tensor)
print(tensor2)

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
n = t.numpy()
print(t,n)

t.add_(3)
print(t,n)

np.add(n, 1, out=n)
print(t,n)