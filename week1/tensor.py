import torch

data = [[1,2],[3,4]]

tensor = torch.tensor(data)

tensor_init = torch.ones_like(tensor)

tensor_init2 = torch.ones(1,5)

print(tensor)

print(tensor_init)

print(tensor_init2)

tensor = tensor.reshape(4,1)

print(tensor)

tensor = tensor.add(1) # broadcasting

tensor = tensor.mul(2)

tensor = tensor.float()

print(tensor)

tensor = torch.matmul(tensor,tensor_init2)

print(tensor)

for i in range(2):

    print(tensor.sum(dim=i))


print(tensor[:,:])

print(tensor[1,:])
print(tensor[1:3,:])
print(tensor[1:3,-3:-1])

print(tensor)
print(tensor.view(10,2))

t2 = tensor.view(20)
t3 = tensor.reshape(20)

print(t2)
print(t3)