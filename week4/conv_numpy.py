import numpy as np
import torch
import torch.nn.functional as F

width = 32
height = 32
image_depth = 3
image_count = 2

channel = 40
filter_size = 5

input_image = np.random.rand(image_count, image_depth, height, width)

filter = np.random.rand(channel, image_depth, filter_size, filter_size)


cols = []
for i in range(image_count):
    for r in range(28):
        for c in range(28):
            cols.append(input_image[i, :, r:r+5, c:c+5].reshape(filter_size**2 * image_depth, 1)) # create a column im2col!!
a = np.concatenate(cols, axis=1)

b = filter.reshape(channel, filter_size**2 * image_depth)

out = np.matmul(b,a)
out = out.reshape(channel, image_count, 28, 28)  # channel is outermost because that's out's actual row structure
out = out.transpose(1, 0, 2, 3)    # -> (image_count, channel, 28, 28), matches F.conv2d exactly   
print(out.shape)

input_image_np = input_image.reshape(2, 3, 32, 32)
filter_np = filter.reshape(40, 3, 5, 5)

input_image = torch.from_numpy(input_image_np).float()
filter_tensor = torch.from_numpy(filter_np).float()

output = F.conv2d(input_image, filter_tensor, stride=1, padding=0)
print(output.shape)

np_out = output.detach().numpy()

print(np.max(out-np_out))