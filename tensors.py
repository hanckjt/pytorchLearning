import torch
import numpy as np

data = [[1, 2], [3, 4]]
print(f"Source data:\n\t{data}\n")

x_data = torch.tensor(data)
x_data = x_data.to("cuda")
print(f"x_data({x_data.device}):\n\t{x_data}\n")

np_array = np.array(data)
print(f"np_array:\n\t{np_array}\n")

x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"x_rand:\n\t{x_rand}\n")
