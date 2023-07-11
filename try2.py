with open('0.txt', 'r') as f:
    file_lines = f.readlines()

# file_lines = file_contents.splitlines()
position_x, position_y, position_z = float(file_lines[10][8:]), float(file_lines[11][8:]), float(file_lines[12][8:])
orientation_x, orientation_y, orientation_z, orientation_w = float(file_lines[14][8:]), float(file_lines[15][8:]), float(file_lines[16][8:]), float(file_lines[17][8:])

print(position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w)

import torch
a = torch.tensor([3, 3, 2])
print(len(a.shape))