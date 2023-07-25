from exact.ops import low_mem_input2rp, low_mem_rp2input
import torch
input = torch.rand(8,8, device='cuda')
kept_acts = 4
dim_reduced_input, rand_mat_size, seed = low_mem_input2rp(input, kept_acts)
input_recovered = low_mem_rp2input(dim_reduced_input, input.shape, seed, rand_mat_size)
