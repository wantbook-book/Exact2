import torch
import torch.nn
import time
import torch.nn.functional as F
import numpy as np
import torch_geometric
import torch_geometric.transforms as T
from exact.layers import QLowMemDropout

class LowMemDropoutNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.weight = torch.nn.Parameter(torch.rand(out_channels, in_channels))
        self.bias = torch.nn.Parameter(torch.rand(out_channels))
        self.dropout_layer = QLowMemDropout(dropout)
    def forward(self, input):
        return self.dropout_layer(F.linear(input, self.weight, self.bias))
        # return self.triton_dropout(input)

def get_memory_usage(gpu, print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated(gpu)
    reserved = torch.cuda.memory_reserved(gpu)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated

def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    # np.prod() calculate the number of elements
    for x in tensors:
        if x.dtype in [torch.int64, torch.long]:
            ret += np.prod(x.size()) * 8
        if x.dtype in [torch.float32, torch.int, torch.int32]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size())
        else:
            print(x.dtype)
            raise ValueError()
    return ret


def test_model_memory(model, input):
    gpu = 0
    MB = 1024 * 1024
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    print('='*20, 'Model and Optimizer Only', '='*20)
    model.cuda(gpu)
    usage = get_memory_usage(gpu, print_info=True)
    print('='*20, 'load data', '='*20)
    input = input.cuda()
    init_mem = get_memory_usage(gpu, print_info=True)
    data_mem = init_mem - usage
    print('='*20, 'before backward', '='*20)
    model.train()
    output = model(input)
    loss = output.sum()
    before_backward = get_memory_usage(gpu, print_info=True)
    act_mem = before_backward - init_mem - compute_tensor_bytes([loss, output])
    res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                        data_mem / MB,
                                                                        act_mem / MB)
    print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
    print(res)
    print('='*20, 'after backward', '='*20)
    loss.backward()
    optimizer.step()
    del loss
    after_backward = get_memory_usage(gpu, print_info=True)
    total_mem = before_backward + (after_backward - init_mem)
    res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                          data_mem / MB,
                                                                          act_mem / MB)
    print(res)
    print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')



def main():
    M,N,K = 1024,1024,1024
    
    net = LowMemDropoutNet(N,K,0.5)
    input = torch.rand(M,N)
    test_model_memory(net, input)

if __name__ == '__main__':
    main()