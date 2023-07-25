import torch
import exact.layers as exact_layers
import exact.cpp_extension.quantization as cpp_quantization
class QLowMemDropoutNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.dropout_layer = exact_layers.QLowMemDropout(dropout)
    def forward(self, input):
        return self.dropout_layer(input)
    
N = 5
input = torch.rand(N, N, dtype=torch.float32, requires_grad=True).cuda()
model = QLowMemDropoutNet(N, N, 0.5).cuda()
output = model(input)
output.retain_grad()
input.retain_grad()
print(output)
loss = output.sum()
loss.backward()
print(output.grad)
print(input.grad)

# data = torch.rand(4,4,device='cuda')

# output, seed = cpp_quantization.low_mem_dropout_forward_cuda(data, 0.5)
# print(output)
# grad_output = torch.ones(4,4, device='cuda', dtype=torch.float32)
# print(grad_output)
# grad_input = cpp_quantization.low_mem_dropout_backward_cuda(grad_output, seed, 0.5)
# print(grad_input)