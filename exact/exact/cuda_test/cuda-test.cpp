#include <iostream>
#include <torch/torch.h>
using namespace std;
// using torch::autograd::Function;
// using torch::autograd::AutogradContext;
// using torch::autograd::tensor_list;
using torch::Tensor;
// using torch::IntArrayRef;
Tensor test_rand_cuda(int N);
std::pair<Tensor, uint64_t> low_mem_dropout_forward_cuda(Tensor data, float p);
std::pair<Tensor, Tensor> low_mem_dropout_backward_cuda(Tensor grad_output, uint64_t seed, float p);
int main(){
    int N = 10;
    Tensor data = torch::rand({4,4}).to(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    Tensor grad_output = torch::ones({4,4}, options);
    auto [output, seed] = low_mem_dropout_forward_cuda(data, 0.5);
    torch::cuda::synchronize();
    auto [grad_input, mask] = low_mem_dropout_backward_cuda(grad_output, seed, 0.5);
    torch::cuda::synchronize();
    cout<<data<<endl;
    cout<<output<<endl;
    cout<<grad_output<<endl;
    cout<<grad_input<<endl;
    cout<<mask<<endl;
    // Tensor a = test_rand_cuda(N);
    // cout << a << endl;

    return 0;
}