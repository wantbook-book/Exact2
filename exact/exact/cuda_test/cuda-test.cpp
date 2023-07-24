#include <iostream>
#include <torch/torch.h>
using namespace std;
// using torch::autograd::Function;
// using torch::autograd::AutogradContext;
// using torch::autograd::tensor_list;
using torch::Tensor;
// using torch::IntArrayRef;
Tensor test_rand_cuda(int N);
std::pair<Tensor, Tensor> low_mem_dropout_forward_cuda(Tensor data, float p);
int main(){
    int N = 10;
    Tensor data = torch::rand({4,4}).to(torch::kCUDA);
    auto [output, mask] = low_mem_dropout_forward_cuda(data, 0.5);
    cout<<data<<endl;
    cout<<output<<endl;
    cout<<mask<<endl;
    // Tensor a = test_rand_cuda(N);
    // cout << a << endl;

    return 0;
}