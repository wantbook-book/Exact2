#include <iostream>
#include <torch/torch.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <chrono>
#include <ATen/Dispatch.h>

// #include <THC/THCAtomics.cuh>
// #include <THC/THCGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ATen/ATen.h>
// #include <ATen/AccumulateType.h>
// #include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
// #include <ATen/cuda/CUDAGraphsUtils.cuh>
// #include <c10/macros/Macros.h>
// #include <ATen/native/TensorIterator.h>
// #include <ATen/native/cuda/Loops.cuh>
using torch::IntArrayRef;
using torch::Tensor;
#define RN_NUM_THREADS 512
__global__ void test_rand_kernel(
    at::cuda::detail::TensorInfo<float, int64_t> output_info,
    std::pair<uint64_t, uint64_t> seeds,
    int N
){
    const int64_t id = blockIdx.x*blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seeds.first, id, seeds.second, &state);
    float rand = curand_uniform(&state);
    if(id < N){
        const int64_t offset = at::cuda::detail::IndexToOffset<float, int64_t, 1>::get(id, output_info);
        output_info.data[offset] = rand;
    }

}

Tensor test_rand_cuda(int N){
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    Tensor output = torch::ones(N, options);
    int64_t block_size = RN_NUM_THREADS;
    dim3 dim_block(block_size);
    dim3 dim_grid((N + block_size - 1) / block_size);
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout<<"seed:"<<seed<<std::endl;
    // seed = 1690117527861138371;
    gen->set_current_seed(1);
    std::pair<uint64_t, uint64_t> rng_engine_inputs;
    {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_engine_inputs(N);
    }
    std::cout<<rng_engine_inputs.first<<" "<<rng_engine_inputs.second<<std::endl;
    auto output_info = at::cuda::detail::getTensorInfo<float, int64_t>(output);

    test_rand_kernel<<<dim_grid, dim_block>>>(output_info, rng_engine_inputs, N);
    return output;
}
#define LOW_MEM_DROPOUT_NUM_THREADS 512
template <typename scalar_t, int ADims, int BDims = ADims>
__global__ void low_mem_dropout_forward_kernel(
    at::cuda::detail::TensorInfo<scalar_t, int64_t> data_info,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> output_info,
    at::cuda::detail::TensorInfo<bool, int64_t> mask_info,
    // at::cuda::detail::TensorInfo<float, int64_t> rand_num_info,
    std::pair<uint64_t, uint64_t> seeds,
    uint64_t N,
    float p
){
    const uint64_t id = blockIdx.x*blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seeds.first, id, seeds.second, &state);
    float rand = curand_uniform(&state);
    if(id < N){
        const int64_t data_offset = at::cuda::detail::IndexToOffset<scalar_t, int64_t, 1>::get(id, data_info);
        const int64_t output_offset = at::cuda::detail::IndexToOffset<scalar_t, int64_t, 1>::get(id, output_info);
        const int64_t mask_offset = at::cuda::detail::IndexToOffset<bool, int64_t, 1>::get(id, mask_info);
        // const int64_t rand_num_offset = at::cuda::detail::IndexToOffset<float, int64_t, 1>::get(id, rand_num_info);
        // rand_num_info.data[rand_num_offset] = rand;
        mask_info.data[mask_offset] = rand>p;
        scalar_t data = data_info.data[data_offset];
        scalar_t output = data*(rand>p);
        output_info.data[output_offset] = output;
    }
}

std::pair<Tensor, Tensor> low_mem_dropout_forward_cuda(Tensor data, float p){
    uint64_t n_elements = 1;
    for (size_t i = 0; i < data.dim(); ++i) {
        n_elements *= data.size(i);
    }
    std::cout<<"n_elements:"<<n_elements<<std::endl;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(data.device());
    Tensor output = torch::empty_like(data, options);
    options = torch::TensorOptions().dtype(torch::kBool).device(data.device());
    Tensor mask = torch::zeros_like(data, options);
    uint64_t block_size = LOW_MEM_DROPOUT_NUM_THREADS;
    unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
    // unsigned int n_blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm;
    dim3 dim_block(block_size);
    dim3 grid((n_elements + block_size - 1) / block_size);
    // grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
    // uint64_t counter_offset = ((n_elements - 1)/(block_size*grid.x)+1);
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen->set_current_seed(seed);
    std::pair<uint64_t, uint64_t> rng_engine_inputs;
    {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_engine_inputs(n_elements);
    }
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "low_mem_dropout_forward", ([&] {
        auto data_info = at::cuda::detail::getTensorInfo<scalar_t, int64_t>(data);
        auto output_info = at::cuda::detail::getTensorInfo<scalar_t, int64_t>(output);
        auto mask_info = at::cuda::detail::getTensorInfo<bool, int64_t>(mask);
        data_info.collapseDims();
        output_info.collapseDims();
        mask_info.collapseDims();
        low_mem_dropout_forward_kernel<scalar_t, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            data_info, output_info, mask_info, //rand_num_info,
            rng_engine_inputs,
            n_elements, p);
    }));
    return std::make_pair(output, mask);

}