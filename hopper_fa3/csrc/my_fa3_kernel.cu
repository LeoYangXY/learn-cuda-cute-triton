#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>

namespace {

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shape must be same");
    TORCH_CHECK(q.dim() == 4, "expect q/k/v shape = [B, H, N, D]");
}

// 最蠢的 kernel：只把 Q 逐元素乘上 sqrt(softmax_scale) 后写回 O。
// 目的只是跑通 "CUDA 编译 + pybind + 调用" 的完整流程，不实现真正的 attention。
template <typename scalar_t>
__global__ void my_fa3_dummy_kernel(
    const scalar_t* __restrict__ q,
    scalar_t* __restrict__ o,
    int64_t numel,
    float scale) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        o[idx] = q[idx] * static_cast<scalar_t>(std::sqrt(scale));
    }
}

}  // namespace

torch::Tensor my_fa3_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double softmax_scale,
    bool causal) {

    check_inputs(q, k, v);

    auto o = torch::empty_like(q);
    const int64_t numel = q.numel();

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "my_fa3_dummy_kernel", ([&] {
        my_fa3_dummy_kernel<scalar_t><<<blocks, threads>>>(
            q.data_ptr<scalar_t>(),
            o.data_ptr<scalar_t>(),
            numel,
            static_cast<float>(softmax_scale));
    }));

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_fa3_forward", &my_fa3_forward, "My FA3 forward (dummy CUDA kernel scaffold)");
}
