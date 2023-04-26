#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t, typename bound_t>
__device__ __forceinline__ scalar_t clamp(const scalar_t v, const bound_t lo, const bound_t hi)
{
    return min(max(v, lo), hi);
}

template <typename scalar_t, bool dense_mode>
__global__ void color_aware_voxel_smooth_get_exp_weight_cuda_kernel(
    const scalar_t *__restrict__ color,
    scalar_t *__restrict__ color_grad,
    scalar_t *__restrict__ grad,
    const size_t sz_i, const size_t sz_j, const size_t sz_k, const size_t sz_feat, const size_t N, const size_t N_vox)
{

    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        const size_t k = index % sz_k;
        const size_t j = index / sz_k % sz_j;
        const size_t i = index / sz_k / sz_j % sz_i;

        float feat = 0;
        feat += (k == 0 ? 0 : -abs(color[index] - color[index - 1]));
        feat += (k == sz_k - 1 ? 0 : -abs(color[index] - color[index + 1]));
        feat += (j == 0 ? 0 : -abs(color[index] - color[index - sz_k]));
        feat += (j == sz_j - 1 ? 0 : -abs(color[index] - color[index + sz_k]));
        feat += (i == 0 ? 0 : -abs(color[index] - color[index - sz_k * sz_j]));
        feat += (i == sz_i - 1 ? 0 : -abs(color[index] - color[index + sz_k * sz_j]));
        color_grad[index] += feat;
    }
}

template <typename scalar_t, bool dense_mode>
__global__ void color_aware_voxel_smooth_add_grad_cuda_kernel(
    const scalar_t *__restrict__ exp_weight,
    const scalar_t *__restrict__ param,
    scalar_t *__restrict__ grad,
    float wx, float wy, float wz,
    const size_t sz_i, const size_t sz_j, const size_t sz_k, const size_t N)
{

    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        const size_t k = index % sz_k;
        const size_t j = index / sz_k % sz_j;
        const size_t i = index / sz_k / sz_j % sz_i;

        float grad_to_add = 0;
        grad_to_add += (k == 0 ? 0 : wz * clamp(param[index] - param[index - 1], -1.f, 1.f));
        grad_to_add += (k == sz_k - 1 ? 0 : wz * clamp(param[index] - param[index + 1], -1.f, 1.f));
        grad_to_add += (j == 0 ? 0 : wy * clamp(param[index] - param[index - sz_k], -1.f, 1.f));
        grad_to_add += (j == sz_j - 1 ? 0 : wy * clamp(param[index] - param[index + sz_k], -1.f, 1.f));
        grad_to_add += (i == 0 ? 0 : wx * clamp(param[index] - param[index - sz_k * sz_j], -1.f, 1.f));
        grad_to_add += (i == sz_i - 1 ? 0 : wz * clamp(param[index] - param[index + sz_k * sz_j], -1.f, 1.f));
        grad[index] += grad_to_add * exp_weight[index];
    }
}

torch::Tensor color_aware_voxel_smooth_get_exp_weight_cuda(torch::Tensor color, torch::Tensor grad)
{
    torch::Tensor color_grad = torch::zeros_like(color);
    const size_t N = color.numel();
    const size_t N_vox = color.numel();
    const size_t sz_feat = color.size(1);
    const size_t sz_i = grad.size(2);
    const size_t sz_j = grad.size(3);
    const size_t sz_k = grad.size(4);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(grad.type(), "color_aware_voxel_smooth_get_exp_weight_cuda", ([&]
                                                                                             { color_aware_voxel_smooth_get_exp_weight_cuda_kernel<scalar_t, true><<<blocks, threads>>>(
                                                                                                   color.data_ptr<scalar_t>(),
                                                                                                   color_grad.data_ptr<scalar_t>(),
                                                                                                   grad.data_ptr<scalar_t>(),
                                                                                                   sz_i, sz_j, sz_k, sz_feat, N, N_vox); }));
    auto exp_weight = torch::exp(torch::mean(color_grad, 1, true));
    return exp_weight;
}

void color_aware_voxel_smooth_add_grad_cuda(torch::Tensor feature, torch::Tensor param, torch::Tensor grad, float wx, float wy, float wz, bool dense_mode, bool activate)
{
    const size_t N_vox = param.numel(); // Number of voxels(or densitys)
    const size_t sz_feat = feature.size(1);
    torch::Tensor color = feature;
    if(activate){
        color = color.sigmoid();
    }
    const size_t sz_i = param.size(2);
    const size_t sz_j = param.size(3);
    const size_t sz_k = param.size(4);
    const int threads = 256;
    const int blocks = (N_vox + threads - 1) / threads;

    wx /= 6;
    wy /= 6;
    wz /= 6;

    auto exp_weight = color_aware_voxel_smooth_get_exp_weight_cuda(color, grad);
    AT_DISPATCH_FLOATING_TYPES(param.type(), "color_aware_voxel_smooth_add_grad_cuda", ([&]
                                                                                        { color_aware_voxel_smooth_add_grad_cuda_kernel<scalar_t, true><<<blocks, threads>>>(
                                                                                              exp_weight.data_ptr<scalar_t>(),
                                                                                              param.data_ptr<scalar_t>(),
                                                                                              grad.data_ptr<scalar_t>(),
                                                                                              wx, wy, wz,
                                                                                              sz_i, sz_j, sz_k, N_vox); }));

}
