#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void color_aware_voxel_smooth_add_grad_cuda(torch::Tensor feature, torch::Tensor param, torch::Tensor grad, float wx, float wy, float wz, bool dense_mode, bool activate);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void color_aware_voxel_smooth_add_grad(torch::Tensor feature, torch::Tensor param, torch::Tensor grad, float wx, float wy, float wz, bool dense_mode, bool activate)
{
    CHECK_INPUT(feature);
    CHECK_INPUT(param);
    CHECK_INPUT(grad);
    color_aware_voxel_smooth_add_grad_cuda(feature, param, grad, wx, wy, wz, dense_mode, activate);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("color_aware_voxel_smooth_add_grad", &color_aware_voxel_smooth_add_grad, "Add color aware total variation grad");
}
