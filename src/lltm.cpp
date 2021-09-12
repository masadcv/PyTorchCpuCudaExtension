#include <torch/extension.h>
#include <vector>
#include "lltm.h"
#include "common.h"

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(input);
    CHECK_CONTIGUOUS_CUDA(weights);
    CHECK_CONTIGUOUS_CUDA(bias);
    CHECK_CONTIGUOUS_CUDA(old_h);
    CHECK_CONTIGUOUS_CUDA(old_cell);

    return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return lltm_cpu_forward(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  if (X.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(grad_h);
    CHECK_CONTIGUOUS_CUDA(grad_cell);
    CHECK_CONTIGUOUS_CUDA(new_cell);
    CHECK_CONTIGUOUS_CUDA(input_gate);
    CHECK_CONTIGUOUS_CUDA(output_gate);
    CHECK_CONTIGUOUS_CUDA(candidate_cell);
    CHECK_CONTIGUOUS_CUDA(X);
    CHECK_CONTIGUOUS_CUDA(gate_weights);
    CHECK_CONTIGUOUS_CUDA(weights);

    return lltm_cuda_backward(
        grad_h, grad_cell, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights, weights);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return lltm_cpu_backward(
      grad_h, grad_cell, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights, weights);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  // lltm
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");

}