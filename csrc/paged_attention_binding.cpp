#include <torch/extension.h>

torch::Tensor paged_decode_attention_cuda_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor sequence_lengths,
    double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Mini LLM Runtime CUDA kernels";
    module.def(
        "paged_decode_attention",
        &paged_decode_attention_cuda_forward,
        "Naive decode-only Paged Attention (CUDA)",
        pybind11::arg("query"),
        pybind11::arg("key_cache"),
        pybind11::arg("value_cache"),
        pybind11::arg("block_table"),
        pybind11::arg("sequence_lengths"),
        pybind11::arg("scale"));
}
