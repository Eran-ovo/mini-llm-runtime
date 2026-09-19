#include <torch/extension.h>

torch::Tensor paged_decode_attention_cuda_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor sequence_lengths,
    double scale,
    bool validate_metadata);

torch::Tensor paged_decode_attention_split_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    double, int64_t, bool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Mini LLM Runtime CUDA kernels";
    module.def("paged_decode_attention_split", &paged_decode_attention_split_forward,
               "Experimental split-KV attention; metadata validation required before unchecked use",
               pybind11::arg("query"), pybind11::arg("key_cache"),
               pybind11::arg("value_cache"), pybind11::arg("block_table"),
               pybind11::arg("sequence_lengths"), pybind11::arg("scale"),
               pybind11::arg("num_splits"), pybind11::arg("validate_metadata") = true);
    module.def(
        "paged_decode_attention",
        &paged_decode_attention_cuda_forward,
        "Naive decode-only Paged Attention (CUDA)",
        pybind11::arg("query"),
        pybind11::arg("key_cache"),
        pybind11::arg("value_cache"),
        pybind11::arg("block_table"),
        pybind11::arg("sequence_lengths"),
        pybind11::arg("scale"),
        pybind11::arg("validate_metadata") = true);
}
