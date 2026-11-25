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

torch::Tensor decode_norm_cuda(const torch::Tensor&, const torch::Tensor&, double);

std::tuple<torch::Tensor, torch::Tensor> decode_rope_cuda(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
torch::Tensor decode_rope_and_write_cuda(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&);
std::tuple<torch::Tensor, torch::Tensor> prefill_rope_and_write_cuda(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&);
torch::Tensor swiglu_cuda(const torch::Tensor&, const torch::Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Mini LLM Runtime CUDA kernels";
    module.def("decode_norm", &decode_norm_cuda, "Fused FP16 RMSNorm (CUDA)");
    module.def("decode_rope", &decode_rope_cuda, "Fused FP16 Decode Q/K RoPE (CUDA)");
    module.def("decode_rope_and_write", &decode_rope_and_write_cuda,
               "Fused FP16 Decode Q/K RoPE and paged K/V write (CUDA)");
    module.def("prefill_rope_and_write", &prefill_rope_and_write_cuda,
               "Fused FP16 packed Prefill Q/K RoPE and paged K/V write (CUDA)");
    module.def("swiglu", &swiglu_cuda, "Fused FP16 SiLU gate and multiply (CUDA)");
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
