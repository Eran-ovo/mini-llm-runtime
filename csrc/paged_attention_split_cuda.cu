// split-KV correctness 实验：分区仍使用 v1 的 scalar load、shared reduction。
// 每个 (request,query_head,split) CTA 输出 FP32 (m,l,a)，第二个 kernel 合并。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void validate_paged_attention_inputs(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    double, bool);

namespace paged_attention_split {
constexpr int D = 64;
constexpr int STATE = D + 2;  // [m,l,a[0],...,a[63]]，始终以 FP32 保存。

__global__ void partial_kernel(
    const __half* q, const __half* k, const __half* v,
    const int32_t* table, const int32_t* lengths, float* states,
    int hq, int hkv, int block_size, int table_width, int splits, float scale)
{
    const int bh = blockIdx.x;
    const int part = blockIdx.y;
    const int d = threadIdx.x;
    const int b = bh / hq;
    const int kvh = (bh % hq) / (hq / hkv);
    const int64_t length = lengths[b];
    const int64_t blocks = (length + block_size - 1) / block_size;
    // 按每个请求自己的有效逻辑 block 数划分，相邻半开区间无重叠、无遗漏。
    // 使用 int64 避免 blocks*part 与末尾 block 对齐时发生 int32 溢出。
    const int64_t start = (blocks * part / splits) * block_size;
    const int64_t block_end = (blocks * (part + 1) / splits) * block_size;
    const int64_t end = block_end < length ? block_end : length;
    const size_t qbase = static_cast<size_t>(bh) * D;
    const float qd = __half2float(q[qbase + d]);
    float m = -INFINITY;
    float l = 0.f;
    float a = 0.f;
    __shared__ float dot[D];
    for (int64_t t = start; t < end; ++t) {
        const int physical = table[static_cast<size_t>(b) * table_width + t / block_size];
        const size_t offset = ((static_cast<size_t>(physical) * hkv + kvh)
                               * block_size + t % block_size) * D + d;
        dot[d] = qd * __half2float(k[offset]);
        __syncthreads();
        for (int stride = D / 2; stride; stride >>= 1) {
            if (d < stride) dot[d] += dot[d + stride];
            __syncthreads();
        }
        const float s = dot[0] * scale;
        const float next_m = fmaxf(m, s);
        const float alpha = expf(m - next_m);
        const float beta = expf(s - next_m);
        l = l * alpha + beta;
        a = a * alpha + beta * __half2float(v[offset]);
        m = next_m;
        __syncthreads();
    }
    const size_t base = (static_cast<size_t>(bh) * splits + part) * STATE;
    // 空分区也必须完整写入中性状态，不能读取 torch::empty 的旧内容。
    if (d == 0) { states[base] = m; states[base + 1] = l; }
    states[base + 2 + d] = a;
}

__global__ void merge_kernel(const float* states, __half* output, int splits)
{
    const int bh = blockIdx.x;
    const int d = threadIdx.x;
    const size_t base = static_cast<size_t>(bh) * splits * STATE;
    float m = -INFINITY;
    for (int j = 0; j < splits; ++j) {
        const float* state = states + base + j * STATE;
        if (state[1] > 0.f) m = fmaxf(m, state[0]);
    }
    float l = 0.f;
    float a = 0.f;
    for (int j = 0; j < splits; ++j) {
        const float* state = states + base + j * STATE;
        // 空分区 l=0，直接跳过，避免任何 -inf - -inf 或 0*NaN。
        if (state[1] > 0.f) {
            const float weight = expf(state[0] - m);
            l += weight * state[1];
            a += weight * state[2 + d];
        }
    }
    output[static_cast<size_t>(bh) * D + d] = __float2half(a / l);
}
}  // namespace paged_attention_split

torch::Tensor paged_decode_attention_split_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor table, torch::Tensor lengths, double scale,
    int64_t splits, bool validate_metadata)
{
    using namespace paged_attention_split;
    TORCH_CHECK(splits >= 1 && splits <= 64, "num_splits must be in [1,64]");
    validate_paged_attention_inputs(q, k, v, table, lengths, scale, validate_metadata);
    c10::cuda::CUDAGuard guard(q.device());
    const int bh = static_cast<int>(q.size(0) * q.size(1));
    // scratch 属于本次调用，其分配、生命周期和 merge 成本都要纳入性能比较。
    auto states = torch::empty({bh, splits, static_cast<int64_t>(STATE)},
                                q.options().dtype(at::kFloat));
    auto output = torch::empty_like(q);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    partial_kernel<<<dim3(bh, splits), D, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        table.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(), states.data_ptr<float>(),
        q.size(1), k.size(1), k.size(2), table.size(1), splits, static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // 同一 PyTorch current stream 保证 partial 完成后 merge 才读取 states。
    merge_kernel<<<bh, D, 0, stream>>>(states.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
