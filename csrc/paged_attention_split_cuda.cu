// 每个 (request, query_head, split) CTA 用多个 warp 扫描 KV。
// CTA 内外均用 FP32 (m,l,a) 合并 online softmax 状态。
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
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_PARTITION = 4;
constexpr int STATE = D + 2;  // [m,l,a[0],...,a[63]]，始终以 FP32 保存。

template <bool SPLIT>
__global__ void partial_kernel(
    const __half* q, const __half* k, const __half* v,
    const int32_t* table, const int32_t* lengths, float* states, __half* output,
    int hq, int hkv, int block_size, int table_width, int splits, float scale)
{
    const int bh = blockIdx.x;
    const int part = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int b = bh / hq;
    const int kvh = (bh % hq) / (hq / hkv);
    const int64_t length = lengths[b];
    const int64_t blocks = (length + block_size - 1) / block_size;
    // 按每个请求自己的有效逻辑 block 数划分，相邻半开区间无重叠、无遗漏。
    // 使用 int64 避免 blocks*part 与末尾 block 对齐时发生 int32 溢出。
    const int64_t start_block = SPLIT ? blocks * part / splits : 0;
    const int64_t end_block = SPLIT ? blocks * (part + 1) / splits : blocks;
    const size_t qbase = static_cast<size_t>(bh) * D + lane;
    const float q0 = __half2float(q[qbase]);
    const float q1 = __half2float(q[qbase + WARP_SIZE]);
    float m = -INFINITY;
    float l = 0.f;
    float a0 = 0.f;
    float a1 = 0.f;
    __shared__ float warp_states[WARPS_PER_PARTITION][STATE];
    for (int64_t block = start_block; block < end_block; ++block) {
        const int physical = table[static_cast<size_t>(b) * table_width + block];
        const size_t cache_base =
            (static_cast<size_t>(physical) * hkv + kvh) * block_size * D + lane;
        const int64_t remaining = length - block * block_size;
        const int tokens = remaining < block_size ? static_cast<int>(remaining) : block_size;
        for (int t = warp; t < tokens; t += WARPS_PER_PARTITION) {
            const size_t offset = cache_base + static_cast<size_t>(t) * D;
            float dot = q0 * __half2float(k[offset])
                      + q1 * __half2float(k[offset + WARP_SIZE]);
            for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, stride);
            }
            const float score = __shfl_sync(0xffffffff, dot, 0) * scale;
            const float next_m = fmaxf(m, score);
            const float alpha = expf(m - next_m);
            const float beta = expf(score - next_m);
            l = l * alpha + beta;
            a0 = a0 * alpha + beta * __half2float(v[offset]);
            a1 = a1 * alpha + beta * __half2float(v[offset + WARP_SIZE]);
            m = next_m;
        }
    }

    if (lane == 0) {
        warp_states[warp][0] = m;
        warp_states[warp][1] = l;
    }
    warp_states[warp][2 + lane] = a0;
    warp_states[warp][2 + lane + WARP_SIZE] = a1;
    __syncthreads();

    if (warp != 0) return;
    m = -INFINITY;
    l = a0 = a1 = 0.f;
    for (const auto& state : warp_states) {
        m = fmaxf(m, state[0]);
    }
    for (const auto& state : warp_states) {
        // 空 warp 保持中性状态，跳过权重计算以避免 exp(-inf - -inf)。
        if (state[1] > 0.f) {
            const float weight = expf(state[0] - m);
            l += weight * state[1];
            a0 += weight * state[2 + lane];
            a1 += weight * state[2 + lane + WARP_SIZE];
        }
    }
    if constexpr (SPLIT) {
        const size_t base = (static_cast<size_t>(bh) * splits + part) * STATE;
        if (lane == 0) {
            states[base] = m;
            states[base + 1] = l;
        }
        states[base + 2 + lane] = a0;
        states[base + 2 + lane + WARP_SIZE] = a1;
    } else {
        output[qbase] = __float2half_rn(a0 / l);
        output[qbase + WARP_SIZE] = __float2half_rn(a1 / l);
    }
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
    auto output = torch::empty_like(q);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto* query = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const auto* key = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    const auto* value = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    auto* result = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
    const auto* block_table = table.data_ptr<int32_t>();
    const auto* sequence_lengths = lengths.data_ptr<int32_t>();
    if (splits == 1) {
        partial_kernel<false><<<bh, WARPS_PER_PARTITION * WARP_SIZE, 0, stream>>>(
            query, key, value, block_table, sequence_lengths, nullptr, result,
            q.size(1), k.size(1), k.size(2), table.size(1), 1, static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    // scratch 属于本次调用，其分配、生命周期和 merge 成本都要纳入性能比较。
    auto states = torch::empty({bh, splits, static_cast<int64_t>(STATE)},
                                q.options().dtype(at::kFloat));
    partial_kernel<true><<<dim3(bh, splits), WARPS_PER_PARTITION * WARP_SIZE, 0, stream>>>(
        query, key, value, block_table, sequence_lengths, states.data_ptr<float>(), nullptr,
        q.size(1), k.size(1), k.size(2), table.size(1), splits, static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // 同一 PyTorch current stream 保证 partial 完成后 merge 才读取 states。
    merge_kernel<<<bh, D, 0, stream>>>(states.data_ptr<float>(),
        result, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
