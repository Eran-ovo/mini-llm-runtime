// Decode-only Paged Attention v1：正确性优先的 FP16 / D=64 CUDA kernel。
//
// 一个 CTA 负责一个 (request, query_head)，64 个线程分别负责一个 head_dim。
// K/V token 串行遍历；每个 token 的 QK 点积使用 shared-memory block reduction，
// 输出使用 FP32 online softmax 累加。这个版本有意不加入 half2、token 并行、
// warp reduction 或 shared-memory K/V staging，作为后续单变量优化的稳定基线。

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace paged_attention_v1 {

constexpr int HEAD_DIM = 64;
constexpr int THREADS = HEAD_DIM;

__global__ void paged_decode_attention_fp16_d64_kernel(
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ sequence_lengths,
    __half* __restrict__ output,
    int num_query_heads,
    int num_kv_heads,
    int total_blocks,
    int block_size,
    int max_blocks_per_request,
    float scale)
{
    const int query_head = blockIdx.x % num_query_heads;
    const int batch_index = blockIdx.x / num_query_heads;
    const int dim = threadIdx.x;
    const int sequence_length = sequence_lengths[batch_index];

    // GQA/MQA：相邻 group_size 个 Query Heads 共享一个 KV Head。
    const int group_size = num_query_heads / num_kv_heads;
    const int kv_head = query_head / group_size;
    const size_t query_offset =
        (static_cast<size_t>(batch_index) * num_query_heads + query_head)
        * HEAD_DIM;
    const float query_value = __half2float(query[query_offset + dim]);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float output_accumulator = 0.0f;
    __shared__ float dot_products[HEAD_DIM];

    for (int token_index = 0; token_index < sequence_length; ++token_index) {
        const int logical_block = token_index / block_size;
        const int block_offset = token_index % block_size;
        const int physical_block = block_table[
            static_cast<size_t>(batch_index) * max_blocks_per_request
            + logical_block];

        // 正常 metadata 不会进入此分支；guard 防止损坏的表触发越界显存访问。
        // v1 用 NaN 标记非法输入，正式稳定 API 后续会加入独立 metadata validator。
        if (physical_block < 0 || physical_block >= total_blocks) {
            output[query_offset + dim] = __float2half(NAN);
            return;
        }

        const size_t cache_offset =
            (((static_cast<size_t>(physical_block) * num_kv_heads + kv_head)
              * block_size + block_offset)
             * HEAD_DIM);
        dot_products[dim] =
            query_value * __half2float(key_cache[cache_offset + dim]);
        __syncthreads();

        // D=64 的朴素 shared-memory reduction。后续版本将单独替换成 warp shuffle。
        for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
            if (dim < stride) {
                dot_products[dim] += dot_products[dim + stride];
            }
            __syncthreads();
        }

        const float score = dot_products[0] * scale;
        const float new_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - new_max);
        const float beta = expf(score - new_max);

        // 与 FlashAttention v8 相同：running max 改变时，历史 sum 和 output
        // 必须同时乘 alpha，当前 token 则以 beta 权重加入。
        running_sum = running_sum * alpha + beta;
        output_accumulator =
            output_accumulator * alpha
            + beta * __half2float(value_cache[cache_offset + dim]);
        running_max = new_max;

        // 防止下一 token 覆盖 shared reduction buffer 时，仍有线程尚未读 score。
        __syncthreads();
    }

    output[query_offset + dim] =
        __float2half(output_accumulator / running_sum);
}

}  // namespace paged_attention_v1

torch::Tensor paged_decode_attention_cuda_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    torch::Tensor sequence_lengths,
    double scale)
{
    using namespace paged_attention_v1;

    TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
                "query/key_cache/value_cache must be CUDA tensors");
    TORCH_CHECK(block_table.is_cuda() && sequence_lengths.is_cuda(),
                "block_table/sequence_lengths must be CUDA tensors");
    TORCH_CHECK(query.scalar_type() == at::kHalf &&
                    key_cache.scalar_type() == at::kHalf &&
                    value_cache.scalar_type() == at::kHalf,
                "query/key_cache/value_cache must be float16");
    TORCH_CHECK(block_table.scalar_type() == at::kInt &&
                    sequence_lengths.scalar_type() == at::kInt,
                "block_table/sequence_lengths must be int32");
    TORCH_CHECK(query.dim() == 3,
                "query must be [batch, query_head, head_dim]");
    TORCH_CHECK(key_cache.dim() == 4,
                "key_cache must be [physical_block, kv_head, block_offset, head_dim]");
    TORCH_CHECK(value_cache.sizes() == key_cache.sizes(),
                "key_cache and value_cache must have the same shape");
    TORCH_CHECK(block_table.dim() == 2,
                "block_table must be [batch, max_blocks]");
    TORCH_CHECK(sequence_lengths.dim() == 1,
                "sequence_lengths must be [batch]");
    TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() &&
                    value_cache.is_contiguous() && block_table.is_contiguous() &&
                    sequence_lengths.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(query.device() == key_cache.device() &&
                    query.device() == value_cache.device() &&
                    query.device() == block_table.device() &&
                    query.device() == sequence_lengths.device(),
                "all inputs must be on the same CUDA device");

    c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t batch64 = query.size(0);
    const int64_t query_heads64 = query.size(1);
    const int64_t head_dim64 = query.size(2);
    const int64_t total_blocks64 = key_cache.size(0);
    const int64_t kv_heads64 = key_cache.size(1);
    const int64_t block_size64 = key_cache.size(2);
    const int64_t max_blocks64 = block_table.size(1);

    TORCH_CHECK(batch64 > 0 && query_heads64 > 0 && kv_heads64 > 0,
                "batch and head counts must be positive");
    TORCH_CHECK(total_blocks64 > 0 && block_size64 > 0 && max_blocks64 > 0,
                "cache/block table dimensions must be positive");
    TORCH_CHECK(head_dim64 == HEAD_DIM && key_cache.size(3) == HEAD_DIM,
                "v1 only supports head_dim=64");
    TORCH_CHECK(query_heads64 % kv_heads64 == 0,
                "num_query_heads must be divisible by num_kv_heads");
    TORCH_CHECK(block_table.size(0) == batch64 &&
                    sequence_lengths.size(0) == batch64,
                "metadata batch dimension must match query");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
                "scale must be finite and positive");
    TORCH_CHECK(batch64 <= INT_MAX && query_heads64 <= INT_MAX &&
                    kv_heads64 <= INT_MAX && total_blocks64 <= INT_MAX &&
                    block_size64 <= INT_MAX && max_blocks64 <= INT_MAX,
                "tensor dimensions exceed int32 kernel limits");
    TORCH_CHECK(batch64 <= INT_MAX / query_heads64,
                "batch * query_heads exceeds CUDA grid limit");

    // v1 是 correctness kernel：把很小的 metadata 同步到 CPU，逐请求验证真正会被
    // 访问的 block ID。正式 benchmark 前必须把这一步移到请求创建/调度边界，不能
    // 让每个 Decode launch 都承担 D2H copy 和同步。
    const auto lengths_cpu = sequence_lengths.to(at::kCPU);
    const auto block_table_cpu = block_table.to(at::kCPU);
    const int32_t* lengths_ptr = lengths_cpu.data_ptr<int32_t>();
    const int32_t* table_ptr = block_table_cpu.data_ptr<int32_t>();
    for (int64_t batch_index = 0; batch_index < batch64; ++batch_index) {
        const int64_t sequence_length = lengths_ptr[batch_index];
        TORCH_CHECK(sequence_length > 0,
                    "decode sequence lengths must be positive");
        const int64_t required_blocks =
            (sequence_length + block_size64 - 1) / block_size64;
        TORCH_CHECK(required_blocks <= max_blocks64,
                    "sequence length requires more blocks than block_table provides");
        for (int64_t logical_block = 0;
             logical_block < required_blocks;
             ++logical_block) {
            const int32_t physical_block = table_ptr[
                batch_index * max_blocks64 + logical_block];
            TORCH_CHECK(physical_block >= 0 && physical_block < total_blocks64,
                        "used block_table entry contains an out-of-range physical block ID");
        }
    }

    auto output = torch::empty_like(query);
    const int batch = static_cast<int>(batch64);
    const int query_heads = static_cast<int>(query_heads64);
    const dim3 grid(static_cast<unsigned>(batch * query_heads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    paged_decode_attention_fp16_d64_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        sequence_lengths.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        query_heads,
        static_cast<int>(kv_heads64),
        static_cast<int>(total_blocks64),
        static_cast<int>(block_size64),
        static_cast<int>(max_blocks64),
        static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
