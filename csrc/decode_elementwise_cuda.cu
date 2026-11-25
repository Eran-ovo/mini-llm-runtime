#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <climits>
#include <cmath>
#include <tuple>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int NORM_THREADS = 128;
constexpr int ELEMENTWISE_THREADS = 256;
constexpr int HEAD_DIM = 64;

__device__ float warp_sum(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void decode_norm_kernel(
    const __half* input,
    const __half* weight,
    __half* output,
    int width,
    float epsilon)
{
    const size_t base = static_cast<size_t>(blockIdx.x) * width;
    const int thread = threadIdx.x;
    const int lane = thread % WARP_SIZE;
    const int warp = thread / WARP_SIZE;
    __shared__ float warp_sums[NORM_THREADS / WARP_SIZE];
    __shared__ float inverse_rms;

    float squares = 0.f;
    for (int index = thread; index < width; index += NORM_THREADS) {
        const float value = __half2float(input[base + index]);
        squares += value * value;
    }
    squares = warp_sum(squares);
    if (lane == 0) warp_sums[warp] = squares;
    __syncthreads();

    if (warp == 0) {
        const float sum = warp_sum(lane < NORM_THREADS / WARP_SIZE ? warp_sums[lane] : 0.f);
        if (lane == 0) inverse_rms = rsqrtf(sum / width + epsilon);
    }
    __syncthreads();

    for (int index = thread; index < width; index += NORM_THREADS) {
        const float value = __half2float(input[base + index]);
        // 保留 PyTorch reference 的两次 FP16 舍入。
        const __half normalized = __float2half_rn(value * inverse_rms);
        output[base + index] = __hmul_rn(normalized, weight[index]);
    }
}

template <bool WRITE_CACHE, bool KEEP_ROTATED_KEYS = false>
__global__ void decode_rope_kernel(
    const __half* query,
    const __half* key,
    const __half* cosine,
    const __half* sine,
    __half* query_output,
    __half* key_output,
    int query_heads,
    int kv_heads,
    const __half* value,
    __half* value_cache,
    const int64_t* blocks,
    const int64_t* slots,
    int64_t total_blocks,
    int64_t block_size,
    int total_vectors,
    __half* rotated_keys)
{
    constexpr int WARPS = KEEP_ROTATED_KEYS ? 4 : 1;
    const int heads = query_heads + kv_heads;
    const int vector = blockIdx.x * WARPS + threadIdx.x / WARP_SIZE;
    if (vector >= total_vectors) return;
    const int batch = vector / heads;
    const int head = vector % heads;
    const int lane = threadIdx.x % WARP_SIZE;
    const bool is_query = head < query_heads;
    const size_t offset =
        (static_cast<size_t>(batch) * (is_query ? query_heads : kv_heads)
        + (is_query ? head : head - query_heads)) * HEAD_DIM + lane;
    const size_t rotary_offset = static_cast<size_t>(batch) * HEAD_DIM + lane;
    const __half* input = is_query ? query : key;
    __half* output = is_query ? query_output : key_output;

    const __half lower = input[offset];
    const __half upper = input[offset + WARP_SIZE];
    const __half rotated_lower = __hadd_rn(
        __hmul_rn(lower, cosine[rotary_offset]),
        __hmul_rn(__hneg(upper), sine[rotary_offset]));
    const __half rotated_upper = __hadd_rn(
        __hmul_rn(upper, cosine[rotary_offset + WARP_SIZE]),
        __hmul_rn(lower, sine[rotary_offset + WARP_SIZE]));
    if constexpr (WRITE_CACHE) {
        if (!is_query) {
            const int64_t block = blocks[batch];
            const int64_t slot = slots[batch];
            if (block < 0 || block >= total_blocks || slot < 0 || slot >= block_size) {
                return;
            }
            const size_t cache_offset =
                ((static_cast<size_t>(block) * kv_heads + head - query_heads)
                * block_size + slot) * HEAD_DIM + lane;
            key_output[cache_offset] = rotated_lower;
            key_output[cache_offset + WARP_SIZE] = rotated_upper;
            value_cache[cache_offset] = value[offset];
            value_cache[cache_offset + WARP_SIZE] = value[offset + WARP_SIZE];
            if constexpr (KEEP_ROTATED_KEYS) {
                rotated_keys[offset] = rotated_lower;
                rotated_keys[offset + WARP_SIZE] = rotated_upper;
            }
            return;
        }
    }
    output[offset] = rotated_lower;
    output[offset + WARP_SIZE] = rotated_upper;
}

__global__ void swiglu_kernel(
    const __half* gate, const __half* up, __half* output, int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float input = __half2float(gate[index]);
        // SiLU 输出先舍入到 FP16，再乘 up，与两个 PyTorch 算子一致。
        const __half activated = __float2half_rn(input / (1.f + expf(-input)));
        output[index] = __hmul_rn(activated, up[index]);
    }
}

void validate_tensor(const torch::Tensor& tensor, const c10::Device& device) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
                tensor.scalar_type() == at::kHalf && tensor.is_contiguous(),
                "decode tensors must be contiguous FP16 on the same CUDA device");
}

void validate_rope(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& cosine, const torch::Tensor& sine)
{
    validate_tensor(query, query.device());
    validate_tensor(key, query.device());
    validate_tensor(cosine, query.device());
    validate_tensor(sine, query.device());
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && cosine.dim() == 4 &&
                cosine.sizes() == sine.sizes(), "invalid decode RoPE tensor ranks");
    const int64_t batch = query.size(0);
    const int64_t query_heads = query.size(1);
    const int64_t kv_heads = key.size(1);
    TORCH_CHECK(batch > 0 && key.size(0) == batch && cosine.size(0) == batch &&
                query_heads > 0 && kv_heads > 0 && cosine.size(1) == 1 &&
                query.size(2) == 1 && key.size(2) == 1 && cosine.size(2) == 1 &&
                query.size(3) == HEAD_DIM && key.size(3) == HEAD_DIM &&
                cosine.size(3) == HEAD_DIM, "decode RoPE expects FP16, q_len=1 and head_dim=64");
    TORCH_CHECK(query_heads <= INT_MAX && kv_heads <= INT_MAX &&
                batch <= INT_MAX / (query_heads + kv_heads), "decode RoPE grid exceeds int32 range");
}

}  // namespace

torch::Tensor decode_norm_cuda(
    const torch::Tensor& input, const torch::Tensor& weight, double epsilon)
{
    validate_tensor(input, input.device());
    validate_tensor(weight, input.device());
    TORCH_CHECK(input.dim() == 3 && input.size(2) > 0 &&
                weight.dim() == 1 && weight.size(0) == input.size(2),
                "RMSNorm expects [batch,sequence,hidden] and [hidden]");
    const int64_t width = input.size(2);
    const int64_t rows = input.numel() / width;
    TORCH_CHECK(rows > 0 && rows <= INT_MAX && width <= INT_MAX - NORM_THREADS &&
                std::isfinite(epsilon) && epsilon > 0,
                "invalid RMSNorm dimensions or epsilon");
    const c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    decode_norm_kernel<<<rows, NORM_THREADS, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        width,
        static_cast<float>(epsilon));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> decode_rope_cuda(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& cosine, const torch::Tensor& sine)
{
    validate_rope(query, key, cosine, sine);
    const c10::cuda::CUDAGuard guard(query.device());
    auto query_output = torch::empty_like(query);
    auto key_output = torch::empty_like(key);
    const int vectors = query.size(0) * (query.size(1) + key.size(1));
    decode_rope_kernel<false><<<vectors,
                                 WARP_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(cosine.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(sine.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(query_output.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(key_output.data_ptr<at::Half>()),
        query.size(1), key.size(1), nullptr, nullptr, nullptr, nullptr, 0, 0, vectors, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {query_output, key_output};
}

template <bool KEEP_ROTATED_KEYS>
std::tuple<torch::Tensor, torch::Tensor> rope_and_write(
    const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& cosine, const torch::Tensor& sine,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& blocks, const torch::Tensor& slots)
{
    validate_rope(query, key, cosine, sine);
    validate_tensor(value, query.device());
    validate_tensor(key_cache, query.device());
    validate_tensor(value_cache, query.device());
    const int64_t batch = query.size(0);
    const int64_t query_heads = query.size(1);
    const int64_t kv_heads = key.size(1);
    TORCH_CHECK(value.sizes() == key.sizes(), "decode K/V shapes must match");
    TORCH_CHECK(key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes() &&
                key_cache.size(0) > 0 && key_cache.size(1) == kv_heads &&
                key_cache.size(2) > 0 && key_cache.size(3) == HEAD_DIM,
                "invalid paged K/V storage layout");
    for (const auto& index : {blocks, slots}) {
        TORCH_CHECK(index.device() == query.device() && index.scalar_type() == at::kLong &&
                    index.is_contiguous() && index.dim() == 1 && index.size(0) == batch,
                    "decode slots must be contiguous int64 vectors on the cache device");
    }
    const c10::cuda::CUDAGuard guard(query.device());
    auto query_output = torch::empty_like(query);
    auto key_output = KEEP_ROTATED_KEYS ? torch::empty_like(key) : torch::Tensor{};
    __half* rotated_keys =
        KEEP_ROTATED_KEYS ? reinterpret_cast<__half*>(key_output.data_ptr<at::Half>()) : nullptr;
    constexpr int WARPS = KEEP_ROTATED_KEYS ? 4 : 1;
    const int vectors = batch * (query_heads + kv_heads);
    const int grid = (vectors - 1) / WARPS + 1;
    decode_rope_kernel<true, KEEP_ROTATED_KEYS><<<
        grid, WARPS * WARP_SIZE, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(cosine.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(sine.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(query_output.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(key_cache.data_ptr<at::Half>()),
        query_heads, kv_heads,
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(value_cache.data_ptr<at::Half>()),
        blocks.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
        key_cache.size(0), key_cache.size(2), vectors, rotated_keys);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {query_output, key_output};
}

torch::Tensor decode_rope_and_write_cuda(
    const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& cosine, const torch::Tensor& sine,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& blocks, const torch::Tensor& slots)
{
    return std::get<0>(rope_and_write<false>(
        query, key, value, cosine, sine, key_cache, value_cache, blocks, slots));
}

std::tuple<torch::Tensor, torch::Tensor> prefill_rope_and_write_cuda(
    const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& cosine, const torch::Tensor& sine,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& blocks, const torch::Tensor& slots)
{
    return rope_and_write<true>(
        query, key, value, cosine, sine, key_cache, value_cache, blocks, slots);
}

torch::Tensor swiglu_cuda(const torch::Tensor& gate, const torch::Tensor& up)
{
    validate_tensor(gate, gate.device());
    validate_tensor(up, gate.device());
    TORCH_CHECK(gate.sizes() == up.sizes(), "SwiGLU gate and up must have the same shape");
    const int64_t count = gate.numel();
    const int64_t blocks = (count + ELEMENTWISE_THREADS - 1) / ELEMENTWISE_THREADS;
    TORCH_CHECK(blocks > 0 && blocks <= INT_MAX, "SwiGLU tensor exceeds grid limits");
    const c10::cuda::CUDAGuard guard(gate.device());
    auto output = torch::empty_like(gate);
    swiglu_kernel<<<blocks, ELEMENTWISE_THREADS, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
