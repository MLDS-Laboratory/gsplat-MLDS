#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

template <typename S>
__device__ inline bool shadow_splat_eval(
    const vec2<S> *__restrict__ means2d,
    const vec3<S> *__restrict__ conics,
    const S *__restrict__ opacities,
    const int32_t g,
    const S px,
    const S py,
    const S min_alpha,
    S &beta,
    S &alpha,
    vec2<S> &delta,
    vec3<S> &conic
) {
    conic = conics[g];
    const vec2<S> xy = means2d[g];
    delta = {xy.x - px, xy.y - py};
    const S sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                    conic.y * delta.x * delta.y;
    beta = __expf(-sigma);
    alpha = min((S)0.999f, opacities[g] * beta);
    return !(sigma < 0.f || alpha < min_alpha);
}

template <typename S>
__device__ inline void shadow_splat_accumulate_grad(
    const int32_t g,
    const int32_t gid,
    const S beta,
    const S alpha,
    const vec2<S> delta,
    const vec3<S> conic,
    const S transmittance_source,
    const S v_alpha,
    const S *__restrict__ opacities,
    const S *__restrict__ v_shadow_num,
    const S *__restrict__ v_shadow_den,
    vec2<S> *__restrict__ v_means2d,
    vec3<S> *__restrict__ v_conics,
    S *__restrict__ v_opacities
) {
    S v_beta = v_shadow_num[gid] * (1.0f - transmittance_source) + v_shadow_den[gid];
    const S raw_alpha = opacities[g] * beta;
    if (raw_alpha <= 0.999f) {
        v_beta += opacities[g] * v_alpha;
        gpuAtomicAdd(v_opacities + g, beta * v_alpha);
    }
    const S v_sigma = -beta * v_beta;
    vec3<S> *v_conic = v_conics + g;
    gpuAtomicAdd(&v_conic->x, 0.5f * v_sigma * delta.x * delta.x);
    gpuAtomicAdd(&v_conic->y, v_sigma * delta.x * delta.y);
    gpuAtomicAdd(&v_conic->z, 0.5f * v_sigma * delta.y * delta.y);
    vec2<S> *v_mean = v_means2d + g;
    gpuAtomicAdd(&v_mean->x, v_sigma * (conic.x * delta.x + conic.y * delta.y));
    gpuAtomicAdd(&v_mean->y, v_sigma * (conic.y * delta.x + conic.z * delta.y));
}

// The forward shadow raster is metadata-only, so its output has no pixel image
// to seed a standard gsplat backward pass.  This kernel replays the exact tile
// order and hard visibility decisions used by rasterize_to_pixels_fwd_kernel,
// then differentiates the selected splat weights and transmittance recurrence.
// Sort/group membership, thresholds, and receiver-bias comparisons are discrete
// forward decisions and therefore intentionally receive zero derivative.
template <typename S>
__global__ void rasterize_to_pixels_bwd_shadow_only_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2<S> *__restrict__ means2d,
    const vec3<S> *__restrict__ conics,
    const S *__restrict__ opacities,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const int32_t *__restrict__ gaussian_ids,
    const S *__restrict__ depths,
    const S *__restrict__ v_shadow_num,
    const S *__restrict__ v_shadow_den,
    const S shadow_alpha_threshold,
    const S shadow_depth_group_eps,
    const bool use_shadow_receiver_bias,
    const S *__restrict__ shadow_receiver_bias,
    vec2<S> *__restrict__ v_means2d,
    vec3<S> *__restrict__ v_conics,
    S *__restrict__ v_opacities
) {
    auto block = cg::this_thread_block();
    const uint32_t camera_id = block.group_index().x;
    const uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    if (i >= image_height || j >= image_width) {
        return;
    }

    tile_offsets += camera_id * tile_height * tile_width;
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    if (range_start >= range_end) {
        return;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    const bool grouped = shadow_depth_group_eps > 0.0f ||
                         (use_shadow_receiver_bias && shadow_receiver_bias != nullptr);
    const uint32_t block_size = block.size();
    const int32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Replay the forward recurrence to obtain its final transmittance.  Groups
    // deliberately do not cross CUDA load-batch boundaries, matching forward.
    S T_final = 1.0f;
    for (int32_t b = 0; b < num_batches; ++b) {
        const int32_t batch_start = range_start + block_size * b;
        const int32_t batch_end = min(range_end, batch_start + (int32_t)block_size);
        int32_t t = batch_start;
        while (t < batch_end) {
            const int32_t g = flatten_ids[t];
            S beta, alpha;
            vec2<S> delta;
            vec3<S> conic;
            if (!shadow_splat_eval(means2d, conics, opacities, g, px, py, shadow_alpha_threshold, beta, alpha, delta, conic)) {
                ++t;
                continue;
            }
            if (!grouped) {
                T_final *= 1.0f - alpha;
                ++t;
                continue;
            }
            const S group_depth = depths[g];
            int32_t u = t;
            while (u < batch_end) {
                const int32_t g_u = flatten_ids[u];
                const int32_t gid_u = packed ? gaussian_ids[g_u] : g_u;
                S depth_gap = depths[g_u] - group_depth;
                if (use_shadow_receiver_bias && shadow_receiver_bias != nullptr) {
                    depth_gap -= shadow_receiver_bias[gid_u];
                }
                if (depth_gap > shadow_depth_group_eps) {
                    break;
                }
                S beta_u, alpha_u;
                vec2<S> delta_u;
                vec3<S> conic_u;
                if (shadow_splat_eval(
                        means2d, conics, opacities, g_u, px, py, shadow_alpha_threshold,
                        beta_u, alpha_u, delta_u, conic_u
                    )) {
                    T_final *= 1.0f - alpha_u;
                }
                ++u;
            }
            t = u;
        }
    }

    // Reverse the same recurrence.  v_T_after is the adjoint of the current
    // suffix transmittance; shadow_num/shadow_den gradients provide the direct
    // ratio-rule contribution from each receiver splat.
    S T_after = T_final;
    S v_T_after = 0.0f;
    for (int32_t b = num_batches - 1; b >= 0; --b) {
        const int32_t batch_start = range_start + block_size * b;
        const int32_t batch_end = min(range_end, batch_start + (int32_t)block_size);
        int32_t t = batch_end;
        while (t > batch_start) {
            const int32_t g_last = flatten_ids[t - 1];
            S beta_last, alpha_last;
            vec2<S> delta_last;
            vec3<S> conic_last;
            if (!shadow_splat_eval(
                    means2d, conics, opacities, g_last, px, py, shadow_alpha_threshold,
                    beta_last, alpha_last, delta_last, conic_last
                )) {
                --t;
                continue;
            }

            if (!grouped) {
                const S T_before = T_after / (1.0f - alpha_last);
                const int32_t gid = packed ? gaussian_ids[g_last] : g_last;
                const S v_alpha = -v_T_after * T_before;
                shadow_splat_accumulate_grad(
                    g_last, gid, beta_last, alpha_last, delta_last, conic_last, T_before, v_alpha,
                    opacities, v_shadow_num, v_shadow_den, v_means2d, v_conics, v_opacities
                );
                v_T_after = v_T_after * (1.0f - alpha_last) - v_shadow_num[gid] * beta_last;
                T_after = T_before;
                --t;
                continue;
            }

            // Reconstruct the final forward group ending at t.  Group membership
            // is fixed by the forward depth/bias comparisons; only its continuous
            // beta/alpha/transmittance values are differentiated.
            S required_depth = -1e30f;
            int32_t group_start = -1;
            for (int32_t q = t - 1; q >= batch_start; --q) {
                const int32_t g_q = flatten_ids[q];
                const int32_t gid_q = packed ? gaussian_ids[g_q] : g_q;
                const S bias_q = (use_shadow_receiver_bias && shadow_receiver_bias != nullptr)
                    ? shadow_receiver_bias[gid_q]
                    : 0.0f;
                required_depth = max(required_depth, depths[g_q] - bias_q - shadow_depth_group_eps);
                if (depths[g_q] < required_depth) {
                    break;
                }
                S beta_q, alpha_q;
                vec2<S> delta_q;
                vec3<S> conic_q;
                if (shadow_splat_eval(
                        means2d, conics, opacities, g_q, px, py, shadow_alpha_threshold,
                        beta_q, alpha_q, delta_q, conic_q
                    )) {
                    group_start = q;
                }
            }
            if (group_start < 0) {
                return;
            }

            S group_product = 1.0f;
            for (int32_t u = group_start; u < t; ++u) {
                const int32_t g_u = flatten_ids[u];
                S beta_u, alpha_u;
                vec2<S> delta_u;
                vec3<S> conic_u;
                if (shadow_splat_eval(
                        means2d, conics, opacities, g_u, px, py, shadow_alpha_threshold,
                        beta_u, alpha_u, delta_u, conic_u
                    )) {
                    group_product *= 1.0f - alpha_u;
                }
            }
            const S T_group = T_after / group_product;
            S v_T_group = v_T_after * group_product;
            for (int32_t u = group_start; u < t; ++u) {
                const int32_t g_u = flatten_ids[u];
                const int32_t gid_u = packed ? gaussian_ids[g_u] : g_u;
                S beta_u, alpha_u;
                vec2<S> delta_u;
                vec3<S> conic_u;
                if (!shadow_splat_eval(
                        means2d, conics, opacities, g_u, px, py, shadow_alpha_threshold,
                        beta_u, alpha_u, delta_u, conic_u
                    )) {
                    continue;
                }
                const S v_alpha = -v_T_after * T_after / (1.0f - alpha_u);
                shadow_splat_accumulate_grad(
                    g_u, gid_u, beta_u, alpha_u, delta_u, conic_u, T_group, v_alpha,
                    opacities, v_shadow_num, v_shadow_den, v_means2d, v_conics, v_opacities
                );
                v_T_group -= v_shadow_num[gid_u] * beta_u;
            }
            T_after = T_group;
            v_T_after = v_T_group;
            t = group_start;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_shadow_only_tensor(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids,
    const torch::Tensor &gaussian_ids,
    const torch::Tensor &depths,
    const torch::Tensor &v_shadow_num,
    const torch::Tensor &v_shadow_den,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const float shadow_alpha_threshold,
    const float shadow_depth_group_eps,
    const bool use_shadow_receiver_bias,
    const at::optional<torch::Tensor> &shadow_receiver_bias
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    GSPLAT_CHECK_INPUT(gaussian_ids);
    GSPLAT_CHECK_INPUT(depths);
    GSPLAT_CHECK_INPUT(v_shadow_num);
    GSPLAT_CHECK_INPUT(v_shadow_den);
    if (shadow_receiver_bias.has_value()) {
        GSPLAT_CHECK_INPUT(shadow_receiver_bias.value());
    }
    const bool packed = means2d.dim() == 2;
    const uint32_t C = tile_offsets.size(0);
    const uint32_t N = packed ? 0 : means2d.size(1);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const uint32_t n_isects = flatten_ids.size(0);
    auto v_means2d = torch::zeros_like(means2d);
    auto v_conics = torch::zeros_like(conics);
    auto v_opacities = torch::zeros_like(opacities);
    if (n_isects == 0) {
        return std::make_tuple(v_means2d, v_conics, v_opacities);
    }
    const dim3 threads = {tile_size, tile_size, 1};
    const dim3 blocks = {C, tile_height, tile_width};
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    rasterize_to_pixels_bwd_shadow_only_kernel<float><<<blocks, threads, 0, stream>>>(
        C, N, n_isects, packed,
        reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
        reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
        opacities.data_ptr<float>(), image_width, image_height, tile_size, tile_width, tile_height,
        tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(), gaussian_ids.data_ptr<int32_t>(),
        depths.data_ptr<float>(), v_shadow_num.data_ptr<float>(), v_shadow_den.data_ptr<float>(),
        shadow_alpha_threshold, shadow_depth_group_eps, use_shadow_receiver_bias,
        shadow_receiver_bias.has_value() ? shadow_receiver_bias.value().data_ptr<float>() : nullptr,
        reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
        reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()), v_opacities.data_ptr<float>()
    );
    return std::make_tuple(v_means2d, v_conics, v_opacities);
}

} // namespace gsplat
