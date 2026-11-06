// ============================================================================
// MedTRT Custom CUDA C++ Medical Image Processing Kernels (`cuda/medical_kernels.cu`)
// Target Architecture: NVIDIA Blackwell (Compute Capability 12.0 / sm_120)
// Compiled via NVRTC 12.8 (`nvrtc64_120_0.dll`) & executed via CUDA Driver API (`nvcuda.dll`)
// ============================================================================

extern "C" {

// ----------------------------------------------------------------------------
// 1. Scalar Grid-Stride HU Windowing & Intensity Normalization (FP32)
//    Maps [a_min, a_max] -> [b_min, b_max] with clamping.
// ----------------------------------------------------------------------------
__global__ void hu_window_normalize_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long n_elements,
    float a_min,
    float a_max,
    float inv_range,
    float b_min,
    float b_scale
) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = idx; i < n_elements; i += stride) {
        float v = input[i];
        v = fminf(fmaxf(v, a_min), a_max);
        output[i] = (v - a_min) * inv_range * b_scale + b_min;
    }
}

// ----------------------------------------------------------------------------
// 2. Vectorized float4 (128-bit) Coalesced HU Windowing & Intensity Normalization
// ----------------------------------------------------------------------------
__global__ void hu_window_normalize_vec4_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long n_elements,
    float a_min,
    float a_max,
    float inv_range,
    float b_min,
    float b_scale
) {
    long long idx4 = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride4 = static_cast<long long>(gridDim.x) * blockDim.x;
    long long n_vec4 = n_elements >> 2;

    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(output);
    float scale = inv_range * b_scale;

    for (long long i = idx4; i < n_vec4; i += stride4) {
        float4 v = in4[i];
        v.x = (fminf(fmaxf(v.x, a_min), a_max) - a_min) * scale + b_min;
        v.y = (fminf(fmaxf(v.y, a_min), a_max) - a_min) * scale + b_min;
        v.z = (fminf(fmaxf(v.z, a_min), a_max) - a_min) * scale + b_min;
        v.w = (fminf(fmaxf(v.w, a_min), a_max) - a_min) * scale + b_min;
        out4[i] = v;
    }

    long long tail_start = n_vec4 << 2;
    for (long long i = tail_start + idx4; i < n_elements; i += stride4) {
        float val = fminf(fmaxf(input[i], a_min), a_max);
        output[i] = (val - a_min) * scale + b_min;
    }
}

// ----------------------------------------------------------------------------
// Helper: Clamp integer index to [0, dim - 1] (border padding mode matching MONAI)
// ----------------------------------------------------------------------------
__device__ __forceinline__ int clamp_idx(int v, int dim) {
    return v < 0 ? 0 : (v >= dim ? dim - 1 : v);
}

// ----------------------------------------------------------------------------
// 3. 3D Affine Trilinear Resampling (Register-Computed Coordinate Mapping)
//    Eliminates MONAI SpatialResample's dense (D*H*W*3) coordinate meshgrid allocation!
//    Memory layout: (C, X, Y, Z) where Z is contiguous (threadIdx.x -> z).
//    Maps output voxel (x_o, y_o, z_o) -> input voxel (x_i, y_i, z_i) via M_3x4.
// ----------------------------------------------------------------------------
__global__ void affine_trilinear_resample_3d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels,
    int in_x, int in_y, int in_z,
    int out_x, int out_y, int out_z,
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23
) {
    int zo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;
    int xo = blockIdx.z * blockDim.z + threadIdx.z;

    if (xo >= out_x || yo >= out_y || zo >= out_z) return;

    float fx = m00 * xo + m01 * yo + m02 * zo + m03;
    float fy = m10 * xo + m11 * yo + m12 * zo + m13;
    float fz = m20 * xo + m21 * yo + m22 * zo + m23;

    // Clamp to valid border range [0, dim - 1] matching grid_sample border padding
    fx = fminf(fmaxf(fx, 0.0f), static_cast<float>(in_x - 1));
    fy = fminf(fmaxf(fy, 0.0f), static_cast<float>(in_y - 1));
    fz = fminf(fmaxf(fz, 0.0f), static_cast<float>(in_z - 1));

    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    int z0 = static_cast<int>(floorf(fz));

    int x1 = clamp_idx(x0 + 1, in_x);
    int y1 = clamp_idx(y0 + 1, in_y);
    int z1 = clamp_idx(z0 + 1, in_z);
    x0 = clamp_idx(x0, in_x);
    y0 = clamp_idx(y0, in_y);
    z0 = clamp_idx(z0, in_z);

    float wx = fx - static_cast<float>(x0);
    float wy = fy - static_cast<float>(y0);
    float wz = fz - static_cast<float>(z0);

    float wx0 = 1.0f - wx, wx1 = wx;
    float wy0 = 1.0f - wy, wy1 = wy;
    float wz0 = 1.0f - wz, wz1 = wz;

    long long in_stride_y = in_z;
    long long in_stride_x = static_cast<long long>(in_y) * in_z;
    long long in_vol = static_cast<long long>(in_x) * in_stride_x;

    long long out_idx_base = (static_cast<long long>(xo) * out_y + yo) * out_z + zo;
    long long out_vol = static_cast<long long>(out_x) * out_y * out_z;

    long long o000 = x0 * in_stride_x + y0 * in_stride_y + z0;
    long long o001 = x0 * in_stride_x + y0 * in_stride_y + z1;
    long long o010 = x0 * in_stride_x + y1 * in_stride_y + z0;
    long long o011 = x0 * in_stride_x + y1 * in_stride_y + z1;
    long long o100 = x1 * in_stride_x + y0 * in_stride_y + z0;
    long long o101 = x1 * in_stride_x + y0 * in_stride_y + z1;
    long long o110 = x1 * in_stride_x + y1 * in_stride_y + z0;
    long long o111 = x1 * in_stride_x + y1 * in_stride_y + z1;

    for (int c = 0; c < channels; ++c) {
        const float* __restrict__ in_c = input + c * in_vol;
        float c00 = in_c[o000] * wz0 + in_c[o001] * wz1;
        float c01 = in_c[o010] * wz0 + in_c[o011] * wz1;
        float c10 = in_c[o100] * wz0 + in_c[o101] * wz1;
        float c11 = in_c[o110] * wz0 + in_c[o111] * wz1;
        float c0 = c00 * wy0 + c01 * wy1;
        float c1 = c10 * wy0 + c11 * wy1;
        output[c * out_vol + out_idx_base] = c0 * wx0 + c1 * wx1;
    }
}

// ----------------------------------------------------------------------------
// 3b. Fused 3D Affine Trilinear Resampling + HU Window Clamping & Normalization
//     Replaces MONAI's Orientationd(RAS) + Spacingd(bilinear) + ScaleIntensityRanged
//     in a single CUDA kernel pass without allocating a 3D coordinate meshgrid!
// ----------------------------------------------------------------------------
__global__ void fused_preprocess_affine_trilinear_hu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_x, int in_y, int in_z,
    int out_x, int out_y, int out_z,
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float a_min, float a_max, float scale, float b_min
) {
    int zo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;
    int xo = blockIdx.z * blockDim.z + threadIdx.z;

    if (xo >= out_x || yo >= out_y || zo >= out_z) return;

    float fx = m00 * xo + m01 * yo + m02 * zo + m03;
    float fy = m10 * xo + m11 * yo + m12 * zo + m13;
    float fz = m20 * xo + m21 * yo + m22 * zo + m23;

    fx = fminf(fmaxf(fx, 0.0f), static_cast<float>(in_x - 1));
    fy = fminf(fmaxf(fy, 0.0f), static_cast<float>(in_y - 1));
    fz = fminf(fmaxf(fz, 0.0f), static_cast<float>(in_z - 1));

    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    int z0 = static_cast<int>(floorf(fz));

    int x1 = clamp_idx(x0 + 1, in_x);
    int y1 = clamp_idx(y0 + 1, in_y);
    int z1 = clamp_idx(z0 + 1, in_z);
    x0 = clamp_idx(x0, in_x);
    y0 = clamp_idx(y0, in_y);
    z0 = clamp_idx(z0, in_z);

    float wx = fx - static_cast<float>(x0);
    float wy = fy - static_cast<float>(y0);
    float wz = fz - static_cast<float>(z0);

    float wx0 = 1.0f - wx, wx1 = wx;
    float wy0 = 1.0f - wy, wy1 = wy;
    float wz0 = 1.0f - wz, wz1 = wz;

    long long in_stride_y = in_z;
    long long in_stride_x = static_cast<long long>(in_y) * in_z;

    float c00 = input[x0 * in_stride_x + y0 * in_stride_y + z0] * wz0 +
                input[x0 * in_stride_x + y0 * in_stride_y + z1] * wz1;
    float c01 = input[x0 * in_stride_x + y1 * in_stride_y + z0] * wz0 +
                input[x0 * in_stride_x + y1 * in_stride_y + z1] * wz1;
    float c10 = input[x1 * in_stride_x + y0 * in_stride_y + z0] * wz0 +
                input[x1 * in_stride_x + y0 * in_stride_y + z1] * wz1;
    float c11 = input[x1 * in_stride_x + y1 * in_stride_y + z0] * wz0 +
                input[x1 * in_stride_x + y1 * in_stride_y + z1] * wz1;

    float val = (c00 * wy0 + c01 * wy1) * wx0 + (c10 * wy0 + c11 * wy1) * wx1;
    val = fminf(fmaxf(val, a_min), a_max);
    long long out_idx = (static_cast<long long>(xo) * out_y + yo) * out_z + zo;
    output[out_idx] = (val - a_min) * scale + b_min;
}

// ----------------------------------------------------------------------------
// 4. 3D Affine Nearest-Neighbor Discrete Label Resampling (uint8)
// ----------------------------------------------------------------------------
__global__ void affine_nearest_resample_3d_u8(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_x, int in_y, int in_z,
    int out_x, int out_y, int out_z,
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23
) {
    int zo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;
    int xo = blockIdx.z * blockDim.z + threadIdx.z;

    if (xo >= out_x || yo >= out_y || zo >= out_z) return;

    float fx = m00 * xo + m01 * yo + m02 * zo + m03;
    float fy = m10 * xo + m11 * yo + m12 * zo + m13;
    float fz = m20 * xo + m21 * yo + m22 * zo + m23;

    int xi = clamp_idx(static_cast<int>( roundf(fx) ), in_x);
    int yi = clamp_idx(static_cast<int>( roundf(fy) ), in_y);
    int zi = clamp_idx(static_cast<int>( roundf(fz) ), in_z);

    long long in_idx = (static_cast<long long>(xi) * in_y + yi) * in_z + zi;
    long long out_idx = (static_cast<long long>(xo) * out_y + yo) * out_z + zo;
    output[out_idx] = input[in_idx];
}

// ----------------------------------------------------------------------------
// 5. Fused 2-Channel Softmax + Affine 3D Trilinear Inverse Resample + Argmax -> uint8
//    Replaces the entire MONAI postprocessing chain:
//      Activationsd(softmax=True) -> Invertd(nearest_interp=False) -> AsDiscreted(argmax=True)
//    in a SINGLE pass without materializing the (4, X*Y*Z) coordinate grid or the
//    intermediate (2, X_orig, Y_orig, Z_orig) float32 resampled probability volume!
//    Note: For 2-class softmax (p0 + p1 = 1), trilinear interpolation of p1 minus p0
//    equals trilinear interpolation of (p1 - p0), so interpolating p1 alone and comparing
//    against 0.5f is algebraically identical to interpolating both channels and taking argmax(p0, p1),
//    cutting global memory reads in half (8 loads instead of 16 loads per output voxel)!
// ----------------------------------------------------------------------------
__global__ void fused_softmax_2ch_p1_f32(
    const float* __restrict__ logits,
    float* __restrict__ prob_ch1,
    long long n_voxels
) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const float* __restrict__ ch0 = logits;
    const float* __restrict__ ch1 = logits + n_voxels;
    for (long long i = idx; i < n_voxels; i += stride) {
        float l0 = ch0[i];
        float l1 = ch1[i];
        float m = fmaxf(l0, l1);
        float e0 = __expf(l0 - m);
        float e1 = __expf(l1 - m);
        prob_ch1[i] = e1 / (e0 + e1);
    }
}

__global__ void fused_invert_trilinear_threshold_u8(
    const float* __restrict__ prob_ch1,
    unsigned char* __restrict__ out_mask,
    unsigned int* __restrict__ fg_voxel_counter,
    int in_x, int in_y, int in_z,
    int out_x, int out_y, int out_z,
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23
) {
    int zo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;
    int xo = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int is_fg = 0u;
    if (xo < out_x && yo < out_y && zo < out_z) {
        float fx = m00 * xo + m01 * yo + m02 * zo + m03;
        float fy = m10 * xo + m11 * yo + m12 * zo + m13;
        float fz = m20 * xo + m21 * yo + m22 * zo + m23;

        fx = fminf(fmaxf(fx, 0.0f), static_cast<float>(in_x - 1));
        fy = fminf(fmaxf(fy, 0.0f), static_cast<float>(in_y - 1));
        fz = fminf(fmaxf(fz, 0.0f), static_cast<float>(in_z - 1));

        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        int z0 = static_cast<int>(floorf(fz));

        int x1 = clamp_idx(x0 + 1, in_x);
        int y1 = clamp_idx(y0 + 1, in_y);
        int z1 = clamp_idx(z0 + 1, in_z);
        x0 = clamp_idx(x0, in_x);
        y0 = clamp_idx(y0, in_y);
        z0 = clamp_idx(z0, in_z);

        float wx = fx - static_cast<float>(x0);
        float wy = fy - static_cast<float>(y0);
        float wz = fz - static_cast<float>(z0);

        float wx0 = 1.0f - wx, wx1 = wx;
        float wy0 = 1.0f - wy, wy1 = wy;
        float wz0 = 1.0f - wz, wz1 = wz;

        long long in_stride_y = in_z;
        long long in_stride_x = static_cast<long long>(in_y) * in_z;

        float c00 = prob_ch1[x0 * in_stride_x + y0 * in_stride_y + z0] * wz0 +
                    prob_ch1[x0 * in_stride_x + y0 * in_stride_y + z1] * wz1;
        float c01 = prob_ch1[x0 * in_stride_x + y1 * in_stride_y + z0] * wz0 +
                    prob_ch1[x0 * in_stride_x + y1 * in_stride_y + z1] * wz1;
        float c10 = prob_ch1[x1 * in_stride_x + y0 * in_stride_y + z0] * wz0 +
                    prob_ch1[x1 * in_stride_x + y0 * in_stride_y + z1] * wz1;
        float c11 = prob_ch1[x1 * in_stride_x + y1 * in_stride_y + z0] * wz0 +
                    prob_ch1[x1 * in_stride_x + y1 * in_stride_y + z1] * wz1;

        float p1 = (c00 * wy0 + c01 * wy1) * wx0 + (c10 * wy0 + c11 * wy1) * wx1;
        is_fg = (p1 >= 0.5f) ? 1u : 0u;

        long long out_idx = (static_cast<long long>(xo) * out_y + yo) * out_z + zo;
        out_mask[out_idx] = static_cast<unsigned char>(is_fg);
    }

    // Warp-level ballot reduction for foreground voxel counting (1 atomicAdd per warp)
    unsigned int mask = __ballot_sync(0xFFFFFFFFu, is_fg != 0u);
    int lane = (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x) & 31;
    if (lane == 0 && mask != 0u && fg_voxel_counter != nullptr) {
        atomicAdd(fg_voxel_counter, static_cast<unsigned int>(__popc(mask)));
    }
}

// ----------------------------------------------------------------------------
// 6. Standalone Voxel Counting Kernel with Warp Ballot + Shared-Memory Reduction
// ----------------------------------------------------------------------------
__global__ void count_nonzero_u8(
    const unsigned char* __restrict__ mask_u8,
    unsigned int* __restrict__ out_count,
    long long n_elements
) {
    __shared__ unsigned int warp_sums[32];
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

    unsigned int local_sum = 0u;
    for (long long i = idx; i < n_elements; i += stride) {
        local_sum += (mask_u8[i] != 0) ? 1u : 0u;
    }

    // Warp-shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, offset);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        unsigned int block_sum = (lane < num_warps) ? warp_sums[lane] : 0u;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xFFFFFFFFu, block_sum, offset);
        }
        if (lane == 0 && block_sum > 0u) {
            atomicAdd(out_count, block_sum);
        }
    }
}

// ----------------------------------------------------------------------------
// 7. 3D Sliding-Window Gaussian-Weighted Patch Accumulation
//    Accumulates a batch of (B, C, roi_x, roi_y, roi_z) patches into output
//    volume (C, vol_x, vol_y, vol_z) and count map (1, vol_x, vol_y, vol_z).
// ----------------------------------------------------------------------------
__global__ void sliding_window_gaussian_accumulate_f32(
    const float* __restrict__ patch_logits,
    const float* __restrict__ gaussian_weight,
    float* __restrict__ out_logits,
    float* __restrict__ out_weights,
    int channels,
    int roi_x, int roi_y, int roi_z,
    int vol_x, int vol_y, int vol_z,
    int start_x, int start_y, int start_z
) {
    int rz = blockIdx.x * blockDim.x + threadIdx.x;
    int ry = blockIdx.y * blockDim.y + threadIdx.y;
    int rx = blockIdx.z * blockDim.z + threadIdx.z;

    if (rx >= roi_x || ry >= roi_y || rz >= roi_z) return;

    int vx = start_x + rx;
    int vy = start_y + ry;
    int vz = start_z + rz;
    if (vx >= vol_x || vy >= vol_y || vz >= vol_z) return;

    long long roi_idx = (static_cast<long long>(rx) * roi_y + ry) * roi_z + rz;
    long long roi_vol = static_cast<long long>(roi_x) * roi_y * roi_z;
    long long vol_idx = (static_cast<long long>(vx) * vol_y + vy) * vol_z + vz;
    long long vol_size = static_cast<long long>(vol_x) * vol_y * vol_z;

    float w = gaussian_weight[roi_idx];
    out_weights[vol_idx] += w;
    for (int c = 0; c < channels; ++c) {
        out_logits[c * vol_size + vol_idx] += patch_logits[c * roi_vol + roi_idx] * w;
    }
}

} // extern "C"
