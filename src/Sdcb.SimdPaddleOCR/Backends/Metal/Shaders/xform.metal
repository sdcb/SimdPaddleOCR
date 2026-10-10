// Port of {se_fused, layernorm, softmax, attn}.comp — SE block, row
// LayerNorm, row softmax, fused MHSA. All fp16 activations, fp32 stats.
#include <metal_stdlib>
using namespace metal;

// ---------------- se_part + se_join ----------------
struct PcSe { uint hw, C, S, P, cvP, R; float alpha, beta; };
// SE split into two dispatches: MSL device atomics are relaxed-only, so the
// Vulkan-style ticket (last-WG-does-stage2) cannot order partial writes vs
// reads. Emit se_part for all S workgroups, rely on the inter-dispatch buffer
// barrier, then se_join (one WG per batch) does stage2: mean → fc1+relu →
// fc2 → hardsigmoid.
kernel void se_part(device const half4* x [[buffer(0)]],
                    device float* part [[buffer(1)]],
                    constant PcSe& p [[buffer(2)]],
                    uint2 wg [[threadgroup_position_in_grid]],
                    uint2 lp2 [[thread_position_in_threadgroup]])
{
    uint lane = lp2.x;
    threadgroup float4 sm[256];

    uint s = wg.x;
    uint b = wg.y;
    uint cv = p.C >> 2;
    uint pb = b * p.S * p.C;
    uint xb = b * p.hw * cv;
    uint strips = 256u / p.cvP;
    uint strip = lane / p.cvP;
    uint cLane = lane - strip * p.cvP;
    float4 acc = float4(0.0);
    uint beg = s * p.P;
    uint end = min(beg + p.P, p.hw);
    if (cLane < cv)
        for (uint px = beg + strip; px < end; px += strips)
            acc += float4(x[xb + px * cv + cLane]);
    sm[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = strips >> 1; st > 0u; st >>= 1) {
        if (strip < st) sm[lane] += sm[lane + st * p.cvP];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (strip == 0u && cLane < cv) {
        uint ob = pb + s * p.C + cLane * 4u;
        part[ob] = sm[lane].x; part[ob + 1u] = sm[lane].y;
        part[ob + 2u] = sm[lane].z; part[ob + 3u] = sm[lane].w;
    }
}

kernel void se_join(device const float* part [[buffer(0)]],
                    device const float* w1 [[buffer(1)]],
                    device const float* b1 [[buffer(2)]],
                    device const float* w2 [[buffer(3)]],
                    device const float* b2 [[buffer(4)]],
                    device half4* o [[buffer(5)]],
                    constant PcSe& p [[buffer(6)]],
                    uint2 wg [[threadgroup_position_in_grid]],
                    uint2 lp2 [[thread_position_in_threadgroup]])
{
    uint lane = lp2.x;
    // C <= 1024, R <= 256 (gated at emit); stages loop in 256-lane strides.
    threadgroup float zf[1024];
    threadgroup float hf[256];

    uint b = wg.x;
    uint cv = p.C >> 2;
    uint pb = b * p.S * p.C;

    for (uint c0 = lane; c0 < p.C; c0 += 256u) {
        float a = 0.0f;
        for (uint si = 0u; si < p.S; si++) a += part[pb + si * p.C + c0];
        zf[c0] = a / float(p.hw);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r0 = lane; r0 < p.R; r0 += 256u) {
        float acc2 = b1[r0];
        for (uint c = 0u; c < p.C; c++)
            acc2 += zf[c] * w1[r0 * p.C + c];
        hf[r0] = max(acc2, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < cv) {
        half v[4];
        for (uint i = 0u; i < 4u; i++) {
            uint c = lane * 4u + i;
            float acc3 = b2[c];
            for (uint r = 0u; r < p.R; r++)
                acc3 += hf[r] * w2[c * p.R + r];
            v[i] = half(clamp(p.alpha * acc3 + p.beta, 0.0f, 1.0f));
        }
        o[b * cv + lane] = half4(v[0], v[1], v[2], v[3]);
    }
}

// ---------------- layernorm ----------------
// Row LayerNorm over the contiguous last axis:
//   y[r,c] = (x[r,c] - mean_r)/sqrt(var_r+eps) * g[c] + b[c]
// One 64-lane group per row, fp32 stats; C <= 64*16.
struct PcLn { uint rows, C; uint epsBits; };
kernel void layernorm(device const half* x [[buffer(0)]],
                      device const float* g [[buffer(1)]],
                      device const float* bt [[buffer(2)]],
                      device half* o [[buffer(3)]],
                      constant PcLn& p [[buffer(4)]],
                      uint wg [[threadgroup_position_in_grid]],
                      uint t [[thread_position_in_threadgroup]])
{
    threadgroup float red[64];
    uint r = wg;
    if (r >= p.rows) return;
    uint base = r * p.C;
    float v[16];
    float s = 0.0f;
    for (uint i = 0u; i < 16u; i++) {
        uint c = t + i * 64u;
        v[i] = c < p.C ? float(x[base + c]) : 0.0f;
        s += v[i];
    }
    // wgSum
    red[t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 32u; k > 0u; k >>= 1) {
        if (t < k) red[t] += red[t + k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / float(p.C);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q = 0.0f;
    for (uint i = 0u; i < 16u; i++) {
        uint c = t + i * 64u;
        float d = c < p.C ? v[i] - mean : 0.0f;
        q += d * d;
    }
    red[t] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 32u; k > 0u; k >>= 1) {
        if (t < k) red[t] += red[t + k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = red[0] / float(p.C);
    float inv = 1.0f / sqrt(var + as_type<float>(p.epsBits));
    for (uint i = 0u; i < 16u; i++) {
        uint c = t + i * 64u;
        if (c < p.C) o[base + c] = half((v[i] - mean) * inv * g[c] + bt[c]);
    }
}

// ---------------- softmax ----------------
// Row softmax on fp16: y[r,:] = softmax(x[r,:]) over the last axis.
struct PcSm { uint rows, cols; };
kernel void softmax(device const half* x [[buffer(0)]],
                    device half* y [[buffer(1)]],
                    constant PcSm& p [[buffer(2)]],
                    uint r [[thread_position_in_grid]])
{
    if (r >= p.rows) return;
    uint b = r * p.cols;
    float m = -3.4e38f;
    for (uint j = 0u; j < p.cols; j++) m = max(m, float(x[b + j]));
    float s = 0.0f;
    for (uint j = 0u; j < p.cols; j++) s += exp(float(x[b + j]) - m);
    float inv = 1.0f / s;
    for (uint j = 0u; j < p.cols; j++) y[b + j] = half(exp(float(x[b + j]) - m) * inv);
}

// ---------------- attn ----------------
// Fused multi-head self-attention on packed qkv [n, T, 3, H, D] fp16:
//   o[n, i, h, :] = sum_j softmax_j(scale * q_i . k_j) * v_j
// written as [n, T, H, D]. Online softmax, fp32 math; D <= 32 (zero-pad).
// Group = 64 queries of one (n, h); K/V staged through threadgroup tiles.
struct PcAttn { uint T, H, D; uint scaleBits; };
kernel void attn(device const half* qkv [[buffer(0)]],
                 device half* o [[buffer(1)]],
                 constant PcAttn& p [[buffer(2)]],
                 uint2 lp2 [[thread_position_in_threadgroup]],
                 uint2 wg [[threadgroup_position_in_grid]])
{
    uint lid = lp2.x;
    threadgroup float ks[64 * 32];
    threadgroup float vs[64 * 32];

    uint nh = wg.y;
    uint n = nh / p.H, h = nh % p.H;
    uint i = wg.x * 64u + lid;
    uint hd = p.H * p.D;
    uint rowStride = 3u * hd;
    bool valid = i < p.T;
    float sc = as_type<float>(p.scaleBits);

    float q[32];
    float acc[32];
    uint qb = (n * p.T + (valid ? i : 0u)) * rowStride + h * p.D;
    for (uint d = 0u; d < 32u; d++) {
        q[d] = (valid && d < p.D) ? float(qkv[qb + d]) * sc : 0.0f;
        acc[d] = 0.0f;
    }
    float m = -3.0e38f, l = 0.0f;

    for (uint j0 = 0u; j0 < p.T; j0 += 64u) {
        for (uint e = lid; e < 64u * 32u; e += 64u) {
            uint jr = e >> 5, d = e & 31u;
            uint j = j0 + jr;
            float kv = 0.0f, vv = 0.0f;
            if (j < p.T && d < p.D) {
                uint kb = (n * p.T + j) * rowStride + hd + h * p.D + d;
                kv = float(qkv[kb]);
                vv = float(qkv[kb + hd]);
            }
            ks[e] = kv;
            vs[e] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint jn = min(64u, p.T - j0);
        for (uint jr = 0u; jr < jn; jr++) {
            float s = 0.0f;
            for (uint d = 0u; d < 32u; d++) s += q[d] * ks[jr * 32u + d];
            float mn = max(m, s);
            float corr = exp(m - mn);
            float pe = exp(s - mn);
            l = l * corr + pe;
            for (uint d = 0u; d < 32u; d++) acc[d] = acc[d] * corr + pe * vs[jr * 32u + d];
            m = mn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (valid) {
        uint ob = (n * p.T + i) * hd + h * p.D;
        float inv = 1.0f / l;
        for (uint d = 0u; d < 32u; d++)
            if (d < p.D) o[ob + d] = half(acc[d] * inv);
    }
}
