// Port of Vulkan Shaders/{nchw2nhwc,sigmoid_out,elem,elem4,addps,affine4}.comp
// Same semantics; fp16 arena, fp32 compute. The GLSL uint-pair reads become
// native half / half4 loads (vec4 binds always sit on 8B-aligned offsets).
#include <metal_stdlib>
using namespace metal;
// erf1 / auxHalf2 live in a_common.metal (compiled into the same library).

// ---------------- nchw2nhwc ----------------
// Graph input convert: fp32 NCHW [n,C,H,W] → fp16 NHWC [n,H*W,Cout].
// Cout may exceed C (zero-padded channels for aligned downstream reads).
struct Pc3 { uint hw, C, Cout; };
// One thread per pixel: per-plane reads stay w-contiguous (fully
// coalesced per channel), writes are back-to-back half4 rows.
kernel void nchw2nhwc(device const float* x [[buffer(0)]],
                      device half* o [[buffer(1)]],
                      constant Pc3& p [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.hw) return;
    ulong inBase = ulong(gid.y) * p.C * p.hw + gid.x;
    ulong outBase = (ulong(gid.y) * p.hw + gid.x) * p.Cout;
    if ((p.Cout & 3u) == 0u)
    {
        for (uint i = 0u; i < p.Cout; i += 4u)
        {
            half4 v;
            for (uint j = 0u; j < 4u; j++)
            {
                uint c = i + j;
                v[j] = c < p.C ? half(x[inBase + c * p.hw]) : half(0.0);
            }
            ((device half4*)(o + outBase + i))[0] = v;
        }
    }
    else
    {
        for (uint i = 0u; i < p.Cout; i++)
            o[outBase + i] = i < p.C ? half(x[inBase + i * p.hw]) : half(0.0);
    }
}

// ---------------- sigmoid_out ----------------
// Final head: fp16 → fp32 into the readback buffer, optional sigmoid.
struct Pc2 { uint n, act; };
kernel void sigmoid_out(device const half* x [[buffer(0)]],
                        device float* o [[buffer(1)]],
                        constant Pc2& p [[buffer(2)]],
                        uint i [[thread_position_in_grid]])
{
    if (i >= p.n) return;
    float v = float(x[i]);
    o[i] = (p.act == 4u) ? 1.0 / (1.0 + exp(-v)) : v;
}

// ---------------- elem / elem4 ----------------
// Generic elementwise on NHWC fp16 activations.
//   unary:  o[i] = f(a[i])                        (op 0-7, 14)
//   binary: o[i] = a[ai(i)] OP b[bi(i)]           (op 8+)
// Broadcast modes per operand (aMode/bMode): 0=full index, 1=channel vector
//   (index % C), 2=scalar (index 0), 3=per-image channel vector (aux =
//   elements per image).
//   op unary: 0=copy 1=relu 2=gelu 3=hardsigmoid(aux) 4=sigmoid 5=erf 6=sqrt
//             14=hardswish (x*hs(x), aux)
//   op binary: 8=add 9=mul 10=sub 11=div 12=max 13=pow
struct PcElem { uint n, C, op, aMode, bMode, aux; };

static inline uint elem_idx(uint i, uint mode, constant PcElem& p)
{
    if (mode == 1u) return i % p.C;
    if (mode == 2u) return 0u;
    if (mode == 3u) return (i / p.aux) * p.C + i % p.C;
    return i;
}

static inline float elem_unary(uint op, float va, uint aux)
{
    switch (op) {
    case 1u: return max(va, 0.0f);
    case 2u: return 0.5f * va * (1.0f + erf1(va * 0.70710678118654752f));
    case 3u: { float2 ab = auxHalf2(aux); return clamp(va * ab.x + ab.y, 0.0f, 1.0f); }
    case 4u: return 1.0f / (1.0f + exp(-va));
    case 5u: return erf1(va);
    case 6u: return sqrt(max(va, 0.0f));
    case 14u: { float2 ab = auxHalf2(aux); return va * clamp(va * ab.x + ab.y, 0.0f, 1.0f); }
    default: return va;
    }
}

kernel void elem(device const half* a [[buffer(0)]],
                 device const half* b [[buffer(1)]],
                 device half* o [[buffer(2)]],
                 constant PcElem& p [[buffer(3)]],
                 uint i [[thread_position_in_grid]])
{
    if (i >= p.n) return;
    float va = float(a[elem_idx(i, p.aMode, p)]);
    float r;
    if (p.op < 8u || p.op == 14u) {
        r = elem_unary(p.op, va, p.aux);
    } else {
        float vb = float(b[elem_idx(i, p.bMode, p)]);
        switch (p.op) {
        case 9u:  r = va * vb; break;
        case 10u: r = va - vb; break;
        case 11u: r = va / vb; break;
        case 12u: r = max(va, vb); break;
        case 13u: r = pow(va, vb); break;
        default:  r = va + vb; break;
        }
    }
    o[i] = half(r);
}

kernel void elem4(device const half4* a [[buffer(0)]],
                  device const half4* b [[buffer(1)]],
                  device half4* o [[buffer(2)]],
                  constant PcElem& p [[buffer(3)]],
                  uint gid [[thread_position_in_grid]])
{
    uint i0 = gid * 4u;
    if (i0 >= p.n) return;

    float4 va;
    {
        uint base = p.aMode == 1u ? i0 % p.C
                  : p.aMode == 3u ? (i0 / p.aux) * p.C + i0 % p.C : i0;
        va = p.aMode == 2u ? float4(float(a[0].x)) : float4(a[base >> 2]);
    }
    float4 r;
    if (p.op < 8u || p.op == 14u) {
        if (p.op == 2u || p.op == 5u) {
            for (int i = 0; i < 4; i++) r[i] = elem_unary(p.op, va[i], p.aux);
        } else if (p.op == 3u || p.op == 14u) {
            float2 ab = auxHalf2(p.aux);
            float4 hs = clamp(va * ab.x + ab.y, 0.0f, 1.0f);
            r = p.op == 14u ? va * hs : hs;
        } else {
            for (int i = 0; i < 4; i++) r[i] = elem_unary(p.op, va[i], p.aux);
        }
    } else {
        float4 vb;
        uint base = p.bMode == 1u ? i0 % p.C
                  : p.bMode == 3u ? (i0 / p.aux) * p.C + i0 % p.C : i0;
        vb = p.bMode == 2u ? float4(float(b[0].x)) : float4(b[base >> 2]);
        if (p.op == 9u) r = va * vb;
        else if (p.op == 10u) r = va - vb;
        else if (p.op == 11u) r = va / vb;
        else if (p.op == 12u) r = max(va, vb);
        else if (p.op == 13u) r = pow(va, vb);
        else r = va + vb;
    }
    o[gid] = half4(r);
}

// ---------------- addps ----------------
// Scaled residual add on NHWC fp16: o[i] = a[i] + x[i] * s[c(i)]
// s is a per-channel fp16 vector with cv4 = C/4 vec4 groups. Requires
// n%4==0 && C%4==0. (per-image SE: si is vec4-indexed within the image)
struct PcAddps { uint n, cv4, perImg4; };
kernel void addps(device const half4* a [[buffer(0)]],
                  device const half4* x [[buffer(1)]],
                  device const half4* s [[buffer(2)]],
                  device half4* o [[buffer(3)]],
                  constant PcAddps& p [[buffer(4)]],
                  uint gid [[thread_position_in_grid]])
{
    uint i0 = gid * 4u;
    if (i0 >= p.n) return;
    uint v = i0 >> 2;
    // cv4 = vec4 count of the channel dim; perImg4 = vec4 count per image
    // (0 = same numel as out → channel-only). Per-batch SE [n,C,1,1]:
    // perImg4>0 → index within image's channel vector.
    uint si = p.perImg4 > 0 ? (v / p.perImg4) * p.cv4 + v % p.cv4 : v % p.cv4;
    o[v] = a[v] + x[v] * s[si];
}

// ---------------- affine4 ----------------
// Per-channel affine on NHWC fp16 (folded BatchNormalization):
//   o[i] = x[i] * s[c4] + t[c4]      i = flat vec4 index, c4 = i % cv
struct PcAff { uint n, cv; };
kernel void affine4(device const half4* x [[buffer(0)]],
                    device const half4* s [[buffer(1)]],
                    device const half4* t [[buffer(2)]],
                    device half4* o [[buffer(3)]],
                    constant PcAff& p [[buffer(4)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint c4 = gid % p.cv;
    o[gid] = x[gid] * s[c4] + t[c4];
}
