using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class SimdKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float x) => x >= 0 ? 1f / (1f + MathF.Exp(-x)) : (float)(MathF.Exp(x) / (1f + MathF.Exp(x)));

    // Abramowitz-Stegun 7.1.26; deterministic fallback because .NET exposes no MathF.Erf.
    private static float Erf(float x)
    {
        float s = x < 0 ? -1f : 1f, a = MathF.Abs(x), t = 1f / (1f + 0.3275911f * a);
        float p = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
        return s * (1f - p * MathF.Exp(-a * a));
    }
}
