using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class MatMul
{
    // Prefer Avx512 only for the REC projection shape below. Broad Avx512
    // MatMul (all shapes / light 4-row tiles) was slower than AVX on Zen 5
    // in paired A/B (~+10–15 ms e2e), so the ladder is intentionally
    // shape-gated rather than "Avx512F ⇒ always Avx512 kernels".
    internal static bool Try(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns, float[]? packedWeights = null)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported && packedWeights is not null &&
            rows >= 8 && (rows & 7) == 0 && inner >= 64 && columns >= 1024)
        {
            // Eight-row packed tile: one ZMM weight load feeds eight FMA chains.
            MatMulRows16PackedAvx512(input, weights, packedWeights, output,
                batch, rows, inner, columns);
            return true;
        }
        else if (Avx.IsSupported)
        {
            if (packedWeights is not null && rows >= 4 && (rows & 3) == 0 && inner >= 64 && columns >= 1024)
            {
                MatMulRows4Packed(input, weights, packedWeights, output, batch, rows, inner, columns);
                return true;
            }
            if (rows >= 4)
            {
                MatMulRows4(input, weights, output, batch, rows, inner, columns);
                return true;
            }
            MatMulRows1(input, weights, output, batch, 0, rows, inner, columns);
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return TryVector(input, weights, output, batch, rows, inner, columns);
        }
        MatMulScalar(input, weights, output, batch, rows, inner, columns);
        return true;
    }
}
