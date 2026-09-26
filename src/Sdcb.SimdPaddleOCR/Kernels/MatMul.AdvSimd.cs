using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
#endif
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class MatMul
{
#if !NETSTANDARD2_0
    // ARM64 NEON tier. The Vector<T> fallback spends one `dup` per (row, k)
    // on broadcasting the input scalar; AdvSimd can instead load four k's of
    // one row in a single ld1.4s and index the lane inside the FMA itself
    // (`fmla acc.4s, w.4s, x.s[j]` via FusedMultiplyAddBySelectedScalar),
    // removing the broadcast chain from the inner loop entirely.

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool TryAdvSimd(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, Span<float> output, int batch, int rows,
        int inner, int columns, float[]? packedWeights = null)
    {
        if (packedWeights is not null && rows >= 4 && inner >= 64 && columns >= 1024)
            MatMulRows4PackedAdvSimd(input, weights, packedWeights, output,
                batch, rows, inner, columns);
        else if (rows >= 4)
            MatMulRows4AdvSimd(input, weights, output, batch, rows, inner, columns);
        else
            MatMulRows1Vector(input, weights, output, batch, 0, rows, inner, columns);
        return true;
    }

    // 4 rows x 16 columns over the 16-column packed tile. Per 4-k block the
    // input side costs one ld1.4s per row (vs four broadcasts in the Vector
    // path); the weight side is sixteen contiguous vec128 loads.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows4PackedAdvSimd(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights,
        Span<float> output, int batch, int rows, int inner, int columns)
    {
        fixed (float* inputPtr = input, weightsPtr = weights,
            packedPtr = packedWeights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                int row = 0;
                for (; row <= rows - 4; row += 4)
                {
                    int inputBase = (b * rows + row) * inner;
                    int outputBase = (b * rows + row) * columns;
                    int col = 0;
                    for (; col <= columns - 16; col += 16)
                    {
                        float* tile = packedPtr + (col / 16) * inner * 16;
                        Vector128<float> a00 = Vector128<float>.Zero, a01 = a00, a02 = a00, a03 = a00;
                        Vector128<float> a10 = Vector128<float>.Zero, a11 = a10, a12 = a10, a13 = a10;
                        Vector128<float> a20 = Vector128<float>.Zero, a21 = a20, a22 = a20, a23 = a20;
                        Vector128<float> a30 = Vector128<float>.Zero, a31 = a30, a32 = a30, a33 = a30;
                        int k = 0;
                        for (; k <= inner - 4; k += 4)
                        {
                            Vector128<float> x0 = AdvSimd.LoadVector128(inputPtr + inputBase + k);
                            Vector128<float> x1 = AdvSimd.LoadVector128(inputPtr + inputBase + inner + k);
                            Vector128<float> x2 = AdvSimd.LoadVector128(inputPtr + inputBase + inner * 2 + k);
                            Vector128<float> x3 = AdvSimd.LoadVector128(inputPtr + inputBase + inner * 3 + k);
                            float* wrow = tile + k * 16;
                            Vector128<float> w00 = AdvSimd.LoadVector128(wrow);
                            Vector128<float> w01 = AdvSimd.LoadVector128(wrow + 4);
                            Vector128<float> w02 = AdvSimd.LoadVector128(wrow + 8);
                            Vector128<float> w03 = AdvSimd.LoadVector128(wrow + 12);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w00, x0, 0);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w01, x0, 0);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w02, x0, 0);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w03, x0, 0);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w00, x1, 0);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w01, x1, 0);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w02, x1, 0);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w03, x1, 0);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w00, x2, 0);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w01, x2, 0);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w02, x2, 0);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w03, x2, 0);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w00, x3, 0);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w01, x3, 0);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w02, x3, 0);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w03, x3, 0);
                            wrow += 16;
                            w00 = AdvSimd.LoadVector128(wrow);
                            w01 = AdvSimd.LoadVector128(wrow + 4);
                            w02 = AdvSimd.LoadVector128(wrow + 8);
                            w03 = AdvSimd.LoadVector128(wrow + 12);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w00, x0, 1);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w01, x0, 1);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w02, x0, 1);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w03, x0, 1);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w00, x1, 1);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w01, x1, 1);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w02, x1, 1);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w03, x1, 1);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w00, x2, 1);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w01, x2, 1);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w02, x2, 1);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w03, x2, 1);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w00, x3, 1);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w01, x3, 1);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w02, x3, 1);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w03, x3, 1);
                            wrow += 16;
                            w00 = AdvSimd.LoadVector128(wrow);
                            w01 = AdvSimd.LoadVector128(wrow + 4);
                            w02 = AdvSimd.LoadVector128(wrow + 8);
                            w03 = AdvSimd.LoadVector128(wrow + 12);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w00, x0, 2);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w01, x0, 2);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w02, x0, 2);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w03, x0, 2);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w00, x1, 2);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w01, x1, 2);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w02, x1, 2);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w03, x1, 2);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w00, x2, 2);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w01, x2, 2);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w02, x2, 2);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w03, x2, 2);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w00, x3, 2);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w01, x3, 2);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w02, x3, 2);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w03, x3, 2);
                            wrow += 16;
                            w00 = AdvSimd.LoadVector128(wrow);
                            w01 = AdvSimd.LoadVector128(wrow + 4);
                            w02 = AdvSimd.LoadVector128(wrow + 8);
                            w03 = AdvSimd.LoadVector128(wrow + 12);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w00, x0, 3);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w01, x0, 3);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w02, x0, 3);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w03, x0, 3);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w00, x1, 3);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w01, x1, 3);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w02, x1, 3);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w03, x1, 3);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w00, x2, 3);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w01, x2, 3);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w02, x2, 3);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w03, x2, 3);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w00, x3, 3);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w01, x3, 3);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w02, x3, 3);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w03, x3, 3);
                        }
                        for (; k < inner; k++)
                        {
                            float* wrow = tile + k * 16;
                            Vector128<float> w0 = AdvSimd.LoadVector128(wrow);
                            Vector128<float> w1 = AdvSimd.LoadVector128(wrow + 4);
                            Vector128<float> w2 = AdvSimd.LoadVector128(wrow + 8);
                            Vector128<float> w3 = AdvSimd.LoadVector128(wrow + 12);
                            Vector64<float> x0 = Vector64.CreateScalarUnsafe(inputPtr[inputBase + k]);
                            Vector64<float> x1 = Vector64.CreateScalarUnsafe(inputPtr[inputBase + inner + k]);
                            Vector64<float> x2 = Vector64.CreateScalarUnsafe(inputPtr[inputBase + inner * 2 + k]);
                            Vector64<float> x3 = Vector64.CreateScalarUnsafe(inputPtr[inputBase + inner * 3 + k]);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a00, w0, x0);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a01, w1, x0);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a02, w2, x0);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a03, w3, x0);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a10, w0, x1);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a11, w1, x1);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a12, w2, x1);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a13, w3, x1);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a20, w0, x2);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a21, w1, x2);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a22, w2, x2);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a23, w3, x2);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a30, w0, x3);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a31, w1, x3);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a32, w2, x3);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a33, w3, x3);
                        }
                        int ob = outputBase + col;
                        AdvSimd.Store(outputPtr + ob, a00); AdvSimd.Store(outputPtr + ob + 4, a01);
                        AdvSimd.Store(outputPtr + ob + 8, a02); AdvSimd.Store(outputPtr + ob + 12, a03);
                        AdvSimd.Store(outputPtr + ob + columns, a10); AdvSimd.Store(outputPtr + ob + columns + 4, a11);
                        AdvSimd.Store(outputPtr + ob + columns + 8, a12); AdvSimd.Store(outputPtr + ob + columns + 12, a13);
                        AdvSimd.Store(outputPtr + ob + columns * 2, a20); AdvSimd.Store(outputPtr + ob + columns * 2 + 4, a21);
                        AdvSimd.Store(outputPtr + ob + columns * 2 + 8, a22); AdvSimd.Store(outputPtr + ob + columns * 2 + 12, a23);
                        AdvSimd.Store(outputPtr + ob + columns * 3, a30); AdvSimd.Store(outputPtr + ob + columns * 3 + 4, a31);
                        AdvSimd.Store(outputPtr + ob + columns * 3 + 8, a32); AdvSimd.Store(outputPtr + ob + columns * 3 + 12, a33);
                    }
                    if (col < columns)
                    {
                        for (int r = 0; r < 4; r++)
                            for (int c = col; c < columns; c++)
                            {
                                float sum = 0;
                                for (int k = 0; k < inner; k++)
                                    sum += inputPtr[inputBase + r * inner + k] * weightsPtr[k * columns + c];
                                outputPtr[outputBase + r * columns + c] = sum;
                            }
                    }
                }
                if (row < rows)
                    MatMulRows1Vector(
                        input.Slice(b * rows * inner, rows * inner), weights,
                        output.Slice(b * rows * columns, rows * columns),
                        1, row, rows, inner, columns);
            }
        }
    }

    // 4x16 unpadded tile reading the weight matrix directly. Same lane-FMA
    // trick; weight loads stride by `columns` per k but stay contiguous
    // across the 16 output columns.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows4AdvSimd(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, Span<float> output, int batch, int rows,
        int inner, int columns)
    {
        fixed (float* inputPtr = input, weightsPtr = weights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                int row = 0;
                for (; row <= rows - 4; row += 4)
                {
                    int inputBase = (b * rows + row) * inner;
                    int col = 0;
                    for (; col <= columns - 16; col += 16)
                    {
                        Vector128<float> a00 = Vector128<float>.Zero, a01 = a00, a02 = a00, a03 = a00;
                        Vector128<float> a10 = Vector128<float>.Zero, a11 = a10, a12 = a10, a13 = a10;
                        Vector128<float> a20 = Vector128<float>.Zero, a21 = a20, a22 = a20, a23 = a20;
                        Vector128<float> a30 = Vector128<float>.Zero, a31 = a30, a32 = a30, a33 = a30;
                        float* weightCursor = weightsPtr + col;
                        int k = 0;
                        for (; k <= inner - 4; k += 4)
                        {
                            Vector128<float> x0 = AdvSimd.LoadVector128(inputPtr + inputBase + k);
                            Vector128<float> x1 = AdvSimd.LoadVector128(inputPtr + inputBase + inner + k);
                            Vector128<float> x2 = AdvSimd.LoadVector128(inputPtr + inputBase + inner * 2 + k);
                            Vector128<float> x3 = AdvSimd.LoadVector128(inputPtr + inputBase + inner * 3 + k);
                            Vector128<float> w0 = AdvSimd.LoadVector128(weightCursor);
                            Vector128<float> w1 = AdvSimd.LoadVector128(weightCursor + 4);
                            Vector128<float> w2 = AdvSimd.LoadVector128(weightCursor + 8);
                            Vector128<float> w3 = AdvSimd.LoadVector128(weightCursor + 12);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, x0, 0);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, x0, 0);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, x0, 0);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, x0, 0);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, x1, 0);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, x1, 0);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, x1, 0);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, x1, 0);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, x2, 0);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, x2, 0);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, x2, 0);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, x2, 0);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, x3, 0);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, x3, 0);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, x3, 0);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, x3, 0);
                            weightCursor += columns;
                            w0 = AdvSimd.LoadVector128(weightCursor);
                            w1 = AdvSimd.LoadVector128(weightCursor + 4);
                            w2 = AdvSimd.LoadVector128(weightCursor + 8);
                            w3 = AdvSimd.LoadVector128(weightCursor + 12);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, x0, 1);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, x0, 1);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, x0, 1);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, x0, 1);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, x1, 1);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, x1, 1);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, x1, 1);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, x1, 1);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, x2, 1);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, x2, 1);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, x2, 1);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, x2, 1);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, x3, 1);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, x3, 1);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, x3, 1);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, x3, 1);
                            weightCursor += columns;
                            w0 = AdvSimd.LoadVector128(weightCursor);
                            w1 = AdvSimd.LoadVector128(weightCursor + 4);
                            w2 = AdvSimd.LoadVector128(weightCursor + 8);
                            w3 = AdvSimd.LoadVector128(weightCursor + 12);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, x0, 2);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, x0, 2);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, x0, 2);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, x0, 2);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, x1, 2);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, x1, 2);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, x1, 2);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, x1, 2);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, x2, 2);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, x2, 2);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, x2, 2);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, x2, 2);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, x3, 2);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, x3, 2);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, x3, 2);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, x3, 2);
                            weightCursor += columns;
                            w0 = AdvSimd.LoadVector128(weightCursor);
                            w1 = AdvSimd.LoadVector128(weightCursor + 4);
                            w2 = AdvSimd.LoadVector128(weightCursor + 8);
                            w3 = AdvSimd.LoadVector128(weightCursor + 12);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, w0, x0, 3);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, w1, x0, 3);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, w2, x0, 3);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, w3, x0, 3);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, w0, x1, 3);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, w1, x1, 3);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, w2, x1, 3);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, w3, x1, 3);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, w0, x2, 3);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, w1, x2, 3);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, w2, x2, 3);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, w3, x2, 3);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, w0, x3, 3);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, w1, x3, 3);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, w2, x3, 3);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, w3, x3, 3);
                            weightCursor += columns;
                        }
                        for (; k < inner; k++)
                        {
                            Vector128<float> w0 = AdvSimd.LoadVector128(weightCursor);
                            Vector128<float> w1 = AdvSimd.LoadVector128(weightCursor + 4);
                            Vector128<float> w2 = AdvSimd.LoadVector128(weightCursor + 8);
                            Vector128<float> w3 = AdvSimd.LoadVector128(weightCursor + 12);
                            Vector64<float> x0 = Vector64.CreateScalarUnsafe(inputPtr[inputBase + k]);
                            Vector64<float> x1 = Vector64.CreateScalarUnsafe(inputPtr[inputBase + inner + k]);
                            Vector64<float> x2 = Vector64.CreateScalarUnsafe(inputPtr[inputBase + inner * 2 + k]);
                            Vector64<float> x3 = Vector64.CreateScalarUnsafe(inputPtr[inputBase + inner * 3 + k]);
                            a00 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a00, w0, x0);
                            a01 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a01, w1, x0);
                            a02 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a02, w2, x0);
                            a03 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a03, w3, x0);
                            a10 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a10, w0, x1);
                            a11 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a11, w1, x1);
                            a12 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a12, w2, x1);
                            a13 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a13, w3, x1);
                            a20 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a20, w0, x2);
                            a21 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a21, w1, x2);
                            a22 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a22, w2, x2);
                            a23 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a23, w3, x2);
                            a30 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a30, w0, x3);
                            a31 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a31, w1, x3);
                            a32 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a32, w2, x3);
                            a33 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a33, w3, x3);
                            weightCursor += columns;
                        }
                        int ob = (b * rows + row) * columns + col;
                        AdvSimd.Store(outputPtr + ob, a00); AdvSimd.Store(outputPtr + ob + 4, a01);
                        AdvSimd.Store(outputPtr + ob + 8, a02); AdvSimd.Store(outputPtr + ob + 12, a03);
                        AdvSimd.Store(outputPtr + ob + columns, a10); AdvSimd.Store(outputPtr + ob + columns + 4, a11);
                        AdvSimd.Store(outputPtr + ob + columns + 8, a12); AdvSimd.Store(outputPtr + ob + columns + 12, a13);
                        AdvSimd.Store(outputPtr + ob + columns * 2, a20); AdvSimd.Store(outputPtr + ob + columns * 2 + 4, a21);
                        AdvSimd.Store(outputPtr + ob + columns * 2 + 8, a22); AdvSimd.Store(outputPtr + ob + columns * 2 + 12, a23);
                        AdvSimd.Store(outputPtr + ob + columns * 3, a30); AdvSimd.Store(outputPtr + ob + columns * 3 + 4, a31);
                        AdvSimd.Store(outputPtr + ob + columns * 3 + 8, a32); AdvSimd.Store(outputPtr + ob + columns * 3 + 12, a33);
                    }
                    for (; col < columns; col += 4)
                    {
                        int width = Math.Min(4, columns - col);
                        Vector128<float> a0 = Vector128<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
                        float* weightCursor = weightsPtr + col;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector128<float> w;
                            if (width == 4)
                                w = AdvSimd.LoadVector128(weightCursor);
                            else
                            {
                                w = Vector128<float>.Zero;
                                for (int lane = 0; lane < width; lane++)
                                    w = w.WithElement(lane, weightCursor[lane]);
                            }
                            int ib = inputBase + k;
                            a0 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a0, w, Vector64.CreateScalarUnsafe(inputPtr[ib]));
                            a1 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a1, w, Vector64.CreateScalarUnsafe(inputPtr[ib + inner]));
                            a2 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a2, w, Vector64.CreateScalarUnsafe(inputPtr[ib + inner * 2]));
                            a3 = AdvSimd.Arm64.FusedMultiplyAddByScalar(a3, w, Vector64.CreateScalarUnsafe(inputPtr[ib + inner * 3]));
                            weightCursor += columns;
                        }
                        int ob = (b * rows + row) * columns + col;
                        for (int lane = 0; lane < width; lane++)
                        {
                            outputPtr[ob + lane] = a0.GetElement(lane);
                            outputPtr[ob + columns + lane] = a1.GetElement(lane);
                            outputPtr[ob + columns * 2 + lane] = a2.GetElement(lane);
                            outputPtr[ob + columns * 3 + lane] = a3.GetElement(lane);
                        }
                    }
                }
                if (row < rows)
                    MatMulRows1Vector(
                        input.Slice(b * rows * inner, rows * inner), weights,
                        output.Slice(b * rows * columns, rows * columns),
                        1, row, rows, inner, columns);
            }
        }
    }
#endif
}
