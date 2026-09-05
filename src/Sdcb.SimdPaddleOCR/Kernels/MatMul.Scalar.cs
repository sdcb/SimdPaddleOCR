using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class MatMul
{
    // Scalar twin of MatMulRows4Vector: 4 rows × 4 columns, lanes = 1.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulScalar(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns)
    {
        fixed (float* inputPtr = input, weightsPtr = weights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                int row = 0;
                for (; row <= rows - 4; row += 4)
                {
                    int col = 0;
                    for (; col <= columns - 4; col += 4)
                    {
                        float a00 = 0, a01 = 0, a02 = 0, a03 = 0;
                        float a10 = 0, a11 = 0, a12 = 0, a13 = 0;
                        float a20 = 0, a21 = 0, a22 = 0, a23 = 0;
                        float a30 = 0, a31 = 0, a32 = 0, a33 = 0;
                        int inputBase = (b * rows + row) * inner;
                        float* weightCursor = weightsPtr + col;
                        for (int k = 0; k < inner; k++)
                        {
                            float v0 = inputPtr[inputBase + k];
                            float v1 = inputPtr[inputBase + inner + k];
                            float v2 = inputPtr[inputBase + inner * 2 + k];
                            float v3 = inputPtr[inputBase + inner * 3 + k];
                            float w0 = weightCursor[0], w1 = weightCursor[1], w2 = weightCursor[2], w3 = weightCursor[3];
                            a00 += v0 * w0; a01 += v0 * w1; a02 += v0 * w2; a03 += v0 * w3;
                            a10 += v1 * w0; a11 += v1 * w1; a12 += v1 * w2; a13 += v1 * w3;
                            a20 += v2 * w0; a21 += v2 * w1; a22 += v2 * w2; a23 += v2 * w3;
                            a30 += v3 * w0; a31 += v3 * w1; a32 += v3 * w2; a33 += v3 * w3;
                            weightCursor += columns;
                        }
                        int ob = (b * rows + row) * columns + col;
                        outputPtr[ob] = a00; outputPtr[ob + 1] = a01; outputPtr[ob + 2] = a02; outputPtr[ob + 3] = a03;
                        outputPtr[ob + columns] = a10; outputPtr[ob + columns + 1] = a11;
                        outputPtr[ob + columns + 2] = a12; outputPtr[ob + columns + 3] = a13;
                        outputPtr[ob + columns * 2] = a20; outputPtr[ob + columns * 2 + 1] = a21;
                        outputPtr[ob + columns * 2 + 2] = a22; outputPtr[ob + columns * 2 + 3] = a23;
                        outputPtr[ob + columns * 3] = a30; outputPtr[ob + columns * 3 + 1] = a31;
                        outputPtr[ob + columns * 3 + 2] = a32; outputPtr[ob + columns * 3 + 3] = a33;
                    }
                    for (; col < columns; col++)
                    {
                        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                        int inputBase = (b * rows + row) * inner;
                        for (int k = 0; k < inner; k++)
                        {
                            float w = weightsPtr[k * columns + col];
                            s0 += inputPtr[inputBase + k] * w;
                            s1 += inputPtr[inputBase + inner + k] * w;
                            s2 += inputPtr[inputBase + inner * 2 + k] * w;
                            s3 += inputPtr[inputBase + inner * 3 + k] * w;
                        }
                        int ob = (b * rows + row) * columns + col;
                        outputPtr[ob] = s0; outputPtr[ob + columns] = s1;
                        outputPtr[ob + columns * 2] = s2; outputPtr[ob + columns * 3] = s3;
                    }
                }
                for (; row < rows; row++)
                {
                    int outputBase = (b * rows + row) * columns;
                    int inputBase = (b * rows + row) * inner;
                    for (int col = 0; col < columns; col++)
                    {
                        float sum = 0;
                        for (int k = 0; k < inner; k++)
                            sum += inputPtr[inputBase + k] * weightsPtr[k * columns + col];
                        outputPtr[outputBase + col] = sum;
                    }
                }
            }
        }
    }
}
