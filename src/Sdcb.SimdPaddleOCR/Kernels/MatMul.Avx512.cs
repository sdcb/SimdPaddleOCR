using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class MatMul
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows16PackedAvx512(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights, Span<float> output,
        int batch, int rows, int inner, int columns)
    {
        fixed (float* inputPtr = input, weightsPtr = weights,
            packedPtr = packedWeights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                int row = 0;
                for (; row <= rows - 16; row += 16)
                {
                    int inputBase = (b * rows + row) * inner;
                    int outputBase = (b * rows + row) * columns;
                    int col = 0;
                    for (; col <= columns - 16; col += 16)
                    {
                        Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
                        Vector512<float> a4 = a0, a5 = a0, a6 = a0, a7 = a0;
                        Vector512<float> a8 = a0, a9 = a0, a10 = a0, a11 = a0;
                        Vector512<float> a12 = a0, a13 = a0, a14 = a0, a15 = a0;
                        float* tile = packedPtr + (col / 16) * inner * 16;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector512<float> w = Avx512F.LoadVector512(tile + k * 16);
                            a0 = AddMul512(a0, Vector512.Create(inputPtr[inputBase + k]), w);
                            a1 = AddMul512(a1, Vector512.Create(inputPtr[inputBase + inner + k]), w);
                            a2 = AddMul512(a2, Vector512.Create(inputPtr[inputBase + inner * 2 + k]), w);
                            a3 = AddMul512(a3, Vector512.Create(inputPtr[inputBase + inner * 3 + k]), w);
                            a4 = AddMul512(a4, Vector512.Create(inputPtr[inputBase + inner * 4 + k]), w);
                            a5 = AddMul512(a5, Vector512.Create(inputPtr[inputBase + inner * 5 + k]), w);
                            a6 = AddMul512(a6, Vector512.Create(inputPtr[inputBase + inner * 6 + k]), w);
                            a7 = AddMul512(a7, Vector512.Create(inputPtr[inputBase + inner * 7 + k]), w);
                            a8 = AddMul512(a8, Vector512.Create(inputPtr[inputBase + inner * 8 + k]), w);
                            a9 = AddMul512(a9, Vector512.Create(inputPtr[inputBase + inner * 9 + k]), w);
                            a10 = AddMul512(a10, Vector512.Create(inputPtr[inputBase + inner * 10 + k]), w);
                            a11 = AddMul512(a11, Vector512.Create(inputPtr[inputBase + inner * 11 + k]), w);
                            a12 = AddMul512(a12, Vector512.Create(inputPtr[inputBase + inner * 12 + k]), w);
                            a13 = AddMul512(a13, Vector512.Create(inputPtr[inputBase + inner * 13 + k]), w);
                            a14 = AddMul512(a14, Vector512.Create(inputPtr[inputBase + inner * 14 + k]), w);
                            a15 = AddMul512(a15, Vector512.Create(inputPtr[inputBase + inner * 15 + k]), w);
                        }
                        int ob = outputBase + col;
                        Store512(output, ob, a0);
                        Store512(output, ob + columns, a1);
                        Store512(output, ob + columns * 2, a2);
                        Store512(output, ob + columns * 3, a3);
                        Store512(output, ob + columns * 4, a4);
                        Store512(output, ob + columns * 5, a5);
                        Store512(output, ob + columns * 6, a6);
                        Store512(output, ob + columns * 7, a7);
                        Store512(output, ob + columns * 8, a8);
                        Store512(output, ob + columns * 9, a9);
                        Store512(output, ob + columns * 10, a10);
                        Store512(output, ob + columns * 11, a11);
                        Store512(output, ob + columns * 12, a12);
                        Store512(output, ob + columns * 13, a13);
                        Store512(output, ob + columns * 14, a14);
                        Store512(output, ob + columns * 15, a15);
                    }
                    if (col < columns)
                    {
                        for (int r = 0; r < 16; r++)
                            for (int c = col; c < columns; c++)
                            {
                                float sum = 0;
                                for (int k = 0; k < inner; k++)
                                    sum += inputPtr[inputBase + r * inner + k] *
                                        weightsPtr[k * columns + c];
                                outputPtr[outputBase + r * columns + c] = sum;
                            }
                    }
                }
                if (row < rows)
                {
                    int inputOffset = (b * rows + row) * inner;
                    int outputOffset = (b * rows + row) * columns;
                    MatMulRows8PackedAvx512(
                        input.Slice(inputOffset, 8 * inner), weights, packedWeights,
                        output.Slice(outputOffset, 8 * columns), 1, 8, inner, columns);
                }
            }
        }
    }

    // REC output projection: A is [T,K], packed B is [K,C] in 16-column
    // tiles, with C in the thousands.  Eight rows amortize every 64-byte
    // weight load over eight FMAs and expose eight independent accumulation
    // chains.  The previous four-row AVX-512 kernel only doubled vector width
    // versus AVX while preserving the same weight reuse, which regressed on
    // Zen 5.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows8PackedAvx512(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights, Span<float> output,
        int batch, int rows, int inner, int columns)
    {
        fixed (float* inputPtr = input, weightsPtr = weights,
            packedPtr = packedWeights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int row = 0; row < rows; row += 8)
                {
                    int inputBase = (b * rows + row) * inner;
                    int outputBase = (b * rows + row) * columns;
                    int col = 0;
                    for (; col <= columns - 16; col += 16)
                    {
                        Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
                        Vector512<float> a4 = a0, a5 = a0, a6 = a0, a7 = a0;
                        float* tile = packedPtr + (col / 16) * inner * 16;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector512<float> w = Avx512F.LoadVector512(tile + k * 16);
                            a0 = AddMul512(a0, Vector512.Create(inputPtr[inputBase + k]), w);
                            a1 = AddMul512(a1, Vector512.Create(inputPtr[inputBase + inner + k]), w);
                            a2 = AddMul512(a2, Vector512.Create(inputPtr[inputBase + inner * 2 + k]), w);
                            a3 = AddMul512(a3, Vector512.Create(inputPtr[inputBase + inner * 3 + k]), w);
                            a4 = AddMul512(a4, Vector512.Create(inputPtr[inputBase + inner * 4 + k]), w);
                            a5 = AddMul512(a5, Vector512.Create(inputPtr[inputBase + inner * 5 + k]), w);
                            a6 = AddMul512(a6, Vector512.Create(inputPtr[inputBase + inner * 6 + k]), w);
                            a7 = AddMul512(a7, Vector512.Create(inputPtr[inputBase + inner * 7 + k]), w);
                        }
                        int ob = outputBase + col;
                        Store512(output, ob, a0);
                        Store512(output, ob + columns, a1);
                        Store512(output, ob + columns * 2, a2);
                        Store512(output, ob + columns * 3, a3);
                        Store512(output, ob + columns * 4, a4);
                        Store512(output, ob + columns * 5, a5);
                        Store512(output, ob + columns * 6, a6);
                        Store512(output, ob + columns * 7, a7);
                    }
                    if (col < columns)
                    {
                        for (int r = 0; r < 8; r++)
                            for (int c = col; c < columns; c++)
                            {
                                float sum = 0;
                                for (int k = 0; k < inner; k++)
                                    sum += inputPtr[inputBase + r * inner + k] *
                                        weightsPtr[k * columns + c];
                                outputPtr[outputBase + r * columns + c] = sum;
                            }
                    }
                }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void MatMulRows1Avx512(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rowStart, int rows, int inner, int columns)
    {
        for (int b = 0; b < batch; b++)
            for (int row = rowStart; row < rows; row++)
            {
                int outputBase = (b * rows + row) * columns;
                output.Slice(outputBase, columns).Clear();
                int inputBase = (b * rows + row) * inner;
                for (int k = 0; k < inner; k++)
                {
                    float value = input[inputBase + k];
                    Vector512<float> broadcast = Vector512.Create(value);
                    int col = 0, weightBase = k * columns;
                    for (; col <= columns - 16; col += 16)
                    {
                        Vector512<float> current = Load512(output, outputBase + col);
                        Vector512<float> weight = Load512(weights, weightBase + col);
                        Store512(output, outputBase + col, AddMul512(current, broadcast, weight));
                    }
                    for (; col < columns; col++) output[outputBase + col] += value * weights[weightBase + col];
                }
            }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows4Avx512(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
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
                    // Cap at 2×Vector512 columns (4 rows × 2 cols = 8 ZMM acc).
                    for (; col <= columns - 32; col += 32)
                    {
                        Vector512<float> a0l = Vector512<float>.Zero, a0h = a0l;
                        Vector512<float> a1l = Vector512<float>.Zero, a1h = a1l;
                        Vector512<float> a2l = Vector512<float>.Zero, a2h = a2l;
                        Vector512<float> a3l = Vector512<float>.Zero, a3h = a3l;
                        float* weightCursor = weightsPtr + col;
                        int inputBase = (b * rows + row) * inner;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector512<float> wLow = Avx512F.LoadVector512(weightCursor);
                            Vector512<float> wHigh = Avx512F.LoadVector512(weightCursor + 16);
                            Vector512<float> v0 = Vector512.Create(inputPtr[inputBase + k]);
                            Vector512<float> v1 = Vector512.Create(inputPtr[inputBase + inner + k]);
                            Vector512<float> v2 = Vector512.Create(inputPtr[inputBase + inner * 2 + k]);
                            Vector512<float> v3 = Vector512.Create(inputPtr[inputBase + inner * 3 + k]);
                            a0l = AddMul512(a0l, v0, wLow); a0h = AddMul512(a0h, v0, wHigh);
                            a1l = AddMul512(a1l, v1, wLow); a1h = AddMul512(a1h, v1, wHigh);
                            a2l = AddMul512(a2l, v2, wLow); a2h = AddMul512(a2h, v2, wHigh);
                            a3l = AddMul512(a3l, v3, wLow); a3h = AddMul512(a3h, v3, wHigh);
                            weightCursor += columns;
                        }
                        int ob = (b * rows + row) * columns + col;
                        Store512(output, ob, a0l); Store512(output, ob + 16, a0h);
                        Store512(output, ob + columns, a1l); Store512(output, ob + columns + 16, a1h);
                        Store512(output, ob + columns * 2, a2l); Store512(output, ob + columns * 2 + 16, a2h);
                        Store512(output, ob + columns * 3, a3l); Store512(output, ob + columns * 3 + 16, a3h);
                    }
                    for (; col < columns; col += 16)
                    {
                        int width = Math.Min(16, columns - col);
                        Vector512<float> a0 = Vector512<float>.Zero, a1 = Vector512<float>.Zero;
                        Vector512<float> a2 = Vector512<float>.Zero, a3 = Vector512<float>.Zero;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector512<float> w = width == 16 ? Load512(weights, k * columns + col) : Vector512<float>.Zero;
                            if (width != 16)
                            {
                                for (int lane = 0; lane < width; lane++)
                                    w = w.WithElement(lane, weights[k * columns + col + lane]);
                            }
                            int ib = (b * rows + row) * inner + k;
                            Vector512<float> v0 = Vector512.Create(input[ib]);
                            Vector512<float> v1 = Vector512.Create(input[ib + inner]);
                            Vector512<float> v2 = Vector512.Create(input[ib + inner * 2]);
                            Vector512<float> v3 = Vector512.Create(input[ib + inner * 3]);
                            a0 = AddMul512(a0, v0, w);
                            a1 = AddMul512(a1, v1, w);
                            a2 = AddMul512(a2, v2, w);
                            a3 = AddMul512(a3, v3, w);
                        }
                        int ob = (b * rows + row) * columns + col;
                        if (width == 16)
                        {
                            Store512(output, ob, a0); Store512(output, ob + columns, a1);
                            Store512(output, ob + columns * 2, a2); Store512(output, ob + columns * 3, a3);
                        }
                        else
                        {
                            for (int lane = 0; lane < width; lane++)
                            {
                                output[ob + lane] = a0.GetElement(lane);
                                output[ob + columns + lane] = a1.GetElement(lane);
                                output[ob + columns * 2 + lane] = a2.GetElement(lane);
                                output[ob + columns * 3 + lane] = a3.GetElement(lane);
                            }
                        }
                    }
                }
                if (row < rows)
                    MatMulRows1Avx512(input.Slice(b * rows * inner), weights,
                        output.Slice(b * rows * columns), 1, row, rows, inner, columns);
            }
        }
    }

    // Packed B tiles stay 16 columns; AVX-512 loads each tile as one Vector512.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows4PackedAvx512(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights, Span<float> output,
        int batch, int rows, int inner, int columns)
    {
        fixed (float* inputPtr = input, weightsPtr = weights,
            packedPtr = packedWeights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int row = 0; row < rows; row += 4)
                {
                    int inputBase = (b * rows + row) * inner;
                    int outputBase = (b * rows + row) * columns;
                    int col = 0;
                    for (; col <= columns - 16; col += 16)
                    {
                        Vector512<float> a0 = Vector512<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
                        float* tile = packedPtr + (col / 16) * inner * 16;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector512<float> w = Avx512F.LoadVector512(tile + k * 16);
                            Vector512<float> v0 = Vector512.Create(inputPtr[inputBase + k]);
                            Vector512<float> v1 = Vector512.Create(inputPtr[inputBase + inner + k]);
                            Vector512<float> v2 = Vector512.Create(inputPtr[inputBase + inner * 2 + k]);
                            Vector512<float> v3 = Vector512.Create(inputPtr[inputBase + inner * 3 + k]);
                            a0 = AddMul512(a0, v0, w);
                            a1 = AddMul512(a1, v1, w);
                            a2 = AddMul512(a2, v2, w);
                            a3 = AddMul512(a3, v3, w);
                        }
                        int ob = outputBase + col;
                        Store512(output, ob, a0);
                        Store512(output, ob + columns, a1);
                        Store512(output, ob + columns * 2, a2);
                        Store512(output, ob + columns * 3, a3);
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
        }
    }
}
