using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class MatMul
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void MatMulRows1(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
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
                    Vector256<float> broadcast = Vector256.Create(value);
                    int col = 0, weightBase = k * columns;
                    for (; col <= columns - 8; col += 8)
                    {
                        Vector256<float> current = Load(output, outputBase + col);
                        Vector256<float> weight = Load(weights, weightBase + col);
                        Store(output, outputBase + col, AddMul(current, broadcast, weight));
                    }
                    for (; col < columns; col++) output[outputBase + col] += value * weights[weightBase + col];
                }
            }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows4(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
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
                    for (; col <= columns - 32; col += 32)
                    {
                        Vector256<float> a0a = Vector256<float>.Zero, a0b = a0a, a0c = a0a, a0d = a0a;
                        Vector256<float> a1a = Vector256<float>.Zero, a1b = a1a, a1c = a1a, a1d = a1a;
                        Vector256<float> a2a = Vector256<float>.Zero, a2b = a2a, a2c = a2a, a2d = a2a;
                        Vector256<float> a3a = Vector256<float>.Zero, a3b = a3a, a3c = a3a, a3d = a3a;
                        float* weightCursor = weightsPtr + col;
                        int inputBase = (b * rows + row) * inner;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector256<float> w0 = Avx.LoadVector256(weightCursor);
                            Vector256<float> w1 = Avx.LoadVector256(weightCursor + 8);
                            Vector256<float> w2 = Avx.LoadVector256(weightCursor + 16);
                            Vector256<float> w3 = Avx.LoadVector256(weightCursor + 24);
                            Vector256<float> v0 = Vector256.Create(inputPtr[inputBase + k]);
                            Vector256<float> v1 = Vector256.Create(inputPtr[inputBase + inner + k]);
                            Vector256<float> v2 = Vector256.Create(inputPtr[inputBase + inner * 2 + k]);
                            Vector256<float> v3 = Vector256.Create(inputPtr[inputBase + inner * 3 + k]);
                            a0a = AddMul(a0a, v0, w0); a0b = AddMul(a0b, v0, w1);
                            a0c = AddMul(a0c, v0, w2); a0d = AddMul(a0d, v0, w3);
                            a1a = AddMul(a1a, v1, w0); a1b = AddMul(a1b, v1, w1);
                            a1c = AddMul(a1c, v1, w2); a1d = AddMul(a1d, v1, w3);
                            a2a = AddMul(a2a, v2, w0); a2b = AddMul(a2b, v2, w1);
                            a2c = AddMul(a2c, v2, w2); a2d = AddMul(a2d, v2, w3);
                            a3a = AddMul(a3a, v3, w0); a3b = AddMul(a3b, v3, w1);
                            a3c = AddMul(a3c, v3, w2); a3d = AddMul(a3d, v3, w3);
                            weightCursor += columns;
                        }
                        int ob = (b * rows + row) * columns + col;
                        Store(output, ob, a0a); Store(output, ob + 8, a0b);
                        Store(output, ob + 16, a0c); Store(output, ob + 24, a0d);
                        Store(output, ob + columns, a1a); Store(output, ob + columns + 8, a1b);
                        Store(output, ob + columns + 16, a1c); Store(output, ob + columns + 24, a1d);
                        Store(output, ob + columns * 2, a2a); Store(output, ob + columns * 2 + 8, a2b);
                        Store(output, ob + columns * 2 + 16, a2c); Store(output, ob + columns * 2 + 24, a2d);
                        Store(output, ob + columns * 3, a3a); Store(output, ob + columns * 3 + 8, a3b);
                        Store(output, ob + columns * 3 + 16, a3c); Store(output, ob + columns * 3 + 24, a3d);
                    }
                    for (; col <= columns - 16; col += 16)
                    {
                        Vector256<float> a0l = Vector256<float>.Zero, a0h = a0l;
                        Vector256<float> a1l = Vector256<float>.Zero, a1h = a1l;
                        Vector256<float> a2l = Vector256<float>.Zero, a2h = a2l;
                        Vector256<float> a3l = Vector256<float>.Zero, a3h = a3l;
                        float* weightCursor = weightsPtr + col;
                        int inputBase = (b * rows + row) * inner;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector256<float> wLow = Avx.LoadVector256(weightCursor);
                            Vector256<float> wHigh = Avx.LoadVector256(weightCursor + 8);
                            Vector256<float> v0 = Vector256.Create(inputPtr[inputBase + k]);
                            Vector256<float> v1 = Vector256.Create(inputPtr[inputBase + inner + k]);
                            Vector256<float> v2 = Vector256.Create(inputPtr[inputBase + inner * 2 + k]);
                            Vector256<float> v3 = Vector256.Create(inputPtr[inputBase + inner * 3 + k]);
                            a0l = AddMul(a0l, v0, wLow); a0h = AddMul(a0h, v0, wHigh);
                            a1l = AddMul(a1l, v1, wLow); a1h = AddMul(a1h, v1, wHigh);
                            a2l = AddMul(a2l, v2, wLow); a2h = AddMul(a2h, v2, wHigh);
                            a3l = AddMul(a3l, v3, wLow); a3h = AddMul(a3h, v3, wHigh);
                            weightCursor += columns;
                        }
                        int ob = (b * rows + row) * columns + col;
                        Store(output, ob, a0l); Store(output, ob + 8, a0h);
                        Store(output, ob + columns, a1l); Store(output, ob + columns + 8, a1h);
                        Store(output, ob + columns * 2, a2l); Store(output, ob + columns * 2 + 8, a2h);
                        Store(output, ob + columns * 3, a3l); Store(output, ob + columns * 3 + 8, a3h);
                    }
                    for (; col < columns; col += 8)
                    {
                        int width = Math.Min(8, columns - col);
                        Vector256<float> a0 = Vector256<float>.Zero, a1 = Vector256<float>.Zero;
                        Vector256<float> a2 = Vector256<float>.Zero, a3 = Vector256<float>.Zero;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector256<float> w = width == 8 ? Load(weights, k * columns + col) : Vector256<float>.Zero;
                            if (width != 8)
                            {
                                for (int lane = 0; lane < width; lane++)
                                    w = w.WithElement(lane, weights[k * columns + col + lane]);
                            }
                            int ib = (b * rows + row) * inner + k;
                            Vector256<float> v0 = Vector256.Create(input[ib]);
                            Vector256<float> v1 = Vector256.Create(input[ib + inner]);
                            Vector256<float> v2 = Vector256.Create(input[ib + inner * 2]);
                            Vector256<float> v3 = Vector256.Create(input[ib + inner * 3]);
                            a0 = AddMul(a0, v0, w);
                            a1 = AddMul(a1, v1, w);
                            a2 = AddMul(a2, v2, w);
                            a3 = AddMul(a3, v3, w);
                        }
                        int ob = (b * rows + row) * columns + col;
                        if (width == 8)
                        {
                            Store(output, ob, a0); Store(output, ob + columns, a1);
                            Store(output, ob + columns * 2, a2); Store(output, ob + columns * 3, a3);
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
                    MatMulRows1(input.Slice(b * rows * inner), weights,
                        output.Slice(b * rows * columns), 1, row, rows, inner, columns);
            }
        }
    }

    // Matrix B is packed in 16-column tiles. The ordinary layout is already
    // contiguous across columns, but a fixed output tile otherwise jumps by
    // `columns` floats for every input k. Packing keeps the tile's complete
    // Kx16 working set contiguous and improves cache locality for REC's large
    // [40,80] x [80,6906] projection.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows4Packed(ReadOnlySpan<float> input,
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
                        Vector256<float> a0l = Vector256<float>.Zero, a0h = a0l;
                        Vector256<float> a1l = Vector256<float>.Zero, a1h = a1l;
                        Vector256<float> a2l = Vector256<float>.Zero, a2h = a2l;
                        Vector256<float> a3l = Vector256<float>.Zero, a3h = a3l;
                        float* tile = packedPtr + (col / 16) * inner * 16;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector256<float> wLow = Avx.LoadVector256(tile + k * 16);
                            Vector256<float> wHigh = Avx.LoadVector256(tile + k * 16 + 8);
                            Vector256<float> v0 = Vector256.Create(inputPtr[inputBase + k]);
                            Vector256<float> v1 = Vector256.Create(inputPtr[inputBase + inner + k]);
                            Vector256<float> v2 = Vector256.Create(inputPtr[inputBase + inner * 2 + k]);
                            Vector256<float> v3 = Vector256.Create(inputPtr[inputBase + inner * 3 + k]);
                            a0l = AddMul(a0l, v0, wLow); a0h = AddMul(a0h, v0, wHigh);
                            a1l = AddMul(a1l, v1, wLow); a1h = AddMul(a1h, v1, wHigh);
                            a2l = AddMul(a2l, v2, wLow); a2h = AddMul(a2h, v2, wHigh);
                            a3l = AddMul(a3l, v3, wLow); a3h = AddMul(a3h, v3, wHigh);
                        }
                        int ob = outputBase + col;
                        Store(output, ob, a0l); Store(output, ob + 8, a0h);
                        Store(output, ob + columns, a1l); Store(output, ob + columns + 8, a1h);
                        Store(output, ob + columns * 2, a2l); Store(output, ob + columns * 2 + 8, a2h);
                        Store(output, ob + columns * 3, a3l); Store(output, ob + columns * 3 + 8, a3h);
                    }
                    if (col < columns)
                    {
                        // The final partial tile is small for the target
                        // graphs; retain the original layout for exact scalar
                        // tail computation.
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
