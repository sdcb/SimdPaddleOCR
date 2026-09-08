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
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static bool TryVector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns, float[]? packedWeights = null)
    {
        if (packedWeights is not null && rows >= 4 && inner >= 64 && columns >= 1024)
        {
            MatMulRows4PackedVector(input, weights, packedWeights, output, batch, rows, inner, columns);
            return true;
        }
        if (rows >= 4)
        {
            MatMulRows4Vector(input, weights, output, batch, rows, inner, columns);
            return true;
        }
        MatMulRows1Vector(input, weights, output, batch, 0, rows, inner, columns);
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void MatMulRows1Vector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rowStart, int rows, int inner, int columns)
    {
        int widthLanes = Vector<float>.Count;
        for (int b = 0; b < batch; b++)
            for (int row = rowStart; row < rows; row++)
            {
                int outputBase = (b * rows + row) * columns;
                output.Slice(outputBase, columns).Clear();
                int inputBase = (b * rows + row) * inner;
                for (int k = 0; k < inner; k++)
                {
                    float value = input[inputBase + k];
                    Vector<float> broadcast = new(value);
                    int col = 0, weightBase = k * columns;
                    for (; col <= columns - widthLanes; col += widthLanes)
                    {
                        Vector<float> current = VectorLoad(output, outputBase + col);
                        Vector<float> weight = VectorLoad(weights, weightBase + col);
                        VectorStore(output, outputBase + col, VectorAddMul(current, weight, broadcast));
                    }
                    for (; col < columns; col++) output[outputBase + col] += value * weights[weightBase + col];
                }
            }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows4Vector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns)
    {
        int widthLanes = Vector<float>.Count;
        int tile = widthLanes * 2;
        fixed (float* inputPtr = input, weightsPtr = weights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                int row = 0;
                for (; row <= rows - 4; row += 4)
                {
                    int col = 0;
                    for (; col <= columns - tile; col += tile)
                    {
                        Vector<float> a0l = Vector<float>.Zero, a0h = a0l;
                        Vector<float> a1l = Vector<float>.Zero, a1h = a1l;
                        Vector<float> a2l = Vector<float>.Zero, a2h = a2l;
                        Vector<float> a3l = Vector<float>.Zero, a3h = a3l;
                        float* weightCursor = weightsPtr + col;
                        int inputBase = (b * rows + row) * inner;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector<float> wLow = VectorLoad(weightCursor);
                            Vector<float> wHigh = VectorLoad(weightCursor + widthLanes);
                            Vector<float> v0 = new(inputPtr[inputBase + k]);
                            Vector<float> v1 = new(inputPtr[inputBase + inner + k]);
                            Vector<float> v2 = new(inputPtr[inputBase + inner * 2 + k]);
                            Vector<float> v3 = new(inputPtr[inputBase + inner * 3 + k]);
                            a0l = VectorAddMul(a0l, wLow, v0); a0h = VectorAddMul(a0h, wHigh, v0);
                            a1l = VectorAddMul(a1l, wLow, v1); a1h = VectorAddMul(a1h, wHigh, v1);
                            a2l = VectorAddMul(a2l, wLow, v2); a2h = VectorAddMul(a2h, wHigh, v2);
                            a3l = VectorAddMul(a3l, wLow, v3); a3h = VectorAddMul(a3h, wHigh, v3);
                            weightCursor += columns;
                        }
                        int ob = (b * rows + row) * columns + col;
                        VectorStore(output, ob, a0l); VectorStore(output, ob + widthLanes, a0h);
                        VectorStore(output, ob + columns, a1l); VectorStore(output, ob + columns + widthLanes, a1h);
                        VectorStore(output, ob + columns * 2, a2l); VectorStore(output, ob + columns * 2 + widthLanes, a2h);
                        VectorStore(output, ob + columns * 3, a3l); VectorStore(output, ob + columns * 3 + widthLanes, a3h);
                    }
                    for (; col < columns; col += widthLanes)
                    {
                        int width = Math.Min(widthLanes, columns - col);
                        Vector<float> a0 = Vector<float>.Zero, a1 = Vector<float>.Zero;
                        Vector<float> a2 = Vector<float>.Zero, a3 = Vector<float>.Zero;
                        for (int k = 0; k < inner; k++)
                        {
                            Vector<float> w = default;
                            if (width == widthLanes)
                                w = VectorLoad(weights, k * columns + col);
                            else
                                for (int lane = 0; lane < width; lane++)
                                    w = w.WithElement(lane, weights[k * columns + col + lane]);
                            int ib = (b * rows + row) * inner + k;
                            Vector<float> v0 = new(input[ib]);
                            Vector<float> v1 = new(input[ib + inner]);
                            Vector<float> v2 = new(input[ib + inner * 2]);
                            Vector<float> v3 = new(input[ib + inner * 3]);
                            a0 = VectorAddMul(a0, w, v0);
                            a1 = VectorAddMul(a1, w, v1);
                            a2 = VectorAddMul(a2, w, v2);
                            a3 = VectorAddMul(a3, w, v3);
                        }
                        int ob = (b * rows + row) * columns + col;
                        if (width == widthLanes)
                        {
                            VectorStore(output, ob, a0);
                            VectorStore(output, ob + columns, a1);
                            VectorStore(output, ob + columns * 2, a2);
                            VectorStore(output, ob + columns * 3, a3);
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
                    MatMulRows1Vector(
                        input.Slice(b * rows * inner, rows * inner), weights,
                        output.Slice(b * rows * columns, rows * columns),
                        1, row, rows, inner, columns);
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulRows4PackedVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights, Span<float> output,
        int batch, int rows, int inner, int columns)
    {
        int widthLanes = Vector<float>.Count;
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
                        int ob = outputBase + col;
                        if (widthLanes == 8)
                        {
                            Vector<float> a0l = Vector<float>.Zero, a0h = a0l;
                            Vector<float> a1l = Vector<float>.Zero, a1h = a1l;
                            Vector<float> a2l = Vector<float>.Zero, a2h = a2l;
                            Vector<float> a3l = Vector<float>.Zero, a3h = a3l;
                            for (int k = 0; k < inner; k++)
                            {
                                Vector<float> wLow = VectorLoad(tile + k * 16);
                                Vector<float> wHigh = VectorLoad(tile + k * 16 + 8);
                                Vector<float> v0 = new(inputPtr[inputBase + k]);
                                Vector<float> v1 = new(inputPtr[inputBase + inner + k]);
                                Vector<float> v2 = new(inputPtr[inputBase + inner * 2 + k]);
                                Vector<float> v3 = new(inputPtr[inputBase + inner * 3 + k]);
                                a0l = VectorAddMul(a0l, wLow, v0); a0h = VectorAddMul(a0h, wHigh, v0);
                                a1l = VectorAddMul(a1l, wLow, v1); a1h = VectorAddMul(a1h, wHigh, v1);
                                a2l = VectorAddMul(a2l, wLow, v2); a2h = VectorAddMul(a2h, wHigh, v2);
                                a3l = VectorAddMul(a3l, wLow, v3); a3h = VectorAddMul(a3h, wHigh, v3);
                            }
                            VectorStore(output, ob, a0l); VectorStore(output, ob + 8, a0h);
                            VectorStore(output, ob + columns, a1l); VectorStore(output, ob + columns + 8, a1h);
                            VectorStore(output, ob + columns * 2, a2l); VectorStore(output, ob + columns * 2 + 8, a2h);
                            VectorStore(output, ob + columns * 3, a3l); VectorStore(output, ob + columns * 3 + 8, a3h);
                        }
                        else
                        {
                            Vector<float> a00 = Vector<float>.Zero, a01 = a00, a02 = a00, a03 = a00;
                            Vector<float> a10 = Vector<float>.Zero, a11 = a10, a12 = a10, a13 = a10;
                            Vector<float> a20 = Vector<float>.Zero, a21 = a20, a22 = a20, a23 = a20;
                            Vector<float> a30 = Vector<float>.Zero, a31 = a30, a32 = a30, a33 = a30;
                            for (int k = 0; k < inner; k++)
                            {
                                Vector<float> w0 = VectorLoad(tile + k * 16);
                                Vector<float> w1 = VectorLoad(tile + k * 16 + 4);
                                Vector<float> w2 = VectorLoad(tile + k * 16 + 8);
                                Vector<float> w3 = VectorLoad(tile + k * 16 + 12);
                                Vector<float> v0 = new(inputPtr[inputBase + k]);
                                Vector<float> v1 = new(inputPtr[inputBase + inner + k]);
                                Vector<float> v2 = new(inputPtr[inputBase + inner * 2 + k]);
                                Vector<float> v3 = new(inputPtr[inputBase + inner * 3 + k]);
                                a00 = VectorAddMul(a00, w0, v0); a01 = VectorAddMul(a01, w1, v0);
                                a02 = VectorAddMul(a02, w2, v0); a03 = VectorAddMul(a03, w3, v0);
                                a10 = VectorAddMul(a10, w0, v1); a11 = VectorAddMul(a11, w1, v1);
                                a12 = VectorAddMul(a12, w2, v1); a13 = VectorAddMul(a13, w3, v1);
                                a20 = VectorAddMul(a20, w0, v2); a21 = VectorAddMul(a21, w1, v2);
                                a22 = VectorAddMul(a22, w2, v2); a23 = VectorAddMul(a23, w3, v2);
                                a30 = VectorAddMul(a30, w0, v3); a31 = VectorAddMul(a31, w1, v3);
                                a32 = VectorAddMul(a32, w2, v3); a33 = VectorAddMul(a33, w3, v3);
                            }
                            VectorStore(output, ob, a00); VectorStore(output, ob + 4, a01);
                            VectorStore(output, ob + 8, a02); VectorStore(output, ob + 12, a03);
                            VectorStore(output, ob + columns, a10); VectorStore(output, ob + columns + 4, a11);
                            VectorStore(output, ob + columns + 8, a12); VectorStore(output, ob + columns + 12, a13);
                            VectorStore(output, ob + columns * 2, a20); VectorStore(output, ob + columns * 2 + 4, a21);
                            VectorStore(output, ob + columns * 2 + 8, a22); VectorStore(output, ob + columns * 2 + 12, a23);
                            VectorStore(output, ob + columns * 3, a30); VectorStore(output, ob + columns * 3 + 4, a31);
                            VectorStore(output, ob + columns * 3 + 8, a32); VectorStore(output, ob + columns * 3 + 12, a33);
                        }
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
}
