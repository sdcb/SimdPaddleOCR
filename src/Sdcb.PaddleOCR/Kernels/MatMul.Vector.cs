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
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static bool TryVector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns)
    {
        int widthLanes = Vector<float>.Count;
        if (rows >= 4 && (rows & 3) == 0 && inner >= 64 && columns >= 1024)
        {
            MatMulRows4Vector(input, weights, output, batch, rows, inner, columns);
            return true;
        }
        for (int b = 0; b < batch; b++)
            for (int row = 0; row < rows; row++)
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
        return true;
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
                for (int row = 0; row < rows; row += 4)
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
        }
    }
}
