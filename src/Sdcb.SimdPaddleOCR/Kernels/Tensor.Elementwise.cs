using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

/// <summary>
/// The elementwise arithmetic a binary kernel applies, expressed once per
/// instruction set. Implementations are empty structs, so the JIT specializes
/// each kernel per operation and folds the calls away entirely: no operation
/// selector survives into the inner loop.
/// </summary>
internal interface IBinaryOp
{
#if !NETSTANDARD2_0
    Vector512<float> Apply(Vector512<float> left, Vector512<float> right);
    Vector256<float> Apply(Vector256<float> left, Vector256<float> right);
#endif
    Vector<float> Apply(Vector<float> left, Vector<float> right);
    float Apply(float left, float right);
}

internal readonly struct AddOp : IBinaryOp
{
#if !NETSTANDARD2_0
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector512<float> Apply(Vector512<float> left, Vector512<float> right) => Avx512F.Add(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector256<float> Apply(Vector256<float> left, Vector256<float> right) => Avx.Add(left, right);
#endif
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<float> Apply(Vector<float> left, Vector<float> right) => left + right;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Apply(float left, float right) => left + right;
}

internal readonly struct SubOp : IBinaryOp
{
#if !NETSTANDARD2_0
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector512<float> Apply(Vector512<float> left, Vector512<float> right) => Avx512F.Subtract(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector256<float> Apply(Vector256<float> left, Vector256<float> right) => Avx.Subtract(left, right);
#endif
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<float> Apply(Vector<float> left, Vector<float> right) => left - right;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Apply(float left, float right) => left - right;
}

internal readonly struct MulOp : IBinaryOp
{
#if !NETSTANDARD2_0
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector512<float> Apply(Vector512<float> left, Vector512<float> right) => Avx512F.Multiply(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector256<float> Apply(Vector256<float> left, Vector256<float> right) => Avx.Multiply(left, right);
#endif
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<float> Apply(Vector<float> left, Vector<float> right) => left * right;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Apply(float left, float right) => left * right;
}

internal readonly struct DivOp : IBinaryOp
{
#if !NETSTANDARD2_0
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector512<float> Apply(Vector512<float> left, Vector512<float> right) => Avx512F.Divide(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector256<float> Apply(Vector256<float> left, Vector256<float> right) => Avx.Divide(left, right);
#endif
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<float> Apply(Vector<float> left, Vector<float> right) => left / right;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Apply(float left, float right) => left / right;
}

internal static partial class SimdKernels
{
    // Splits large elementwise work across the intra-op budget. Chunk
    // boundaries are 16-float aligned so every interior chunk keeps the
    // single-threaded kernels' vector/scalar grid: results are bit-identical
    // to the sequential call regardless of worker count.
    private const int ParallelElementwiseThreshold = 1 << 18;

    private static void ForEachChunk(int length, int threads, nint a, nint b, nint o,
        Action<nint, nint, nint, int, int> body)
    {
        int workers = Math.Min(threads, 16);
        Parallel.For(0, workers, worker =>
        {
            int begin = (int)((long)length * worker / workers) & ~15;
            int end = worker == workers - 1 ? length : (int)((long)length * (worker + 1) / workers) & ~15;
            if (end > begin) body(a, b, o, begin, end - begin);
        });
    }

    public static void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Elementwise<AddOp>(left, right, output);
    public static void Sub(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Elementwise<SubOp>(left, right, output);
    public static void Mul(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Elementwise<MulOp>(left, right, output);
    public static void Div(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) => Elementwise<DivOp>(left, right, output);

    public static void Add(ReadOnlySpan<float> left, float right, Span<float> output) => ElementwiseScalar<AddOp>(left, right, output);
    public static void Sub(ReadOnlySpan<float> left, float right, Span<float> output) => ElementwiseScalar<SubOp>(left, right, output);
    public static void Mul(ReadOnlySpan<float> left, float right, Span<float> output) => ElementwiseScalar<MulOp>(left, right, output);
    public static void Div(ReadOnlySpan<float> left, float right, Span<float> output) => ElementwiseScalar<DivOp>(left, right, output);

    /// <summary>
    /// Elementwise <typeparamref name="TOp"/> over two equal-length spans,
    /// splitting the work across up to <paramref name="threads"/> workers.
    /// </summary>
    public static unsafe void ElementwiseParallel<TOp>(ReadOnlySpan<float> left, ReadOnlySpan<float> right,
        Span<float> output, int threads) where TOp : struct, IBinaryOp
    {
        int n = output.Length;
        if (threads <= 1 || n < ParallelElementwiseThreshold) { Elementwise<TOp>(left, right, output); return; }
        fixed (float* leftPtr = left, rightPtr = right, outputPtr = output)
            ForEachChunk(n, threads, (nint)leftPtr, (nint)rightPtr, (nint)outputPtr, (la, ra, oa, begin, count) =>
                Elementwise<TOp>(new ReadOnlySpan<float>((float*)la + begin, count),
                    new ReadOnlySpan<float>((float*)ra + begin, count),
                    new Span<float>((float*)oa + begin, count)));
    }

    /// <summary>Elementwise <typeparamref name="TOp"/> over two equal-length spans.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void Elementwise<TOp>(ReadOnlySpan<float> left, ReadOnlySpan<float> right,
        Span<float> output) where TOp : struct, IBinaryOp
    {
        TOp op = default;
        int i = 0, n = output.Length;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            fixed (float* leftPtr = left, rightPtr = right, outputPtr = output)
            {
                for (; i <= n - 64; i += 64)
                {
                    Avx512F.Store(outputPtr + i, op.Apply(Avx512F.LoadVector512(leftPtr + i), Avx512F.LoadVector512(rightPtr + i)));
                    Avx512F.Store(outputPtr + i + 16, op.Apply(Avx512F.LoadVector512(leftPtr + i + 16), Avx512F.LoadVector512(rightPtr + i + 16)));
                    Avx512F.Store(outputPtr + i + 32, op.Apply(Avx512F.LoadVector512(leftPtr + i + 32), Avx512F.LoadVector512(rightPtr + i + 32)));
                    Avx512F.Store(outputPtr + i + 48, op.Apply(Avx512F.LoadVector512(leftPtr + i + 48), Avx512F.LoadVector512(rightPtr + i + 48)));
                }
                for (; i <= n - 16; i += 16) Avx512F.Store(outputPtr + i, op.Apply(Avx512F.LoadVector512(leftPtr + i), Avx512F.LoadVector512(rightPtr + i)));
            }
        }
        else if (Avx.IsSupported)
        {
            fixed (float* leftPtr = left, rightPtr = right, outputPtr = output)
            {
                for (; i <= n - 32; i += 32)
                {
                    Avx.Store(outputPtr + i, op.Apply(Avx.LoadVector256(leftPtr + i), Avx.LoadVector256(rightPtr + i)));
                    Avx.Store(outputPtr + i + 8, op.Apply(Avx.LoadVector256(leftPtr + i + 8), Avx.LoadVector256(rightPtr + i + 8)));
                    Avx.Store(outputPtr + i + 16, op.Apply(Avx.LoadVector256(leftPtr + i + 16), Avx.LoadVector256(rightPtr + i + 16)));
                    Avx.Store(outputPtr + i + 24, op.Apply(Avx.LoadVector256(leftPtr + i + 24), Avx.LoadVector256(rightPtr + i + 24)));
                }
                for (; i <= n - 8; i += 8) Avx.Store(outputPtr + i, op.Apply(Avx.LoadVector256(leftPtr + i), Avx.LoadVector256(rightPtr + i)));
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            int unroll = width * 4;
            fixed (float* leftPtr = left, rightPtr = right, outputPtr = output)
            {
                for (; i <= n - unroll; i += unroll)
                {
                    VecStore(outputPtr + i, op.Apply(VecLoad(leftPtr + i), VecLoad(rightPtr + i)));
                    VecStore(outputPtr + i + width, op.Apply(VecLoad(leftPtr + i + width), VecLoad(rightPtr + i + width)));
                    VecStore(outputPtr + i + width * 2, op.Apply(VecLoad(leftPtr + i + width * 2), VecLoad(rightPtr + i + width * 2)));
                    VecStore(outputPtr + i + width * 3, op.Apply(VecLoad(leftPtr + i + width * 3), VecLoad(rightPtr + i + width * 3)));
                }
                for (; i <= n - width; i += width)
                    VecStore(outputPtr + i, op.Apply(VecLoad(leftPtr + i), VecLoad(rightPtr + i)));
            }
        }
        for (; i < n; i++) output[i] = op.Apply(left[i], right[i]);
    }

    /// <summary>Elementwise <typeparamref name="TOp"/> against a broadcast right-hand scalar.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void ElementwiseScalar<TOp>(ReadOnlySpan<float> left, float right,
        Span<float> output) where TOp : struct, IBinaryOp
    {
        TOp op = default;
        int i = 0, n = output.Length;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            Vector512<float> scalar = Vector512.Create(right);
            fixed (float* leftPtr = left, outputPtr = output)
            {
                for (; i <= n - 64; i += 64)
                {
                    Avx512F.Store(outputPtr + i, op.Apply(Avx512F.LoadVector512(leftPtr + i), scalar));
                    Avx512F.Store(outputPtr + i + 16, op.Apply(Avx512F.LoadVector512(leftPtr + i + 16), scalar));
                    Avx512F.Store(outputPtr + i + 32, op.Apply(Avx512F.LoadVector512(leftPtr + i + 32), scalar));
                    Avx512F.Store(outputPtr + i + 48, op.Apply(Avx512F.LoadVector512(leftPtr + i + 48), scalar));
                }
                for (; i <= n - 16; i += 16) Avx512F.Store(outputPtr + i, op.Apply(Avx512F.LoadVector512(leftPtr + i), scalar));
            }
        }
        else if (Avx.IsSupported)
        {
            Vector256<float> scalar = Vector256.Create(right);
            fixed (float* leftPtr = left, outputPtr = output)
            {
                for (; i <= n - 32; i += 32)
                {
                    Avx.Store(outputPtr + i, op.Apply(Avx.LoadVector256(leftPtr + i), scalar));
                    Avx.Store(outputPtr + i + 8, op.Apply(Avx.LoadVector256(leftPtr + i + 8), scalar));
                    Avx.Store(outputPtr + i + 16, op.Apply(Avx.LoadVector256(leftPtr + i + 16), scalar));
                    Avx.Store(outputPtr + i + 24, op.Apply(Avx.LoadVector256(leftPtr + i + 24), scalar));
                }
                for (; i <= n - 8; i += 8) Avx.Store(outputPtr + i, op.Apply(Avx.LoadVector256(leftPtr + i), scalar));
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            int unroll = width * 4;
            Vector<float> scalar = new(right);
            fixed (float* leftPtr = left, outputPtr = output)
            {
                for (; i <= n - unroll; i += unroll)
                {
                    VecStore(outputPtr + i, op.Apply(VecLoad(leftPtr + i), scalar));
                    VecStore(outputPtr + i + width, op.Apply(VecLoad(leftPtr + i + width), scalar));
                    VecStore(outputPtr + i + width * 2, op.Apply(VecLoad(leftPtr + i + width * 2), scalar));
                    VecStore(outputPtr + i + width * 3, op.Apply(VecLoad(leftPtr + i + width * 3), scalar));
                }
                for (; i <= n - width; i += width)
                    VecStore(outputPtr + i, op.Apply(VecLoad(leftPtr + i), scalar));
            }
        }
        for (; i < n; i++) output[i] = op.Apply(left[i], right);
    }

    /// <summary>
    /// Elementwise <typeparamref name="TOp"/> against a per-channel scalar
    /// broadcast over each [batch, channel] plane.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void ElementwiseChannel<TOp>(ReadOnlySpan<float> left, ReadOnlySpan<float> channels,
        Span<float> output, int batch, int channelCount, int plane) where TOp : struct, IBinaryOp
    {
        TOp op = default;
        fixed (float* leftPtr = left, channelPtr = channels, outputPtr = output)
        {
            for (int bc = 0; bc < batch * channelCount; bc++)
            {
                float* src = leftPtr + bc * plane, dst = outputPtr + bc * plane;
                float channelValue = channels[bc % channelCount];
                int i = 0;
                #if !NETSTANDARD2_0
                if (Avx512F.IsSupported)
                {
                    Vector512<float> scalar = Vector512.Create(channelValue);
                    for (; i <= plane - 16; i += 16) Avx512F.Store(dst + i, op.Apply(Avx512F.LoadVector512(src + i), scalar));
                }
                else if (Avx.IsSupported)
                {
                    Vector256<float> scalar = Vector256.Create(channelValue);
                    for (; i <= plane - 8; i += 8) Avx.Store(dst + i, op.Apply(Avx.LoadVector256(src + i), scalar));
                }
                else
#endif
                if (Vector.IsHardwareAccelerated)
                {
                    int width = Vector<float>.Count;
                    Vector<float> scalar = new(channelValue);
                    for (; i <= plane - width; i += width) VecStore(dst + i, op.Apply(VecLoad(src + i), scalar));
                }
                for (; i < plane; i++) dst[i] = op.Apply(src[i], channelValue);
            }
        }
    }
}
