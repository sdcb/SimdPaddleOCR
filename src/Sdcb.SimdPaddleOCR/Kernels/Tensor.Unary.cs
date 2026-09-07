using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.Kernels;

/// <summary>
/// One activation, named so the parallel chunk splitter can be shared without
/// a runtime operation selector. Implementations are empty structs, so each
/// instantiation binds directly to its kernel.
/// </summary>
internal interface IUnaryOp
{
    void Apply(ReadOnlySpan<float> input, Span<float> output);
}

internal readonly struct ReluOp : IUnaryOp
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Apply(ReadOnlySpan<float> input, Span<float> output) => SimdKernels.Relu(input, output);
}

internal readonly struct SigmoidOp : IUnaryOp
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Apply(ReadOnlySpan<float> input, Span<float> output) => SimdKernels.Sigmoid(input, output);
}

internal readonly struct ErfOp : IUnaryOp
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Apply(ReadOnlySpan<float> input, Span<float> output) => SimdKernels.Erf(input, output);
}

internal readonly struct SqrtOp : IUnaryOp
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Apply(ReadOnlySpan<float> input, Span<float> output) => SimdKernels.Sqrt(input, output);
}

internal readonly struct GeluOp : IUnaryOp
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Apply(ReadOnlySpan<float> input, Span<float> output) => SimdKernels.Gelu(input, output);
}

internal static partial class SimdKernels
{
    public static void ReluParallel(ReadOnlySpan<float> input, Span<float> output, int threads) => UnaryParallel<ReluOp>(input, output, threads);
    public static void SigmoidParallel(ReadOnlySpan<float> input, Span<float> output, int threads) => UnaryParallel<SigmoidOp>(input, output, threads);
    public static void ErfParallel(ReadOnlySpan<float> input, Span<float> output, int threads) => UnaryParallel<ErfOp>(input, output, threads);
    public static void SqrtParallel(ReadOnlySpan<float> input, Span<float> output, int threads) => UnaryParallel<SqrtOp>(input, output, threads);
    public static void GeluParallel(ReadOnlySpan<float> input, Span<float> output, int threads) => UnaryParallel<GeluOp>(input, output, threads);

    private static unsafe void UnaryParallel<TOp>(ReadOnlySpan<float> input, Span<float> output, int threads)
        where TOp : struct, IUnaryOp
    {
        int n = input.Length;
        if (threads <= 1 || n < ParallelElementwiseThreshold) { default(TOp).Apply(input, output); return; }
        fixed (float* inputPtr = input, outputPtr = output)
            ForEachChunk(n, threads, (nint)inputPtr, 0, (nint)outputPtr, (ia, _, oa, begin, count) =>
                default(TOp).Apply(new ReadOnlySpan<float>((float*)ia + begin, count),
                    new Span<float>((float*)oa + begin, count)));
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void Relu(ReadOnlySpan<float> input, Span<float> output)
    {
        int i = 0, n = output.Length;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            Vector512<float> z = Vector512<float>.Zero;
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 16; i += 16)
                    Avx512F.Store(outputPtr + i, Avx512F.Max(Avx512F.LoadVector512(inputPtr + i), z));
            }
        }
        else if (Avx.IsSupported)
        {
            Vector256<float> z = Vector256<float>.Zero;
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 8; i += 8)
                    Avx.Store(outputPtr + i, Avx.Max(Avx.LoadVector256(inputPtr + i), z));
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            Vector<float> z = Vector<float>.Zero;
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - width; i += width)
                    VecStore(outputPtr + i, Vector.Max(VecLoad(inputPtr + i), z));
            }
        }
        for (; i < n; i++) output[i] = input[i] > 0 ? input[i] : 0;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void Sigmoid(ReadOnlySpan<float> input, Span<float> output)
    {
        int i = 0, n = output.Length;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                Vector512<float> one = V512One;
                Vector512<float> absMask = V512AbsMask;
                Vector512<float> signMask = V512SignMask;
                for (; i <= n - 16; i += 16)
                {
                    Vector512<float> value = Avx512F.LoadVector512(inputPtr + i);
                    Vector512<float> negative = And512(value, signMask);
                    Vector512<float> magnitude = And512(value, absMask);
                    Vector512<float> exponent = ExpApproxVector512(Avx512F.Subtract(Vector512<float>.Zero, magnitude));
                    Vector512<float> positiveResult = Avx512F.Divide(one, Avx512F.Add(one, exponent));
                    Vector512<float> result = Avx512F.Subtract(one, positiveResult);
                    result = Avx512F.BlendVariable(positiveResult, result, negative);
                    Avx512F.Store(outputPtr + i, result);
                }
            }
        }
        else if (Avx2.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                Vector256<float> one = Vector256.Create(1f);
                Vector256<float> absMask = VAbsMask;
                Vector256<float> signMask = VSignMask;
                for (; i <= n - 8; i += 8)
                {
                    Vector256<float> value = Avx.LoadVector256(inputPtr + i);
                    Vector256<float> negative = Avx.And(value, signMask);
                    Vector256<float> magnitude = Avx.And(value, absMask);
                    Vector256<float> exponent = ExpApproxVector(Avx.Subtract(Vector256<float>.Zero, magnitude));
                    Vector256<float> positiveResult = Avx.Divide(one, Avx.Add(one, exponent));
                    Vector256<float> result = Avx.Subtract(one, positiveResult);
                    result = Avx.BlendVariable(positiveResult, result, negative);
                    Avx.Store(outputPtr + i, result);
                }
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - width; i += width)
                    VecStore(outputPtr + i, SigmoidExactVector(VecLoad(inputPtr + i)));
            }
        }
        for (; i < n; i++) output[i] = Sigmoid(input[i]);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void Erf(ReadOnlySpan<float> input, Span<float> output)
    {
        int i = 0, n = output.Length;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 16; i += 16)
                    Avx512F.Store(outputPtr + i, ErfVector512(Avx512F.LoadVector512(inputPtr + i)));
            }
        }
        else if (Avx.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 8; i += 8)
                    Avx.Store(outputPtr + i, ErfVector(Avx.LoadVector256(inputPtr + i)));
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - width; i += width)
                    VecStore(outputPtr + i, ErfExactVector(VecLoad(inputPtr + i)));
            }
        }
        else
        {
            for (; i <= n - 4; i += 4)
            {
                output[i] = Erf(input[i]);
                output[i + 1] = Erf(input[i + 1]);
                output[i + 2] = Erf(input[i + 2]);
                output[i + 3] = Erf(input[i + 3]);
            }
        }
        for (; i < n; i++) output[i] = Erf(input[i]);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void Sqrt(ReadOnlySpan<float> input, Span<float> output)
    {
        int i = 0, n = output.Length;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 16; i += 16)
                    Avx512F.Store(outputPtr + i, Avx512F.Sqrt(Avx512F.LoadVector512(inputPtr + i)));
            }
        }
        else if (Avx.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 8; i += 8)
                    Avx.Store(outputPtr + i, Avx.Sqrt(Avx.LoadVector256(inputPtr + i)));
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - width; i += width)
                    VecStore(outputPtr + i, Vector.SquareRoot(VecLoad(inputPtr + i)));
            }
        }
        for (; i < n; i++) output[i] = MathF.Sqrt(input[i]);
    }
}
