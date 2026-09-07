using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class SimdKernels
{
    /// <summary>Hard-sigmoid with an AVX fast path.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void HardSigmoid(ReadOnlySpan<float> input, Span<float> output,
        float alpha, float beta)
    {
        int i = 0, n = output.Length;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            Vector512<float> va = Vector512.Create(alpha);
            Vector512<float> vb = Vector512.Create(beta);
            Vector512<float> zero = Vector512<float>.Zero;
            Vector512<float> one = Vector512.Create(1f);
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 16; i += 16)
                {
                    Vector512<float> value = Avx512F.Add(Avx512F.Multiply(Avx512F.LoadVector512(inputPtr + i), va), vb);
                    value = Avx512F.Max(zero, Avx512F.Min(one, value));
                    Avx512F.Store(outputPtr + i, value);
                }
            }
        }
        else if (Avx.IsSupported)
        {
            Vector256<float> va = Vector256.Create(alpha);
            Vector256<float> vb = Vector256.Create(beta);
            Vector256<float> zero = Vector256<float>.Zero;
            Vector256<float> one = Vector256.Create(1f);
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 8; i += 8)
                {
                    Vector256<float> value = Avx.Add(Avx.Multiply(Avx.LoadVector256(inputPtr + i), va), vb);
                    value = Avx.Max(zero, Avx.Min(one, value));
                    Avx.Store(outputPtr + i, value);
                }
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            Vector<float> va = new(alpha);
            Vector<float> vb = new(beta);
            Vector<float> zero = Vector<float>.Zero;
            Vector<float> one = new Vector<float>(1f);
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - width; i += width)
                {
                    Vector<float> value = VecLoad(inputPtr + i) * va + vb;
                    value = Vector.Max(zero, Vector.Min(one, value));
                    VecStore(outputPtr + i, value);
                }
            }
        }
        for (; i < n; i++)
        {
            float value = alpha * input[i] + beta;
            output[i] = MathCompat.Clamp(value, 0f, 1f);
        }
    }

    /// <summary>Fused hard-swish (x * clamp(alpha*x + beta, 0, 1)).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void HardSwish(ReadOnlySpan<float> input, Span<float> output,
        float alpha, float beta)
    {
        int i = 0, n = output.Length;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            Vector512<float> va = Vector512.Create(alpha);
            Vector512<float> vb = Vector512.Create(beta);
            Vector512<float> zero = Vector512<float>.Zero;
            Vector512<float> one = Vector512.Create(1f);
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 16; i += 16)
                {
                    Vector512<float> value = Avx512F.LoadVector512(inputPtr + i);
                    Vector512<float> gate = Avx512F.Add(Avx512F.Multiply(value, va), vb);
                    gate = Avx512F.Max(zero, Avx512F.Min(one, gate));
                    Avx512F.Store(outputPtr + i, Avx512F.Multiply(value, gate));
                }
            }
        }
        else if (Avx.IsSupported)
        {
            Vector256<float> va = Vector256.Create(alpha);
            Vector256<float> vb = Vector256.Create(beta);
            Vector256<float> zero = Vector256<float>.Zero;
            Vector256<float> one = Vector256.Create(1f);
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - 8; i += 8)
                {
                    Vector256<float> value = Avx.LoadVector256(inputPtr + i);
                    Vector256<float> gate = Avx.Add(Avx.Multiply(value, va), vb);
                    gate = Avx.Max(zero, Avx.Min(one, gate));
                    Avx.Store(outputPtr + i, Avx.Multiply(value, gate));
                }
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            Vector<float> va = new(alpha);
            Vector<float> vb = new(beta);
            Vector<float> zero = Vector<float>.Zero;
            Vector<float> one = new Vector<float>(1f);
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= n - width; i += width)
                {
                    Vector<float> value = VecLoad(inputPtr + i);
                    Vector<float> gate = Vector.Max(zero, Vector.Min(one, value * va + vb));
                    VecStore(outputPtr + i, value * gate);
                }
            }
        }
        for (; i < n; i++)
        {
            float value = input[i];
            float gate = MathCompat.Clamp(alpha * value + beta, 0f, 1f);
            output[i] = value * gate;
        }
    }

    /// <summary>2x2 max-pool with one-cell end padding and unit stride.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void MaxPool2x2PadEnd(ReadOnlySpan<float> input, Span<float> output,
        int batch, int channels, int height, int width)
    {
        int plane = checked(height * width);
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < channels; c++)
                    {
                        int offset = (b * channels + c) * plane;
                        for (int y = 0; y < height - 1; y++)
                        {
                            int row = offset + y * width, next = row + width, x = 0;
                            for (; x <= width - 17; x += 16)
                            {
                                Vector512<float> a = Avx512F.LoadVector512(inputPtr + row + x);
                                Vector512<float> b0 = Avx512F.LoadVector512(inputPtr + row + x + 1);
                                Vector512<float> c0 = Avx512F.LoadVector512(inputPtr + next + x);
                                Vector512<float> d = Avx512F.LoadVector512(inputPtr + next + x + 1);
                                Vector512<float> value = Avx512F.Max(Avx512F.Max(a, b0), Avx512F.Max(c0, d));
                                Avx512F.Store(outputPtr + row + x, value);
                            }
                            for (; x < width - 1; x++)
                                output[row + x] = MathF.Max(MathF.Max(input[row + x], input[row + x + 1]),
                                    MathF.Max(input[next + x], input[next + x + 1]));
                            output[row + width - 1] = MathF.Max(input[row + width - 1], input[next + width - 1]);
                        }
                        int last = offset + (height - 1) * width, lx = 0;
                        for (; lx < width - 1; lx++) output[last + lx] = MathF.Max(input[last + lx], input[last + lx + 1]);
                        output[last + width - 1] = input[last + width - 1];
                    }
            }
            return;
        }
        else if (Avx.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < channels; c++)
                    {
                        int offset = (b * channels + c) * plane;
                        for (int y = 0; y < height - 1; y++)
                        {
                            int row = offset + y * width, next = row + width, x = 0;
                            for (; x <= width - 9; x += 8)
                            {
                                Vector256<float> a = Avx.LoadVector256(inputPtr + row + x);
                                Vector256<float> b0 = Avx.LoadVector256(inputPtr + row + x + 1);
                                Vector256<float> c0 = Avx.LoadVector256(inputPtr + next + x);
                                Vector256<float> d = Avx.LoadVector256(inputPtr + next + x + 1);
                                Vector256<float> value = Avx.Max(Avx.Max(a, b0), Avx.Max(c0, d));
                                Avx.Store(outputPtr + row + x, value);
                            }
                            for (; x < width - 1; x++)
                                output[row + x] = MathF.Max(MathF.Max(input[row + x], input[row + x + 1]),
                                    MathF.Max(input[next + x], input[next + x + 1]));
                            output[row + width - 1] = MathF.Max(input[row + width - 1], input[next + width - 1]);
                        }
                        int last = offset + (height - 1) * width, lx = 0;
                        for (; lx < width - 1; lx++) output[last + lx] = MathF.Max(input[last + lx], input[last + lx + 1]);
                        output[last + width - 1] = input[last + width - 1];
                    }
            }
            return;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int widthLanes = Vector<float>.Count;
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < channels; c++)
                    {
                        int offset = (b * channels + c) * plane;
                        for (int y = 0; y < height - 1; y++)
                        {
                            int row = offset + y * width, next = row + width, x = 0;
                            for (; x <= width - (widthLanes + 1); x += widthLanes)
                            {
                                Vector<float> a = VecLoad(inputPtr + row + x);
                                Vector<float> b0 = VecLoad(inputPtr + row + x + 1);
                                Vector<float> c0 = VecLoad(inputPtr + next + x);
                                Vector<float> d = VecLoad(inputPtr + next + x + 1);
                                VecStore(outputPtr + row + x, Vector.Max(Vector.Max(a, b0), Vector.Max(c0, d)));
                            }
                            for (; x < width - 1; x++)
                                output[row + x] = MathF.Max(MathF.Max(input[row + x], input[row + x + 1]),
                                    MathF.Max(input[next + x], input[next + x + 1]));
                            output[row + width - 1] = MathF.Max(input[row + width - 1], input[next + width - 1]);
                        }
                        int last = offset + (height - 1) * width, lx = 0;
                        for (; lx < width - 1; lx++) output[last + lx] = MathF.Max(input[last + lx], input[last + lx + 1]);
                        output[last + width - 1] = input[last + width - 1];
                    }
            }
            return;
        }
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                int offset = (b * channels + c) * plane;
                for (int y = 0; y < height - 1; y++)
                {
                    int row = offset + y * width, next = row + width;
                    for (int x = 0; x < width - 1; x++)
                        output[row + x] = MathF.Max(MathF.Max(input[row + x], input[row + x + 1]),
                            MathF.Max(input[next + x], input[next + x + 1]));
                    output[row + width - 1] = MathF.Max(input[row + width - 1], input[next + width - 1]);
                }
                int last = offset + (height - 1) * width;
                for (int x = 0; x < width - 1; x++) output[last + x] = MathF.Max(input[last + x], input[last + x + 1]);
                output[last + width - 1] = input[last + width - 1];
            }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void SoftmaxContiguous(ReadOnlySpan<float> input, Span<float> output,
        int rowCount, int axisCount)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported && axisCount >= 16)
        {
            fixed (float* inputPtr = input, outputPtr = output)
                for (int row = 0; row < rowCount; row++)
                {
                    float* inputRow = inputPtr + row * axisCount, outputRow = outputPtr + row * axisCount;
                    Vector512<float> maxVector = Vector512.Create(float.NegativeInfinity);
                    int maxVecEnd = axisCount & ~15;
                    for (int i = 0; i < maxVecEnd; i += 16)
                        maxVector = Avx512F.Max(maxVector, Avx512F.LoadVector512(inputRow + i));
                    Vector256<float> max256 = Avx.Max(maxVector.GetLower(), maxVector.GetUpper());
                    Vector128<float> max128 = Sse.Max(max256.GetLower(), max256.GetUpper());
                    max128 = Sse.Max(max128, Sse.Shuffle(max128, max128, 0x4E));
                    max128 = Sse.Max(max128, Sse.Shuffle(max128, max128, 0xB1));
                    float max = max128.ToScalar();
                    for (int i = maxVecEnd; i < axisCount; i++) max = MathF.Max(max, inputRow[i]);
                    Vector512<float> vmax = Vector512.Create(max);
                    int iVec = 0;
                    for (; iVec <= axisCount - 16; iVec += 16)
                        Avx512F.Store(outputRow + iVec, ExpApproxVector512(Avx512F.Subtract(Avx512F.LoadVector512(inputRow + iVec), vmax)));
                    for (int i = iVec; i < axisCount; i++) outputRow[i] = MathF.Exp(inputRow[i] - max);
                    Vector512<float> sumVector = Vector512<float>.Zero;
                    int sumVecEnd = 0;
                    for (; sumVecEnd <= axisCount - 16; sumVecEnd += 16)
                        sumVector = Avx512F.Add(sumVector, Avx512F.LoadVector512(outputRow + sumVecEnd));
                    float sum = Vector512.Sum(sumVector);
                    for (int i = sumVecEnd; i < axisCount; i++) sum += outputRow[i];
                    Vector512<float> inv = Vector512.Create(1f / sum);
                    iVec = 0;
                    for (; iVec <= axisCount - 16; iVec += 16)
                        Avx512F.Store(outputRow + iVec, Avx512F.Multiply(Avx512F.LoadVector512(outputRow + iVec), inv));
                    for (int i = iVec; i < axisCount; i++) outputRow[i] /= sum;
                }
        }
        else if (Avx2.IsSupported && axisCount >= 8)
        {
            fixed (float* inputPtr = input, outputPtr = output)
                for (int row = 0; row < rowCount; row++)
                {
                    float* inputRow = inputPtr + row * axisCount, outputRow = outputPtr + row * axisCount;
                    Vector256<float> maxVector = Vector256.Create(float.NegativeInfinity);
                    int maxVecEnd = axisCount & ~7;
                    for (int i = 0; i < maxVecEnd; i += 8)
                        maxVector = Avx.Max(maxVector, Avx.LoadVector256(inputRow + i));
                    float max = maxVector.GetElement(0);
                    for (int lane = 1; lane < 8; lane++) max = MathF.Max(max, maxVector.GetElement(lane));
                    for (int i = maxVecEnd; i < axisCount; i++) max = MathF.Max(max, inputRow[i]);
                    Vector256<float> vmax = Vector256.Create(max);
                    int iVec = 0;
                    for (; iVec <= axisCount - 8; iVec += 8)
                        Avx.Store(outputRow + iVec, ExpApproxVector(Avx.Subtract(Avx.LoadVector256(inputRow + iVec), vmax)));
                    for (int i = iVec; i < axisCount; i++) outputRow[i] = MathF.Exp(inputRow[i] - max);
                    // The denominator uses a vectorized 8-lane accumulation with a
                    // fixed lane-order horizontal reduction: deterministic across
                    // runs, but no longer bit-matched to the C engine's scalar sum
                    // (only the low bits of reported CTC scores differ).
                    Vector256<float> sumVector = Vector256<float>.Zero;
                    int sumVecEnd = 0;
                    for (; sumVecEnd <= axisCount - 8; sumVecEnd += 8)
                        sumVector = Avx.Add(sumVector, Avx.LoadVector256(outputRow + sumVecEnd));
                    float sum = 0f;
                    for (int lane = 0; lane < 8; lane++) sum += sumVector.GetElement(lane);
                    for (int i = sumVecEnd; i < axisCount; i++) sum += outputRow[i];
                    Vector256<float> inv = Vector256.Create(1f / sum);
                    iVec = 0;
                    for (; iVec <= axisCount - 8; iVec += 8)
                        Avx.Store(outputRow + iVec, Avx.Multiply(Avx.LoadVector256(outputRow + iVec), inv));
                    for (int i = iVec; i < axisCount; i++) outputRow[i] /= sum;
                }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated && axisCount >= Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            fixed (float* inputPtr = input, outputPtr = output)
                for (int row = 0; row < rowCount; row++)
                {
                    float* inputRow = inputPtr + row * axisCount, outputRow = outputPtr + row * axisCount;
                    Vector<float> maxVector = new(float.NegativeInfinity);
                    int i = 0;
                    for (; i <= axisCount - width; i += width)
                        maxVector = Vector.Max(maxVector, VecLoad(inputRow + i));
                    float max = maxVector[0];
                    for (int lane = 1; lane < width; lane++) max = MathF.Max(max, maxVector.GetElement(lane));
                    for (; i < axisCount; i++) max = MathF.Max(max, inputRow[i]);
                    Vector<float> vmax = new(max);
                    i = 0;
                    for (; i <= axisCount - width; i += width)
                        VecStore(outputRow + i, ExpExactVector(VecLoad(inputRow + i) - vmax));
                    for (; i < axisCount; i++) outputRow[i] = MathF.Exp(inputRow[i] - max);
                    Vector<float> sumVector = Vector<float>.Zero;
                    int sumEnd = 0;
                    for (; sumEnd <= axisCount - width; sumEnd += width)
                        sumVector += VecLoad(outputRow + sumEnd);
                    float sum = 0f;
                    for (int lane = 0; lane < width; lane++) sum += sumVector.GetElement(lane);
                    for (int j = sumEnd; j < axisCount; j++) sum += outputRow[j];
                    Vector<float> inv = new(1f / sum);
                    i = 0;
                    for (; i <= axisCount - width; i += width)
                        VecStore(outputRow + i, VecLoad(outputRow + i) * inv);
                    for (; i < axisCount; i++) outputRow[i] /= sum;
                }
        }
        else
        {
            for (int row = 0; row < rowCount; row++)
            {
                int offset = row * axisCount; float max = input[offset];
                for (int i = 1; i < axisCount; i++) max = MathF.Max(max, input[offset + i]);
                float sum = 0;
                for (int i = 0; i < axisCount; i++) { float e = MathF.Exp(input[offset + i] - max); output[offset + i] = e; sum += e; }
                for (int i = 0; i < axisCount; i++) output[offset + i] /= sum;
            }
        }
    }

    // Computes just Softmax(argmax(logits)). This is the only probability CTC
    // decoding consumes, so avoid writing, rereading, and normalizing the
    // entire (typically many-thousand-class) row.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe float SoftmaxMaximumProbability(ReadOnlySpan<float> logits,
        float maximum)
    {
        int i = 0;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported && logits.Length >= 16)
        {
            Vector512<float> vmax = Vector512.Create(maximum);
            Vector512<float> sum0 = Vector512<float>.Zero, sum1 = Vector512<float>.Zero;
            fixed (float* ptr = logits)
            {
                for (; i <= logits.Length - 32; i += 32)
                {
                    sum0 = Avx512F.Add(sum0,
                        ExpApproxVector512(Avx512F.Subtract(Avx512F.LoadVector512(ptr + i), vmax)));
                    sum1 = Avx512F.Add(sum1,
                        ExpApproxVector512(Avx512F.Subtract(Avx512F.LoadVector512(ptr + i + 16), vmax)));
                }
                if (i <= logits.Length - 16)
                {
                    sum0 = Avx512F.Add(sum0,
                        ExpApproxVector512(Avx512F.Subtract(Avx512F.LoadVector512(ptr + i), vmax)));
                    i += 16;
                }
            }
            float sum = Vector512.Sum(Avx512F.Add(sum0, sum1));
            for (; i < logits.Length; i++) sum += MathF.Exp(logits[i] - maximum);
            return 1f / sum;
        }
        else if (Avx2.IsSupported && logits.Length >= 8)
        {
            Vector256<float> vmax = Vector256.Create(maximum);
            Vector256<float> sum0 = Vector256<float>.Zero, sum1 = Vector256<float>.Zero;
            fixed (float* ptr = logits)
            {
                for (; i <= logits.Length - 16; i += 16)
                {
                    sum0 = Avx.Add(sum0,
                        ExpApproxVector(Avx.Subtract(Avx.LoadVector256(ptr + i), vmax)));
                    sum1 = Avx.Add(sum1,
                        ExpApproxVector(Avx.Subtract(Avx.LoadVector256(ptr + i + 8), vmax)));
                }
                if (i <= logits.Length - 8)
                {
                    sum0 = Avx.Add(sum0,
                        ExpApproxVector(Avx.Subtract(Avx.LoadVector256(ptr + i), vmax)));
                    i += 8;
                }
            }
            Vector256<float> sums = Avx.Add(sum0, sum1);
            float sum = 0;
            for (int lane = 0; lane < 8; lane++) sum += sums.GetElement(lane);
            for (; i < logits.Length; i++) sum += MathF.Exp(logits[i] - maximum);
            return 1f / sum;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated && logits.Length >= Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            Vector<float> vmax = new(maximum);
            Vector<float> sum0 = Vector<float>.Zero, sum1 = Vector<float>.Zero;
            fixed (float* ptr = logits)
            {
                for (; i <= logits.Length - width * 2; i += width * 2)
                {
                    sum0 += ExpExactVector(VecLoad(ptr + i) - vmax);
                    sum1 += ExpExactVector(VecLoad(ptr + i + width) - vmax);
                }
                if (i <= logits.Length - width)
                {
                    sum0 += ExpExactVector(VecLoad(ptr + i) - vmax);
                    i += width;
                }
            }
            Vector<float> sums = sum0 + sum1;
            float sum = 0;
            for (int lane = 0; lane < width; lane++)
                sum += sums.GetElement(lane);
            for (; i < logits.Length; i++) sum += MathF.Exp(logits[i] - maximum);
            return 1f / sum;
        }
        else
        {
            float scalarSum = 0;
            for (; i < logits.Length; i++) scalarSum += MathF.Exp(logits[i] - maximum);
            return 1f / scalarSum;
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe void Gelu(ReadOnlySpan<float> input, Span<float> output)
    {
        int i = 0;
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= input.Length - 32; i += 32)
                {
                    Vector512<float> value0 = Avx512F.LoadVector512(inputPtr + i);
                    Vector512<float> value1 = Avx512F.LoadVector512(inputPtr + i + 16);
                    Vector512<float> activated0 = Avx512F.Add(ErfVector512(Avx512F.Multiply(value0, V512InvSqrtTwo)), V512One);
                    Vector512<float> activated1 = Avx512F.Add(ErfVector512(Avx512F.Multiply(value1, V512InvSqrtTwo)), V512One);
                    activated0 = Avx512F.Multiply(value0, activated0);
                    activated1 = Avx512F.Multiply(value1, activated1);
                    Avx512F.Store(outputPtr + i, Avx512F.Multiply(activated0, V512Half));
                    Avx512F.Store(outputPtr + i + 16, Avx512F.Multiply(activated1, V512Half));
                }
                for (; i <= input.Length - 16; i += 16)
                {
                    Vector512<float> value = Avx512F.LoadVector512(inputPtr + i);
                    Vector512<float> activated = Avx512F.Add(ErfVector512(Avx512F.Multiply(value, V512InvSqrtTwo)), V512One);
                    Avx512F.Store(outputPtr + i, Avx512F.Multiply(Avx512F.Multiply(value, activated), V512Half));
                }
            }
        }
        else if (Avx.IsSupported)
        {
            fixed (float* inputPtr = input, outputPtr = output)
            {
                // Keep two independent vectors in flight so the polynomial
                // dependency chain in ErfVector can overlap on AVX2/FMA
                // execution ports. Each element is still evaluated and
                // stored in its original order.
                for (; i <= input.Length - 16; i += 16)
                {
                    Vector256<float> value0 = Avx.LoadVector256(inputPtr + i);
                    Vector256<float> value1 = Avx.LoadVector256(inputPtr + i + 8);
                    Vector256<float> activated0 = Avx.Add(ErfVector(Avx.Multiply(value0, VInvSqrtTwo)), VOne);
                    Vector256<float> activated1 = Avx.Add(ErfVector(Avx.Multiply(value1, VInvSqrtTwo)), VOne);
                    activated0 = Avx.Multiply(value0, activated0);
                    activated1 = Avx.Multiply(value1, activated1);
                    Avx.Store(outputPtr + i, Avx.Multiply(activated0, VHalf));
                    Avx.Store(outputPtr + i + 8, Avx.Multiply(activated1, VHalf));
                }
            }
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            Vector<float> invSqrtTwo = new(0.70710678118654752f);
            Vector<float> one = new Vector<float>(1f);
            Vector<float> half = new(0.5f);
            fixed (float* inputPtr = input, outputPtr = output)
            {
                for (; i <= input.Length - width; i += width)
                {
                    Vector<float> value = VecLoad(inputPtr + i);
                    Vector<float> activated = ErfExactVector(value * invSqrtTwo) + one;
                    VecStore(outputPtr + i, value * activated * half);
                }
            }
        }
        for (; i < input.Length; i++) { float v = input[i], e = Erf(v * 0.70710678118654752f) + 1f; output[i] = v * e * 0.5f; }
    }
}
