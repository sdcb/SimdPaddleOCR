using System.Numerics;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class ConvDenseStride1
{
    // AVX2 channel-blocked dense kernel for the detector's large 5x5/7x7
    // projections.  The graph remains NCHW; only the output tile is kept in
    // an NHWC scratch buffer while eight output channels are accumulated in a
    // vector.  This replaces eight scalar weight broadcasts with one vector
    // load per tap and preserves the original ci,ky,kx accumulation order.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe bool TryPacked8(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> rawWeights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int intraOpThreads)
    {
#if !NETSTANDARD2_0
        if (!Avx2.IsSupported || !Fma.IsSupported || Vector<float>.Count != 8 ||
            batch <= 0 || inputChannels < 16 || outputChannels < 8 ||
            (outputChannels & 7) != 0 ||
            !((kernelH == 5 && kernelW == 5) || (kernelH == 7 && kernelW == 7)))
            return false;
        int xStart = MathCompat.Clamp(padLeft, 0, outputWidth);
        int xEnd = MathCompat.Clamp(width - kernelW + 1 + padLeft, xStart, outputWidth);
        int yStart = MathCompat.Clamp(padTop, 0, outputHeight);
        int yEnd = MathCompat.Clamp(height - kernelH + 1 + padTop, yStart, outputHeight);
        if (xEnd - xStart < 4 || yEnd <= yStart)
            return false;
        int taps = checked(kernelH * kernelW), inputPlane = checked(height * width),
            outputPlane = checked(outputHeight * outputWidth), blocks = outputChannels / 8;
        long work = checked((long)batch * outputChannels * inputChannels * taps * outputPlane);
        if (work < 4_000_000)
            return false;
        if (packedWeights.Length < checked(outputChannels * inputChannels * taps))
            return false;
        float[] scratch = ArrayPool<float>.Shared.Rent(checked(batch * outputPlane * outputChannels));
        try
        {
            fixed (float* inputPtr = input, packedPtr = packedWeights, rawPtr = rawWeights,
                biasPtr = bias, outputPtr = output, scratchPtr = scratch)
            {
                nint inputAddress = (nint)inputPtr, packedAddress = (nint)packedPtr,
                    rawAddress = (nint)rawPtr, biasAddress = (nint)biasPtr,
                    scratchAddress = (nint)scratchPtr;
                // For the detector's 32-channel 5x5/7x7 blocks, the NCHW
                // spatial kernel has eight independent output-channel workers
                // while this layout has only four. Its extra scratch transpose
                // loses the gain from a packed weight load, so let the NCHW
                // implementation handle undersubscribed channel blocks.
                if (intraOpThreads > blocks)
                    return false;
                int workers = intraOpThreads > 1 && blocks > 1 && work >= 4_000_000
                    ? Math.Min(intraOpThreads, blocks) : 1;
                Action<int> runWorker = worker =>
                {
                    float* inputBase = (float*)inputAddress, packedBase = (float*)packedAddress,
                        rawBase = (float*)rawAddress, biasBase = (float*)biasAddress,
                        scratchBase = (float*)scratchAddress;
                    int beginBlock = blocks * worker / workers;
                    int endBlock = blocks * (worker + 1) / workers;
                    for (int b = 0; b < batch; b++)
                    {
                        float* inputBatch = inputBase + (long)b * inputChannels * inputPlane;
                        float* scratchBatch = scratchBase + (long)b * outputPlane * outputChannels;
                        for (int block = beginBlock; block < endBlock; block++)
                        {
                            int co = block * 8;
                            float* blockWeights = packedBase + (long)block * inputChannels * taps * 8;
                            for (int y = 0; y < outputHeight; y++)
                            {
                                int kyMin = Math.Max(0, padTop - y);
                                int kyMax = Math.Min(kernelH, height - y + padTop);
                                float* scratchRow = scratchBatch + (long)y * outputWidth * outputChannels + co;
                                for (int x = 0; x < xStart; x++)
                                    for (int q = 0; q < 8; q++)
                                        PackedEdgePixel(inputBatch, rawBase, biasBase,
                                            scratchRow + (long)x * outputChannels + q,
                                            inputChannels, height, width, outputWidth, outputChannels,
                                            kernelH, kernelW, padTop, padLeft, co + q, x, y);
                                int x4 = xStart;
                                if (kyMax > kyMin)
                                {
                                    int kyCount = kyMax - kyMin;
                                    for (; x4 <= xEnd - 4; x4 += 4)
                                    {
                                        Vector256<float> a0, a1, a2, a3;
                                        // The four spatial accumulators each
                                        // contain the eight output channels.
                                        // Initialize them from the same bias
                                        // vector; lanes are output channels.
                                        Vector256<float> biasVector = biasBase == null
                                            ? Vector256<float>.Zero : Avx.LoadVector256(biasBase + co);
                                        a0 = biasVector; a1 = biasVector; a2 = biasVector; a3 = biasVector;
                                        for (int ci = 0; ci < inputChannels; ci++)
                                        {
                                            int inputChannel = checked(ci * inputPlane);
                                            int packedChannel = checked(ci * taps * 8 + kyMin * kernelW * 8);
                                            for (int ky = kyMin; ky < kyMax; ky++)
                                            {
                                                int inputRow = checked((y - padTop + ky) * width + x4 - padLeft);
                                                for (int kx = 0; kx < kernelW; kx++)
                                                {
                                                    Vector256<float> weightsVector = Avx.LoadVector256(blockWeights + packedChannel);
                                                    a0 = Fma.MultiplyAdd(Vector256.Create(inputBatch[inputChannel + inputRow + kx]), weightsVector, a0);
                                                    a1 = Fma.MultiplyAdd(Vector256.Create(inputBatch[inputChannel + inputRow + kx + 1]), weightsVector, a1);
                                                    a2 = Fma.MultiplyAdd(Vector256.Create(inputBatch[inputChannel + inputRow + kx + 2]), weightsVector, a2);
                                                    a3 = Fma.MultiplyAdd(Vector256.Create(inputBatch[inputChannel + inputRow + kx + 3]), weightsVector, a3);
                                                    packedChannel += 8;
                                                }
                                            }
                                        }
                                        Avx.Store(scratchRow + (long)x4 * outputChannels, a0);
                                        Avx.Store(scratchRow + (long)(x4 + 1) * outputChannels, a1);
                                        Avx.Store(scratchRow + (long)(x4 + 2) * outputChannels, a2);
                                        Avx.Store(scratchRow + (long)(x4 + 3) * outputChannels, a3);
                                    }
                                    for (; x4 < xEnd; x4++)
                                        for (int q = 0; q < 8; q++)
                                            PackedEdgePixel(inputBatch, rawBase, biasBase,
                                                scratchRow + (long)x4 * outputChannels + q,
                                                inputChannels, height, width, outputWidth, outputChannels,
                                                kernelH, kernelW, padTop, padLeft, co + q, x4, y);
                                }
                                for (int x = Math.Max(x4, xEnd); x < outputWidth; x++)
                                    for (int q = 0; q < 8; q++)
                                        PackedEdgePixel(inputBatch, rawBase, biasBase,
                                            scratchRow + (long)x * outputChannels + q,
                                            inputChannels, height, width, outputWidth, outputChannels,
                                            kernelH, kernelW, padTop, padLeft, co + q, x, y);
                            }
                        }
                    }
                };
                if (workers > 1) Parallel.For(0, workers, runWorker);
                else runWorker(0);

                // Convert the spatial-major scratch back to the graph's NCHW
                // layout.  This is a single streaming pass after all channel
                // blocks have completed and is much cheaper than the blocked
                // convolution it replaces.
                for (int b = 0; b < batch; b++)
                {
                    float* scratchBatch = scratchPtr + (long)b * outputPlane * outputChannels;
                    float* outputBatch = outputPtr + (long)b * outputPlane * outputChannels;
                    for (int co = 0; co < outputChannels; co++)
                    {
                        float* dst = outputBatch + (long)co * outputPlane;
                        for (int spatial = 0; spatial < outputPlane; spatial++)
                            dst[spatial] = scratchBatch[(long)spatial * outputChannels + co];
                    }
                }
            }
            return true;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
#else
        return false;
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackedEdgePixel(float* input, float* rawWeights,
        float* bias, float* output, int inputChannels, int height, int width,
        int outputWidth, int outputChannels, int kernelH, int kernelW,
        int padTop, int padLeft, int co, int x, int y)
    {
        float sum = bias == null ? 0f : bias[co];
        long weightBase = (long)co * inputChannels * kernelH * kernelW;
        for (int ci = 0; ci < inputChannels; ci++)
            for (int ky = 0; ky < kernelH; ky++)
            {
                int iy = y - padTop + ky;
                if ((uint)iy >= (uint)height) continue;
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int ix = x - padLeft + kx;
                    if ((uint)ix >= (uint)width) continue;
                    sum += input[(long)ci * height * width + (long)iy * width + ix] *
                        rawWeights[weightBase + (long)ci * kernelH * kernelW + ky * kernelW + kx];
                }
            }
        output[0] = sum;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void DenseStride1OctUnsafe(float* input, float* weights, float* bias,
        float* output, int inputChannels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int co, int xStart, int xEnd,
        int yBegin = 0, int yEnd = -1)
    {
        if (yEnd < 0) yEnd = outputHeight;
        int weightsPerOut = inputChannels * kernelH * kernelW;
        float* w0 = weights + (long)co * weightsPerOut;
        float* w1 = w0 + weightsPerOut, w2 = w1 + weightsPerOut, w3 = w2 + weightsPerOut;
        float* w4 = w3 + weightsPerOut, w5 = w4 + weightsPerOut, w6 = w5 + weightsPerOut, w7 = w6 + weightsPerOut;
        int outPlane = outputHeight * outputWidth;
        float* out0 = output + (long)co * outPlane;
        float* out1 = out0 + outPlane, out2 = out1 + outPlane, out3 = out2 + outPlane;
        float* out4 = out3 + outPlane, out5 = out4 + outPlane, out6 = out5 + outPlane, out7 = out6 + outPlane;
        Vector256<float> b0 = Vector256.Create(bias == null ? 0f : bias[co]);
        Vector256<float> b1 = Vector256.Create(bias == null ? 0f : bias[co + 1]);
        Vector256<float> b2 = Vector256.Create(bias == null ? 0f : bias[co + 2]);
        Vector256<float> b3 = Vector256.Create(bias == null ? 0f : bias[co + 3]);
        Vector256<float> b4 = Vector256.Create(bias == null ? 0f : bias[co + 4]);
        Vector256<float> b5 = Vector256.Create(bias == null ? 0f : bias[co + 5]);
        Vector256<float> b6 = Vector256.Create(bias == null ? 0f : bias[co + 6]);
        Vector256<float> b7 = Vector256.Create(bias == null ? 0f : bias[co + 7]);
        for (int y = yBegin; y < yEnd; y++)
        {
            int kyMin = Math.Max(0, padTop - y), kyMax = Math.Min(kernelH, height - y + padTop);
            if (kyMax <= kyMin)
            {
                for (int x = 0; x < outputWidth; x++)
                    for (int q = 0; q < 8; q++)
                        DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                            outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
                continue;
            }
            for (int x = 0; x < xStart; x++)
                for (int q = 0; q < 8; q++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
            int x8 = xStart, kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            int weightRowSkip = kyMin * kernelW, weightRowRemainder = (kernelH - kyMax) * kernelW;
            for (; x8 <= xEnd - 8; x8 += 8)
            {
                Vector256<float> a0 = b0, a1 = b1, a2 = b2, a3 = b3;
                Vector256<float> a4 = b4, a5 = b5, a6 = b6, a7 = b7;
                float* inputChannel = rowBase + x8;
                float* wc0 = w0 + weightRowSkip, wc1 = w1 + weightRowSkip;
                float* wc2 = w2 + weightRowSkip, wc3 = w3 + weightRowSkip;
                float* wc4 = w4 + weightRowSkip, wc5 = w5 + weightRowSkip;
                float* wc6 = w6 + weightRowSkip, wc7 = w7 + weightRowSkip;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    float* tapRow = inputChannel;
                    for (int ky = 0; ky < kyCount; ky++)
                    {
                        for (int kx = 0; kx < kernelW; kx++)
                        {
                            Vector256<float> value = Avx.LoadVector256(tapRow + kx);
                            a0 = AddMul(a0, value, Vector256.Create(wc0[kx]));
                            a1 = AddMul(a1, value, Vector256.Create(wc1[kx]));
                            a2 = AddMul(a2, value, Vector256.Create(wc2[kx]));
                            a3 = AddMul(a3, value, Vector256.Create(wc3[kx]));
                            a4 = AddMul(a4, value, Vector256.Create(wc4[kx]));
                            a5 = AddMul(a5, value, Vector256.Create(wc5[kx]));
                            a6 = AddMul(a6, value, Vector256.Create(wc6[kx]));
                            a7 = AddMul(a7, value, Vector256.Create(wc7[kx]));
                        }
                        tapRow += width;
                        wc0 += kernelW; wc1 += kernelW; wc2 += kernelW; wc3 += kernelW;
                        wc4 += kernelW; wc5 += kernelW; wc6 += kernelW; wc7 += kernelW;
                    }
                    inputChannel += height * width;
                    wc0 += weightRowSkip + weightRowRemainder; wc1 += weightRowSkip + weightRowRemainder;
                    wc2 += weightRowSkip + weightRowRemainder; wc3 += weightRowSkip + weightRowRemainder;
                    wc4 += weightRowSkip + weightRowRemainder; wc5 += weightRowSkip + weightRowRemainder;
                    wc6 += weightRowSkip + weightRowRemainder; wc7 += weightRowSkip + weightRowRemainder;
                }
                int outOffset = y * outputWidth + x8;
                Avx.Store(out0 + outOffset, a0); Avx.Store(out1 + outOffset, a1);
                Avx.Store(out2 + outOffset, a2); Avx.Store(out3 + outOffset, a3);
                Avx.Store(out4 + outOffset, a4); Avx.Store(out5 + outOffset, a5);
                Avx.Store(out6 + outOffset, a6); Avx.Store(out7 + outOffset, a7);
            }
            for (int x = x8; x < outputWidth; x++)
                for (int q = 0; q < 8; q++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void DenseStride1QuadUnsafe(float* input, float* weights, float* bias,
        float* output, int inputChannels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int co, int xStart, int xEnd)
    {
        int weightsPerOut = inputChannels * kernelH * kernelW;
        float* w0 = weights + (long)co * weightsPerOut;
        float* w1 = w0 + weightsPerOut, w2 = w1 + weightsPerOut, w3 = w2 + weightsPerOut;
        float* out0 = output + (long)co * outputHeight * outputWidth;
        float* out1 = out0 + outputHeight * outputWidth, out2 = out1 + outputHeight * outputWidth,
            out3 = out2 + outputHeight * outputWidth;
        Vector256<float> bias0 = Vector256.Create(bias == null ? 0f : bias[co]);
        Vector256<float> bias1 = Vector256.Create(bias == null ? 0f : bias[co + 1]);
        Vector256<float> bias2 = Vector256.Create(bias == null ? 0f : bias[co + 2]);
        Vector256<float> bias3 = Vector256.Create(bias == null ? 0f : bias[co + 3]);
        for (int y = 0; y < outputHeight; y++)
        {
            // Rows near the top/bottom simply restrict the valid tap range;
            // the restricted loop adds taps in the same (ci,ky,kx) order the
            // scalar edge path uses, so results are identical.
            int kyMin = Math.Max(0, padTop - y);
            int kyMax = Math.Min(kernelH, height - y + padTop);
            if (kyMax <= kyMin)
            {
                for (int x = 0; x < outputWidth; x++)
                    for (int q = 0; q < 4; q++)
                        DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                            outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
                continue;
            }
            for (int x = 0; x < xStart; x++)
                for (int q = 0; q < 4; q++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
            int x16 = xStart;
            int kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            int weightRowSkip = kyMin * kernelW, weightRowRemainder = (kernelH - kyMax) * kernelW;
            for (; x16 <= xEnd - 16; x16 += 16)
            {
                Vector256<float> a0l = bias0, a0h = bias0;
                Vector256<float> a1l = bias1, a1h = bias1;
                Vector256<float> a2l = bias2, a2h = bias2;
                Vector256<float> a3l = bias3, a3h = bias3;
                float* inputChannel = rowBase + x16;
                float* wc0 = w0 + weightRowSkip, wc1 = w1 + weightRowSkip;
                float* wc2 = w2 + weightRowSkip, wc3 = w3 + weightRowSkip;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    float* tapRow = inputChannel;
                    for (int ky = 0; ky < kyCount; ky++)
                    {
                        for (int kx = 0; kx < kernelW; kx++)
                        {
                            Vector256<float> valueLow = Avx.LoadVector256(tapRow + kx);
                            Vector256<float> valueHigh = Avx.LoadVector256(tapRow + kx + 8);
                            a0l = AddMul(a0l, valueLow, Vector256.Create(wc0[kx]));
                            a0h = AddMul(a0h, valueHigh, Vector256.Create(wc0[kx]));
                            a1l = AddMul(a1l, valueLow, Vector256.Create(wc1[kx]));
                            a1h = AddMul(a1h, valueHigh, Vector256.Create(wc1[kx]));
                            a2l = AddMul(a2l, valueLow, Vector256.Create(wc2[kx]));
                            a2h = AddMul(a2h, valueHigh, Vector256.Create(wc2[kx]));
                            a3l = AddMul(a3l, valueLow, Vector256.Create(wc3[kx]));
                            a3h = AddMul(a3h, valueHigh, Vector256.Create(wc3[kx]));
                        }
                        tapRow += width;
                        wc0 += kernelW; wc1 += kernelW; wc2 += kernelW; wc3 += kernelW;
                    }
                    inputChannel += height * width;
                    wc0 += weightRowSkip + weightRowRemainder; wc1 += weightRowSkip + weightRowRemainder;
                    wc2 += weightRowSkip + weightRowRemainder; wc3 += weightRowSkip + weightRowRemainder;
                }
                int outOffset = y * outputWidth + x16;
                Avx.Store(out0 + outOffset, a0l); Avx.Store(out0 + outOffset + 8, a0h);
                Avx.Store(out1 + outOffset, a1l); Avx.Store(out1 + outOffset + 8, a1h);
                Avx.Store(out2 + outOffset, a2l); Avx.Store(out2 + outOffset + 8, a2h);
                Avx.Store(out3 + outOffset, a3l); Avx.Store(out3 + outOffset + 8, a3h);
            }
            int x8 = x16;
            for (; x8 <= xEnd - 8; x8 += 8)
            {
                Vector256<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                float* inputChannel = rowBase + x8;
                float* wc0 = w0 + weightRowSkip, wc1 = w1 + weightRowSkip,
                    wc2 = w2 + weightRowSkip, wc3 = w3 + weightRowSkip;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    float* tapRow = inputChannel;
                    for (int ky = 0; ky < kyCount; ky++)
                    {
                        for (int kx = 0; kx < kernelW; kx++)
                        {
                            Vector256<float> value = Avx.LoadVector256(tapRow + kx);
                            a0 = AddMul(a0, value, Vector256.Create(wc0[kx]));
                            a1 = AddMul(a1, value, Vector256.Create(wc1[kx]));
                            a2 = AddMul(a2, value, Vector256.Create(wc2[kx]));
                            a3 = AddMul(a3, value, Vector256.Create(wc3[kx]));
                        }
                        tapRow += width;
                        wc0 += kernelW; wc1 += kernelW; wc2 += kernelW; wc3 += kernelW;
                    }
                    inputChannel += height * width;
                    wc0 += weightRowSkip + weightRowRemainder; wc1 += weightRowSkip + weightRowRemainder;
                    wc2 += weightRowSkip + weightRowRemainder; wc3 += weightRowSkip + weightRowRemainder;
                }
                int outOffset = y * outputWidth + x8;
                Avx.Store(out0 + outOffset, a0); Avx.Store(out1 + outOffset, a1);
                Avx.Store(out2 + outOffset, a2); Avx.Store(out3 + outOffset, a3);
            }
            for (int x = x8; x < outputWidth; x++)
                for (int q = 0; q < 4; q++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co + q, y, x);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void DenseStride1SingleUnsafe(float* input, float* weights, float* bias,
        float* output, int inputChannels, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int co, int xStart, int xEnd)
    {
        int weightsPerOut = inputChannels * kernelH * kernelW;
        float* w0 = weights + (long)co * weightsPerOut;
        float* out0 = output + (long)co * outputHeight * outputWidth;
        Vector256<float> bias0 = Vector256.Create(bias == null ? 0f : bias[co]);
        for (int y = 0; y < outputHeight; y++)
        {
            int kyMin = Math.Max(0, padTop - y);
            int kyMax = Math.Min(kernelH, height - y + padTop);
            if (kyMax <= kyMin)
            {
                for (int x = 0; x < outputWidth; x++)
                    DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                        outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co, y, x);
                continue;
            }
            for (int x = 0; x < xStart; x++)
                DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                    outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co, y, x);
            int x8 = xStart;
            int kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            int weightRowSkip = kyMin * kernelW, weightRowRemainder = (kernelH - kyMax) * kernelW;
            for (; x8 <= xEnd - 8; x8 += 8)
            {
                Vector256<float> a0 = bias0;
                float* inputChannel = rowBase + x8;
                float* wc0 = w0 + weightRowSkip;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    float* tapRow = inputChannel;
                    for (int ky = 0; ky < kyCount; ky++)
                    {
                        for (int kx = 0; kx < kernelW; kx++)
                            a0 = AddMul(a0, Avx.LoadVector256(tapRow + kx), Vector256.Create(wc0[kx]));
                        tapRow += width;
                        wc0 += kernelW;
                    }
                    inputChannel += height * width;
                    wc0 += weightRowSkip + weightRowRemainder;
                }
                Avx.Store(out0 + y * outputWidth + x8, a0);
            }
            for (int x = x8; x < outputWidth; x++)
                DenseEdgePixel(input, weights, bias, output, inputChannels, height, width,
                    outputHeight, outputWidth, kernelH, kernelW, padTop, padLeft, co, y, x);
        }
    }
}
