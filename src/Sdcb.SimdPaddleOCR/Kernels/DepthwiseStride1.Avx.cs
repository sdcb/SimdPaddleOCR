using System.Numerics;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class DepthwiseStride1
{
    // Fixed 9x9 AVX2 interior. The generic channel kernel keeps a runtime
    // kx loop in its hottest path; spelling out the nine taps lets Tier-1 keep
    // the weight broadcasts and input loads in a tight FMA chain. Boundary
    // pixels retain the exact scalar tap-skipping path.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Depthwise9x9ChannelsUnsafe(float* input, float* weights,
        float* bias, float* output, int channels, int height, int width)
    {
        int plane = checked(height * width);
        for (int c = 0; c < channels; c++)
        {
            float* inputChannel = input + (long)c * plane;
            float* outputChannel = output + (long)c * plane;
            float* channelWeights = weights + (long)c * 81;
            float initial = bias == null ? 0f : bias[c];
            Vector256<float> vBias = Vector256.Create(initial);
            for (int y = 0; y < height; y++)
            {
                int kyMin = Math.Max(0, 4 - y), kyMax = Math.Min(9, height - y + 4);
                for (int x = 0; x < 4; x++)
                    DepthwiseEdgePixel(inputChannel, channelWeights, initial, outputChannel,
                        height, width, width, 9, 9, 4, 4, y, x);
                int x8 = 4;
                int interiorEnd = width - 4;
                // Bias the row pointer by the left pad so x=4 starts at the
                // first valid input column (the generic kernel uses the same
                // convention for its interior vectors).
                float* rowBase = inputChannel + (long)(y - 4 + kyMin) * width - 4;
                float* weightBase = channelWeights + kyMin * 9;
                for (; x8 <= interiorEnd - 8; x8 += 8)
                {
                    Vector256<float> acc = vBias;
                    float* tapRow = rowBase + x8;
                    float* wc = weightBase;
                    for (int ky = kyMin; ky < kyMax; ky++)
                    {
                        acc = AddMul(acc, Avx.LoadVector256(tapRow), wc[0]);
                        acc = AddMul(acc, Avx.LoadVector256(tapRow + 1), wc[1]);
                        acc = AddMul(acc, Avx.LoadVector256(tapRow + 2), wc[2]);
                        acc = AddMul(acc, Avx.LoadVector256(tapRow + 3), wc[3]);
                        acc = AddMul(acc, Avx.LoadVector256(tapRow + 4), wc[4]);
                        acc = AddMul(acc, Avx.LoadVector256(tapRow + 5), wc[5]);
                        acc = AddMul(acc, Avx.LoadVector256(tapRow + 6), wc[6]);
                        acc = AddMul(acc, Avx.LoadVector256(tapRow + 7), wc[7]);
                        acc = AddMul(acc, Avx.LoadVector256(tapRow + 8), wc[8]);
                        tapRow += width;
                        wc += 9;
                    }
                    Avx.Store(outputChannel + y * width + x8, acc);
                }
                for (; x8 < interiorEnd; x8++)
                    DepthwiseEdgePixel(inputChannel, channelWeights, initial, outputChannel,
                        height, width, width, 9, 9, 4, 4, y, x8);
                for (int x = Math.Max(interiorEnd, 4); x < width; x++)
                    DepthwiseEdgePixel(inputChannel, channelWeights, initial, outputChannel,
                        height, width, width, 9, 9, 4, 4, y, x);
            }
        }
    }

    // Channel-blocked AVX2 path for the 9x9 depthwise kernels used by the
    // medium/small detector.  NCHW activations are transposed once into an
    // NHWC scratch buffer, allowing one vector load for eight channels and a
    // contiguous packed weight vector per tap.  The output is transposed back
    // after the convolution so the rest of the graph keeps its original ABI.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DepthwisePackedPixel(float* input, float* packedWeights,
        float* bias, float* output, int rowInput, int channels, int width, int y, int x,
        int kyMin, int kyMax, int block)
    {
        int co = block * 8;
        Vector256<float> acc = bias == null ? Vector256<float>.Zero : Avx.LoadVector256(bias + co);
        int packedTap = block * 81 + kyMin * 9;
        for (int ky = kyMin; ky < kyMax; ky++)
        {
            int iy = y - 4 + ky;
            int inputRow = rowInput + (iy - y) * width * channels + co;
            for (int kx = 0; kx < 9; kx++, packedTap++)
            {
                int ix = x - 4 + kx;
                if ((uint)ix >= (uint)width) continue;
                acc = Fma.MultiplyAdd(Avx.LoadVector256(input + inputRow + ix * channels),
                    Avx.LoadVector256(packedWeights + packedTap * 8), acc);
            }
        }
        Avx.Store(output + co, acc);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool TryPacked9(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels,
        int height, int width, int outputHeight, int outputWidth, int intraOpThreads)
    {
#if !NETSTANDARD2_0
        if (!Avx2.IsSupported || !Fma.IsSupported || Vector<float>.Count != 8 ||
            batch <= 0 || channels < 8 || (channels & 7) != 0 ||
            outputHeight != height || outputWidth != width || height < 9 || width < 9 ||
            packedWeights.Length < checked(channels * 81))
            return false;
        int plane = checked(height * width), blocks = channels / 8;
        long work = checked((long)batch * channels * plane * 81);
        // The old spatial-vector kernel is faster once the map is large: it
        // streams contiguous x vectors and avoids two full NCHW/NHWC passes.
        // The channel-blocked path is reserved for the small detector maps,
        // where edge handling otherwise leaves too little vectorized interior.
        if (plane > 4096 || work < 1_000_000)
            return false;
        int rows = checked(batch * plane);
        float[] inputNhwc = ArrayPool<float>.Shared.Rent(checked(rows * channels));
        float[] outputNhwc = ArrayPool<float>.Shared.Rent(checked(rows * channels));
        try
        {
            fixed (float* inputPtr = input, packedPtr = packedWeights,
                biasPtr = bias, outputPtr = output, inputWork = inputNhwc, outputWork = outputNhwc)
            {
                nint inputAddress = (nint)inputPtr, packedAddress = (nint)packedPtr,
                    biasAddress = (nint)biasPtr, inputWorkAddress = (nint)inputWork,
                    outputWorkAddress = (nint)outputWork;
                int workers = intraOpThreads > 1 ? Math.Min(intraOpThreads, rows) : 1;
                Action<int> transposeWorker = worker =>
                {
                    float* srcBase = (float*)inputAddress, dstBase = (float*)inputWorkAddress;
                    int begin = rows * worker / workers, end = rows * (worker + 1) / workers;
                    for (int row = begin; row < end; row++)
                    {
                        int b = row / plane, spatial = row - b * plane;
                        float* src = srcBase + (long)b * channels * plane + spatial;
                        float* dst = dstBase + (long)row * channels;
                        for (int c = 0; c < channels; c++, src += plane)
                            dst[c] = *src;
                    }
                };
                if (workers > 1) Parallel.For(0, workers, transposeWorker);
                else transposeWorker(0);

                // Compute one complete image row per work item.  `rows` above
                // is the number of spatial points used by the NCHW<->NHWC
                // transposes; using it here would repeat every output row for
                // every x coordinate (and inflate the work by ~width).
                int computeRows = checked(batch * height);
                int computeWorkers = intraOpThreads > 1 ? Math.Min(intraOpThreads, computeRows) : 1;
                Action<int> computeWorker = worker =>
                {
                    float* inputBase = (float*)inputWorkAddress, packedBase = (float*)packedAddress,
                        biasBase = (float*)biasAddress, outputBase = (float*)outputWorkAddress;
                    int begin = computeRows * worker / computeWorkers, end = computeRows * (worker + 1) / computeWorkers;
                    for (int row = begin; row < end; row++)
                    {
                        int b = row / height, y = row - b * height;
                        int xStart = 4, xEnd = width - 4;
                        int rowInput = checked(b * plane * channels + y * width * channels);
                        int rowOutput = checked(b * plane * channels + y * width * channels);
                        int kyMin = Math.Max(0, 4 - y), kyMax = Math.Min(9, height - y + 4);
                        for (int x = 0; x < xStart; x++)
                            for (int block = 0; block < blocks; block++)
                                DepthwisePackedPixel(inputBase, packedBase, biasBase,
                                    outputBase + rowOutput + x * channels, rowInput, channels,
                                    width, y, x, kyMin, kyMax, block);
                        int x4 = xStart;
                        for (; x4 <= xEnd - 4; x4 += 4)
                        {
                            int inputX = x4 - 4;
                            for (int block = 0; block < blocks; block++)
                            {
                                int co = block * 8;
                                Vector256<float> acc0 = biasBase == null ? Vector256<float>.Zero : Avx.LoadVector256(biasBase + co);
                                Vector256<float> acc1 = acc0, acc2 = acc0, acc3 = acc0;
                                int packedTap = block * 81 + kyMin * 9;
                                for (int ky = kyMin; ky < kyMax; ky++)
                                {
                                    int inputRow = rowInput + (ky - 4) * width * channels + inputX * channels + co;
                                    for (int kx = 0; kx < 9; kx++)
                                    {
                                        Vector256<float> weightsVector = Avx.LoadVector256(packedBase + packedTap * 8);
                                        acc0 = Fma.MultiplyAdd(Avx.LoadVector256(inputBase + inputRow + kx * channels), weightsVector, acc0);
                                        acc1 = Fma.MultiplyAdd(Avx.LoadVector256(inputBase + inputRow + (kx * channels) + channels), weightsVector, acc1);
                                        acc2 = Fma.MultiplyAdd(Avx.LoadVector256(inputBase + inputRow + (kx * channels) + channels * 2), weightsVector, acc2);
                                        acc3 = Fma.MultiplyAdd(Avx.LoadVector256(inputBase + inputRow + (kx * channels) + channels * 3), weightsVector, acc3);
                                        packedTap++;
                                    }
                                }
                                Avx.Store(outputBase + rowOutput + x4 * channels + co, acc0);
                                Avx.Store(outputBase + rowOutput + (x4 + 1) * channels + co, acc1);
                                Avx.Store(outputBase + rowOutput + (x4 + 2) * channels + co, acc2);
                                Avx.Store(outputBase + rowOutput + (x4 + 3) * channels + co, acc3);
                            }
                        }
                        for (; x4 < width; x4++)
                            for (int block = 0; block < blocks; block++)
                                DepthwisePackedPixel(inputBase, packedBase, biasBase,
                                    outputBase + rowOutput + x4 * channels, rowInput, channels,
                                    width, y, x4, kyMin, kyMax, block);
                    }
                };
                if (computeWorkers > 1) Parallel.For(0, computeWorkers, computeWorker);
                else computeWorker(0);

                // Restore NCHW with a streaming channel pass.
                for (int b = 0; b < batch; b++)
                {
                    float* srcBase = outputWork + (long)b * plane * channels;
                    float* dstBase = outputPtr + (long)b * plane * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        float* dst = dstBase + (long)c * plane;
                        for (int s = 0; s < plane; s++)
                            dst[s] = srcBase[(long)s * channels + c];
                    }
                }
            }
            return true;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(inputNhwc);
            ArrayPool<float>.Shared.Return(outputNhwc);
        }
#else
        return false;
#endif
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void DepthwiseStride1ChannelUnsafe(float* input, float* weights,
        float bias, float* output, int height, int width, int outputHeight, int outputWidth,
        int kernelH, int kernelW, int padTop, int padLeft, int xStart, int xEnd)
    {
        Vector256<float> vBias = Vector256.Create(bias);
        for (int y = 0; y < outputHeight; y++)
        {
            int kyMin = Math.Max(0, padTop - y);
            int kyMax = Math.Min(kernelH, height - y + padTop);
            if (kyMax <= kyMin)
            {
                for (int x = 0; x < outputWidth; x++)
                    DepthwiseEdgePixel(input, weights, bias, output, height, width,
                        outputWidth, kernelH, kernelW, padTop, padLeft, y, x);
                continue;
            }
            for (int x = 0; x < xStart; x++)
                DepthwiseEdgePixel(input, weights, bias, output, height, width,
                    outputWidth, kernelH, kernelW, padTop, padLeft, y, x);
            int kyCount = kyMax - kyMin;
            float* rowBase = input + (long)(y - padTop + kyMin) * width - padLeft;
            float* weightBase = weights + kyMin * kernelW;
            int x32 = xStart;
            for (; x32 <= xEnd - 32; x32 += 32)
            {
                Vector256<float> a0 = vBias, a1 = vBias, a2 = vBias, a3 = vBias;
                float* tapRow = rowBase + x32;
                float* wc = weightBase;
                for (int ky = 0; ky < kyCount; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                    {
                        Vector256<float> weight = Vector256.Create(wc[kx]);
                        float* tap = tapRow + kx;
                        a0 = AddMul(a0, Avx.LoadVector256(tap), weight);
                        a1 = AddMul(a1, Avx.LoadVector256(tap + 8), weight);
                        a2 = AddMul(a2, Avx.LoadVector256(tap + 16), weight);
                        a3 = AddMul(a3, Avx.LoadVector256(tap + 24), weight);
                    }
                    tapRow += width;
                    wc += kernelW;
                }
                float* dst = output + y * outputWidth + x32;
                Avx.Store(dst, a0); Avx.Store(dst + 8, a1);
                Avx.Store(dst + 16, a2); Avx.Store(dst + 24, a3);
            }
            int x16 = x32;
            for (; x16 <= xEnd - 16; x16 += 16)
            {
                Vector256<float> a0 = vBias, a1 = vBias;
                float* tapRow = rowBase + x16;
                float* wc = weightBase;
                for (int ky = 0; ky < kyCount; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                    {
                        Vector256<float> weight = Vector256.Create(wc[kx]);
                        a0 = AddMul(a0, Avx.LoadVector256(tapRow + kx), weight);
                        a1 = AddMul(a1, Avx.LoadVector256(tapRow + kx + 8), weight);
                    }
                    tapRow += width;
                    wc += kernelW;
                }
                Avx.Store(output + y * outputWidth + x16, a0);
                Avx.Store(output + y * outputWidth + x16 + 8, a1);
            }
            for (; x16 <= xEnd - 8; x16 += 8)
            {
                Vector256<float> a0 = vBias;
                float* tapRow = rowBase + x16;
                float* wc = weightBase;
                for (int ky = 0; ky < kyCount; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                        a0 = AddMul(a0, Avx.LoadVector256(tapRow + kx), Vector256.Create(wc[kx]));
                    tapRow += width;
                    wc += kernelW;
                }
                Avx.Store(output + y * outputWidth + x16, a0);
            }
            for (int x = x16; x < outputWidth; x++)
                DepthwiseEdgePixel(input, weights, bias, output, height, width,
                    outputWidth, kernelH, kernelW, padTop, padLeft, y, x);
        }
    }
}
