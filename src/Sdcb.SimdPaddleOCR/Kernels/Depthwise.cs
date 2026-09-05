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

internal static partial class Depthwise
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static bool Try3x3(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels, int height, int width)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            if (batch == 1 && height >= 50 && width >= 10)
            {
                Depthwise3x3OutputVectorAvx512Unsafe(input, weights, bias, output, channels, height, width);
                return true;
            }
            int plane = checked(height * width);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = (b * channels + c) * plane;
                    ReadOnlySpan<float> source = input.Slice(channelOffset, plane);
                    Span<float> destination = output.Slice(channelOffset, plane);
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    Vector512<float> initialVector = Vector512.Create(initial);
                    int i = 0;
                    for (; i <= plane - 16; i += 16) Store512(destination, i, initialVector);
                    for (; i < plane; i++) destination[i] = initial;
                    int weightBase = c * 9;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int yBegin = ky == 0 ? 1 : 0;
                        int yEnd = ky == 2 ? height - 1 : height;
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int xBegin = kx == 0 ? 1 : 0;
                            int xEnd = kx == 2 ? width - 1 : width;
                            float weight = weights[weightBase + ky * 3 + kx];
                            for (int y = yBegin; y < yEnd; y++)
                            {
                                int row = y * width;
                                int sourceRow = (y + ky - 1) * width;
                                int x = xBegin;
                                Vector512<float> vectorWeight = Vector512.Create(weight);
                                for (; x <= xEnd - 16; x += 16)
                                {
                                    Vector512<float> current = Load512(destination, row + x);
                                    Vector512<float> value = Load512(source, sourceRow + x + kx - 1);
                                    Store512(destination, row + x, AddMul512(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                    destination[row + x] += source[sourceRow + x + kx - 1] * weight;
                            }
                        }
                    }
                }
            return true;
        }
        else if (Avx.IsSupported)
        {
            if (batch == 1 && height >= 50 && width >= 10)
            {
                Depthwise3x3OutputVectorUnsafe(input, weights, bias, output, channels, height, width);
                return true;
            }
            int plane = checked(height * width);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = (b * channels + c) * plane;
                    ReadOnlySpan<float> source = input.Slice(channelOffset, plane);
                    Span<float> destination = output.Slice(channelOffset, plane);
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    Vector256<float> initialVector = Vector256.Create(initial);
                    int i = 0;
                    for (; i <= plane - 8; i += 8) Store(destination, i, initialVector);
                    for (; i < plane; i++) destination[i] = initial;
                    int weightBase = c * 9;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int yBegin = ky == 0 ? 1 : 0;
                        int yEnd = ky == 2 ? height - 1 : height;
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int xBegin = kx == 0 ? 1 : 0;
                            int xEnd = kx == 2 ? width - 1 : width;
                            float weight = weights[weightBase + ky * 3 + kx];
                            for (int y = yBegin; y < yEnd; y++)
                            {
                                int row = y * width;
                                int sourceRow = (y + ky - 1) * width;
                                int x = xBegin;
                                Vector256<float> vectorWeight = Vector256.Create(weight);
                                for (; x <= xEnd - 8; x += 8)
                                {
                                    Vector256<float> current = Load(destination, row + x);
                                    Vector256<float> value = Load(source, sourceRow + x + kx - 1);
                                    Store(destination, row + x, AddMul(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                    destination[row + x] += source[sourceRow + x + kx - 1] * weight;
                            }
                        }
                    }
                }
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return Try3x3Vector(input, weights, bias, output, batch, channels, height, width);
        }
        Depthwise3x3Scalar(input, weights, bias, output, batch, channels, height, width);
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool Try5x5(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels, int height, int width,
        int intraOpThreads = 1)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            // The C runtime shards depthwise convolutions by output channel.  This
            // matters for the detector's 64-channel 5x5 blocks (roughly 77M MACs)
            // when the four-worker OCR configuration is used.  Keep the scalar
            // path and small tensors single-threaded to avoid nested scheduling
            // overhead.
            long work = (long)channels * height * width * 25;
            if (CanShardChannels(intraOpThreads, batch, channels, work))
            {
                int workers = Math.Min(intraOpThreads, channels);
                fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
                {
                    nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                        biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                    int inputLength = input.Length, weightsLength = weights.Length,
                        biasLength = bias.Length, outputLength = output.Length;
                    int plane = checked(height * width);
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = channels * worker / workers;
                        int end = channels * (worker + 1) / workers;
                        int count = end - begin;
                        if (count <= 0) return;
                        ReadOnlySpan<float> inSpan = new ReadOnlySpan<float>((void*)inputAddress, inputLength)
                            .Slice(begin * plane, count * plane);
                        ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                            .Slice(begin * 25, count * 25);
                        ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                            : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                            .Slice(begin * plane, count * plane);
                        Depthwise5x5Avx512Unsafe(inSpan, weightSpan, biasSpan, outSpan, 1, count, height, width);
                    });
                }
                return true;
            }
            if (batch == 1 && height >= 10 && width >= 10)
            {
                Depthwise5x5OutputVectorAvx512Unsafe(input, weights, bias, output, channels, height, width);
                return true;
            }
            Depthwise5x5Avx512Unsafe(input, weights, bias, output, batch, channels, height, width);
            return true;
        }
        else if (Avx.IsSupported)
        {
            // The C runtime shards depthwise convolutions by output channel.  This
            // matters for the detector's 64-channel 5x5 blocks (roughly 77M MACs)
            // when the four-worker OCR configuration is used.  Keep the scalar
            // path and small tensors single-threaded to avoid nested scheduling
            // overhead.
            long work = (long)channels * height * width * 25;
            if (CanShardChannels(intraOpThreads, batch, channels, work))
            {
                int workers = Math.Min(intraOpThreads, channels);
                fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
                {
                    nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                        biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                    int inputLength = input.Length, weightsLength = weights.Length,
                        biasLength = bias.Length, outputLength = output.Length;
                    int plane = checked(height * width);
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = channels * worker / workers;
                        int end = channels * (worker + 1) / workers;
                        int count = end - begin;
                        if (count <= 0) return;
                        ReadOnlySpan<float> inSpan = new ReadOnlySpan<float>((void*)inputAddress, inputLength)
                            .Slice(begin * plane, count * plane);
                        ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                            .Slice(begin * 25, count * 25);
                        ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                            : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                            .Slice(begin * plane, count * plane);
                        Depthwise5x5Unsafe(inSpan, weightSpan, biasSpan, outSpan, 1, count, height, width);
                    });
                }
                return true;
            }
            if (batch == 1 && height >= 10 && width >= 10)
            {
                Depthwise5x5OutputVectorUnsafe(input, weights, bias, output, channels, height, width);
                return true;
            }
            Depthwise5x5Unsafe(input, weights, bias, output, batch, channels, height, width);
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
            return Try5x5Vector(input, weights, bias, output, batch, channels, height, width, intraOpThreads);
        Depthwise5x5Scalar(input, weights, bias, output, batch, channels, height, width);
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool Try7x7(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels, int height, int width,
        int intraOpThreads = 1)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            long work = (long)channels * height * width * 49;
            if (CanShardChannels(intraOpThreads, batch, channels, work))
            {
                int workers = Math.Min(intraOpThreads, channels);
                fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
                {
                    nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                        biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                    int inputLength = input.Length, weightsLength = weights.Length,
                        biasLength = bias.Length, outputLength = output.Length;
                    int plane = checked(height * width);
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = channels * worker / workers;
                        int end = channels * (worker + 1) / workers;
                        int count = end - begin;
                        if (count <= 0) return;
                        ReadOnlySpan<float> inSpan = new ReadOnlySpan<float>((void*)inputAddress, inputLength)
                            .Slice(begin * plane, count * plane);
                        ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                            .Slice(begin * 49, count * 49);
                        ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                            : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                            .Slice(begin * plane, count * plane);
                        Depthwise7x7Avx512Unsafe(inSpan, weightSpan, biasSpan, outSpan,
                            1, count, height, width);
                    });
                }
                return true;
            }
            Depthwise7x7Avx512Unsafe(input, weights, bias, output, batch, channels, height, width);
            return true;
        }
        else if (Avx.IsSupported)
        {
            long work = (long)channels * height * width * 49;
            if (CanShardChannels(intraOpThreads, batch, channels, work))
            {
                int workers = Math.Min(intraOpThreads, channels);
                fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
                {
                    nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                        biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                    int inputLength = input.Length, weightsLength = weights.Length,
                        biasLength = bias.Length, outputLength = output.Length;
                    int plane = checked(height * width);
                    Parallel.For(0, workers, worker =>
                    {
                        int begin = channels * worker / workers;
                        int end = channels * (worker + 1) / workers;
                        int count = end - begin;
                        if (count <= 0) return;
                        ReadOnlySpan<float> inSpan = new ReadOnlySpan<float>((void*)inputAddress, inputLength)
                            .Slice(begin * plane, count * plane);
                        ReadOnlySpan<float> weightSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                            .Slice(begin * 49, count * 49);
                        ReadOnlySpan<float> biasSpan = biasLength == 0 ? []
                            : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                        Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                            .Slice(begin * plane, count * plane);
                        Depthwise7x7Unsafe(inSpan, weightSpan, biasSpan, outSpan,
                            1, count, height, width);
                    });
                }
                return true;
            }
            Depthwise7x7Unsafe(input, weights, bias, output, batch, channels, height, width);
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
            return Try7x7Vector(input, weights, bias, output, batch, channels, height, width,
                intraOpThreads);
        return false;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static bool Try3x3Stride2(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels,
        int inputHeight, int inputWidth, int outputHeight, int outputWidth)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int inputOffset = (b * channels + c) * inputPlane;
                    int outputOffset = (b * channels + c) * outputPlane;
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    for (int i = 0; i < outputPlane; i++) output[outputOffset + i] = initial;
                    int wb = c * 9;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int yBegin = ky == 0 ? 1 : 0;
                        // With stride two the final output row can still consume the
                        // final input row (for ky==2), so unlike the unit-stride
                        // kernel the end is not outputHeight-1.
                        int yEnd = outputHeight;
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int xBegin = kx == 0 ? 1 : 0;
                            // The rightmost output column may map to the last input
                            // column at stride two; leave bounds checking to the
                            // scalar tail instead of dropping that column entirely.
                            int xEnd = outputWidth;
                            float weight = weights[wb + ky * 3 + kx];
                            for (int oy = yBegin; oy < yEnd; oy++)
                            {
                                int sourceY = oy * 2 - 1 + ky;
                                if ((uint)sourceY >= (uint)inputHeight) continue;
                                int row = oy * outputWidth;
                                int sourceRow = inputOffset + sourceY * inputWidth;
                                int x = xBegin;
                                int vectorEnd = xEnd;
                                while (vectorEnd - x >= 8 && 2 * (vectorEnd - 1) - 1 + kx + 14 >= inputWidth) vectorEnd--;
                                for (; x <= vectorEnd - 16; x += 16)
                                {
                                    int sourceX = 2 * x - 1 + kx;
                                    Vector512<float> current = Load512(output, outputOffset + row + x);
                                    Vector512<float> value = LoadStride2512(input.Slice(sourceRow, inputWidth), sourceX);
                                    Store512(output, outputOffset + row + x, AddMul512(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                {
                                    int sourceX = 2 * x - 1 + kx;
                                    if ((uint)sourceX < (uint)inputWidth)
                                        output[outputOffset + row + x] += input[sourceRow + sourceX] * weight;
                                }
                            }
                        }
                    }
                }
            return true;
        }
        else if (Avx.IsSupported)
        {
            int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int inputOffset = (b * channels + c) * inputPlane;
                    int outputOffset = (b * channels + c) * outputPlane;
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    for (int i = 0; i < outputPlane; i++) output[outputOffset + i] = initial;
                    int wb = c * 9;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int yBegin = ky == 0 ? 1 : 0;
                        // With stride two the final output row can still consume the
                        // final input row (for ky==2), so unlike the unit-stride
                        // kernel the end is not outputHeight-1.
                        int yEnd = outputHeight;
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int xBegin = kx == 0 ? 1 : 0;
                            // The rightmost output column may map to the last input
                            // column at stride two; leave bounds checking to the
                            // scalar tail instead of dropping that column entirely.
                            int xEnd = outputWidth;
                            float weight = weights[wb + ky * 3 + kx];
                            for (int oy = yBegin; oy < yEnd; oy++)
                            {
                                int sourceY = oy * 2 - 1 + ky;
                                if ((uint)sourceY >= (uint)inputHeight) continue;
                                int row = oy * outputWidth;
                                int sourceRow = inputOffset + sourceY * inputWidth;
                                int x = xBegin;
                                int vectorEnd = xEnd;
                                while (vectorEnd - x >= 8 && 2 * (vectorEnd - 1) - 1 + kx + 14 >= inputWidth) vectorEnd--;
                                for (; x <= vectorEnd - 8; x += 8)
                                {
                                    int sourceX = 2 * x - 1 + kx;
                                    Vector256<float> current = Load(output, outputOffset + row + x);
                                    Vector256<float> value = LoadStride2(input.Slice(sourceRow, inputWidth), sourceX);
                                    Store(output, outputOffset + row + x, AddMul(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                {
                                    int sourceX = 2 * x - 1 + kx;
                                    if ((uint)sourceX < (uint)inputWidth)
                                        output[outputOffset + row + x] += input[sourceRow + sourceX] * weight;
                                }
                            }
                        }
                    }
                }
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return Try3x3Stride2Vector(input, weights, bias, output, batch, channels,
                inputHeight, inputWidth, outputHeight, outputWidth);
        }
        Depthwise3x3Stride2Scalar(input, weights, bias, output, batch, channels,
            inputHeight, inputWidth, outputHeight, outputWidth);
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static bool Try3x3StrideHeight2(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int inputHeight, int inputWidth,
        int outputHeight, int outputWidth)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int inputOffset = (b * channels + c) * inputPlane;
                    int outputOffset = (b * channels + c) * outputPlane;
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    for (int i = 0; i < outputPlane; i++) output[outputOffset + i] = initial;
                    int wb = c * 9;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int yBegin = ky == 0 ? 1 : 0, yEnd = outputHeight;
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int xBegin = kx == 0 ? 1 : 0, xEnd = kx == 2 ? outputWidth - 1 : outputWidth;
                            float weight = weights[wb + ky * 3 + kx];
                            for (int oy = yBegin; oy < yEnd; oy++)
                            {
                                int sourceY = oy * 2 - 1 + ky;
                                if ((uint)sourceY >= (uint)inputHeight) continue;
                                int row = oy * outputWidth, sourceRow = inputOffset + sourceY * inputWidth;
                                int x = xBegin;
                                for (; x <= xEnd - 16; x += 16)
                                {
                                    Vector512<float> current = Load512(output, outputOffset + row + x);
                                    Vector512<float> value = Load512(input.Slice(sourceRow, inputWidth), x + kx - 1);
                                    Store512(output, outputOffset + row + x, AddMul512(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                    output[outputOffset + row + x] += input[sourceRow + x + kx - 1] * weight;
                            }
                        }
                    }
                }
            return true;
        }
        else if (Avx.IsSupported)
        {
            int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int inputOffset = (b * channels + c) * inputPlane;
                    int outputOffset = (b * channels + c) * outputPlane;
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    for (int i = 0; i < outputPlane; i++) output[outputOffset + i] = initial;
                    int wb = c * 9;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int yBegin = ky == 0 ? 1 : 0, yEnd = outputHeight;
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int xBegin = kx == 0 ? 1 : 0, xEnd = kx == 2 ? outputWidth - 1 : outputWidth;
                            float weight = weights[wb + ky * 3 + kx];
                            for (int oy = yBegin; oy < yEnd; oy++)
                            {
                                int sourceY = oy * 2 - 1 + ky;
                                if ((uint)sourceY >= (uint)inputHeight) continue;
                                int row = oy * outputWidth, sourceRow = inputOffset + sourceY * inputWidth;
                                int x = xBegin;
                                for (; x <= xEnd - 8; x += 8)
                                {
                                    Vector256<float> current = Load(output, outputOffset + row + x);
                                    Vector256<float> value = Load(input.Slice(sourceRow, inputWidth), x + kx - 1);
                                    Store(output, outputOffset + row + x, AddMul(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                    output[outputOffset + row + x] += input[sourceRow + x + kx - 1] * weight;
                            }
                        }
                    }
                }
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return Try3x3StrideHeight2Vector(input, weights, bias, output, batch, channels,
                inputHeight, inputWidth, outputHeight, outputWidth);
        }
        return false;
    }

    /// <summary>
    /// Depthwise 5×5 with stride (2,1) and pad (2,2). Used by small-model REC
    /// (e.g. g=64 in=64×5×80 → 64×3×80); without this path it falls to scalar Conv.
    /// Accumulation order matches unit-stride depthwise / scalar: bias, then ky→kx +=.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static bool Try5x5StrideHeight2(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int inputHeight, int inputWidth,
        int outputHeight, int outputWidth)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int inputOffset = (b * channels + c) * inputPlane;
                    int outputOffset = (b * channels + c) * outputPlane;
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    for (int i = 0; i < outputPlane; i++) output[outputOffset + i] = initial;
                    int wb = c * 25;
                    for (int ky = 0; ky < 5; ky++)
                    {
                        // sourceY = oy*2 - 2 + ky; ky<2 leaves oy=0 out of range.
                        int yBegin = ky < 2 ? 1 : 0, yEnd = outputHeight;
                        for (int kx = 0; kx < 5; kx++)
                        {
                            int xBegin = kx < 2 ? 2 - kx : 0;
                            int xEnd = kx > 2 ? outputWidth - (kx - 2) : outputWidth;
                            float weight = weights[wb + ky * 5 + kx];
                            for (int oy = yBegin; oy < yEnd; oy++)
                            {
                                int sourceY = oy * 2 - 2 + ky;
                                if ((uint)sourceY >= (uint)inputHeight) continue;
                                int row = oy * outputWidth, sourceRow = inputOffset + sourceY * inputWidth;
                                int x = xBegin;
                                for (; x <= xEnd - 16; x += 16)
                                {
                                    Vector512<float> current = Load512(output, outputOffset + row + x);
                                    Vector512<float> value = Load512(input.Slice(sourceRow, inputWidth), x + kx - 2);
                                    Store512(output, outputOffset + row + x, AddMul512(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                    output[outputOffset + row + x] += input[sourceRow + x + kx - 2] * weight;
                            }
                        }
                    }
                }
            return true;
        }
        else if (Avx.IsSupported)
        {
            int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int inputOffset = (b * channels + c) * inputPlane;
                    int outputOffset = (b * channels + c) * outputPlane;
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    for (int i = 0; i < outputPlane; i++) output[outputOffset + i] = initial;
                    int wb = c * 25;
                    for (int ky = 0; ky < 5; ky++)
                    {
                        int yBegin = ky < 2 ? 1 : 0, yEnd = outputHeight;
                        for (int kx = 0; kx < 5; kx++)
                        {
                            int xBegin = kx < 2 ? 2 - kx : 0;
                            int xEnd = kx > 2 ? outputWidth - (kx - 2) : outputWidth;
                            float weight = weights[wb + ky * 5 + kx];
                            for (int oy = yBegin; oy < yEnd; oy++)
                            {
                                int sourceY = oy * 2 - 2 + ky;
                                if ((uint)sourceY >= (uint)inputHeight) continue;
                                int row = oy * outputWidth, sourceRow = inputOffset + sourceY * inputWidth;
                                int x = xBegin;
                                for (; x <= xEnd - 8; x += 8)
                                {
                                    Vector256<float> current = Load(output, outputOffset + row + x);
                                    Vector256<float> value = Load(input.Slice(sourceRow, inputWidth), x + kx - 2);
                                    Store(output, outputOffset + row + x, AddMul(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                    output[outputOffset + row + x] += input[sourceRow + x + kx - 2] * weight;
                            }
                        }
                    }
                }
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return Try5x5StrideHeight2Vector(input, weights, bias, output, batch, channels,
                inputHeight, inputWidth, outputHeight, outputWidth);
        }
        return false;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static bool Try1x5(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels, int height, int width)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported)
        {
            int plane = checked(height * width);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int offset = (b * channels + c) * plane;
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    for (int i = 0; i < plane; i++) output[offset + i] = initial;
                    int wb = c * 5;
                    for (int kx = 0; kx < 5; kx++)
                    {
                        int xBegin = kx < 2 ? 2 - kx : 0;
                        int xEnd = kx > 2 ? width - (kx - 2) : width;
                        float weight = weights[wb + kx];
                        for (int y = 0; y < height; y++)
                        {
                            int row = y * width, x = xBegin;
                            for (; x <= xEnd - 16; x += 16)
                            {
                                Vector512<float> current = Load512(output, offset + row + x);
                                Vector512<float> value = Load512(input.Slice(offset + row, width), x + kx - 2);
                                Store512(output, offset + row + x, AddMul512(current, value, weight));
                            }
                            for (; x < xEnd; x++)
                                output[offset + row + x] += input[offset + row + x + kx - 2] * weight;
                        }
                    }
                }
            return true;
        }
        else if (Avx.IsSupported)
        {
            int plane = checked(height * width);
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    int offset = (b * channels + c) * plane;
                    float initial = bias.IsEmpty ? 0f : bias[c];
                    for (int i = 0; i < plane; i++) output[offset + i] = initial;
                    int wb = c * 5;
                    for (int kx = 0; kx < 5; kx++)
                    {
                        int xBegin = kx < 2 ? 2 - kx : 0;
                        int xEnd = kx > 2 ? width - (kx - 2) : width;
                        float weight = weights[wb + kx];
                        for (int y = 0; y < height; y++)
                        {
                            int row = y * width, x = xBegin;
                            for (; x <= xEnd - 8; x += 8)
                            {
                                Vector256<float> current = Load(output, offset + row + x);
                                Vector256<float> value = Load(input.Slice(offset + row, width), x + kx - 2);
                                Store(output, offset + row + x, AddMul(current, value, weight));
                            }
                            for (; x < xEnd; x++)
                                output[offset + row + x] += input[offset + row + x + kx - 2] * weight;
                        }
                    }
                }
            return true;
        }
        else
#endif
        if (Vector.IsHardwareAccelerated)
        {
            return Try1x5Vector(input, weights, bias, output, batch, channels, height, width);
        }
        return false;
    }
}
