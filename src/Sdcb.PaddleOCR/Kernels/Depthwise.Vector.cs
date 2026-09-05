using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Depthwise
{
    internal static bool Try3x3Vector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels, int height, int width)
    {
        int plane = checked(height * width);
        int widthLanes = Vector<float>.Count;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                int channelOffset = (b * channels + c) * plane;
                ReadOnlySpan<float> source = input.Slice(channelOffset, plane);
                Span<float> destination = output.Slice(channelOffset, plane);
                float initial = bias.IsEmpty ? 0f : bias[c];
                Vector<float> initialVector = new(initial);
                int i = 0;
                for (; i <= plane - widthLanes; i += widthLanes)
                    VectorStore(destination, i, initialVector);
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
                            for (; x <= xEnd - widthLanes; x += widthLanes)
                            {
                                Vector<float> current = VectorLoad(destination, row + x);
                                Vector<float> value = VectorLoad(source, sourceRow + x + kx - 1);
                                VectorStore(destination, row + x, VectorAddMul(current, value, weight));
                            }
                            for (; x < xEnd; x++)
                                destination[row + x] += source[sourceRow + x + kx - 1] * weight;
                        }
                    }
                }
            }
        return true;
    }

    internal static unsafe bool Try5x5Vector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int height, int width, int intraOpThreads)
    {
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
                    Try5x5Vector(inSpan, weightSpan, biasSpan, outSpan, 1, count, height, width, 1);
                });
            }
            return true;
        }

        int planeSize = checked(height * width);
        int widthLanes = Vector<float>.Count;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                int offset = (b * channels + c) * planeSize;
                ReadOnlySpan<float> source = input.Slice(offset, planeSize);
                Span<float> destination = output.Slice(offset, planeSize);
                float initial = bias.IsEmpty ? 0f : bias[c];
                Vector<float> initialVector = new(initial);
                int i = 0;
                for (; i <= planeSize - widthLanes; i += widthLanes)
                    VectorStore(destination, i, initialVector);
                for (; i < planeSize; i++) destination[i] = initial;
                int weightBase = c * 25;
                for (int ky = 0; ky < 5; ky++)
                {
                    int yBegin = ky < 2 ? 2 - ky : 0;
                    int yEnd = ky > 2 ? height - (ky - 2) : height;
                    for (int kx = 0; kx < 5; kx++)
                    {
                        int xBegin = kx < 2 ? 2 - kx : 0;
                        int xEnd = kx > 2 ? width - (kx - 2) : width;
                        float weight = weights[weightBase + ky * 5 + kx];
                        for (int y = yBegin; y < yEnd; y++)
                        {
                            int row = y * width;
                            int sourceRow = (y + ky - 2) * width;
                            int x = xBegin;
                            for (; x <= xEnd - widthLanes; x += widthLanes)
                            {
                                Vector<float> current = VectorLoad(destination, row + x);
                                Vector<float> value = VectorLoad(source, sourceRow + x + kx - 2);
                                VectorStore(destination, row + x, VectorAddMul(current, value, weight));
                            }
                            for (; x < xEnd; x++)
                                destination[row + x] += source[sourceRow + x + kx - 2] * weight;
                        }
                    }
                }
            }
        return true;
    }

    internal static unsafe bool Try7x7Vector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int height, int width, int intraOpThreads)
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
                    Try7x7Vector(inSpan, weightSpan, biasSpan, outSpan,
                        1, count, height, width, 1);
                });
            }
            return true;
        }

        int planeSize = checked(height * width);
        int widthLanes = Vector<float>.Count;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                int offset = (b * channels + c) * planeSize;
                ReadOnlySpan<float> source = input.Slice(offset, planeSize);
                Span<float> destination = output.Slice(offset, planeSize);
                float initial = bias.IsEmpty ? 0f : bias[c];
                Vector<float> initialVector = new(initial);
                int weightBase = c * 49;
                for (int y = 0; y < height; y++)
                {
                    int row = y * width, x = 0;
                    for (; x < width && x < 3; x++)
                        Depthwise7x7ScalarVector(source, destination, weights, weightBase,
                            initial, width, height, x, y);
                    if (y >= 3 && y + 3 < height)
                    {
                        for (; x + widthLanes - 1 <= width - 4; x += widthLanes)
                        {
                            Vector<float> value = initialVector;
                            for (int ky = 0; ky < 7; ky++)
                            {
                                int sourceRow = (y + ky - 3) * width + x - 3;
                                int kernelRow = weightBase + ky * 7;
                                value = VectorAddMul(value, VectorLoad(source, sourceRow), weights[kernelRow]);
                                value = VectorAddMul(value, VectorLoad(source, sourceRow + 1), weights[kernelRow + 1]);
                                value = VectorAddMul(value, VectorLoad(source, sourceRow + 2), weights[kernelRow + 2]);
                                value = VectorAddMul(value, VectorLoad(source, sourceRow + 3), weights[kernelRow + 3]);
                                value = VectorAddMul(value, VectorLoad(source, sourceRow + 4), weights[kernelRow + 4]);
                                value = VectorAddMul(value, VectorLoad(source, sourceRow + 5), weights[kernelRow + 5]);
                                value = VectorAddMul(value, VectorLoad(source, sourceRow + 6), weights[kernelRow + 6]);
                            }
                            VectorStore(destination, row + x, value);
                        }
                    }
                    for (; x < width; x++)
                        Depthwise7x7ScalarVector(source, destination, weights, weightBase,
                            initial, width, height, x, y);
                }
            }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Depthwise7x7ScalarVector(ReadOnlySpan<float> source, Span<float> destination,
        ReadOnlySpan<float> weights, int weightBase, float initial, int width, int height, int x, int y)
    {
        float value = initial;
        for (int ky = 0; ky < 7; ky++)
        {
            int iy = y + ky - 3;
            if ((uint)iy >= (uint)height) continue;
            int sourceRow = iy * width;
            for (int kx = 0; kx < 7; kx++)
            {
                int ix = x + kx - 3;
                if ((uint)ix < (uint)width)
                    value += source[sourceRow + ix] * weights[weightBase + ky * 7 + kx];
            }
        }
        destination[y * width + x] = value;
    }

    internal static bool Try3x3Stride2Vector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int inputHeight, int inputWidth, int outputHeight, int outputWidth)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        int widthLanes = Vector<float>.Count;
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
                    int yEnd = outputHeight;
                    for (int kx = 0; kx < 3; kx++)
                    {
                        int xBegin = kx == 0 ? 1 : 0;
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
                            while (vectorEnd - x >= widthLanes &&
                                2 * (vectorEnd - 1) - 1 + kx + (widthLanes * 2 - 2) >= inputWidth)
                                vectorEnd--;
                            for (; x <= vectorEnd - widthLanes; x += widthLanes)
                            {
                                int sourceX = 2 * x - 1 + kx;
                                Vector<float> current = VectorLoad(output, outputOffset + row + x);
                                Vector<float> value = VectorLoadStride2(input.Slice(sourceRow, inputWidth), sourceX);
                                VectorStore(output, outputOffset + row + x, VectorAddMul(current, value, weight));
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

    internal static bool Try3x3StrideHeight2Vector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int inputHeight, int inputWidth, int outputHeight, int outputWidth)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        int widthLanes = Vector<float>.Count;
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
                            for (; x <= xEnd - widthLanes; x += widthLanes)
                            {
                                Vector<float> current = VectorLoad(output, outputOffset + row + x);
                                Vector<float> value = VectorLoad(input.Slice(sourceRow, inputWidth), x + kx - 1);
                                VectorStore(output, outputOffset + row + x, VectorAddMul(current, value, weight));
                            }
                            for (; x < xEnd; x++)
                                output[outputOffset + row + x] += input[sourceRow + x + kx - 1] * weight;
                        }
                    }
                }
            }
        return true;
    }

    internal static bool Try5x5StrideHeight2Vector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int inputHeight, int inputWidth, int outputHeight, int outputWidth)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        int widthLanes = Vector<float>.Count;
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
                            for (; x <= xEnd - widthLanes; x += widthLanes)
                            {
                                Vector<float> current = VectorLoad(output, outputOffset + row + x);
                                Vector<float> value = VectorLoad(input.Slice(sourceRow, inputWidth), x + kx - 2);
                                VectorStore(output, outputOffset + row + x, VectorAddMul(current, value, weight));
                            }
                            for (; x < xEnd; x++)
                                output[outputOffset + row + x] += input[sourceRow + x + kx - 2] * weight;
                        }
                    }
                }
            }
        return true;
    }

    internal static bool Try1x5Vector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels, int height, int width)
    {
        int plane = checked(height * width);
        int widthLanes = Vector<float>.Count;
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
                        for (; x <= xEnd - widthLanes; x += widthLanes)
                        {
                            Vector<float> current = VectorLoad(output, offset + row + x);
                            Vector<float> value = VectorLoad(input.Slice(offset + row, width), x + kx - 2);
                            VectorStore(output, offset + row + x, VectorAddMul(current, value, weight));
                        }
                        for (; x < xEnd; x++)
                            output[offset + row + x] += input[offset + row + x + kx - 2] * weight;
                    }
                }
            }
        return true;
    }
}
