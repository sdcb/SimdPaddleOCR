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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Depthwise3x3Scalar(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels, int height, int width)
    {
        int plane = checked(height * width);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                int channelOffset = (b * channels + c) * plane;
                ReadOnlySpan<float> source = input.Slice(channelOffset, plane);
                Span<float> destination = output.Slice(channelOffset, plane);
                float initial = bias.IsEmpty ? 0f : bias[c];
                destination.Fill(initial);
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
                            for (int x = xBegin; x < xEnd; x++)
                                destination[row + x] += source[sourceRow + x + kx - 1] * weight;
                        }
                    }
                }
            }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Depthwise5x5Scalar(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels, int height, int width)
    {
        int plane = checked(height * width);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                int offset = (b * channels + c) * plane;
                ReadOnlySpan<float> source = input.Slice(offset, plane);
                Span<float> destination = output.Slice(offset, plane);
                destination.Fill(bias.IsEmpty ? 0f : bias[c]);
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
                            for (int x = xBegin; x < xEnd; x++)
                                destination[row + x] += source[sourceRow + x + kx - 2] * weight;
                        }
                    }
                }
            }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Depthwise3x3Scalar(float* source, float* destination,
        float* kernel, float initial, int width, int height, int x, int y)
    {
        float value = initial;
        for (int ky = 0; ky < 3; ky++)
        {
            int iy = y + ky - 1;
            if ((uint)iy >= (uint)height) continue;
            float* sourceRow = source + iy * width;
            for (int kx = 0; kx < 3; kx++)
            {
                int ix = x + kx - 1;
                if ((uint)ix < (uint)width) value += sourceRow[ix] * kernel[ky * 3 + kx];
            }
        }
        destination[y * width + x] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Depthwise5x5Scalar(float* source, float* destination,
        float* kernel, float initial, int width, int height, int x, int y)
    {
        float value = initial;
        for (int ky = 0; ky < 5; ky++)
        {
            int iy = y + ky - 2;
            if ((uint)iy >= (uint)height) continue;
            float* sourceRow = source + iy * width;
            for (int kx = 0; kx < 5; kx++)
            {
                int ix = x + kx - 2;
                if ((uint)ix < (uint)width) value += sourceRow[ix] * kernel[ky * 5 + kx];
            }
        }
        destination[y * width + x] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Depthwise7x7Scalar(float* source, float* destination,
        float* kernel, float initial, int width, int height, int x, int y)
    {
        float value = initial;
        for (int ky = 0; ky < 7; ky++)
        {
            int iy = y + ky - 3;
            if ((uint)iy >= (uint)height) continue;
            float* sourceRow = source + iy * width;
            for (int kx = 0; kx < 7; kx++)
            {
                int ix = x + kx - 3;
                if ((uint)ix < (uint)width)
                    value += sourceRow[ix] * kernel[ky * 7 + kx];
            }
        }
        destination[y * width + x] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Depthwise3x3Stride2Scalar(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int channels,
        int inputHeight, int inputWidth, int outputHeight, int outputWidth)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
            {
                int inputOffset = (b * channels + c) * inputPlane;
                int outputOffset = (b * channels + c) * outputPlane;
                float initial = bias.IsEmpty ? 0f : bias[c];
                output.Slice(outputOffset, outputPlane).Fill(initial);
                int wb = c * 9;
                for (int ky = 0; ky < 3; ky++)
                    for (int kx = 0; kx < 3; kx++)
                    {
                        float weight = weights[wb + ky * 3 + kx];
                        for (int oy = 0; oy < outputHeight; oy++)
                        {
                            int sourceY = oy * 2 - 1 + ky;
                            if ((uint)sourceY >= (uint)inputHeight) continue;
                            int row = oy * outputWidth;
                            int sourceRow = inputOffset + sourceY * inputWidth;
                            for (int ox = 0; ox < outputWidth; ox++)
                            {
                                int sourceX = ox * 2 - 1 + kx;
                                if ((uint)sourceX < (uint)inputWidth)
                                    output[outputOffset + row + ox] += input[sourceRow + sourceX] * weight;
                            }
                        }
                    }
            }
    }
}
