using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Depthwise
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Depthwise3x3OutputVectorUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int channels, int height, int width)
    {
        int plane = checked(height * width);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int c = 0; c < channels; c++)
            {
                float* source = inputPtr + c * plane;
                float* destination = outputPtr + c * plane;
                float initial = biasPtr == null ? 0f : biasPtr[c];
                Vector256<float> initialVector = Vector256.Create(initial);
                int weightBase = c * 9;
                for (int y = 0; y < height; y++)
                {
                    int row = y * width, x = 0;
                    for (; x < width && x < 1; x++)
                        Depthwise3x3Scalar(source, destination, weightsPtr + weightBase,
                            initial, width, height, x, y);
                    if (y > 0 && y + 1 < height)
                    {
                        for (; x + 7 <= width - 2; x += 8)
                        {
                            Vector256<float> value = initialVector;
                            float* row0 = source + (y - 1) * width + x - 1;
                            float* row1 = row0 + width, row2 = row1 + width;
                            value = AddMul(value, Avx.LoadVector256(row0), weightsPtr[weightBase]);
                            value = AddMul(value, Avx.LoadVector256(row0 + 1), weightsPtr[weightBase + 1]);
                            value = AddMul(value, Avx.LoadVector256(row0 + 2), weightsPtr[weightBase + 2]);
                            value = AddMul(value, Avx.LoadVector256(row1), weightsPtr[weightBase + 3]);
                            value = AddMul(value, Avx.LoadVector256(row1 + 1), weightsPtr[weightBase + 4]);
                            value = AddMul(value, Avx.LoadVector256(row1 + 2), weightsPtr[weightBase + 5]);
                            value = AddMul(value, Avx.LoadVector256(row2), weightsPtr[weightBase + 6]);
                            value = AddMul(value, Avx.LoadVector256(row2 + 1), weightsPtr[weightBase + 7]);
                            value = AddMul(value, Avx.LoadVector256(row2 + 2), weightsPtr[weightBase + 8]);
                            Avx.Store(destination + row + x, value);
                        }
                    }
                    for (; x < width; x++)
                        Depthwise3x3Scalar(source, destination, weightsPtr + weightBase,
                            initial, width, height, x, y);
                }
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Depthwise5x5OutputVectorUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int channels, int height, int width)
    {
        int plane = checked(height * width);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int c = 0; c < channels; c++)
            {
                float* source = inputPtr + c * plane;
                float* destination = outputPtr + c * plane;
                float initial = biasPtr == null ? 0f : biasPtr[c];
                Vector256<float> initialVector = Vector256.Create(initial);
                int weightBase = c * 25;
                for (int y = 0; y < height; y++)
                {
                    int row = y * width, x = 0;
                    for (; x < width && x < 2; x++)
                        Depthwise5x5Scalar(source, destination, weightsPtr + weightBase,
                            initial, width, height, x, y);
                    if (y >= 2 && y + 2 < height)
                    {
                        for (; x + 7 <= width - 3; x += 8)
                        {
                            Vector256<float> value = initialVector;
                            for (int ky = 0; ky < 5; ky++)
                            {
                                float* sourceRow = source + (y + ky - 2) * width + x - 2;
                                float* kernel = weightsPtr + weightBase + ky * 5;
                                value = AddMul(value, Avx.LoadVector256(sourceRow), kernel[0]);
                                value = AddMul(value, Avx.LoadVector256(sourceRow + 1), kernel[1]);
                                value = AddMul(value, Avx.LoadVector256(sourceRow + 2), kernel[2]);
                                value = AddMul(value, Avx.LoadVector256(sourceRow + 3), kernel[3]);
                                value = AddMul(value, Avx.LoadVector256(sourceRow + 4), kernel[4]);
                            }
                            Avx.Store(destination + row + x, value);
                        }
                    }
                    for (; x < width; x++)
                        Depthwise5x5Scalar(source, destination, weightsPtr + weightBase,
                            initial, width, height, x, y);
                }
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Depthwise5x5Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int height, int width)
    {
        int plane = checked(height * width);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    float* source = inputPtr + (b * channels + c) * plane;
                    float* destination = outputPtr + (b * channels + c) * plane;
                    float initial = biasPtr == null ? 0f : biasPtr[c];
                    Vector256<float> initialVector = Vector256.Create(initial);
                    int i = 0;
                    for (; i <= plane - 8; i += 8) Avx.Store(destination + i, initialVector);
                    for (; i < plane; i++) destination[i] = initial;
                    int weightBase = c * 25;
                    for (int ky = 0; ky < 5; ky++)
                    {
                        int yBegin = ky < 2 ? 2 - ky : 0;
                        int yEnd = ky > 2 ? height - (ky - 2) : height;
                        for (int kx = 0; kx < 5; kx++)
                        {
                            int xBegin = kx < 2 ? 2 - kx : 0;
                            int xEnd = kx > 2 ? width - (kx - 2) : width;
                            float weight = weightsPtr[weightBase + ky * 5 + kx];
                            Vector256<float> weightVector = Vector256.Create(weight);
                            for (int y = yBegin; y < yEnd; y++)
                            {
                                int row = y * width;
                                float* sourceRow = source + (y + ky - 2) * width;
                                float* destinationRow = destination + row;
                                int x = xBegin;
                                for (; x <= xEnd - 16; x += 16)
                                {
                                    Vector256<float> currentLow = Avx.LoadVector256(destinationRow + x);
                                    Vector256<float> currentHigh = Avx.LoadVector256(destinationRow + x + 8);
                                    Vector256<float> valueLow = Avx.LoadVector256(sourceRow + x + kx - 2);
                                    Vector256<float> valueHigh = Avx.LoadVector256(sourceRow + x + kx + 6);
                                    Avx.Store(destinationRow + x, AddMul(currentLow, valueLow, weightVector));
                                    Avx.Store(destinationRow + x + 8, AddMul(currentHigh, valueHigh, weightVector));
                                }
                                for (; x <= xEnd - 8; x += 8)
                                {
                                    Vector256<float> current = Avx.LoadVector256(destinationRow + x);
                                    Vector256<float> value = Avx.LoadVector256(sourceRow + x + kx - 2);
                                    Avx.Store(destinationRow + x, AddMul(current, value, weightVector));
                                }
                                for (; x < xEnd; x++)
                                    destinationRow[x] += sourceRow[x + kx - 2] * weight;
                            }
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Depthwise7x7Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int channels, int height, int width)
    {
        int plane = checked(height * width);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                {
                    float* source = inputPtr + (b * channels + c) * plane;
                    float* destination = outputPtr + (b * channels + c) * plane;
                    float initial = biasPtr == null ? 0f : biasPtr[c];
                    float* kernel = weightsPtr + c * 49;
                    Vector256<float> initialVector = Vector256.Create(initial);
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        for (; x < width && x < 3; x++)
                            Depthwise7x7Scalar(source, destination, kernel, initial,
                                width, height, x, y);
                        if (y >= 3 && y + 3 < height)
                        {
                            for (; x + 7 <= width - 4; x += 8)
                            {
                                Vector256<float> value = initialVector;
                                for (int ky = 0; ky < 7; ky++)
                                {
                                    float* sourceRow = source + (y + ky - 3) * width + x - 3;
                                    float* kernelRow = kernel + ky * 7;
                                    value = AddMul(value, Avx.LoadVector256(sourceRow), kernelRow[0]);
                                    value = AddMul(value, Avx.LoadVector256(sourceRow + 1), kernelRow[1]);
                                    value = AddMul(value, Avx.LoadVector256(sourceRow + 2), kernelRow[2]);
                                    value = AddMul(value, Avx.LoadVector256(sourceRow + 3), kernelRow[3]);
                                    value = AddMul(value, Avx.LoadVector256(sourceRow + 4), kernelRow[4]);
                                    value = AddMul(value, Avx.LoadVector256(sourceRow + 5), kernelRow[5]);
                                    value = AddMul(value, Avx.LoadVector256(sourceRow + 6), kernelRow[6]);
                                }
                                Avx.Store(destination + row + x, value);
                            }
                        }
                        for (; x < width; x++)
                            Depthwise7x7Scalar(source, destination, kernel, initial,
                                width, height, x, y);
                    }
                }
        }
    }
}
