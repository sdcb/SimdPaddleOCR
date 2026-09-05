using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Stride2
{
    internal static bool Try(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels)
    {
        int plane = checked(height * width);
        if (Avx.IsSupported)
        {
            if ((outputChannels & 7) == 0)
            {
                Conv2x2PadEndEightOutputsUnsafe(input, weights, bias, output, batch, inputChannels,
                    height, width, outputChannels);
                return true;
            }
            if ((outputChannels & 3) == 0)
            {
                Conv2x2PadEndFourOutputsUnsafe(input, weights, bias, output, batch, inputChannels,
                    height, width, outputChannels);
                return true;
            }
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co++)
                {
                    int outputOffset = (b * outputChannels + co) * plane;
                    float initial = bias.IsEmpty ? 0f : bias[co];
                    Vector256<float> initialVector = Vector256.Create(initial);
                    int i = 0;
                    for (; i <= plane - 8; i += 8) Store(output, outputOffset + i, initialVector);
                    for (; i < plane; i++) output[outputOffset + i] = initial;
                    int weightBase = co * inputChannels * 4;
                    int inputBatch = b * inputChannels * plane;
                    for (int ci = 0; ci < inputChannels; ci++)
                    {
                        ReadOnlySpan<float> source = input.Slice(inputBatch + ci * plane, plane);
                        int channelWeights = weightBase + ci * 4;
                        for (int ky = 0; ky < 2; ky++)
                            for (int kx = 0; kx < 2; kx++)
                            {
                                int yEnd = height - ky, xEnd = width - kx;
                                float weight = weights[channelWeights + ky * 2 + kx];
                                for (int y = 0; y < yEnd; y++)
                                {
                                    int row = y * width, sourceRow = (y + ky) * width;
                                    int x = 0;
                                    for (; x <= xEnd - 8; x += 8)
                                    {
                                        Vector256<float> current = Load(output, outputOffset + row + x);
                                        Vector256<float> value = Load(source, sourceRow + x + kx);
                                        Store(output, outputOffset + row + x, AddMul(current, value, weight));
                                    }
                                    for (; x < xEnd; x++)
                                        output[outputOffset + row + x] += source[sourceRow + x + kx] * weight;
                                }
                            }
                    }
                }
            return true;
        }
        else if (Vector.IsHardwareAccelerated)
        {
            return TryVector(input, weights, bias, output, batch, inputChannels,
                height, width, outputChannels);
        }
        Conv2x2PadEndScalar(input, weights, bias, output, batch, inputChannels, height, width, outputChannels);
        return true;
    }
}
