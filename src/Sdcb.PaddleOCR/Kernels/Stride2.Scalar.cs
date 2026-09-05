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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv2x2PadEndScalar(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels)
    {
        int plane = checked(height * width);
        int weightsPerOutput = checked(inputChannels * 4);
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co++)
            {
                int outputOffset = (b * outputChannels + co) * plane;
                float initial = bias.IsEmpty ? 0f : bias[co];
                output.Slice(outputOffset, plane).Fill(initial);
                int weightBase = co * weightsPerOutput;
                int inputBatch = b * inputChannels * plane;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    ReadOnlySpan<float> source = input.Slice(inputBatch + ci * plane, plane);
                    int channelWeights = weightBase + ci * 4;
                    for (int ky = 0; ky < 2; ky++)
                        for (int kx = 0; kx < 2; kx++)
                        {
                            float weight = weights[channelWeights + ky * 2 + kx];
                            int yEnd = height - ky, xEnd = width - kx;
                            for (int y = 0; y < yEnd; y++)
                            {
                                int row = y * width, sourceRow = (y + ky) * width;
                                for (int x = 0; x < xEnd; x++)
                                    output[outputOffset + row + x] += source[sourceRow + x + kx] * weight;
                            }
                        }
                }
            }
    }
}
