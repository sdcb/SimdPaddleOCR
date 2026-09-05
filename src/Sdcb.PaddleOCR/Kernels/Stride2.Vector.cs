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
    internal static bool TryVector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels)
    {
        int plane = checked(height * width);
        int widthLanes = Vector<float>.Count;
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co++)
            {
                int outputOffset = (b * outputChannels + co) * plane;
                float initial = bias.IsEmpty ? 0f : bias[co];
                Vector<float> initialVector = new(initial);
                int i = 0;
                for (; i <= plane - widthLanes; i += widthLanes)
                    VectorStore(output, outputOffset + i, initialVector);
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
                                for (; x <= xEnd - widthLanes; x += widthLanes)
                                {
                                    Vector<float> current = VectorLoad(output, outputOffset + row + x);
                                    Vector<float> value = VectorLoad(source, sourceRow + x + kx);
                                    VectorStore(output, outputOffset + row + x, VectorAddMul(current, value, weight));
                                }
                                for (; x < xEnd; x++)
                                    output[outputOffset + row + x] += source[sourceRow + x + kx] * weight;
                            }
                        }
                }
            }
        return true;
    }
}
