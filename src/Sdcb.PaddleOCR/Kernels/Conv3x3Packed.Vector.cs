using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Conv3x3Packed
{
    internal static unsafe bool TryVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels, int intraOpThreads)
    {
        int plane = checked(height * width), blocks = outputChannels / 8;
        int widthLanes = Vector<float>.Count, weightsPerInput = 9 * 8;
        if (intraOpThreads > 1 && batch == 1 && blocks >= 2 &&
            (long)outputChannels * inputChannels * plane * 9 >= IntraOpMinWork)
        {
            int workers = Math.Min(intraOpThreads, blocks);
            fixed (float* inputPtr = input, weightsPtr = packedWeights,
                biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = packedWeights.Length,
                    biasLength = bias.Length, outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    int begin = blocks * worker / workers, end = blocks * (worker + 1) / workers;
                    if (end <= begin) return;
                    int count = (end - begin) * 8;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> wSpan = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(begin * inputChannels * 9 * 8, (end - begin) * inputChannels * 9 * 8);
                    ReadOnlySpan<float> bSpan = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin * 8, count);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * 8 * plane, count * plane);
                    TryVector(inSpan, wSpan, bSpan, outSpan, 1, inputChannels,
                        height, width, count, 1);
                });
            }
            return true;
        }

        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* w0 = weightsPtr + (co / 8) * inputChannels * weightsPerInput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane;
                    float* o1 = o0 + plane, o2 = o1 + plane, o3 = o2 + plane;
                    float* o4 = o3 + plane, o5 = o4 + plane, o6 = o5 + plane, o7 = o6 + plane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4], b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6], b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        while (x < width)
                        {
                            bool vector = y > 0 && y + 1 < height && x > 0 && x + widthLanes < width;
                            if (vector)
                            {
                                Vector<float> a0 = new(b0), a1 = new(b1), a2 = new(b2), a3 = new(b3);
                                Vector<float> a4 = new(b4), a5 = new(b5), a6 = new(b6), a7 = new(b7);
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* source = batchInput + ci * plane;
                                    float* weights0 = w0 + ci * weightsPerInput;
                                    float* row0 = source + (y - 1) * width + x - 1;
                                    float* row1 = row0 + width, row2 = row1 + width;
                                    Vector<float> value = VectorLoad(row0);
                                    a0 = VectorAddMul(a0, value, weights0[0]); a1 = VectorAddMul(a1, value, weights0[1]);
                                    a2 = VectorAddMul(a2, value, weights0[2]); a3 = VectorAddMul(a3, value, weights0[3]);
                                    a4 = VectorAddMul(a4, value, weights0[4]); a5 = VectorAddMul(a5, value, weights0[5]);
                                    a6 = VectorAddMul(a6, value, weights0[6]); a7 = VectorAddMul(a7, value, weights0[7]);
                                    value = VectorLoad(row0 + 1);
                                    a0 = VectorAddMul(a0, value, weights0[8]); a1 = VectorAddMul(a1, value, weights0[9]);
                                    a2 = VectorAddMul(a2, value, weights0[10]); a3 = VectorAddMul(a3, value, weights0[11]);
                                    a4 = VectorAddMul(a4, value, weights0[12]); a5 = VectorAddMul(a5, value, weights0[13]);
                                    a6 = VectorAddMul(a6, value, weights0[14]); a7 = VectorAddMul(a7, value, weights0[15]);
                                    value = VectorLoad(row0 + 2);
                                    a0 = VectorAddMul(a0, value, weights0[16]); a1 = VectorAddMul(a1, value, weights0[17]);
                                    a2 = VectorAddMul(a2, value, weights0[18]); a3 = VectorAddMul(a3, value, weights0[19]);
                                    a4 = VectorAddMul(a4, value, weights0[20]); a5 = VectorAddMul(a5, value, weights0[21]);
                                    a6 = VectorAddMul(a6, value, weights0[22]); a7 = VectorAddMul(a7, value, weights0[23]);
                                    value = VectorLoad(row1);
                                    a0 = VectorAddMul(a0, value, weights0[24]); a1 = VectorAddMul(a1, value, weights0[25]);
                                    a2 = VectorAddMul(a2, value, weights0[26]); a3 = VectorAddMul(a3, value, weights0[27]);
                                    a4 = VectorAddMul(a4, value, weights0[28]); a5 = VectorAddMul(a5, value, weights0[29]);
                                    a6 = VectorAddMul(a6, value, weights0[30]); a7 = VectorAddMul(a7, value, weights0[31]);
                                    value = VectorLoad(row1 + 1);
                                    a0 = VectorAddMul(a0, value, weights0[32]); a1 = VectorAddMul(a1, value, weights0[33]);
                                    a2 = VectorAddMul(a2, value, weights0[34]); a3 = VectorAddMul(a3, value, weights0[35]);
                                    a4 = VectorAddMul(a4, value, weights0[36]); a5 = VectorAddMul(a5, value, weights0[37]);
                                    a6 = VectorAddMul(a6, value, weights0[38]); a7 = VectorAddMul(a7, value, weights0[39]);
                                    value = VectorLoad(row1 + 2);
                                    a0 = VectorAddMul(a0, value, weights0[40]); a1 = VectorAddMul(a1, value, weights0[41]);
                                    a2 = VectorAddMul(a2, value, weights0[42]); a3 = VectorAddMul(a3, value, weights0[43]);
                                    a4 = VectorAddMul(a4, value, weights0[44]); a5 = VectorAddMul(a5, value, weights0[45]);
                                    a6 = VectorAddMul(a6, value, weights0[46]); a7 = VectorAddMul(a7, value, weights0[47]);
                                    value = VectorLoad(row2);
                                    a0 = VectorAddMul(a0, value, weights0[48]); a1 = VectorAddMul(a1, value, weights0[49]);
                                    a2 = VectorAddMul(a2, value, weights0[50]); a3 = VectorAddMul(a3, value, weights0[51]);
                                    a4 = VectorAddMul(a4, value, weights0[52]); a5 = VectorAddMul(a5, value, weights0[53]);
                                    a6 = VectorAddMul(a6, value, weights0[54]); a7 = VectorAddMul(a7, value, weights0[55]);
                                    value = VectorLoad(row2 + 1);
                                    a0 = VectorAddMul(a0, value, weights0[56]); a1 = VectorAddMul(a1, value, weights0[57]);
                                    a2 = VectorAddMul(a2, value, weights0[58]); a3 = VectorAddMul(a3, value, weights0[59]);
                                    a4 = VectorAddMul(a4, value, weights0[60]); a5 = VectorAddMul(a5, value, weights0[61]);
                                    a6 = VectorAddMul(a6, value, weights0[62]); a7 = VectorAddMul(a7, value, weights0[63]);
                                    value = VectorLoad(row2 + 2);
                                    a0 = VectorAddMul(a0, value, weights0[64]); a1 = VectorAddMul(a1, value, weights0[65]);
                                    a2 = VectorAddMul(a2, value, weights0[66]); a3 = VectorAddMul(a3, value, weights0[67]);
                                    a4 = VectorAddMul(a4, value, weights0[68]); a5 = VectorAddMul(a5, value, weights0[69]);
                                    a6 = VectorAddMul(a6, value, weights0[70]); a7 = VectorAddMul(a7, value, weights0[71]);
                                }
                                VectorStore(o0 + row + x, a0); VectorStore(o1 + row + x, a1);
                                VectorStore(o2 + row + x, a2); VectorStore(o3 + row + x, a3);
                                VectorStore(o4 + row + x, a4); VectorStore(o5 + row + x, a5);
                                VectorStore(o6 + row + x, a6); VectorStore(o7 + row + x, a7);
                                x += widthLanes;
                            }
                            else
                            {
                                float s0 = b0, s1 = b1, s2 = b2, s3 = b3, s4 = b4, s5 = b5, s6 = b6, s7 = b7;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* source = batchInput + ci * plane;
                                    float* weights0 = w0 + ci * weightsPerInput;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = y + ky - 1;
                                        if ((uint)iy >= (uint)height) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = x + kx - 1;
                                            if ((uint)ix >= (uint)width) continue;
                                            float value = source[iy * width + ix];
                                            int wi = (ky * 3 + kx) * 8;
                                            s0 += value * weights0[wi]; s1 += value * weights0[wi + 1];
                                            s2 += value * weights0[wi + 2]; s3 += value * weights0[wi + 3];
                                            s4 += value * weights0[wi + 4]; s5 += value * weights0[wi + 5];
                                            s6 += value * weights0[wi + 6]; s7 += value * weights0[wi + 7];
                                        }
                                    }
                                }
                                o0[row + x] = s0; o1[row + x] = s1; o2[row + x] = s2; o3[row + x] = s3;
                                o4[row + x] = s4; o5[row + x] = s5; o6[row + x] = s6; o7[row + x] = s7;
                                x++;
                            }
                        }
                    }
                }
        }
        return true;
    }
}
