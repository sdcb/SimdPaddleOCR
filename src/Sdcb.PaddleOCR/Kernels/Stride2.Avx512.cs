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
    private static unsafe void Conv2x2PadEndEightOutputsAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerOutput = checked(inputChannels * 4);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* w0 = weightsPtr + co * weightsPerOutput;
                    float* w1 = w0 + weightsPerOutput, w2 = w1 + weightsPerOutput;
                    float* w3 = w2 + weightsPerOutput, w4 = w3 + weightsPerOutput;
                    float* w5 = w4 + weightsPerOutput, w6 = w5 + weightsPerOutput;
                    float* w7 = w6 + weightsPerOutput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane;
                    float* o1 = o0 + plane, o2 = o1 + plane, o3 = o2 + plane;
                    float* o4 = o3 + plane, o5 = o4 + plane, o6 = o5 + plane, o7 = o6 + plane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co];
                    float b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2];
                    float b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4];
                    float b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6];
                    float b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        for (; x <= width - 17; x += 16)
                        {
                            Vector512<float> a0 = Vector512.Create(b0); Vector512<float> a1 = Vector512.Create(b1);
                            Vector512<float> a2 = Vector512.Create(b2); Vector512<float> a3 = Vector512.Create(b3);
                            Vector512<float> a4 = Vector512.Create(b4); Vector512<float> a5 = Vector512.Create(b5);
                            Vector512<float> a6 = Vector512.Create(b6); Vector512<float> a7 = Vector512.Create(b7);
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float* srcRow = batchInput + ci * plane + y * width;
                                int wb = ci * 4;
                                Vector512<float> v00 = Avx512F.LoadVector512(srcRow + x);
                                Vector512<float> v01 = Avx512F.LoadVector512(srcRow + x + 1);
                                a0 = AddMul512(a0, v00, w0[wb]); a0 = AddMul512(a0, v01, w0[wb + 1]);
                                a1 = AddMul512(a1, v00, w1[wb]); a1 = AddMul512(a1, v01, w1[wb + 1]);
                                a2 = AddMul512(a2, v00, w2[wb]); a2 = AddMul512(a2, v01, w2[wb + 1]);
                                a3 = AddMul512(a3, v00, w3[wb]); a3 = AddMul512(a3, v01, w3[wb + 1]);
                                a4 = AddMul512(a4, v00, w4[wb]); a4 = AddMul512(a4, v01, w4[wb + 1]);
                                a5 = AddMul512(a5, v00, w5[wb]); a5 = AddMul512(a5, v01, w5[wb + 1]);
                                a6 = AddMul512(a6, v00, w6[wb]); a6 = AddMul512(a6, v01, w6[wb + 1]);
                                a7 = AddMul512(a7, v00, w7[wb]); a7 = AddMul512(a7, v01, w7[wb + 1]);
                                if (y + 1 < height)
                                {
                                    srcRow += width;
                                    Vector512<float> v10 = Avx512F.LoadVector512(srcRow + x);
                                    Vector512<float> v11 = Avx512F.LoadVector512(srcRow + x + 1);
                                    a0 = AddMul512(a0, v10, w0[wb + 2]); a0 = AddMul512(a0, v11, w0[wb + 3]);
                                    a1 = AddMul512(a1, v10, w1[wb + 2]); a1 = AddMul512(a1, v11, w1[wb + 3]);
                                    a2 = AddMul512(a2, v10, w2[wb + 2]); a2 = AddMul512(a2, v11, w2[wb + 3]);
                                    a3 = AddMul512(a3, v10, w3[wb + 2]); a3 = AddMul512(a3, v11, w3[wb + 3]);
                                    a4 = AddMul512(a4, v10, w4[wb + 2]); a4 = AddMul512(a4, v11, w4[wb + 3]);
                                    a5 = AddMul512(a5, v10, w5[wb + 2]); a5 = AddMul512(a5, v11, w5[wb + 3]);
                                    a6 = AddMul512(a6, v10, w6[wb + 2]); a6 = AddMul512(a6, v11, w6[wb + 3]);
                                    a7 = AddMul512(a7, v10, w7[wb + 2]); a7 = AddMul512(a7, v11, w7[wb + 3]);
                                }
                            }
                            Avx512F.Store(o0 + row + x, a0); Avx512F.Store(o1 + row + x, a1);
                            Avx512F.Store(o2 + row + x, a2); Avx512F.Store(o3 + row + x, a3);
                            Avx512F.Store(o4 + row + x, a4); Avx512F.Store(o5 + row + x, a5);
                            Avx512F.Store(o6 + row + x, a6); Avx512F.Store(o7 + row + x, a7);
                        }
                        for (; x < width; x++)
                        {
                            float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
                            float s4 = b4, s5 = b5, s6 = b6, s7 = b7;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float* src = batchInput + ci * plane;
                                int wb = ci * 4;
                                float v = src[y * width + x];
                                s0 += v * w0[wb]; s1 += v * w1[wb]; s2 += v * w2[wb]; s3 += v * w3[wb];
                                s4 += v * w4[wb]; s5 += v * w5[wb]; s6 += v * w6[wb]; s7 += v * w7[wb];
                                if (x + 1 < width)
                                {
                                    v = src[y * width + x + 1];
                                    s0 += v * w0[wb + 1]; s1 += v * w1[wb + 1]; s2 += v * w2[wb + 1]; s3 += v * w3[wb + 1];
                                    s4 += v * w4[wb + 1]; s5 += v * w5[wb + 1]; s6 += v * w6[wb + 1]; s7 += v * w7[wb + 1];
                                }
                                if (y + 1 < height)
                                {
                                    v = src[(y + 1) * width + x];
                                    s0 += v * w0[wb + 2]; s1 += v * w1[wb + 2]; s2 += v * w2[wb + 2]; s3 += v * w3[wb + 2];
                                    s4 += v * w4[wb + 2]; s5 += v * w5[wb + 2]; s6 += v * w6[wb + 2]; s7 += v * w7[wb + 2];
                                    if (x + 1 < width)
                                    {
                                        v = src[(y + 1) * width + x + 1];
                                        s0 += v * w0[wb + 3]; s1 += v * w1[wb + 3]; s2 += v * w2[wb + 3]; s3 += v * w3[wb + 3];
                                        s4 += v * w4[wb + 3]; s5 += v * w5[wb + 3]; s6 += v * w6[wb + 3]; s7 += v * w7[wb + 3];
                                    }
                                }
                            }
                            o0[row + x] = s0; o1[row + x] = s1; o2[row + x] = s2; o3[row + x] = s3;
                            o4[row + x] = s4; o5[row + x] = s5; o6[row + x] = s6; o7[row + x] = s7;
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Conv2x2PadEndFourOutputsAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerOutput = checked(inputChannels * 4);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 4)
                {
                    float* w0 = weightsPtr + co * weightsPerOutput;
                    float* w1 = w0 + weightsPerOutput, w2 = w1 + weightsPerOutput;
                    float* w3 = w2 + weightsPerOutput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane;
                    float* o1 = o0 + plane, o2 = o1 + plane, o3 = o2 + plane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co];
                    float b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2];
                    float b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        for (; x <= width - 17; x += 16)
                        {
                            Vector512<float> a0 = Vector512.Create(b0); Vector512<float> a1 = Vector512.Create(b1);
                            Vector512<float> a2 = Vector512.Create(b2); Vector512<float> a3 = Vector512.Create(b3);
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float* src = batchInput + ci * plane;
                                int wb = ci * 4;
                                float* srcRow = src + y * width;
                                Vector512<float> v00 = Avx512F.LoadVector512(srcRow + x);
                                Vector512<float> v01 = Avx512F.LoadVector512(srcRow + x + 1);
                                a0 = AddMul512(a0, v00, w0[wb]); a0 = AddMul512(a0, v01, w0[wb + 1]);
                                a1 = AddMul512(a1, v00, w1[wb]); a1 = AddMul512(a1, v01, w1[wb + 1]);
                                a2 = AddMul512(a2, v00, w2[wb]); a2 = AddMul512(a2, v01, w2[wb + 1]);
                                a3 = AddMul512(a3, v00, w3[wb]); a3 = AddMul512(a3, v01, w3[wb + 1]);
                                if (y + 1 < height)
                                {
                                    srcRow += width;
                                    Vector512<float> v10 = Avx512F.LoadVector512(srcRow + x);
                                    Vector512<float> v11 = Avx512F.LoadVector512(srcRow + x + 1);
                                    a0 = AddMul512(a0, v10, w0[wb + 2]); a0 = AddMul512(a0, v11, w0[wb + 3]);
                                    a1 = AddMul512(a1, v10, w1[wb + 2]); a1 = AddMul512(a1, v11, w1[wb + 3]);
                                    a2 = AddMul512(a2, v10, w2[wb + 2]); a2 = AddMul512(a2, v11, w2[wb + 3]);
                                    a3 = AddMul512(a3, v10, w3[wb + 2]); a3 = AddMul512(a3, v11, w3[wb + 3]);
                                }
                            }
                            Avx512F.Store(o0 + row + x, a0); Avx512F.Store(o1 + row + x, a1);
                            Avx512F.Store(o2 + row + x, a2); Avx512F.Store(o3 + row + x, a3);
                        }
                        for (; x < width; x++)
                        {
                            float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                float* src = batchInput + ci * plane; int wb = ci * 4;
                                float v = src[y * width + x];
                                s0 += v * w0[wb]; s1 += v * w1[wb]; s2 += v * w2[wb]; s3 += v * w3[wb];
                                if (x + 1 < width)
                                {
                                    v = src[y * width + x + 1];
                                    s0 += v * w0[wb + 1]; s1 += v * w1[wb + 1]; s2 += v * w2[wb + 1]; s3 += v * w3[wb + 1];
                                }
                                if (y + 1 < height)
                                {
                                    v = src[(y + 1) * width + x];
                                    s0 += v * w0[wb + 2]; s1 += v * w1[wb + 2]; s2 += v * w2[wb + 2]; s3 += v * w3[wb + 2];
                                    if (x + 1 < width)
                                    {
                                        v = src[(y + 1) * width + x + 1];
                                        s0 += v * w0[wb + 3]; s1 += v * w1[wb + 3]; s2 += v * w2[wb + 3]; s3 += v * w3[wb + 3];
                                    }
                                }
                            }
                            o0[row + x] = s0; o1[row + x] = s1; o2[row + x] = s2; o3[row + x] = s3;
                        }
                    }
                }
        }
    }
}
