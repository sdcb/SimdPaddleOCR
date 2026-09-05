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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Conv3x3SixteenOutputsPackedAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerInput = 9 * 8;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 16)
                {
                    float* w0 = weightsPtr + (co / 8) * inputChannels * weightsPerInput;
                    float* w8 = w0 + inputChannels * weightsPerInput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane;
                    float* o1 = o0 + plane, o2 = o1 + plane, o3 = o2 + plane;
                    float* o4 = o3 + plane, o5 = o4 + plane, o6 = o5 + plane, o7 = o6 + plane;
                    float* o8 = o7 + plane, o9 = o8 + plane, o10 = o9 + plane, o11 = o10 + plane;
                    float* o12 = o11 + plane, o13 = o12 + plane, o14 = o13 + plane, o15 = o14 + plane;
                    Vector512<float> bias0 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co]);
                    Vector512<float> bias1 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                    Vector512<float> bias2 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                    Vector512<float> bias3 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                    Vector512<float> bias4 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 4]);
                    Vector512<float> bias5 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 5]);
                    Vector512<float> bias6 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 6]);
                    Vector512<float> bias7 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 7]);
                    Vector512<float> bias8 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 8]);
                    Vector512<float> bias9 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 9]);
                    Vector512<float> bias10 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 10]);
                    Vector512<float> bias11 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 11]);
                    Vector512<float> bias12 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 12]);
                    Vector512<float> bias13 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 13]);
                    Vector512<float> bias14 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 14]);
                    Vector512<float> bias15 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 15]);
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        while (x < width)
                        {
                            bool vector = y > 0 && y + 1 < height && x > 0 && x + 16 < width;
                            if (vector)
                            {
                                Vector512<float> a0 = bias0, a1 = bias1, a2 = bias2, a3 = bias3;
                                Vector512<float> a4 = bias4, a5 = bias5, a6 = bias6, a7 = bias7;
                                Vector512<float> a8 = bias8, a9 = bias9, a10 = bias10, a11 = bias11;
                                Vector512<float> a12 = bias12, a13 = bias13, a14 = bias14, a15 = bias15;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* source = batchInput + ci * plane;
                                    float* weights0 = w0 + ci * weightsPerInput;
                                    float* weights8 = w8 + ci * weightsPerInput;
                                    float* row0 = source + (y - 1) * width + x - 1;
                                    float* row1 = row0 + width, row2 = row1 + width;
                                    Vector512<float> value = Avx512F.LoadVector512(row0);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8);
                                    value = Avx512F.LoadVector512(row0 + 1);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0 + 8);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8 + 8);
                                    value = Avx512F.LoadVector512(row0 + 2);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0 + 16);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8 + 16);
                                    value = Avx512F.LoadVector512(row1);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0 + 24);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8 + 24);
                                    value = Avx512F.LoadVector512(row1 + 1);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0 + 32);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8 + 32);
                                    value = Avx512F.LoadVector512(row1 + 2);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0 + 40);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8 + 40);
                                    value = Avx512F.LoadVector512(row2);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0 + 48);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8 + 48);
                                    value = Avx512F.LoadVector512(row2 + 1);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0 + 56);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8 + 56);
                                    value = Avx512F.LoadVector512(row2 + 2);
                                    AddEightPacked512(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5,
                                        ref a6, ref a7, value, weights0 + 64);
                                    AddEightPacked512(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13,
                                        ref a14, ref a15, value, weights8 + 64);
                                }
                                Avx512F.Store(o0 + row + x, a0); Avx512F.Store(o1 + row + x, a1);
                                Avx512F.Store(o2 + row + x, a2); Avx512F.Store(o3 + row + x, a3);
                                Avx512F.Store(o4 + row + x, a4); Avx512F.Store(o5 + row + x, a5);
                                Avx512F.Store(o6 + row + x, a6); Avx512F.Store(o7 + row + x, a7);
                                Avx512F.Store(o8 + row + x, a8); Avx512F.Store(o9 + row + x, a9);
                                Avx512F.Store(o10 + row + x, a10); Avx512F.Store(o11 + row + x, a11);
                                Avx512F.Store(o12 + row + x, a12); Avx512F.Store(o13 + row + x, a13);
                                Avx512F.Store(o14 + row + x, a14); Avx512F.Store(o15 + row + x, a15);
                                x += 16;
                            }
                            else
                            {
                                float s0 = biasPtr == null ? 0f : biasPtr[co], s1 = biasPtr == null ? 0f : biasPtr[co + 1];
                                float s2 = biasPtr == null ? 0f : biasPtr[co + 2], s3 = biasPtr == null ? 0f : biasPtr[co + 3];
                                float s4 = biasPtr == null ? 0f : biasPtr[co + 4], s5 = biasPtr == null ? 0f : biasPtr[co + 5];
                                float s6 = biasPtr == null ? 0f : biasPtr[co + 6], s7 = biasPtr == null ? 0f : biasPtr[co + 7];
                                float s8 = biasPtr == null ? 0f : biasPtr[co + 8], s9 = biasPtr == null ? 0f : biasPtr[co + 9];
                                float s10 = biasPtr == null ? 0f : biasPtr[co + 10], s11 = biasPtr == null ? 0f : biasPtr[co + 11];
                                float s12 = biasPtr == null ? 0f : biasPtr[co + 12], s13 = biasPtr == null ? 0f : biasPtr[co + 13];
                                float s14 = biasPtr == null ? 0f : biasPtr[co + 14], s15 = biasPtr == null ? 0f : biasPtr[co + 15];
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* weights0 = w0 + ci * weightsPerInput;
                                    float* weights8 = w8 + ci * weightsPerInput;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = y + ky - 1;
                                        if ((uint)iy >= (uint)height) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = x + kx - 1;
                                            if ((uint)ix >= (uint)width) continue;
                                            float v = batchInput[ci * plane + iy * width + ix];
                                            int k = ky * 3 + kx;
                                            s0 += v * weights0[k * 8]; s1 += v * weights0[k * 8 + 1];
                                            s2 += v * weights0[k * 8 + 2]; s3 += v * weights0[k * 8 + 3];
                                            s4 += v * weights0[k * 8 + 4]; s5 += v * weights0[k * 8 + 5];
                                            s6 += v * weights0[k * 8 + 6]; s7 += v * weights0[k * 8 + 7];
                                            s8 += v * weights8[k * 8]; s9 += v * weights8[k * 8 + 1];
                                            s10 += v * weights8[k * 8 + 2]; s11 += v * weights8[k * 8 + 3];
                                            s12 += v * weights8[k * 8 + 4]; s13 += v * weights8[k * 8 + 5];
                                            s14 += v * weights8[k * 8 + 6]; s15 += v * weights8[k * 8 + 7];
                                        }
                                    }
                                }
                                o0[row + x] = s0; o1[row + x] = s1; o2[row + x] = s2; o3[row + x] = s3;
                                o4[row + x] = s4; o5[row + x] = s5; o6[row + x] = s6; o7[row + x] = s7;
                                o8[row + x] = s8; o9[row + x] = s9; o10[row + x] = s10; o11[row + x] = s11;
                                o12[row + x] = s12; o13[row + x] = s13; o14[row + x] = s14; o15[row + x] = s15;
                                x++;
                            }
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Conv3x3EightOutputsPackedAvx512Unsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width), weightsPerInput = 9 * 8;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* w = weightsPtr + (co / 8) * inputChannels * weightsPerInput;
                    float* o0 = outputPtr + (b * outputChannels + co) * plane;
                    float* o1 = o0 + plane, o2 = o1 + plane, o3 = o2 + plane;
                    float* o4 = o3 + plane, o5 = o4 + plane, o6 = o5 + plane, o7 = o6 + plane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4], b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6], b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    Vector512<float> vb0 = Vector512.Create(b0), vb1 = Vector512.Create(b1);
                    Vector512<float> vb2 = Vector512.Create(b2), vb3 = Vector512.Create(b3);
                    Vector512<float> vb4 = Vector512.Create(b4), vb5 = Vector512.Create(b5);
                    Vector512<float> vb6 = Vector512.Create(b6), vb7 = Vector512.Create(b7);
                    float* batchInput = inputPtr + b * inputChannels * plane;
                    for (int y = 0; y < height; y++)
                    {
                        int row = y * width, x = 0;
                        while (x < width)
                        {
                            bool vector = y > 0 && y + 1 < height && x > 0 && x + 16 < width;
                            if (vector)
                            {
                                Vector512<float> a0 = vb0, a1 = vb1;
                                Vector512<float> a2 = vb2, a3 = vb3;
                                Vector512<float> a4 = vb4, a5 = vb5;
                                Vector512<float> a6 = vb6, a7 = vb7;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    float* wc = w + ci * weightsPerInput;
                                    int ix = x - 1;
                                    float* row0 = src + (y - 1) * width + ix;
                                    float* row1 = row0 + width, row2 = row1 + width;
                                    Vector512<float> v0 = Avx512F.LoadVector512(row0), v1 = Avx512F.LoadVector512(row0 + 1), v2 = Avx512F.LoadVector512(row0 + 2);
                                    float* weights = wc;
                                    a0 = AddMul512(a0, v0, weights[0]); a1 = AddMul512(a1, v0, weights[1]);
                                    a2 = AddMul512(a2, v0, weights[2]); a3 = AddMul512(a3, v0, weights[3]);
                                    a4 = AddMul512(a4, v0, weights[4]); a5 = AddMul512(a5, v0, weights[5]);
                                    a6 = AddMul512(a6, v0, weights[6]); a7 = AddMul512(a7, v0, weights[7]);
                                    weights += 8;
                                    a0 = AddMul512(a0, v1, weights[0]); a1 = AddMul512(a1, v1, weights[1]);
                                    a2 = AddMul512(a2, v1, weights[2]); a3 = AddMul512(a3, v1, weights[3]);
                                    a4 = AddMul512(a4, v1, weights[4]); a5 = AddMul512(a5, v1, weights[5]);
                                    a6 = AddMul512(a6, v1, weights[6]); a7 = AddMul512(a7, v1, weights[7]);
                                    weights += 8;
                                    a0 = AddMul512(a0, v2, weights[0]); a1 = AddMul512(a1, v2, weights[1]);
                                    a2 = AddMul512(a2, v2, weights[2]); a3 = AddMul512(a3, v2, weights[3]);
                                    a4 = AddMul512(a4, v2, weights[4]); a5 = AddMul512(a5, v2, weights[5]);
                                    a6 = AddMul512(a6, v2, weights[6]); a7 = AddMul512(a7, v2, weights[7]);
                                    weights += 8;
                                    v0 = Avx512F.LoadVector512(row1); v1 = Avx512F.LoadVector512(row1 + 1); v2 = Avx512F.LoadVector512(row1 + 2);
                                    a0 = AddMul512(a0, v0, weights[0]); a1 = AddMul512(a1, v0, weights[1]);
                                    a2 = AddMul512(a2, v0, weights[2]); a3 = AddMul512(a3, v0, weights[3]);
                                    a4 = AddMul512(a4, v0, weights[4]); a5 = AddMul512(a5, v0, weights[5]);
                                    a6 = AddMul512(a6, v0, weights[6]); a7 = AddMul512(a7, v0, weights[7]);
                                    weights += 8;
                                    a0 = AddMul512(a0, v1, weights[0]); a1 = AddMul512(a1, v1, weights[1]);
                                    a2 = AddMul512(a2, v1, weights[2]); a3 = AddMul512(a3, v1, weights[3]);
                                    a4 = AddMul512(a4, v1, weights[4]); a5 = AddMul512(a5, v1, weights[5]);
                                    a6 = AddMul512(a6, v1, weights[6]); a7 = AddMul512(a7, v1, weights[7]);
                                    weights += 8;
                                    a0 = AddMul512(a0, v2, weights[0]); a1 = AddMul512(a1, v2, weights[1]);
                                    a2 = AddMul512(a2, v2, weights[2]); a3 = AddMul512(a3, v2, weights[3]);
                                    a4 = AddMul512(a4, v2, weights[4]); a5 = AddMul512(a5, v2, weights[5]);
                                    a6 = AddMul512(a6, v2, weights[6]); a7 = AddMul512(a7, v2, weights[7]);
                                    weights += 8;
                                    v0 = Avx512F.LoadVector512(row2); v1 = Avx512F.LoadVector512(row2 + 1); v2 = Avx512F.LoadVector512(row2 + 2);
                                    a0 = AddMul512(a0, v0, weights[0]); a1 = AddMul512(a1, v0, weights[1]);
                                    a2 = AddMul512(a2, v0, weights[2]); a3 = AddMul512(a3, v0, weights[3]);
                                    a4 = AddMul512(a4, v0, weights[4]); a5 = AddMul512(a5, v0, weights[5]);
                                    a6 = AddMul512(a6, v0, weights[6]); a7 = AddMul512(a7, v0, weights[7]);
                                    weights += 8;
                                    a0 = AddMul512(a0, v1, weights[0]); a1 = AddMul512(a1, v1, weights[1]);
                                    a2 = AddMul512(a2, v1, weights[2]); a3 = AddMul512(a3, v1, weights[3]);
                                    a4 = AddMul512(a4, v1, weights[4]); a5 = AddMul512(a5, v1, weights[5]);
                                    a6 = AddMul512(a6, v1, weights[6]); a7 = AddMul512(a7, v1, weights[7]);
                                    weights += 8;
                                    a0 = AddMul512(a0, v2, weights[0]); a1 = AddMul512(a1, v2, weights[1]);
                                    a2 = AddMul512(a2, v2, weights[2]); a3 = AddMul512(a3, v2, weights[3]);
                                    a4 = AddMul512(a4, v2, weights[4]); a5 = AddMul512(a5, v2, weights[5]);
                                    a6 = AddMul512(a6, v2, weights[6]); a7 = AddMul512(a7, v2, weights[7]);
                                }
                                Avx512F.Store(o0 + row + x, a0); Avx512F.Store(o1 + row + x, a1);
                                Avx512F.Store(o2 + row + x, a2); Avx512F.Store(o3 + row + x, a3);
                                Avx512F.Store(o4 + row + x, a4); Avx512F.Store(o5 + row + x, a5);
                                Avx512F.Store(o6 + row + x, a6); Avx512F.Store(o7 + row + x, a7);
                                x += 16;
                            }
                            else
                            {
                                float s0 = b0, s1 = b1, s2 = b2, s3 = b3, s4 = b4, s5 = b5, s6 = b6, s7 = b7;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * plane;
                                    float* wc = w + ci * weightsPerInput;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = y + ky - 1;
                                        if ((uint)iy >= (uint)height) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = x + kx - 1;
                                            if ((uint)ix >= (uint)width) continue;
                                            float value = src[iy * width + ix];
                                            float* weights = wc + (ky * 3 + kx) * 8;
                                            s0 += value * weights[0]; s1 += value * weights[1];
                                            s2 += value * weights[2]; s3 += value * weights[3];
                                            s4 += value * weights[4]; s5 += value * weights[5];
                                            s6 += value * weights[6]; s7 += value * weights[7];
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
    }
}
