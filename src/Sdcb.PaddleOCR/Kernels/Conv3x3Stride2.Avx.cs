using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Conv3x3Stride2
{
    // Stride-two convolutions are common in the detector.  Accumulate four
    // output channels together so each gathered input patch is loaded once
    // instead of once per output channel.
    private static void Conv3x3Stride2FourOutputs(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int inputHeight, int inputWidth,
        int outputHeight, int outputWidth, int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth);
        int outputPlane = checked(outputHeight * outputWidth);
        int weightsPerOutput = checked(inputChannels * 9);
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co += 4)
            {
                int inputBatch = b * inputChannels * inputPlane;
                int outputBatch = b * outputChannels * outputPlane;
                int output0 = outputBatch + co * outputPlane;
                int output1 = output0 + outputPlane;
                int output2 = output1 + outputPlane;
                int output3 = output2 + outputPlane;
                int weightBase0 = co * weightsPerOutput;
                int weightBase1 = weightBase0 + weightsPerOutput;
                int weightBase2 = weightBase1 + weightsPerOutput;
                int weightBase3 = weightBase2 + weightsPerOutput;
                float bias0 = bias.IsEmpty ? 0f : bias[co];
                float bias1 = bias.IsEmpty ? 0f : bias[co + 1];
                float bias2 = bias.IsEmpty ? 0f : bias[co + 2];
                float bias3 = bias.IsEmpty ? 0f : bias[co + 3];
                for (int oy = 0; oy < outputHeight; oy++)
                {
                    int row = oy * outputWidth;
                    for (int x = 0; x < outputWidth;)
                    {
                        // Interior vectors require every lane of the gathered
                        // source to be in bounds for the current kernel column.
                        bool canVector = x + 8 <= outputWidth;
                        if (canVector)
                        {
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int first = 2 * x - 1 + kx;
                                if (first < 0 || first + 14 >= inputWidth) { canVector = false; break; }
                            }
                        }
                        if (canVector)
                        {
                            Vector256<float> a0 = Vector256.Create(bias0);
                            Vector256<float> a1 = Vector256.Create(bias1);
                            Vector256<float> a2 = Vector256.Create(bias2);
                            Vector256<float> a3 = Vector256.Create(bias3);
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                int wb0 = weightBase0 + ci * 9;
                                int wb1 = weightBase1 + ci * 9;
                                int wb2 = weightBase2 + ci * 9;
                                int wb3 = weightBase3 + ci * 9;
                                ReadOnlySpan<float> source = input.Slice(inputBatch + ci * inputPlane, inputPlane);
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    int sourceY = oy * 2 - 1 + ky;
                                    if ((uint)sourceY >= (uint)inputHeight) continue;
                                    int sourceOffset = sourceY * inputWidth + 2 * x - 1;
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        Vector256<float> value = LoadStride2(source, sourceOffset + kx);
                                        a0 = AddMul(a0, value, weights[wb0 + ky * 3 + kx]);
                                        a1 = AddMul(a1, value, weights[wb1 + ky * 3 + kx]);
                                        a2 = AddMul(a2, value, weights[wb2 + ky * 3 + kx]);
                                        a3 = AddMul(a3, value, weights[wb3 + ky * 3 + kx]);
                                    }
                                }
                            }
                            Store(output, output0 + row + x, a0);
                            Store(output, output1 + row + x, a1);
                            Store(output, output2 + row + x, a2);
                            Store(output, output3 + row + x, a3);
                            x += 8;
                            continue;
                        }

                        for (int lane = 0; lane < Math.Min(8, outputWidth - x); lane++)
                        {
                            int ox = x + lane;
                            float s0 = bias0, s1 = bias1, s2 = bias2, s3 = bias3;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                int wb0 = weightBase0 + ci * 9;
                                int wb1 = weightBase1 + ci * 9;
                                int wb2 = weightBase2 + ci * 9;
                                int wb3 = weightBase3 + ci * 9;
                                int sourceBase = inputBatch + ci * inputPlane;
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    int iy = oy * 2 - 1 + ky;
                                    if ((uint)iy >= (uint)inputHeight) continue;
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        int ix = ox * 2 - 1 + kx;
                                        if ((uint)ix >= (uint)inputWidth) continue;
                                        float v = input[sourceBase + iy * inputWidth + ix];
                                        s0 += v * weights[wb0 + ky * 3 + kx];
                                        s1 += v * weights[wb1 + ky * 3 + kx];
                                        s2 += v * weights[wb2 + ky * 3 + kx];
                                        s3 += v * weights[wb3 + ky * 3 + kx];
                                    }
                                }
                            }
                            output[output0 + row + ox] = s0;
                            output[output1 + row + ox] = s1;
                            output[output2 + row + ox] = s2;
                            output[output3 + row + ox] = s3;
                        }
                        x += 8;
                    }
                }
            }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv3x3Stride2EightOutputsUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int inputHeight, int inputWidth,
        int outputHeight, int outputWidth, int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        int weightsPerOutput = checked(inputChannels * 9);
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* w0 = weightsPtr + co * weightsPerOutput;
                    float* w1 = w0 + weightsPerOutput, w2 = w1 + weightsPerOutput;
                    float* w3 = w2 + weightsPerOutput, w4 = w3 + weightsPerOutput;
                    float* w5 = w4 + weightsPerOutput, w6 = w5 + weightsPerOutput, w7 = w6 + weightsPerOutput;
                    float* o0 = outputPtr + ((b * outputChannels + co) * outputPlane);
                    float* o1 = o0 + outputPlane, o2 = o1 + outputPlane, o3 = o2 + outputPlane;
                    float* o4 = o3 + outputPlane, o5 = o4 + outputPlane, o6 = o5 + outputPlane, o7 = o6 + outputPlane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4], b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6], b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    float* batchInput = inputPtr + b * inputChannels * inputPlane;
                    for (int oy = 0; oy < outputHeight; oy++)
                    {
                        int rowOffset = oy * outputWidth;
                        int ox = 0;
                        while (ox < outputWidth)
                        {
                            bool fullHeight = oy != 0 && oy * 2 + 1 < inputHeight;
                            bool fullWidth = ox != 0 && ox + 8 <= outputWidth && ((ox + 7) * 2 + 1 < inputWidth);
                            if (fullHeight && fullWidth)
                            {
                                Vector256<float> a0 = Vector256.Create(b0); Vector256<float> a1 = Vector256.Create(b1);
                                Vector256<float> a2 = Vector256.Create(b2); Vector256<float> a3 = Vector256.Create(b3);
                                Vector256<float> a4 = Vector256.Create(b4); Vector256<float> a5 = Vector256.Create(b5);
                                Vector256<float> a6 = Vector256.Create(b6); Vector256<float> a7 = Vector256.Create(b7);
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * inputPlane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        float* srcRow = src + (oy * 2 - 1 + ky) * inputWidth;
                                        int ix = ox * 2 - 1;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            Vector256<float> value = LoadStride2(srcRow + ix + kx);
                                            int wi = wb + ky * 3 + kx;
                                            a0 = AddMul(a0, value, w0[wi]); a1 = AddMul(a1, value, w1[wi]);
                                            a2 = AddMul(a2, value, w2[wi]); a3 = AddMul(a3, value, w3[wi]);
                                            a4 = AddMul(a4, value, w4[wi]); a5 = AddMul(a5, value, w5[wi]);
                                            a6 = AddMul(a6, value, w6[wi]); a7 = AddMul(a7, value, w7[wi]);
                                        }
                                    }
                                }
                                Avx.Store(o0 + rowOffset + ox, a0); Avx.Store(o1 + rowOffset + ox, a1);
                                Avx.Store(o2 + rowOffset + ox, a2); Avx.Store(o3 + rowOffset + ox, a3);
                                Avx.Store(o4 + rowOffset + ox, a4); Avx.Store(o5 + rowOffset + ox, a5);
                                Avx.Store(o6 + rowOffset + ox, a6); Avx.Store(o7 + rowOffset + ox, a7);
                                ox += 8;
                            }
                            else
                            {
                                float s0 = b0, s1 = b1, s2 = b2, s3 = b3, s4 = b4, s5 = b5, s6 = b6, s7 = b7;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * inputPlane;
                                    int wb = ci * 9;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = oy * 2 - 1 + ky;
                                        if ((uint)iy >= (uint)inputHeight) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = ox * 2 - 1 + kx;
                                            if ((uint)ix >= (uint)inputWidth) continue;
                                            float v = src[iy * inputWidth + ix]; int wi = wb + ky * 3 + kx;
                                            s0 += v * w0[wi]; s1 += v * w1[wi]; s2 += v * w2[wi]; s3 += v * w3[wi];
                                            s4 += v * w4[wi]; s5 += v * w5[wi]; s6 += v * w6[wi]; s7 += v * w7[wi];
                                        }
                                    }
                                }
                                o0[rowOffset + ox] = s0; o1[rowOffset + ox] = s1; o2[rowOffset + ox] = s2; o3[rowOffset + ox] = s3;
                                o4[rowOffset + ox] = s4; o5[rowOffset + ox] = s5; o6[rowOffset + ox] = s6; o7[rowOffset + ox] = s7;
                                ox++;
                            }
                        }
                    }
                }
        }
    }

    private static void Conv3x3Stride2EightOutputs(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int inputHeight, int inputWidth,
        int outputHeight, int outputWidth, int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        int weightsPerOutput = checked(inputChannels * 9);
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co += 8)
            {
                int inputBatch = b * inputChannels * inputPlane, outputBatch = b * outputChannels * outputPlane;
                int o0 = outputBatch + (co + 0) * outputPlane, o1 = o0 + outputPlane;
                int o2 = o1 + outputPlane, o3 = o2 + outputPlane, o4 = o3 + outputPlane;
                int o5 = o4 + outputPlane, o6 = o5 + outputPlane, o7 = o6 + outputPlane;
                int wb0 = (co + 0) * weightsPerOutput, wb1 = (co + 1) * weightsPerOutput;
                int wb2 = (co + 2) * weightsPerOutput, wb3 = (co + 3) * weightsPerOutput;
                int wb4 = (co + 4) * weightsPerOutput, wb5 = (co + 5) * weightsPerOutput;
                int wb6 = (co + 6) * weightsPerOutput, wb7 = (co + 7) * weightsPerOutput;
                Vector256<float> vb0 = Vector256.Create(bias.IsEmpty ? 0f : bias[co + 0]);
                Vector256<float> vb1 = Vector256.Create(bias.IsEmpty ? 0f : bias[co + 1]);
                Vector256<float> vb2 = Vector256.Create(bias.IsEmpty ? 0f : bias[co + 2]);
                Vector256<float> vb3 = Vector256.Create(bias.IsEmpty ? 0f : bias[co + 3]);
                Vector256<float> vb4 = Vector256.Create(bias.IsEmpty ? 0f : bias[co + 4]);
                Vector256<float> vb5 = Vector256.Create(bias.IsEmpty ? 0f : bias[co + 5]);
                Vector256<float> vb6 = Vector256.Create(bias.IsEmpty ? 0f : bias[co + 6]);
                Vector256<float> vb7 = Vector256.Create(bias.IsEmpty ? 0f : bias[co + 7]);
                for (int oy = 0; oy < outputHeight; oy++)
                {
                    int row = oy * outputWidth;
                    for (int x = 0; x < outputWidth;)
                    {
                        bool canVector = x + 8 <= outputWidth;
                        if (canVector)
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int first = 2 * x - 1 + kx;
                                if (first < 0 || first + 14 >= inputWidth) { canVector = false; break; }
                            }
                        if (canVector)
                        {
                            Vector256<float> a0 = vb0; Vector256<float> a1 = vb1; Vector256<float> a2 = vb2; Vector256<float> a3 = vb3;
                            Vector256<float> a4 = vb4; Vector256<float> a5 = vb5; Vector256<float> a6 = vb6; Vector256<float> a7 = vb7;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                int w0 = wb0 + ci * 9, w1 = wb1 + ci * 9, w2 = wb2 + ci * 9, w3 = wb3 + ci * 9;
                                int w4 = wb4 + ci * 9, w5 = wb5 + ci * 9, w6 = wb6 + ci * 9, w7 = wb7 + ci * 9;
                                ReadOnlySpan<float> source = input.Slice(inputBatch + ci * inputPlane, inputPlane);
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    int sy = oy * 2 - 1 + ky;
                                    if ((uint)sy >= (uint)inputHeight) continue;
                                    int sourceOffset = sy * inputWidth + 2 * x - 1;
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        Vector256<float> value = LoadStride2(source, sourceOffset + kx);
                                        a0 = AddMul(a0, value, weights[w0 + ky * 3 + kx]); a1 = AddMul(a1, value, weights[w1 + ky * 3 + kx]);
                                        a2 = AddMul(a2, value, weights[w2 + ky * 3 + kx]); a3 = AddMul(a3, value, weights[w3 + ky * 3 + kx]);
                                        a4 = AddMul(a4, value, weights[w4 + ky * 3 + kx]); a5 = AddMul(a5, value, weights[w5 + ky * 3 + kx]);
                                        a6 = AddMul(a6, value, weights[w6 + ky * 3 + kx]); a7 = AddMul(a7, value, weights[w7 + ky * 3 + kx]);
                                    }
                                }
                            }
                            Store(output, o0 + row + x, a0); Store(output, o1 + row + x, a1);
                            Store(output, o2 + row + x, a2); Store(output, o3 + row + x, a3);
                            Store(output, o4 + row + x, a4); Store(output, o5 + row + x, a5);
                            Store(output, o6 + row + x, a6); Store(output, o7 + row + x, a7);
                            x += 8; continue;
                        }
                        for (int lane = 0; lane < Math.Min(8, outputWidth - x); lane++)
                        {
                            int ox = x + lane; float s0 = vb0.GetElement(0), s1 = vb1.GetElement(0), s2 = vb2.GetElement(0), s3 = vb3.GetElement(0);
                            float s4 = vb4.GetElement(0), s5 = vb5.GetElement(0), s6 = vb6.GetElement(0), s7 = vb7.GetElement(0);
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                int w0 = wb0 + ci * 9, w1 = wb1 + ci * 9, w2 = wb2 + ci * 9, w3 = wb3 + ci * 9;
                                int w4 = wb4 + ci * 9, w5 = wb5 + ci * 9, w6 = wb6 + ci * 9, w7 = wb7 + ci * 9;
                                int sourceBase = inputBatch + ci * inputPlane;
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    int iy = oy * 2 - 1 + ky; if ((uint)iy >= (uint)inputHeight) continue;
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        int ix = ox * 2 - 1 + kx; if ((uint)ix >= (uint)inputWidth) continue;
                                        float v = input[sourceBase + iy * inputWidth + ix];
                                        s0 += v * weights[w0 + ky * 3 + kx]; s1 += v * weights[w1 + ky * 3 + kx]; s2 += v * weights[w2 + ky * 3 + kx]; s3 += v * weights[w3 + ky * 3 + kx];
                                        s4 += v * weights[w4 + ky * 3 + kx]; s5 += v * weights[w5 + ky * 3 + kx]; s6 += v * weights[w6 + ky * 3 + kx]; s7 += v * weights[w7 + ky * 3 + kx];
                                    }
                                }
                            }
                            output[o0 + row + ox] = s0; output[o1 + row + ox] = s1; output[o2 + row + ox] = s2; output[o3 + row + ox] = s3;
                            output[o4 + row + ox] = s4; output[o5 + row + ox] = s5; output[o6 + row + ox] = s6; output[o7 + row + ox] = s7;
                        }
                        x += 8;
                    }
                }
            }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv3x3Stride2SixteenOutputsPackedUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth,
        int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        const int weightsPerInput = 9 * 8;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 16)
                {
                    float* w0 = weightsPtr + (co / 8) * inputChannels * weightsPerInput;
                    float* w8 = w0 + inputChannels * weightsPerInput;
                    float* o0 = outputPtr + (b * outputChannels + co) * outputPlane;
                    float* o1 = o0 + outputPlane, o2 = o1 + outputPlane, o3 = o2 + outputPlane;
                    float* o4 = o3 + outputPlane, o5 = o4 + outputPlane, o6 = o5 + outputPlane, o7 = o6 + outputPlane;
                    float* o8 = o7 + outputPlane, o9 = o8 + outputPlane, o10 = o9 + outputPlane, o11 = o10 + outputPlane;
                    float* o12 = o11 + outputPlane, o13 = o12 + outputPlane, o14 = o13 + outputPlane, o15 = o14 + outputPlane;
                    Vector256<float> b0 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co]);
                    Vector256<float> b1 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                    Vector256<float> b2 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                    Vector256<float> b3 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                    Vector256<float> b4 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 4]);
                    Vector256<float> b5 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 5]);
                    Vector256<float> b6 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 6]);
                    Vector256<float> b7 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 7]);
                    Vector256<float> b8 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 8]);
                    Vector256<float> b9 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 9]);
                    Vector256<float> b10 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 10]);
                    Vector256<float> b11 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 11]);
                    Vector256<float> b12 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 12]);
                    Vector256<float> b13 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 13]);
                    Vector256<float> b14 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 14]);
                    Vector256<float> b15 = Vector256.Create(biasPtr == null ? 0f : biasPtr[co + 15]);
                    float* batchInput = inputPtr + b * inputChannels * inputPlane;
                    for (int oy = 0; oy < outputHeight; oy++)
                    {
                        int row = oy * outputWidth, ox = 0;
                        while (ox < outputWidth)
                        {
                            bool vector = oy > 0 && oy * 2 + 1 < inputHeight && ox > 0 && ox + 8 <= outputWidth &&
                                (ox + 7) * 2 + 1 < inputWidth;
                            if (vector)
                            {
                                Vector256<float> a0 = b0, a1 = b1, a2 = b2, a3 = b3;
                                Vector256<float> a4 = b4, a5 = b5, a6 = b6, a7 = b7;
                                Vector256<float> a8 = b8, a9 = b9, a10 = b10, a11 = b11;
                                Vector256<float> a12 = b12, a13 = b13, a14 = b14, a15 = b15;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* source = batchInput + ci * inputPlane;
                                    float* weights0 = w0 + ci * weightsPerInput;
                                    float* weights8 = w8 + ci * weightsPerInput;
                                    int sourceOffset = ox * 2 - 1;
                                    float* row0 = source + (oy * 2 - 1) * inputWidth + sourceOffset;
                                    float* row1 = row0 + inputWidth, row2 = row1 + inputWidth;
                                    Vector256<float> value = LoadStride2(row0);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8);
                                    value = LoadStride2(row0 + 1);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0 + 8);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8 + 8);
                                    value = LoadStride2(row0 + 2);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0 + 16);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8 + 16);
                                    value = LoadStride2(row1);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0 + 24);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8 + 24);
                                    value = LoadStride2(row1 + 1);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0 + 32);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8 + 32);
                                    value = LoadStride2(row1 + 2);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0 + 40);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8 + 40);
                                    value = LoadStride2(row2);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0 + 48);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8 + 48);
                                    value = LoadStride2(row2 + 1);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0 + 56);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8 + 56);
                                    value = LoadStride2(row2 + 2);
                                    AddEightPacked(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, value, weights0 + 64);
                                    AddEightPacked(ref a8, ref a9, ref a10, ref a11, ref a12, ref a13, ref a14, ref a15, value, weights8 + 64);
                                }
                                Avx.Store(o0 + row + ox, a0); Avx.Store(o1 + row + ox, a1);
                                Avx.Store(o2 + row + ox, a2); Avx.Store(o3 + row + ox, a3);
                                Avx.Store(o4 + row + ox, a4); Avx.Store(o5 + row + ox, a5);
                                Avx.Store(o6 + row + ox, a6); Avx.Store(o7 + row + ox, a7);
                                Avx.Store(o8 + row + ox, a8); Avx.Store(o9 + row + ox, a9);
                                Avx.Store(o10 + row + ox, a10); Avx.Store(o11 + row + ox, a11);
                                Avx.Store(o12 + row + ox, a12); Avx.Store(o13 + row + ox, a13);
                                Avx.Store(o14 + row + ox, a14); Avx.Store(o15 + row + ox, a15);
                                ox += 8;
                            }
                            else
                            {
                                float s0 = b0.GetElement(0), s1 = b1.GetElement(0), s2 = b2.GetElement(0), s3 = b3.GetElement(0);
                                float s4 = b4.GetElement(0), s5 = b5.GetElement(0), s6 = b6.GetElement(0), s7 = b7.GetElement(0);
                                float s8 = b8.GetElement(0), s9 = b9.GetElement(0), s10 = b10.GetElement(0), s11 = b11.GetElement(0);
                                float s12 = b12.GetElement(0), s13 = b13.GetElement(0), s14 = b14.GetElement(0), s15 = b15.GetElement(0);
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* source = batchInput + ci * inputPlane;
                                    float* weights0 = w0 + ci * weightsPerInput;
                                    float* weights8 = w8 + ci * weightsPerInput;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = oy * 2 - 1 + ky;
                                        if ((uint)iy >= (uint)inputHeight) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = ox * 2 - 1 + kx;
                                            if ((uint)ix >= (uint)inputWidth) continue;
                                            float value = source[iy * inputWidth + ix];
                                            int k = (ky * 3 + kx) * 8;
                                            s0 += value * weights0[k]; s1 += value * weights0[k + 1]; s2 += value * weights0[k + 2]; s3 += value * weights0[k + 3];
                                            s4 += value * weights0[k + 4]; s5 += value * weights0[k + 5]; s6 += value * weights0[k + 6]; s7 += value * weights0[k + 7];
                                            s8 += value * weights8[k]; s9 += value * weights8[k + 1]; s10 += value * weights8[k + 2]; s11 += value * weights8[k + 3];
                                            s12 += value * weights8[k + 4]; s13 += value * weights8[k + 5]; s14 += value * weights8[k + 6]; s15 += value * weights8[k + 7];
                                        }
                                    }
                                }
                                o0[row + ox] = s0; o1[row + ox] = s1; o2[row + ox] = s2; o3[row + ox] = s3;
                                o4[row + ox] = s4; o5[row + ox] = s5; o6[row + ox] = s6; o7[row + ox] = s7;
                                o8[row + ox] = s8; o9[row + ox] = s9; o10[row + ox] = s10; o11[row + ox] = s11;
                                o12[row + ox] = s12; o13[row + ox] = s13; o14[row + ox] = s14; o15[row + ox] = s15;
                                ox++;
                            }
                        }
                    }
                }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv3x3Stride2EightOutputsPackedUnsafe(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth,
        int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        const int weightsPerInput = 9 * 8;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* w = weightsPtr + (co / 8) * inputChannels * weightsPerInput;
                    float* o0 = outputPtr + (b * outputChannels + co) * outputPlane;
                    float* o1 = o0 + outputPlane, o2 = o1 + outputPlane, o3 = o2 + outputPlane;
                    float* o4 = o3 + outputPlane, o5 = o4 + outputPlane, o6 = o5 + outputPlane, o7 = o6 + outputPlane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4], b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6], b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    Vector256<float> vb0 = Vector256.Create(b0), vb1 = Vector256.Create(b1);
                    Vector256<float> vb2 = Vector256.Create(b2), vb3 = Vector256.Create(b3);
                    Vector256<float> vb4 = Vector256.Create(b4), vb5 = Vector256.Create(b5);
                    Vector256<float> vb6 = Vector256.Create(b6), vb7 = Vector256.Create(b7);
                    float* batchInput = inputPtr + b * inputChannels * inputPlane;
                    for (int oy = 0; oy < outputHeight; oy++)
                    {
                        int row = oy * outputWidth, ox = 0;
                        while (ox < outputWidth)
                        {
                            bool vector = oy > 0 && oy * 2 + 1 < inputHeight && ox > 0 && ox + 8 <= outputWidth &&
                                (ox + 7) * 2 + 1 < inputWidth;
                            if (vector)
                            {
                                Vector256<float> a0 = vb0, a1 = vb1;
                                Vector256<float> a2 = vb2, a3 = vb3;
                                Vector256<float> a4 = vb4, a5 = vb5;
                                Vector256<float> a6 = vb6, a7 = vb7;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * inputPlane;
                                    float* wc = w + ci * weightsPerInput;
                                    int sourceOffset = ox * 2 - 1;
                                    float* row0 = src + (oy * 2 - 1) * inputWidth + sourceOffset;
                                    float* row1 = row0 + inputWidth, row2 = row1 + inputWidth;
                                    Vector256<float> v0 = LoadStride2(row0), v1 = LoadStride2(row0 + 1), v2 = LoadStride2(row0 + 2);
                                    float* weights = wc;
                                    a0 = AddMul(a0, v0, weights[0]); a1 = AddMul(a1, v0, weights[1]);
                                    a2 = AddMul(a2, v0, weights[2]); a3 = AddMul(a3, v0, weights[3]);
                                    a4 = AddMul(a4, v0, weights[4]); a5 = AddMul(a5, v0, weights[5]);
                                    a6 = AddMul(a6, v0, weights[6]); a7 = AddMul(a7, v0, weights[7]);
                                    weights += 8;
                                    a0 = AddMul(a0, v1, weights[0]); a1 = AddMul(a1, v1, weights[1]);
                                    a2 = AddMul(a2, v1, weights[2]); a3 = AddMul(a3, v1, weights[3]);
                                    a4 = AddMul(a4, v1, weights[4]); a5 = AddMul(a5, v1, weights[5]);
                                    a6 = AddMul(a6, v1, weights[6]); a7 = AddMul(a7, v1, weights[7]);
                                    weights += 8;
                                    a0 = AddMul(a0, v2, weights[0]); a1 = AddMul(a1, v2, weights[1]);
                                    a2 = AddMul(a2, v2, weights[2]); a3 = AddMul(a3, v2, weights[3]);
                                    a4 = AddMul(a4, v2, weights[4]); a5 = AddMul(a5, v2, weights[5]);
                                    a6 = AddMul(a6, v2, weights[6]); a7 = AddMul(a7, v2, weights[7]);
                                    weights += 8;
                                    v0 = LoadStride2(row1); v1 = LoadStride2(row1 + 1); v2 = LoadStride2(row1 + 2);
                                    a0 = AddMul(a0, v0, weights[0]); a1 = AddMul(a1, v0, weights[1]);
                                    a2 = AddMul(a2, v0, weights[2]); a3 = AddMul(a3, v0, weights[3]);
                                    a4 = AddMul(a4, v0, weights[4]); a5 = AddMul(a5, v0, weights[5]);
                                    a6 = AddMul(a6, v0, weights[6]); a7 = AddMul(a7, v0, weights[7]);
                                    weights += 8;
                                    a0 = AddMul(a0, v1, weights[0]); a1 = AddMul(a1, v1, weights[1]);
                                    a2 = AddMul(a2, v1, weights[2]); a3 = AddMul(a3, v1, weights[3]);
                                    a4 = AddMul(a4, v1, weights[4]); a5 = AddMul(a5, v1, weights[5]);
                                    a6 = AddMul(a6, v1, weights[6]); a7 = AddMul(a7, v1, weights[7]);
                                    weights += 8;
                                    a0 = AddMul(a0, v2, weights[0]); a1 = AddMul(a1, v2, weights[1]);
                                    a2 = AddMul(a2, v2, weights[2]); a3 = AddMul(a3, v2, weights[3]);
                                    a4 = AddMul(a4, v2, weights[4]); a5 = AddMul(a5, v2, weights[5]);
                                    a6 = AddMul(a6, v2, weights[6]); a7 = AddMul(a7, v2, weights[7]);
                                    weights += 8;
                                    v0 = LoadStride2(row2); v1 = LoadStride2(row2 + 1); v2 = LoadStride2(row2 + 2);
                                    a0 = AddMul(a0, v0, weights[0]); a1 = AddMul(a1, v0, weights[1]);
                                    a2 = AddMul(a2, v0, weights[2]); a3 = AddMul(a3, v0, weights[3]);
                                    a4 = AddMul(a4, v0, weights[4]); a5 = AddMul(a5, v0, weights[5]);
                                    a6 = AddMul(a6, v0, weights[6]); a7 = AddMul(a7, v0, weights[7]);
                                    weights += 8;
                                    a0 = AddMul(a0, v1, weights[0]); a1 = AddMul(a1, v1, weights[1]);
                                    a2 = AddMul(a2, v1, weights[2]); a3 = AddMul(a3, v1, weights[3]);
                                    a4 = AddMul(a4, v1, weights[4]); a5 = AddMul(a5, v1, weights[5]);
                                    a6 = AddMul(a6, v1, weights[6]); a7 = AddMul(a7, v1, weights[7]);
                                    weights += 8;
                                    a0 = AddMul(a0, v2, weights[0]); a1 = AddMul(a1, v2, weights[1]);
                                    a2 = AddMul(a2, v2, weights[2]); a3 = AddMul(a3, v2, weights[3]);
                                    a4 = AddMul(a4, v2, weights[4]); a5 = AddMul(a5, v2, weights[5]);
                                    a6 = AddMul(a6, v2, weights[6]); a7 = AddMul(a7, v2, weights[7]);
                                }
                                Avx.Store(o0 + row + ox, a0); Avx.Store(o1 + row + ox, a1);
                                Avx.Store(o2 + row + ox, a2); Avx.Store(o3 + row + ox, a3);
                                Avx.Store(o4 + row + ox, a4); Avx.Store(o5 + row + ox, a5);
                                Avx.Store(o6 + row + ox, a6); Avx.Store(o7 + row + ox, a7);
                                ox += 8;
                            }
                            else
                            {
                                float s0 = b0, s1 = b1, s2 = b2, s3 = b3, s4 = b4, s5 = b5, s6 = b6, s7 = b7;
                                for (int ci = 0; ci < inputChannels; ci++)
                                {
                                    float* src = batchInput + ci * inputPlane;
                                    float* wc = w + ci * weightsPerInput;
                                    for (int ky = 0; ky < 3; ky++)
                                    {
                                        int iy = oy * 2 - 1 + ky;
                                        if ((uint)iy >= (uint)inputHeight) continue;
                                        for (int kx = 0; kx < 3; kx++)
                                        {
                                            int ix = ox * 2 - 1 + kx;
                                            if ((uint)ix >= (uint)inputWidth) continue;
                                            float value = src[iy * inputWidth + ix];
                                            float* weights = wc + (ky * 3 + kx) * 8;
                                            s0 += value * weights[0]; s1 += value * weights[1]; s2 += value * weights[2]; s3 += value * weights[3];
                                            s4 += value * weights[4]; s5 += value * weights[5]; s6 += value * weights[6]; s7 += value * weights[7];
                                        }
                                    }
                                }
                                o0[row + ox] = s0; o1[row + ox] = s1; o2[row + ox] = s2; o3[row + ox] = s3;
                                o4[row + ox] = s4; o5[row + ox] = s5; o6[row + ox] = s6; o7[row + ox] = s7;
                                ox++;
                            }
                        }
                    }
                }
        }
    }
}
