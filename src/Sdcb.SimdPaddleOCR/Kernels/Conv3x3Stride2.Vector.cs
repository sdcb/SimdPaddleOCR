using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Conv3x3Stride2
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe bool TryVector(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int inputHeight, int inputWidth, int outputHeight, int outputWidth, int outputChannels,
        int intraOpThreads)
    {
        int inputPlane = checked(inputHeight * inputWidth);
        int outputPlane = checked(outputHeight * outputWidth);
        int weightsPerOutput = checked(inputChannels * 9);
        int widthLanes = Vector<float>.Count;
        if (intraOpThreads > 1 && batch == 1 && outputChannels >= 8 &&
            (long)outputChannels * weightsPerOutput * outputPlane >= IntraOpMinWork)
        {
            int workers = Math.Min(intraOpThreads, outputChannels / 4);
            fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
            {
                nint inputAddress = (nint)inputPtr, weightsAddress = (nint)weightsPtr,
                    biasAddress = (nint)biasPtr, outputAddress = (nint)outputPtr;
                int inputLength = input.Length, weightsLength = weights.Length, biasLength = bias.Length,
                    outputLength = output.Length;
                Parallel.For(0, workers, worker =>
                {
                    int begin = (outputChannels * worker / workers) & ~3;
                    int end = worker == workers - 1 ? outputChannels : (outputChannels * (worker + 1) / workers) & ~3;
                    if (end <= begin) return;
                    int count = end - begin;
                    ReadOnlySpan<float> inSpan = new((void*)inputAddress, inputLength);
                    ReadOnlySpan<float> w = new ReadOnlySpan<float>((void*)weightsAddress, weightsLength)
                        .Slice(begin * weightsPerOutput, count * weightsPerOutput);
                    ReadOnlySpan<float> b = biasLength == 0 ? []
                        : new ReadOnlySpan<float>((void*)biasAddress, biasLength).Slice(begin, count);
                    Span<float> outSpan = new Span<float>((void*)outputAddress, outputLength)
                        .Slice(begin * outputPlane, count * outputPlane);
                    TryVector(inSpan, w, b, outSpan, 1, inputChannels, inputHeight,
                        inputWidth, outputHeight, outputWidth, count, 1);
                });
            }
        }
        else if ((outputChannels & 7) == 0)
        {
            fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
            {
                for (int b = 0; b < batch; b++)
                    for (int co = 0; co < outputChannels; co += 8)
                    {
                        float* w0 = weightsPtr + co * weightsPerOutput;
                        float* w1 = w0 + weightsPerOutput, w2 = w1 + weightsPerOutput, w3 = w2 + weightsPerOutput;
                        float* w4 = w3 + weightsPerOutput, w5 = w4 + weightsPerOutput, w6 = w5 + weightsPerOutput, w7 = w6 + weightsPerOutput;
                        float* o0 = outputPtr + (b * outputChannels + co) * outputPlane;
                        float* o1 = o0 + outputPlane, o2 = o1 + outputPlane, o3 = o2 + outputPlane;
                        float* o4 = o3 + outputPlane, o5 = o4 + outputPlane, o6 = o5 + outputPlane, o7 = o6 + outputPlane;
                        float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                        float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                        float b4 = biasPtr == null ? 0f : biasPtr[co + 4], b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                        float b6 = biasPtr == null ? 0f : biasPtr[co + 6], b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                        Vector<float> vb0 = new(b0), vb1 = new(b1), vb2 = new(b2), vb3 = new(b3);
                        Vector<float> vb4 = new(b4), vb5 = new(b5), vb6 = new(b6), vb7 = new(b7);
                        float* batchInput = inputPtr + b * inputChannels * inputPlane;
                        for (int oy = 0; oy < outputHeight; oy++)
                        {
                            int rowOffset = oy * outputWidth, ox = 0;
                            while (ox < outputWidth)
                            {
                                bool fullHeight = oy != 0 && oy * 2 + 1 < inputHeight;
                                bool fullWidth = ox != 0 && ox + widthLanes <= outputWidth &&
                                    (ox + widthLanes - 1) * 2 + 1 < inputWidth;
                                if (fullHeight && fullWidth)
                                {
                                    Vector<float> a0 = vb0, a1 = vb1, a2 = vb2, a3 = vb3;
                                    Vector<float> a4 = vb4, a5 = vb5, a6 = vb6, a7 = vb7;
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
                                                Vector<float> value = VectorLoadStride2(srcRow + ix + kx);
                                                int wi = wb + ky * 3 + kx;
                                                a0 = VectorAddMul(a0, value, w0[wi]); a1 = VectorAddMul(a1, value, w1[wi]);
                                                a2 = VectorAddMul(a2, value, w2[wi]); a3 = VectorAddMul(a3, value, w3[wi]);
                                                a4 = VectorAddMul(a4, value, w4[wi]); a5 = VectorAddMul(a5, value, w5[wi]);
                                                a6 = VectorAddMul(a6, value, w6[wi]); a7 = VectorAddMul(a7, value, w7[wi]);
                                            }
                                        }
                                    }
                                    VectorStore(o0 + rowOffset + ox, a0); VectorStore(o1 + rowOffset + ox, a1);
                                    VectorStore(o2 + rowOffset + ox, a2); VectorStore(o3 + rowOffset + ox, a3);
                                    VectorStore(o4 + rowOffset + ox, a4); VectorStore(o5 + rowOffset + ox, a5);
                                    VectorStore(o6 + rowOffset + ox, a6); VectorStore(o7 + rowOffset + ox, a7);
                                    ox += widthLanes;
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
        else if ((outputChannels & 3) == 0)
        {
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
                            int dualLanes = widthLanes * 2;
                            bool canDual = CanStride2Width(x, dualLanes, outputWidth, inputWidth);
                            bool canVector = !canDual && CanStride2Width(x, widthLanes, outputWidth, inputWidth);
                            if (canDual)
                            {
                                Vector<float> a0 = new(bias0), a1 = new(bias1), a2 = new(bias2), a3 = new(bias3);
                                Vector<float> c0 = new(bias0), c1 = new(bias1), c2 = new(bias2), c3 = new(bias3);
                                int nextOffset = 2 * widthLanes;
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
                                            Vector<float> valueLow = VectorLoadStride2(source, sourceOffset + kx);
                                            Vector<float> valueHigh = VectorLoadStride2(source, sourceOffset + kx + nextOffset);
                                            float w0 = weights[wb0 + ky * 3 + kx], w1 = weights[wb1 + ky * 3 + kx];
                                            float w2 = weights[wb2 + ky * 3 + kx], w3 = weights[wb3 + ky * 3 + kx];
                                            a0 = VectorAddMul(a0, valueLow, w0); c0 = VectorAddMul(c0, valueHigh, w0);
                                            a1 = VectorAddMul(a1, valueLow, w1); c1 = VectorAddMul(c1, valueHigh, w1);
                                            a2 = VectorAddMul(a2, valueLow, w2); c2 = VectorAddMul(c2, valueHigh, w2);
                                            a3 = VectorAddMul(a3, valueLow, w3); c3 = VectorAddMul(c3, valueHigh, w3);
                                        }
                                    }
                                }
                                VectorStore(output, output0 + row + x, a0);
                                VectorStore(output, output1 + row + x, a1);
                                VectorStore(output, output2 + row + x, a2);
                                VectorStore(output, output3 + row + x, a3);
                                VectorStore(output, output0 + row + x + widthLanes, c0);
                                VectorStore(output, output1 + row + x + widthLanes, c1);
                                VectorStore(output, output2 + row + x + widthLanes, c2);
                                VectorStore(output, output3 + row + x + widthLanes, c3);
                                x += dualLanes;
                            }
                            else if (canVector)
                            {
                                Vector<float> a0 = new(bias0), a1 = new(bias1), a2 = new(bias2), a3 = new(bias3);
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
                                            Vector<float> value = VectorLoadStride2(source, sourceOffset + kx);
                                            a0 = VectorAddMul(a0, value, weights[wb0 + ky * 3 + kx]);
                                            a1 = VectorAddMul(a1, value, weights[wb1 + ky * 3 + kx]);
                                            a2 = VectorAddMul(a2, value, weights[wb2 + ky * 3 + kx]);
                                            a3 = VectorAddMul(a3, value, weights[wb3 + ky * 3 + kx]);
                                        }
                                    }
                                }
                                VectorStore(output, output0 + row + x, a0);
                                VectorStore(output, output1 + row + x, a1);
                                VectorStore(output, output2 + row + x, a2);
                                VectorStore(output, output3 + row + x, a3);
                                x += widthLanes;
                            }
                            else
                            {
                                for (int lane = 0; lane < Math.Min(widthLanes, outputWidth - x); lane++)
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
                                x += widthLanes;
                            }
                        }
                    }
                }
        }
        else
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co++)
                {
                    int outputOffset = (b * outputChannels + co) * outputPlane;
                    float initial = bias.IsEmpty ? 0f : bias[co];
                    Vector<float> initialVector = new(initial);
                    int i = 0;
                    for (; i <= outputPlane - widthLanes; i += widthLanes)
                        VectorStore(output, outputOffset + i, initialVector);
                    for (; i < outputPlane; i++) output[outputOffset + i] = initial;
                    int weightBase = co * weightsPerOutput;
                    int inputBatch = b * inputChannels * inputPlane;
                    for (int ci = 0; ci < inputChannels; ci++)
                    {
                        ReadOnlySpan<float> source = input.Slice(inputBatch + ci * inputPlane, inputPlane);
                        int channelWeights = weightBase + ci * 9;
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++)
                            {
                                float weight = weights[channelWeights + ky * 3 + kx];
                                for (int oy = 0; oy < outputHeight; oy++)
                                {
                                    int sourceY = oy * 2 - 1 + ky;
                                    if ((uint)sourceY >= (uint)inputHeight) continue;
                                    int row = oy * outputWidth;
                                    int x = 0;
                                    for (; x < outputWidth && (2 * x - 1 + kx < 0); x++) { }
                                    int vectorEnd = outputWidth;
                                    while (vectorEnd > x &&
                                        2 * (vectorEnd - 1) - 1 + kx + (widthLanes * 2 - 1) >= inputWidth)
                                        vectorEnd--;
                                    for (; x <= vectorEnd - widthLanes; x += widthLanes)
                                    {
                                        int sourceX = 2 * x - 1 + kx;
                                        Vector<float> value = VectorLoadStride2(source, sourceY * inputWidth + sourceX);
                                        Vector<float> current = VectorLoad(output, outputOffset + row + x);
                                        VectorStore(output, outputOffset + row + x, VectorAddMul(current, value, weight));
                                    }
                                    for (; x < vectorEnd; x++)
                                    {
                                        int sourceX = 2 * x - 1 + kx;
                                        output[outputOffset + row + x] += source[sourceY * inputWidth + sourceX] * weight;
                                    }
                                    for (; x < outputWidth; x++)
                                    {
                                        int sourceX = 2 * x - 1 + kx;
                                        if ((uint)sourceX < (uint)inputWidth)
                                            output[outputOffset + row + x] += source[sourceY * inputWidth + sourceX] * weight;
                                    }
                                }
                            }
                    }
                }
        }
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv3x3Stride2EightOutputsVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int inputHeight, int inputWidth,
        int outputHeight, int outputWidth, int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        int weightsPerOutput = checked(inputChannels * 9), widthLanes = Vector<float>.Count;
        fixed (float* inputPtr = input, weightsPtr = weights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* w0 = weightsPtr + co * weightsPerOutput;
                    float* w1 = w0 + weightsPerOutput, w2 = w1 + weightsPerOutput, w3 = w2 + weightsPerOutput;
                    float* w4 = w3 + weightsPerOutput, w5 = w4 + weightsPerOutput, w6 = w5 + weightsPerOutput, w7 = w6 + weightsPerOutput;
                    float* o0 = outputPtr + (b * outputChannels + co) * outputPlane;
                    float* o1 = o0 + outputPlane, o2 = o1 + outputPlane, o3 = o2 + outputPlane;
                    float* o4 = o3 + outputPlane, o5 = o4 + outputPlane, o6 = o5 + outputPlane, o7 = o6 + outputPlane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4], b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6], b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    Vector<float> vb0 = new(b0), vb1 = new(b1), vb2 = new(b2), vb3 = new(b3);
                    Vector<float> vb4 = new(b4), vb5 = new(b5), vb6 = new(b6), vb7 = new(b7);
                    float* batchInput = inputPtr + b * inputChannels * inputPlane;
                    for (int oy = 0; oy < outputHeight; oy++)
                    {
                        int rowOffset = oy * outputWidth, ox = 0;
                        while (ox < outputWidth)
                        {
                            bool fullHeight = oy != 0 && oy * 2 + 1 < inputHeight;
                            bool fullWidth = ox != 0 && ox + widthLanes <= outputWidth &&
                                (ox + widthLanes - 1) * 2 + 1 < inputWidth;
                            if (fullHeight && fullWidth)
                            {
                                Vector<float> a0 = vb0, a1 = vb1, a2 = vb2, a3 = vb3;
                                Vector<float> a4 = vb4, a5 = vb5, a6 = vb6, a7 = vb7;
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
                                            Vector<float> value = VectorLoadStride2(srcRow + ix + kx);
                                            int wi = wb + ky * 3 + kx;
                                            a0 = VectorAddMul(a0, value, w0[wi]); a1 = VectorAddMul(a1, value, w1[wi]);
                                            a2 = VectorAddMul(a2, value, w2[wi]); a3 = VectorAddMul(a3, value, w3[wi]);
                                            a4 = VectorAddMul(a4, value, w4[wi]); a5 = VectorAddMul(a5, value, w5[wi]);
                                            a6 = VectorAddMul(a6, value, w6[wi]); a7 = VectorAddMul(a7, value, w7[wi]);
                                        }
                                    }
                                }
                                VectorStore(o0 + rowOffset + ox, a0); VectorStore(o1 + rowOffset + ox, a1);
                                VectorStore(o2 + rowOffset + ox, a2); VectorStore(o3 + rowOffset + ox, a3);
                                VectorStore(o4 + rowOffset + ox, a4); VectorStore(o5 + rowOffset + ox, a5);
                                VectorStore(o6 + rowOffset + ox, a6); VectorStore(o7 + rowOffset + ox, a7);
                                ox += widthLanes;
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

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void Conv3x3Stride2FourOutputsVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int inputHeight, int inputWidth,
        int outputHeight, int outputWidth, int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth);
        int outputPlane = checked(outputHeight * outputWidth);
        int weightsPerOutput = checked(inputChannels * 9);
        int widthLanes = Vector<float>.Count;
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
                        bool canVector = x + widthLanes <= outputWidth;
                        if (canVector)
                        {
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int first = 2 * x - 1 + kx;
                                if (first < 0 || first + (widthLanes * 2 - 2) >= inputWidth)
                                {
                                    canVector = false;
                                    break;
                                }
                            }
                        }
                        if (canVector)
                        {
                            Vector<float> a0 = new(bias0), a1 = new(bias1), a2 = new(bias2), a3 = new(bias3);
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
                                        Vector<float> value = VectorLoadStride2(source, sourceOffset + kx);
                                        a0 = VectorAddMul(a0, value, weights[wb0 + ky * 3 + kx]);
                                        a1 = VectorAddMul(a1, value, weights[wb1 + ky * 3 + kx]);
                                        a2 = VectorAddMul(a2, value, weights[wb2 + ky * 3 + kx]);
                                        a3 = VectorAddMul(a3, value, weights[wb3 + ky * 3 + kx]);
                                    }
                                }
                            }
                            VectorStore(output, output0 + row + x, a0);
                            VectorStore(output, output1 + row + x, a1);
                            VectorStore(output, output2 + row + x, a2);
                            VectorStore(output, output3 + row + x, a3);
                            x += widthLanes;
                            continue;
                        }

                        for (int lane = 0; lane < Math.Min(widthLanes, outputWidth - x); lane++)
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
                        x += widthLanes;
                    }
                }
            }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void TryPackedVector(ReadOnlySpan<float> input,
        ReadOnlySpan<float> packedWeights, ReadOnlySpan<float> bias, Span<float> output, int batch,
        int inputChannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth,
        int outputChannels)
    {
        int inputPlane = checked(inputHeight * inputWidth), outputPlane = checked(outputHeight * outputWidth);
        const int weightsPerInput = 9 * 8;
        int widthLanes = Vector<float>.Count;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* w = weightsPtr + (co / 8) * inputChannels * weightsPerInput;
                    float* blockBias = biasPtr == null ? null : biasPtr + co;
                    float* o0 = outputPtr + (b * outputChannels + co) * outputPlane;
                    float* o1 = o0 + outputPlane, o2 = o1 + outputPlane, o3 = o2 + outputPlane;
                    float* o4 = o3 + outputPlane, o5 = o4 + outputPlane, o6 = o5 + outputPlane, o7 = o6 + outputPlane;
                    float b0 = biasPtr == null ? 0f : biasPtr[co], b1 = biasPtr == null ? 0f : biasPtr[co + 1];
                    float b2 = biasPtr == null ? 0f : biasPtr[co + 2], b3 = biasPtr == null ? 0f : biasPtr[co + 3];
                    float b4 = biasPtr == null ? 0f : biasPtr[co + 4], b5 = biasPtr == null ? 0f : biasPtr[co + 5];
                    float b6 = biasPtr == null ? 0f : biasPtr[co + 6], b7 = biasPtr == null ? 0f : biasPtr[co + 7];
                    float* batchInput = inputPtr + b * inputChannels * inputPlane;
                    for (int oy = 0; oy < outputHeight; oy++)
                    {
                        int row = oy * outputWidth, ox = 0, dualLanes = widthLanes * 2;
                        bool fullHeight = oy > 0 && oy * 2 + 1 < inputHeight;
                        float* tileRow = batchInput + (oy * 2 - 1) * inputWidth;
                        while (ox < outputWidth)
                        {
                            bool dual = fullHeight && ox > 0 &&
                                ox + dualLanes <= outputWidth &&
                                (ox + dualLanes - 1) * 2 + 1 < inputWidth;
                            bool vector = !dual && fullHeight && ox > 0 &&
                                ox + widthLanes <= outputWidth &&
                                (ox + widthLanes - 1) * 2 + 1 < inputWidth;
                            if (dual)
                            {
                                PackedStride2TileEightDual(tileRow + ox * 2 - 1, w, blockBias, o0 + row + ox,
                                    outputPlane, inputChannels, inputPlane, inputWidth);
                                ox += dualLanes;
                            }
                            else if (vector)
                            {
                                PackedStride2TileEight(tileRow + ox * 2 - 1, w, blockBias, o0 + row + ox,
                                    outputPlane, inputChannels, inputPlane, inputWidth);
                                ox += widthLanes;
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
                                            s0 += value * weights[0]; s1 += value * weights[1];
                                            s2 += value * weights[2]; s3 += value * weights[3];
                                            s4 += value * weights[4]; s5 += value * weights[5];
                                            s6 += value * weights[6]; s7 += value * weights[7];
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

    // One interior tile: Vector<float>.Count output columns for eight packed
    // output channels. Kept as its own method so the inliner budget covers the
    // nine taps; `source` is input row (2*oy-1), column (2*ox-1) of channel 0.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void PackedStride2TileEight(float* source, float* packed, float* bias,
        float* output, int outputPlane, int inputChannels, int inputPlane, int inputWidth)
    {
        Vector<float> a0 = new(bias == null ? 0f : bias[0]), a1 = new(bias == null ? 0f : bias[1]);
        Vector<float> a2 = new(bias == null ? 0f : bias[2]), a3 = new(bias == null ? 0f : bias[3]);
        Vector<float> a4 = new(bias == null ? 0f : bias[4]), a5 = new(bias == null ? 0f : bias[5]);
        Vector<float> a6 = new(bias == null ? 0f : bias[6]), a7 = new(bias == null ? 0f : bias[7]);
        for (int ci = 0; ci < inputChannels; ci++, source += inputPlane, packed += 9 * 8)
        {
            float* row1 = source + inputWidth, row2 = row1 + inputWidth;
            AccumulatePackedStride2Eight(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
                source, row1, row2, packed);
        }
        VectorStore(output, a0); VectorStore(output + outputPlane, a1);
        VectorStore(output + 2 * outputPlane, a2); VectorStore(output + 3 * outputPlane, a3);
        VectorStore(output + 4 * outputPlane, a4); VectorStore(output + 5 * outputPlane, a5);
        VectorStore(output + 6 * outputPlane, a6); VectorStore(output + 7 * outputPlane, a7);
    }

    // Two adjacent interior tiles sharing each weight load.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void PackedStride2TileEightDual(float* source, float* packed, float* bias,
        float* output, int outputPlane, int inputChannels, int inputPlane, int inputWidth)
    {
        int widthLanes = Vector<float>.Count, next = 2 * widthLanes;
        Vector<float> a0 = new(bias == null ? 0f : bias[0]), a1 = new(bias == null ? 0f : bias[1]);
        Vector<float> a2 = new(bias == null ? 0f : bias[2]), a3 = new(bias == null ? 0f : bias[3]);
        Vector<float> a4 = new(bias == null ? 0f : bias[4]), a5 = new(bias == null ? 0f : bias[5]);
        Vector<float> a6 = new(bias == null ? 0f : bias[6]), a7 = new(bias == null ? 0f : bias[7]);
        Vector<float> c0 = a0, c1 = a1, c2 = a2, c3 = a3, c4 = a4, c5 = a5, c6 = a6, c7 = a7;
        for (int ci = 0; ci < inputChannels; ci++, source += inputPlane, packed += 9 * 8)
        {
            float* row1 = source + inputWidth, row2 = row1 + inputWidth;
            AccumulatePackedStride2EightDual(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
                ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
                source, row1, row2, next, packed);
        }
        VectorStore(output, a0); VectorStore(output + widthLanes, c0);
        output += outputPlane; VectorStore(output, a1); VectorStore(output + widthLanes, c1);
        output += outputPlane; VectorStore(output, a2); VectorStore(output + widthLanes, c2);
        output += outputPlane; VectorStore(output, a3); VectorStore(output + widthLanes, c3);
        output += outputPlane; VectorStore(output, a4); VectorStore(output + widthLanes, c4);
        output += outputPlane; VectorStore(output, a5); VectorStore(output + widthLanes, c5);
        output += outputPlane; VectorStore(output, a6); VectorStore(output + widthLanes, c6);
        output += outputPlane; VectorStore(output, a7); VectorStore(output + widthLanes, c7);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool CanStride2Width(int x, int lanes, int outputWidth, int inputWidth)
    {
        if (x + lanes > outputWidth) return false;
        for (int kx = 0; kx < 3; kx++)
        {
            int first = 2 * x - 1 + kx;
            if (first < 0 || first + (lanes * 2 - 2) >= inputWidth)
                return false;
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void AccumulatePackedStride2Eight(
        ref Vector<float> a0, ref Vector<float> a1, ref Vector<float> a2, ref Vector<float> a3,
        ref Vector<float> a4, ref Vector<float> a5, ref Vector<float> a6, ref Vector<float> a7,
        float* row0, float* row1, float* row2, float* wc)
    {
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row0), wc);
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row0 + 1), wc + 8);
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row0 + 2), wc + 16);
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row1), wc + 24);
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row1 + 1), wc + 32);
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row1 + 2), wc + 40);
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row2), wc + 48);
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row2 + 1), wc + 56);
        VectorAddMulPacked8(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, VectorLoadStride2(row2 + 2), wc + 64);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void AccumulatePackedStride2EightDual(
        ref Vector<float> a0, ref Vector<float> a1, ref Vector<float> a2, ref Vector<float> a3,
        ref Vector<float> a4, ref Vector<float> a5, ref Vector<float> a6, ref Vector<float> a7,
        ref Vector<float> c0, ref Vector<float> c1, ref Vector<float> c2, ref Vector<float> c3,
        ref Vector<float> c4, ref Vector<float> c5, ref Vector<float> c6, ref Vector<float> c7,
        float* row0, float* row1, float* row2, int next, float* wc)
    {
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row0), VectorLoadStride2(row0 + next), wc);
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row0 + 1), VectorLoadStride2(row0 + 1 + next), wc + 8);
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row0 + 2), VectorLoadStride2(row0 + 2 + next), wc + 16);
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row1), VectorLoadStride2(row1 + next), wc + 24);
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row1 + 1), VectorLoadStride2(row1 + 1 + next), wc + 32);
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row1 + 2), VectorLoadStride2(row1 + 2 + next), wc + 40);
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row2), VectorLoadStride2(row2 + next), wc + 48);
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row2 + 1), VectorLoadStride2(row2 + 1 + next), wc + 56);
        VectorAddMulPacked8x2(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7,
            ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7,
            VectorLoadStride2(row2 + 2), VectorLoadStride2(row2 + 2 + next), wc + 64);
    }
}
