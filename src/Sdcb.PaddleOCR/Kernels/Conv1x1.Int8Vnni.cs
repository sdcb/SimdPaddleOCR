using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.PaddleOCR.Kernels.SimdOps;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class Conv1x1
{
    /// <summary>
    /// Quantizes NCHW activations once to tile-scaled unsigned bytes, then uses
    /// VPDPBUSD for eight output channels. Adding 128 to symmetric signed
    /// activations is compensated by the precomputed per-output weight sums.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe bool Conv1x1PackedEightOutputsInt8VnniUnsafe(
        ReadOnlySpan<float> input, PackedConv1x1Int8 packedWeights,
        ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels,
        int intraOpThreads)
    {
        int plane = checked(height * width);
        const int TileSpatial = 16;
        int tileCount = (plane + TileSpatial - 1) / TileSpatial;
        byte[] quantizedArray = PooledArrays.Rent<byte>(checked(plane * inputChannels));
        float[] scaleArray = PooledArrays.Rent<float>(tileCount);
        try
        {
            fixed (float* inputPtr = input, outputPtr = output, biasPtr = bias,
                weightScalePtr = packedWeights.Scales)
            fixed (byte* quantizedPtr = quantizedArray, weightPtr = packedWeights.Weights)
            fixed (int* sumPtr = packedWeights.Sums)
            fixed (float* inputScalePtr = scaleArray)
            {
                for (int b = 0; b < batch; b++)
                {
                    int inputBatch = b * inputChannels * plane;
                    for (int tile = 0; tile < tileCount; tile++)
                    {
                        int start = tile * TileSpatial, end = Math.Min(plane, start + TileSpatial);
                        float absMax = 0f;
                        for (int ci = 0; ci < inputChannels; ci++)
                        {
                            float* source = inputPtr + inputBatch + ci * plane + start;
                            for (int spatial = start; spatial < end; spatial++)
                            {
                                float value = *source++;
                                if (!float.IsFinite(value)) return false;
                                absMax = MathF.Max(absMax, MathF.Abs(value));
                            }
                        }

                        float inputScale = absMax > 0f ? absMax / 127f : 0f;
                        inputScalePtr[tile] = inputScale;
                        float inverseScale = absMax > 0f ? 127f / absMax : 0f;
                        for (int spatial = start; spatial < end; spatial++)
                        {
                            byte* destination = quantizedPtr + spatial * inputChannels;
                            float* source = inputPtr + inputBatch + spatial;
                            for (int ci = 0; ci < inputChannels; ci++)
                            {
                                int q = Math.Clamp((int)MathF.Round(*source * inverseScale), -127, 127);
                                destination[ci] = (byte)(q + 128);
                                source += plane;
                            }
                        }
                    }

                    int blocks = outputChannels / 8;
                    long work = (long)outputChannels * inputChannels * plane;
                    int workers = work >= 1_000_000 ? Math.Min(Math.Max(1, intraOpThreads), blocks) : 1;
                    if (workers == 1)
                    {
                        ProcessInt8VnniBlocks(quantizedPtr, weightPtr, weightScalePtr, sumPtr,
                            biasPtr, outputPtr + b * outputChannels * plane, inputScalePtr,
                            inputChannels, plane, outputChannels, tileCount, 0, blocks);
                    }
                    else
                    {
                        nint qAddress = (nint)quantizedPtr, wAddress = (nint)weightPtr;
                        nint wsAddress = (nint)weightScalePtr, sumAddress = (nint)sumPtr;
                        nint biasAddress = (nint)biasPtr;
                        nint outputAddress = (nint)(outputPtr + b * outputChannels * plane);
                        nint scaleAddress = (nint)inputScalePtr;
                        Parallel.For(0, workers, worker =>
                        {
                            int begin = blocks * worker / workers;
                            int end = blocks * (worker + 1) / workers;
                            ProcessInt8VnniBlocks((byte*)qAddress, (byte*)wAddress, (float*)wsAddress,
                                (int*)sumAddress, (float*)biasAddress, (float*)outputAddress,
                                (float*)scaleAddress, inputChannels, plane, outputChannels,
                                tileCount, begin, end);
                        });
                    }
                }
            }
            return true;
        }
        finally
        {
            PooledArrays.Return(quantizedArray);
            PooledArrays.Return(scaleArray);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ProcessInt8VnniBlocks(
        byte* quantized, byte* packedWeights, float* weightScales, int* weightSums,
        float* bias, float* output, float* inputScales,
        int inputChannels, int plane, int outputChannels, int tileCount,
        int beginBlock, int endBlock)
    {
        int groups = inputChannels / 4;
        float* values = stackalloc float[32];
        for (int block = beginBlock; block < endBlock; block++)
        {
            int co = block * 8;
            Vector256<float> weightScale = Avx.LoadVector256(weightScales + co);
            Vector256<float> biasVector = bias == null
                ? Vector256<float>.Zero
                : Avx.LoadVector256(bias + co);
            Vector256<int> correction = Avx2.MultiplyLow(
                Avx2.LoadVector256(weightSums + co), Vector256.Create(-128));
            byte* blockWeights = packedWeights + block * groups * 32;
            for (int tile = 0; tile < tileCount; tile++)
            {
                int start = tile * 16, end = Math.Min(plane, start + 16);
                Vector256<float> dequantScale = Avx.Multiply(
                    weightScale, Vector256.Create(inputScales[tile]));
                int spatial = start;
                for (; spatial <= end - 4; spatial += 4)
                {
                    Vector256<int> accumulator0 = correction, accumulator1 = correction;
                    Vector256<int> accumulator2 = correction, accumulator3 = correction;
                    byte* activation0 = quantized + spatial * inputChannels;
                    byte* activation1 = activation0 + inputChannels;
                    byte* activation2 = activation1 + inputChannels;
                    byte* activation3 = activation2 + inputChannels;
                    byte* weights = blockWeights;
                    for (int group = 0; group < groups; group++)
                    {
                        Vector256<sbyte> weightVector =
                            Unsafe.ReadUnaligned<Vector256<sbyte>>(weights);
                        accumulator0 = AvxVnni.MultiplyWideningAndAdd(accumulator0,
                            Vector256.Create(Unsafe.ReadUnaligned<int>(activation0)).AsByte(), weightVector);
                        accumulator1 = AvxVnni.MultiplyWideningAndAdd(accumulator1,
                            Vector256.Create(Unsafe.ReadUnaligned<int>(activation1)).AsByte(), weightVector);
                        accumulator2 = AvxVnni.MultiplyWideningAndAdd(accumulator2,
                            Vector256.Create(Unsafe.ReadUnaligned<int>(activation2)).AsByte(), weightVector);
                        accumulator3 = AvxVnni.MultiplyWideningAndAdd(accumulator3,
                            Vector256.Create(Unsafe.ReadUnaligned<int>(activation3)).AsByte(), weightVector);
                        activation0 += 4; activation1 += 4; activation2 += 4; activation3 += 4;
                        weights += 32;
                    }
                    Avx.Store(values, Avx.Add(
                        Avx.Multiply(Avx.ConvertToVector256Single(accumulator0), dequantScale), biasVector));
                    Avx.Store(values + 8, Avx.Add(
                        Avx.Multiply(Avx.ConvertToVector256Single(accumulator1), dequantScale), biasVector));
                    Avx.Store(values + 16, Avx.Add(
                        Avx.Multiply(Avx.ConvertToVector256Single(accumulator2), dequantScale), biasVector));
                    Avx.Store(values + 24, Avx.Add(
                        Avx.Multiply(Avx.ConvertToVector256Single(accumulator3), dequantScale), biasVector));
                    for (int lane = 0; lane < 8; lane++)
                    {
                        float* destination = output + (co + lane) * plane + spatial;
                        destination[0] = values[lane];
                        destination[1] = values[8 + lane];
                        destination[2] = values[16 + lane];
                        destination[3] = values[24 + lane];
                    }
                }
                for (; spatial < end; spatial++)
                {
                    Vector256<int> accumulator = correction;
                    byte* activation = quantized + spatial * inputChannels;
                    byte* weights = blockWeights;
                    for (int group = 0; group < groups; group++)
                    {
                        Vector256<byte> repeatedActivation =
                            Vector256.Create(Unsafe.ReadUnaligned<int>(activation)).AsByte();
                        Vector256<sbyte> weightVector =
                            Unsafe.ReadUnaligned<Vector256<sbyte>>(weights);
                        accumulator = AvxVnni.MultiplyWideningAndAdd(
                            accumulator, repeatedActivation, weightVector);
                        activation += 4;
                        weights += 32;
                    }
                    Vector256<float> result = Avx.Add(
                        Avx.Multiply(Avx.ConvertToVector256Single(accumulator), dequantScale),
                        biasVector);
                    Avx.Store(values, result);
                    for (int lane = 0; lane < 8; lane++)
                        output[(co + lane) * plane + spatial] = values[lane];
                }
            }
        }
    }
}
