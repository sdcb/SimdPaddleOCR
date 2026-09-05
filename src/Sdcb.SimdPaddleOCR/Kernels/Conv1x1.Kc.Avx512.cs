using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

using static Sdcb.SimdPaddleOCR.Kernels.SimdOps;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Conv1x1
{
    /// <summary>
    /// Packed 8-OC Avx512 with KC blocking so a weight panel stays in L1 across
    /// the full spatial sweep (classic GEMM microkernel order).
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Conv1x1PackedEightOutputsKcAvx512Unsafe(
        ReadOnlySpan<float> input, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<float> output,
        int batch, int inputChannels, int height, int width, int outputChannels)
    {
        int plane = checked(height * width);
        const int kc = 32;
        fixed (float* inputPtr = input, weightsPtr = packedWeights, biasPtr = bias, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                int inputBatch = b * inputChannels * plane;
                for (int co = 0; co < outputChannels; co += 8)
                {
                    float* output0 = outputPtr + (b * outputChannels + co) * plane;
                    float* output1 = output0 + plane;
                    float* output2 = output1 + plane;
                    float* output3 = output2 + plane;
                    float* output4 = output3 + plane;
                    float* output5 = output4 + plane;
                    float* output6 = output5 + plane;
                    float* output7 = output6 + plane;
                    float* wBlock0 = weightsPtr + (co / 4) * inputChannels * 4;
                    float* wBlock1 = wBlock0 + inputChannels * 4;
                    // Init outputs with bias across full plane.
                    Vector512<float> bias0 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co]);
                    Vector512<float> bias1 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 1]);
                    Vector512<float> bias2 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 2]);
                    Vector512<float> bias3 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 3]);
                    Vector512<float> bias4 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 4]);
                    Vector512<float> bias5 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 5]);
                    Vector512<float> bias6 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 6]);
                    Vector512<float> bias7 = Vector512.Create(biasPtr == null ? 0f : biasPtr[co + 7]);
                    int spatial = 0;
                    for (; spatial <= plane - 16; spatial += 16)
                    {
                        Avx512F.Store(output0 + spatial, bias0);
                        Avx512F.Store(output1 + spatial, bias1);
                        Avx512F.Store(output2 + spatial, bias2);
                        Avx512F.Store(output3 + spatial, bias3);
                        Avx512F.Store(output4 + spatial, bias4);
                        Avx512F.Store(output5 + spatial, bias5);
                        Avx512F.Store(output6 + spatial, bias6);
                        Avx512F.Store(output7 + spatial, bias7);
                    }
                    for (; spatial < plane; spatial++)
                    {
                        output0[spatial] = biasPtr == null ? 0f : biasPtr[co];
                        output1[spatial] = biasPtr == null ? 0f : biasPtr[co + 1];
                        output2[spatial] = biasPtr == null ? 0f : biasPtr[co + 2];
                        output3[spatial] = biasPtr == null ? 0f : biasPtr[co + 3];
                        output4[spatial] = biasPtr == null ? 0f : biasPtr[co + 4];
                        output5[spatial] = biasPtr == null ? 0f : biasPtr[co + 5];
                        output6[spatial] = biasPtr == null ? 0f : biasPtr[co + 6];
                        output7[spatial] = biasPtr == null ? 0f : biasPtr[co + 7];
                    }

                    for (int ci0 = 0; ci0 < inputChannels; ci0 += kc)
                    {
                        int ciEnd = ci0 + kc < inputChannels ? ci0 + kc : inputChannels;
                        spatial = 0;
                        for (; spatial <= plane - 16; spatial += 16)
                        {
                            Vector512<float> a0 = Avx512F.LoadVector512(output0 + spatial);
                            Vector512<float> a1 = Avx512F.LoadVector512(output1 + spatial);
                            Vector512<float> a2 = Avx512F.LoadVector512(output2 + spatial);
                            Vector512<float> a3 = Avx512F.LoadVector512(output3 + spatial);
                            Vector512<float> a4 = Avx512F.LoadVector512(output4 + spatial);
                            Vector512<float> a5 = Avx512F.LoadVector512(output5 + spatial);
                            Vector512<float> a6 = Avx512F.LoadVector512(output6 + spatial);
                            Vector512<float> a7 = Avx512F.LoadVector512(output7 + spatial);
                            float* inputChannel = inputPtr + inputBatch + spatial + (long)ci0 * plane;
                            float* w0 = wBlock0 + ci0 * 4;
                            float* w1 = wBlock1 + ci0 * 4;
                            for (int ci = ci0; ci < ciEnd; ci++)
                            {
                                Vector512<float> v = Avx512F.LoadVector512(inputChannel);
                                a0 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(w0[0]), a0);
                                a1 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(w0[1]), a1);
                                a2 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(w0[2]), a2);
                                a3 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(w0[3]), a3);
                                a4 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(w1[0]), a4);
                                a5 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(w1[1]), a5);
                                a6 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(w1[2]), a6);
                                a7 = Avx512F.FusedMultiplyAdd(v, Vector512.Create(w1[3]), a7);
                                inputChannel += plane;
                                w0 += 4; w1 += 4;
                            }
                            Avx512F.Store(output0 + spatial, a0);
                            Avx512F.Store(output1 + spatial, a1);
                            Avx512F.Store(output2 + spatial, a2);
                            Avx512F.Store(output3 + spatial, a3);
                            Avx512F.Store(output4 + spatial, a4);
                            Avx512F.Store(output5 + spatial, a5);
                            Avx512F.Store(output6 + spatial, a6);
                            Avx512F.Store(output7 + spatial, a7);
                        }
                        for (; spatial < plane; spatial++)
                        {
                            float a0 = output0[spatial], a1 = output1[spatial];
                            float a2 = output2[spatial], a3 = output3[spatial];
                            float a4 = output4[spatial], a5 = output5[spatial];
                            float a6 = output6[spatial], a7 = output7[spatial];
                            float* inputChannel = inputPtr + inputBatch + spatial + (long)ci0 * plane;
                            float* w0 = wBlock0 + ci0 * 4;
                            float* w1 = wBlock1 + ci0 * 4;
                            for (int ci = ci0; ci < ciEnd; ci++)
                            {
                                float v = *inputChannel;
                                a0 += v * w0[0]; a1 += v * w0[1]; a2 += v * w0[2]; a3 += v * w0[3];
                                a4 += v * w1[0]; a5 += v * w1[1]; a6 += v * w1[2]; a7 += v * w1[3];
                                inputChannel += plane; w0 += 4; w1 += 4;
                            }
                            output0[spatial] = a0; output1[spatial] = a1;
                            output2[spatial] = a2; output3[spatial] = a3;
                            output4[spatial] = a4; output5[spatial] = a5;
                            output6[spatial] = a6; output7[spatial] = a7;
                        }
                    }
                }
            }
        }
    }
}
