using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class KernelCorrectnessTests
{
    [Fact]
    public void Conv3x3Stride2_PackedVectorMatchesReference()
    {
        const int batch = 1, ic = 3, oc = 8, ih = 11, iw = 13;
        int oh = ih / 2, ow = iw / 2;
        float[] input = Ramp(batch * ic * ih * iw, 0.01f);
        float[] weights = Ramp(oc * ic * 9, 0.02f);
        float[] bias = Ramp(oc, 0.03f);
        float[] packed = PackConv3x3(weights, oc, ic);

        float[] expected = new float[batch * oc * oh * ow];
        Conv3x3Stride2Ref(input, weights, bias, expected, batch, ic, ih, iw, oh, ow, oc);

        float[] actual = new float[expected.Length];
        Assert.True(Conv3x3Stride2.TryPacked(input, packed, bias, actual, batch, ic, ih, iw, oh, ow, oc));
        AssertClose(expected, actual);

        float[] vectorActual = new float[expected.Length];
        Conv3x3Stride2.TryPackedVector(input, packed, bias, vectorActual, batch, ic, ih, iw, oh, ow, oc);
        AssertClose(expected, vectorActual);
    }

    [Fact]
    public void Conv3x3Stride2_VectorPathMatchesReference()
    {
        const int batch = 1, ic = 2, oc = 4, ih = 9, iw = 10;
        int oh = ih / 2, ow = iw / 2;
        float[] input = Ramp(batch * ic * ih * iw, -0.04f);
        float[] weights = Ramp(oc * ic * 9, 0.05f);
        float[] bias = Ramp(oc, -0.01f);

        float[] expected = new float[batch * oc * oh * ow];
        Conv3x3Stride2Ref(input, weights, bias, expected, batch, ic, ih, iw, oh, ow, oc);

        float[] actual = new float[expected.Length];
        Assert.True(Conv3x3Stride2.Try(input, weights, bias, actual, batch, ic, ih, iw, oh, ow, oc));
        AssertClose(expected, actual);
    }

    internal static float[] Ramp(int length, float scale)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (i % 17 - 8) * scale;
        return values;
    }

    internal static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual,
        float rtol = 2e-5f, float atol = 2e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = atol + rtol * Math.Max(Math.Abs(e), Math.Abs(a));
            if (Math.Abs(e - a) > tol)
                Assert.Fail($"index {i}: expected {e}, actual {a}, tol {tol}");
        }
    }

    internal static float[] PackConv3x3(ReadOnlySpan<float> weights, int outputChannels, int inputChannels)
    {
        float[] packed = new float[outputChannels * inputChannels * 9];
        int blocks = outputChannels / 8;
        for (int block = 0; block < blocks; block++)
            for (int ci = 0; ci < inputChannels; ci++)
                for (int k = 0; k < 9; k++)
                    for (int lane = 0; lane < 8; lane++)
                        packed[((block * inputChannels + ci) * 9 + k) * 8 + lane] =
                            weights[((block * 8 + lane) * inputChannels + ci) * 9 + k];
        return packed;
    }

    internal static void Conv3x3Stride2Ref(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int inputHeight, int inputWidth, int outputHeight, int outputWidth, int outputChannels)
    {
        int inputPlane = inputHeight * inputWidth, outputPlane = outputHeight * outputWidth;
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co++)
                for (int oy = 0; oy < outputHeight; oy++)
                    for (int ox = 0; ox < outputWidth; ox++)
                    {
                        float sum = bias.IsEmpty ? 0f : bias[co];
                        for (int ci = 0; ci < inputChannels; ci++)
                            for (int ky = 0; ky < 3; ky++)
                                for (int kx = 0; kx < 3; kx++)
                                {
                                    int iy = oy * 2 - 1 + ky, ix = ox * 2 - 1 + kx;
                                    if ((uint)iy >= (uint)inputHeight || (uint)ix >= (uint)inputWidth)
                                        continue;
                                    float v = input[((b * inputChannels + ci) * inputPlane) + iy * inputWidth + ix];
                                    sum += v * weights[((co * inputChannels + ci) * 9) + ky * 3 + kx];
                                }
                        output[((b * outputChannels + co) * outputPlane) + oy * outputWidth + ox] = sum;
                    }
    }
}
