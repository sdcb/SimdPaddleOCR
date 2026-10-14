using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class KernelCorrectnessTests
{
    [Theory]
    [InlineData(5, 32, 40)]
    [InlineData(4, 64, 1040)]
    [InlineData(5, 64, 1040)]
    [InlineData(4, 32, 39)]
    [InlineData(6, 32, 37)]
    [InlineData(7, 96, 1037)]
    public void MatMul_VectorMatchesReference(int rows, int inner, int columns)
    {
        const int batch = 2;
        float[] input = Ramp(batch * rows * inner, 0.01f);
        float[] weights = Ramp(inner * columns, 0.02f);
        float[]? packed = columns >= 16 ? PackMatMul(weights, inner, columns) : null;

        float[] expected = new float[batch * rows * columns];
        MatMulRef(input, weights, expected, batch, rows, inner, columns);

        float[] actual = new float[expected.Length];
        Assert.True(MatMul.Try(input, weights, actual, batch, rows, inner, columns, packed));
        AssertClose(expected, actual, rtol: 5e-5f, atol: 5e-5f);

        float[] vectorActual = new float[expected.Length];
        Assert.True(MatMul.TryVector(input, weights, vectorActual, batch, rows, inner, columns, packed));
        AssertClose(expected, vectorActual, rtol: 5e-5f, atol: 5e-5f);
    }

    [Fact]
    public void MatMul_Rows8PackedHandlesPartialColumnTile()
    {
        const int batch = 1, rows = 8, inner = 64, columns = 1050;
        float[] input = Ramp(batch * rows * inner, 0.01f);
        float[] weights = Ramp(inner * columns, -0.02f);
        float[] packed = PackMatMul(weights, inner, columns);
        float[] expected = new float[batch * rows * columns];
        MatMulRef(input, weights, expected, batch, rows, inner, columns);
        float[] actual = new float[expected.Length];
        Assert.True(MatMul.Try(input, weights, actual, batch, rows, inner, columns, packed));
        AssertClose(expected, actual, rtol: 5e-5f, atol: 5e-5f);
    }

    [Theory]
    [InlineData(4, 64, 1040)]
    [InlineData(5, 96, 1037)]
    [InlineData(3, 64, 1030)]
    [InlineData(7, 96, 6624)]
    public void MatMul_ArgMaxMatchesReference(int rows, int inner, int columns)
    {
        // Random data: a ramp over the inner dim produces near-tied column scores,
        // and argmax then flips between rounding orders (FMA vs mul+add).
        const int batch = 2;
        float[] input = RandomFill(batch * rows * inner, 1f, seed: 1234);
        float[] weights = RandomFill(inner * columns, 0.25f, seed: 5678);
        float[] bias = RandomFill(columns, 0.5f, seed: 9012);
        float[] packed = PackMatMul(weights, inner, columns);

        float[] logits = new float[batch * rows * columns];
        MatMulRef(input, weights, logits, batch, rows, inner, columns);
        int[] expectedIndex = new int[batch * rows];
        float[] expectedScore = new float[batch * rows];
        for (int r = 0; r < batch * rows; r++)
        {
            int best = 0;
            float max = logits[r * columns] + bias[0];
            for (int c = 1; c < columns; c++)
            {
                float value = logits[r * columns + c] + bias[c];
                if (value > max) { max = value; best = c; }
            }
            expectedIndex[r] = best;
            expectedScore[r] = max;
        }

        int[] indices = new int[batch * rows];
        float[] scores = new float[batch * rows];
        Assert.True(MatMul.TryArgMax(input, weights, bias, indices, scores,
            batch, rows, inner, columns, packed, threads: 3));
        Assert.Equal(expectedIndex, indices);
        AssertClose(expectedScore, scores, rtol: 5e-5f, atol: 5e-5f);
    }

    internal static float[] Ramp(int length, float scale)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (i % 17 - 8) * scale;
        return values;
    }

    internal static float[] RandomFill(int length, float amplitude, int seed)
    {
        var rng = new Random(seed);
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1) * amplitude;
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

    internal static float[] PackConv1x1(ReadOnlySpan<float> weights, int outputChannels, int inputChannels, int tile)
    {
        float[] packed = new float[outputChannels * inputChannels];
        int blocks = outputChannels / tile;
        for (int block = 0; block < blocks; block++)
            for (int ci = 0; ci < inputChannels; ci++)
                for (int lane = 0; lane < tile; lane++)
                    packed[(block * inputChannels + ci) * tile + lane] =
                        weights[(block * tile + lane) * inputChannels + ci];
        return packed;
    }

    internal static float[] PackConv1x1Oc16(ReadOnlySpan<float> weights, int outputChannels, int inputChannels)
    {
        int coutPadded = (outputChannels + 15) & ~15;
        float[] packed = new float[inputChannels * coutPadded];
        for (int ci = 0; ci < inputChannels; ci++)
            for (int co = 0; co < outputChannels; co++)
                packed[ci * coutPadded + co] = weights[co * inputChannels + ci];
        return packed;
    }

    internal static void Conv1x1Ref(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels)
    {
        int plane = height * width;
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co++)
                for (int s = 0; s < plane; s++)
                {
                    float sum = bias.IsEmpty ? 0f : bias[co];
                    for (int ci = 0; ci < inputChannels; ci++)
                        sum += input[(b * inputChannels + ci) * plane + s] * weights[co * inputChannels + ci];
                    output[(b * outputChannels + co) * plane + s] = sum;
                }
    }

    private static void Depthwise9Ref(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int channels, int h, int w)
    {
        int plane = h * w;
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float sum = bias[c];
                    for (int ky = 0; ky < 9; ky++)
                    {
                        int iy = y - 4 + ky;
                        if ((uint)iy >= (uint)h) continue;
                        for (int kx = 0; kx < 9; kx++)
                        {
                            int ix = x - 4 + kx;
                            if ((uint)ix < (uint)w)
                                sum += input[c * plane + iy * w + ix] * weights[c * 81 + ky * 9 + kx];
                        }
                    }
                    output[c * plane + y * w + x] = sum;
                }
    }

    private static void Dense5Ref(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int ic, int oc, int h, int w, int kernel)
    {
        int plane = h * w, pad = kernel / 2, taps = kernel * kernel;
        for (int co = 0; co < oc; co++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float sum = bias[co];
                    for (int ci = 0; ci < ic; ci++)
                        for (int ky = 0; ky < kernel; ky++)
                        {
                            int iy = y - pad + ky;
                            if ((uint)iy >= (uint)h) continue;
                            for (int kx = 0; kx < kernel; kx++)
                            {
                                int ix = x - pad + kx;
                                if ((uint)ix < (uint)w)
                                    sum += input[ci * plane + iy * w + ix] *
                                        weights[(co * ic + ci) * taps + ky * kernel + kx];
                            }
                        }
                    output[co * plane + y * w + x] = sum;
                }
    }

    private static void Stride2Ref(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int height, int width, int outputChannels)
    {
        int plane = height * width;
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co++)
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        float sum = bias.IsEmpty ? 0f : bias[co];
                        for (int ci = 0; ci < inputChannels; ci++)
                            for (int ky = 0; ky < 2; ky++)
                                for (int kx = 0; kx < 2; kx++)
                                {
                                    int iy = y + ky, ix = x + kx;
                                    if ((uint)iy < (uint)height && (uint)ix < (uint)width)
                                        sum += input[(b * inputChannels + ci) * plane + iy * width + ix] *
                                            weights[(co * inputChannels + ci) * 4 + ky * 2 + kx];
                                }
                        output[(b * outputChannels + co) * plane + y * width + x] = sum;
                    }
    }

    internal static float[] PackMatMul(ReadOnlySpan<float> weights, int inner, int columns)
    {
        int fullTiles = columns / 16;
        float[] packed = new float[fullTiles * inner * 16];
        for (int tile = 0; tile < fullTiles; tile++)
            for (int k = 0; k < inner; k++)
                weights.Slice(k * columns + tile * 16, 16).CopyTo(packed.AsSpan((tile * inner + k) * 16, 16));
        return packed;
    }

    internal static void MatMulRef(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns)
    {
        for (int b = 0; b < batch; b++)
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < columns; col++)
                {
                    float sum = 0;
                    for (int k = 0; k < inner; k++)
                        sum += input[(b * rows + row) * inner + k] * weights[k * columns + col];
                    output[(b * rows + row) * columns + col] = sum;
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

    internal static void ConvTranspose2x2Ref(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int batch, int inputChannels,
        int inputHeight, int inputWidth, int outputChannels)
    {
        int inputPlane = inputHeight * inputWidth;
        int outputWidth = inputWidth * 2, outputHeight = inputHeight * 2;
        int outputPlane = outputHeight * outputWidth;
        for (int b = 0; b < batch; b++)
            for (int co = 0; co < outputChannels; co++)
            {
                int dstBase = (b * outputChannels + co) * outputPlane;
                float initial = bias.IsEmpty ? 0f : bias[co];
                for (int i = 0; i < outputPlane; i++)
                    output[dstBase + i] = initial;
                for (int ci = 0; ci < inputChannels; ci++)
                {
                    int srcBase = (b * inputChannels + ci) * inputPlane;
                    int wb = (ci * outputChannels + co) * 4;
                    for (int iy = 0; iy < inputHeight; iy++)
                        for (int ix = 0; ix < inputWidth; ix++)
                        {
                            float v = input[srcBase + iy * inputWidth + ix];
                            int ox = ix * 2, oy = iy * 2;
                            output[dstBase + oy * outputWidth + ox] += v * weights[wb];
                            output[dstBase + oy * outputWidth + ox + 1] += v * weights[wb + 1];
                            output[dstBase + (oy + 1) * outputWidth + ox] += v * weights[wb + 2];
                            output[dstBase + (oy + 1) * outputWidth + ox + 1] += v * weights[wb + 3];
                        }
                }
            }
    }

    internal static float ErfPolyRef(float x)
    {
        float a = MathF.Abs(x);
        float result;
        if (a >= 4f)
            result = 1f;
        else if (a < 1f)
        {
            float s = a * a;
            float p = 1.0590875083315439e-6f;
            p = s * p + -1.3906452410711274e-5f;
            p = s * p + 1.1955437252428243e-4f;
            p = s * p + -8.542475960079766e-4f;
            p = s * p + 5.223771899427153e-3f;
            p = s * p + -2.686612888828867e-2f;
            p = s * p + 1.128379122662546e-1f;
            p = s * p + -3.761263888304388e-1f;
            p = s * p + 1.1283791670929921f;
            result = a * p;
        }
        else if (a < 2f)
        {
            float z = a - 1.5f;
            float p = -2.400667527836574e-3f;
            p = z * p + -3.8855162788028288e-3f;
            p = z * p + 1.9332860298601401e-2f;
            p = z * p + -1.501360723487143e-2f;
            p = z * p + -4.4599369472787913e-2f;
            p = z * p + 1.3876136033191724e-1f;
            p = z * p + -1.7839541988688759e-1f;
            p = z * p + 1.1893013063163335e-1f;
            result = z * p + 9.661051464140682e-1f;
        }
        else
        {
            float z = a - 3f;
            float p = -8.875076532056503e-5f;
            p = z * p + 3.880528563007222e-4f;
            p = z * p + -7.781201071142843e-4f;
            p = z * p + 1.0255980996254569e-3f;
            p = z * p + -1.0307062838910627e-3f;
            p = z * p + 7.858608012972956e-4f;
            p = z * p + -4.193605385522123e-4f;
            p = z * p + 1.3951109720745927e-4f;
            result = z * p + 9.999779388683872e-1f;
        }
        return x < 0f ? -result : result;
    }
}
