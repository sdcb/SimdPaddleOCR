#if NET10_0_OR_GREATER && !USE_NS20_LIBRARY
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class VulkanConstantAliasTests
{
    [Fact]
    public void DescriptorOffsetMustLeaveANonEmptyRange()
    {
        var binding = new GpuBind(GpuGraphModel.RoleArena, 16);
        Assert.Equal(48UL, binding.ResolveOffset(32, 64));
        Assert.Throws<InvalidOperationException>(() => binding.ResolveOffset(48, 64));
        Assert.Throws<InvalidOperationException>(() => binding.ResolveOffset(64, 64));
        Assert.Throws<OverflowException>(() => binding.ResolveOffset(ulong.MaxValue, 64));
    }

    [Fact]
    public void ReshapedConstantFeedsChannelBias()
    {
        VkDevice? device = GpuBackend.TryGetDevice();
        Assert.SkipWhen(device is null, "A supported Vulkan device is required.");

        using Stream fixture = typeof(VulkanConstantAliasTests).Assembly
            .GetManifestResourceStream("ConstantReshapeBias.onnx")!;
        using Model model = Model.Load(fixture);
        var compiled = new CompiledModel(model, 1);
        try
        {
            using var graph = new GpuDetGraph(device!, compiled);
            int[] shape = [1, 3, 640, 640];
            GpuSchedule schedule = graph.Prepare(shape);
            GpuBind bias = Assert.Single(schedule.Recs, rec => rec.Tag == "binAdd n2").Binds[1];

            // Check the schedule before any dispatch so a regression cannot submit an invalid descriptor.
            Assert.Equal(0UL, bias.ByteOffset);
            Assert.NotSame(GpuGraphModel.RoleArena, bias.Buf);
            Assert.True(bias.Buf.Size >= 16 * sizeof(ushort));

            // Conv(zero input) + Reshape([0..15], [1,16,1,1]) broadcasts each bias over a channel.
            const int channelSize = 320 * 320;
            float[] zeros = new float[3 * 640 * 640];
            float[] nchw = new float[16 * channelSize];
            for (int run = 0; run < 3; run++)
            {
                ReadOnlySpan<float> output = graph.Run(shape, zeros);
                Assert.Equal(nchw.Length, output.Length);
                // The internal GPU runner returns its physical NHWC buffer.
                Nhwc.NhwcToNchw(output, nchw, batch: 1, channels: 16, plane: channelSize, threads: 1);
                for (int c = 0; c < 16; c++)
                    Assert.Equal(-1, nchw.AsSpan(c * channelSize, channelSize).IndexOfAnyExcept((float)c));
            }
        }
        finally { compiled.Dispose(); }
    }
}
#endif
