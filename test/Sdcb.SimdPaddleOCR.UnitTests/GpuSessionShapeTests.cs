#if NET10_0_OR_GREATER && !USE_NS20_LIBRARY
using Sdcb.SimdPaddleOCR.Backends;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.Models.TextLineOrientation;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class GpuSessionShapeTests
{
    [Fact]
    public void RecognitionShapesRemainCorrectAcrossBatchesAndCacheEviction()
    {
        using Stream stream = ChineseV6TinyModel.Recognition.OpenRead();
        using Model model = Model.Load(stream);
        CompiledModel compiled = new(model, 1);
        try
        {
            using var session = new TestSession(compiled);
            int[][] initial = [[1, 3, 48, 32], [3, 3, 48, 96], [2, 3, 48, 64]];
            AssertPlans(session, compiled, initial);
            AssertPlans(session, compiled, initial.Reverse().ToArray());
            for (int i = 1; i <= 140; i++)
                AssertPlans(session, compiled, [[1, 3, 48, i * 32]]);
            var cache = (System.Collections.IDictionary)typeof(GpuSessionBase)
                .GetField("_ctcPlans", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .GetValue(session)!;
            Assert.Equal(128, cache.Count);
            AssertPlans(session, compiled, initial);
            AssertPlans(session, compiled, [[1, 3, 48, 32], [16, 3, 48, 320]]);
            AssertPlans(session, compiled, [[1, 3, 48, 32]]);
        }
        finally { compiled.Dispose(); }
    }

    [Fact]
    public void ClassificationProjectionTracksChangingBatchSize()
    {
        using Stream stream = TextLineOrientationModel.OpenRead();
        using Model model = Model.Load(stream);
        CompiledModel compiled = new(model, 1);
        try
        {
            using var session = new TestSession(compiled);
            foreach (int batch in new[] { 1, 8, 2, 8, 1 })
            {
                int[] shape = [batch, 3, 80, 160];
                session.Reshape(shape);
                Assert.True(session.TryRunUntilCtcProjection(session.InputData, out var ops));
                Assert.Equal(batch, ops.RowCount);
                Assert.Equal(2, ops.Columns);
                Assert.Equal(compiled.ResolveShapesFor(shape)[compiled.OutputIndex], session.OutputShape.Dimensions);
            }
        }
        finally { compiled.Dispose(); }
    }

    private static void AssertPlans(TestSession session, CompiledModel compiled, int[][] shapes)
    {
        Span<float> input = session.ReshapeMany(shapes);
        Assert.Equal(shapes.Sum(s => s.Aggregate(1, (a, b) => a * b)), input.Length);
        Assert.True(session.TryResolveManyHead(out int[] rows, out CtcHead head));
        Assert.Equal(shapes.Length, rows.Length);
        for (int i = 0; i < shapes.Length; i++)
        {
            int[] output = compiled.ResolveShapesFor(shapes[i])[compiled.OutputIndex];
            Assert.Equal(output[^2], rows[i]);
            Assert.Equal(output[^1], head.Columns);
        }
        Assert.Equal(compiled.ResolveShapesFor(shapes[0])[compiled.OutputIndex], session.OutputShape.Dimensions);
        Assert.True(session.TryRunUntilCtcProjection(session.InputData, out var ops));
        Assert.Equal(shapes[0][0] * rows[0], ops.RowCount);
        Assert.Equal(head.Columns, ops.Columns);
    }

    [Fact]
    public void DetectionOutputShapeDoesNotRequireCtcProjection()
    {
        using Stream stream = ChineseV6TinyModel.Detection.OpenRead();
        using Model model = Model.Load(stream);
        CompiledModel compiled = new(model, 1);
        try
        {
            using var session = new TestSession(compiled);
            foreach (int side in new[] { 64, 96, 64 })
            {
                int[] shape = [1, 3, side, side];
                session.ReshapeMany([shape]);
                Assert.False(session.TryResolveManyHead(out _, out _));
                Assert.Equal(compiled.ResolveShapesFor(shape)[compiled.OutputIndex], session.OutputShape.Dimensions);
            }
        }
        finally { compiled.Dispose(); }
    }

    private sealed class TestSession(CompiledModel compiled) : GpuSessionBase(new ShapeOnlyRunner(), compiled, "test");

    private sealed class ShapeOnlyRunner : IOcrGraphRunner
    {
        public ReadOnlySpan<float> Run(int[] inputShape, ReadOnlySpan<float> input, int nodeLimit = int.MaxValue, int outTensor = -1) => [];
        public bool RunMany(IReadOnlyList<int[]> shapes, ReadOnlySpan<float> input, int nodeLimit, int outTensor, CtcUnitsReady onReady) => throw new NotSupportedException();
        public void Dispose() { }
    }
}
#endif
