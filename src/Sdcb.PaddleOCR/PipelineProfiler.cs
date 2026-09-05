using System.Text;

namespace Sdcb.PaddleOCR;

/// <summary>
/// Shared per-call profiling core: a static enabled flag, cumulative tick
/// counters and per-call counters indexed by a small fixed slot set. All
/// pipeline stages (DET preprocess/graph/postprocess, CLS, REC) report into
/// this single facility so the benchmark can diff cumulative snapshots per
/// image without paying for profiling when it is disabled.
/// </summary>
internal static class PipelineProfiler
{
    // Stage slots. Keep in sync with PipelineProfiler.StageNames.
    internal const int DetPreprocess = 0;
    internal const int DetGraph = 1;
    internal const int DetPostprocess = 2;
    internal const int Crop = 3;
    internal const int ClsPreprocess = 4;
    internal const int ClsGraph = 5;
    internal const int RecPreprocess = 6;
    internal const int RecGraph = 7;
    internal const int RecPostprocess = 8;
    internal const int LinesWall = 9;
    // Fine-grained bookkeeping slots, added to explain per-line / per-image
    // overhead that the coarse stages above cannot account for.
    internal const int CropSetup = 10;      // per-image: GetSize scan + workspace rents
    internal const int RecAcquire = 11;     // per-line REC: width select + cache get + rent + ArrayPool rent
    internal const int ClsAcquire = 12;     // per-line CLS: rent + ArrayPool rent
    internal const int RecDecode = 13;      // per-line REC: CTC decode + string alloc + ArrayPool return
    internal const int RecRelease = 14;     // per-line REC: ArrayPool return + session return
    internal const int ClsRelease = 15;     // per-line CLS: ArrayPool return + session return
    internal const int RecCacheGet = 16;    // per-line REC: width select
    internal const int RecRent = 17;        // per-line REC: bucket.Rent (may build session on miss)
    internal const int RecPool = 18;        // per-line REC: input scratch ensure (was ArrayPool.Rent)
    internal const int RecReshape = 19;     // per-line REC: session.Reshape
    internal const int DetUnclip = 20;      // nested in det_postprocess: convex outward offset
    internal const int StageCount = 21;

    internal static readonly string[] StageNames =
    [
        "det_preprocess", "det_graph", "det_postprocess", "crop",
        "cls_preprocess", "cls_graph", "rec_preprocess", "rec_graph",
        "rec_postprocess", "lines_wall",
        "crop_setup", "rec_acquire", "cls_acquire", "rec_decode",
        "rec_release", "cls_release", "rec_cache_get", "rec_rent", "rec_pool",
        "rec_reshape", "det_unclip"
    ];

    private static bool s_enabled;
    private static readonly long[] s_ticks = new long[StageCount];
    private static readonly long[] s_calls = new long[StageCount];

    internal static bool Enabled => s_enabled;

    internal static void Enable(bool enabled)
    {
        s_enabled = enabled;
        if (enabled)
        {
            Array.Clear(s_ticks, 0, s_ticks.Length);
            Array.Clear(s_calls, 0, s_calls.Length);
        }
    }

    internal static long Now() => System.Diagnostics.Stopwatch.GetTimestamp();

    internal static void Add(int stage, long started)
    {
        long elapsed = System.Diagnostics.Stopwatch.GetTimestamp() - started;
        Interlocked.Add(ref s_ticks[stage], elapsed);
        Interlocked.Increment(ref s_calls[stage]);
    }

    /// <summary>Returns cumulative (milliseconds, call count) per stage.</summary>
    internal static (double Milliseconds, long Calls)[] Snapshot()
    {
        double scale = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        var result = new (double, long)[StageCount];
        for (int i = 0; i < result.Length; i++)
            result[i] = (Interlocked.Read(ref s_ticks[i]) * scale, Interlocked.Read(ref s_calls[i]));
        return result;
    }
}
