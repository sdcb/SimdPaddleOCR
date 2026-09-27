using System.Diagnostics;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

/// <summary>One storage-buffer binding: a fixed (shared, immutable) buffer or
/// a <see cref="GpuGraphModel"/> role sentinel resolved per session.</summary>
internal readonly record struct GpuBind(VkBuffer Buf, ulong ByteOffset);

internal sealed class GpuRec
{
    public required VkPipeline Pipe;
    public required GpuBind[] Binds;
    public required byte[] Pc;
    public uint Gx, Gy;
    public string Tag = "";
}

/// <summary>
/// Compiled dispatch list for one (N,H,W,nodeLimit,outTensor): managed data
/// only. Arena/IO requirements are sizes, not handles, so a session growing
/// its arena never invalidates anything.
/// </summary>
internal sealed class GpuSchedule
{
    public readonly record struct Key(int N, int H, int W, int NodeLimit, int OutTensor);

    public required GpuRec[] Recs;
    public int OutElems;
    public long InElems, ArenaElems;
    public long[] Off = [];
    public int[] Alias = [];
    public long[] Numel = [];
    public int[][] Shapes = [];
    public long Im2colOff;
    public long LastTick;
}

/// <summary>Streamed-run callback: units [first, first+count) of result
/// (unit i at offsets[i]) are complete. Return false to stop receiving
/// further batches.</summary>
internal delegate bool GpuUnitsReady(float[] result, int[] offsets, int first, int count);

/// <summary>
/// Per-session GPU runtime over a shared <see cref="GpuGraphModel"/>: owns
/// the fp16 arena, fp32 input/output buffers, partials scratch, a private
/// command pool with a few command buffers and fences. Each run re-records
/// from the schedule (push descriptors; resettable descriptor pool fallback)
/// so there are no per-shape Vulkan objects at all, and sessions never
/// contend beyond the queue-submit lock — CPU work of concurrent sessions
/// (upload, record, readback) overlaps freely.
/// </summary>
internal sealed unsafe class GpuDetGraph : IDisposable
{
    private readonly VkDevice _dev;
    private readonly GpuGraphModel _model;
    private IntPtr _pool, _descPool;
    private readonly List<IntPtr> _cmds = [], _fences = [];
    private uint _descPoolSets;
    private VkBuffer? _arena, _in, _out, _part;
    private long _arenaElems, _inElems, _outElems;
    private float* _inMap, _outMap;
    private IntPtr _queryPool;
    private int _queryCap, _queryCount;
    private GpuSchedule? _last;
    private bool _disposed;

    private static readonly bool s_prof = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_PROF") == "1";
    private static readonly bool s_noBar = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_NOBAR") == "1";
    private static readonly bool s_dbgTime = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_TIME") == "1";
    private static readonly bool s_onlyHead = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_ONLYHEAD") == "1";
    private static readonly int s_only =
        int.TryParse(Environment.GetEnvironmentVariable("SIMD_OCR_GPU_ONLY"), out int o) ? o : -1;
    private static readonly int s_trunc =
        int.TryParse(Environment.GetEnvironmentVariable("SIMD_OCR_GPU_TRUNCATE"), out int t) ? t : -1;

    public GpuDetGraph(VkDevice dev, CompiledModel compiled)
    {
        _dev = dev;
        _model = GpuGraphModel.Acquire(dev, compiled);
        try
        {
            _pool = dev.NewCommandPool();
            EnsureCommandBuffers(1);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private void EnsureCommandBuffers(int count)
    {
        while (_cmds.Count < count)
        {
            _cmds.Add(_dev.AllocateCommandBuffer(_pool));
            _fences.Add(_dev.NewFence());
        }
    }

    /// <summary>Compile (or fetch) the schedule for a shape without running it —
    /// lets callers probe GPU support before committing to the GPU path.</summary>
    public GpuSchedule Prepare(int[] inputShape, int nodeLimit = int.MaxValue, int outTensor = -1)
        => _model.GetSchedule(inputShape, nodeLimit, outTensor);

    /// <summary>Run the graph: input fp32 NCHW [n,C,H,W] → fp32 [graph output].
    /// nodeLimit truncates the emit loop (nodes beyond it are not dispatched) and
    /// outTensor overrides the readback tensor — used to stop the REC graph before
    /// the vocab projection (activations readback) instead of the graph output.
    /// </summary>
    public ReadOnlySpan<float> Run(int[] inputShape, ReadOnlySpan<float> input,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        GpuSchedule s = _model.GetSchedule(inputShape, nodeLimit, outTensor);
        return RunCore([s], input, out _).AsSpan(0, s.OutElems);
    }

    /// <summary>
    /// Runs several shapes of the same graph (inputs back to back in
    /// <paramref name="input"/>, unit i = numel(shapes[i]) floats) and hands
    /// the activations to <paramref name="onReady"/> in unit order as they
    /// land. The units go out in a few submissions: the GPU starts on the
    /// first while the rest are still being recorded, and the caller's CPU
    /// tail (CTC ArgMax) for early units overlaps the GPU work of later ones.
    /// Returns false when <paramref name="onReady"/> declined a batch; every
    /// submission has completed by the time this returns or throws.
    /// </summary>
    public bool RunMany(IReadOnlyList<int[]> shapes, ReadOnlySpan<float> input,
        int nodeLimit, int outTensor, GpuUnitsReady onReady)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var sched = new GpuSchedule[shapes.Count];
        for (int i = 0; i < sched.Length; i++)
            sched[i] = _model.GetSchedule(shapes[i], nodeLimit, outTensor);
        bool accepted = true;
        RunCore(sched, input, out _, (result, offsets, first, count) =>
            accepted = accepted && onReady(result, offsets, first, count));
        return accepted;
    }

    // in/out regions of consecutive units start on 256-byte boundaries
    // (covers minStorageBufferOffsetAlignment on every desktop GPU)
    private static long Align64(long floats) => (floats + 63) & ~63L;

    private float[] RunCore(GpuSchedule[] sched, ReadOnlySpan<float> input, out int[] outOffsets,
        GpuUnitsReady? sink = null)
    {
        long t0 = Stopwatch.GetTimestamp();
        int n = sched.Length;
        var inBase = new long[n];
        outOffsets = new int[n];
        var outBase = new long[n];
        long inTotal = 0, outTotal = 0, arena = 0;
        int outElems = 0;
        for (int i = 0; i < n; i++)
        {
            inBase[i] = inTotal;
            inTotal = Align64(inTotal + sched[i].InElems);
            outBase[i] = outTotal;
            outTotal = Align64(outTotal + sched[i].OutElems);
            outOffsets[i] = outElems;
            outElems += sched[i].OutElems;
            arena = Math.Max(arena, sched[i].ArenaElems);
        }
        if (n == 1) inTotal = Math.Max(inTotal, input.Length);
        // Several units: record them level-interleaved — dispatch k of every
        // unit, then ONE barrier. Units are independent (own arena, in/out and
        // partials regions), so they run concurrently on the GPU (occupancy
        // for small batches) and the barrier count stays at one unit's worth.
        // Sequences may differ per shape (size-gated kernel choices); a unit
        // simply stops once its own list ends. Units are grouped into waves —
        // one command buffer + submission each — cut where a streaming caller
        // wants early results or where summed arenas exceed a budget. Waves
        // reuse the same regions: submission order plus the trailing barrier
        // of each wave orders them on the queue.
        bool interleave = n > 1 && !s_prof && !s_onlyHead && s_only < 0 && s_trunc < 0 && !s_noBar;
        var arenaBase = new long[n];
        var waveStart = new List<int> { 0 };
        int maxWave = 1;
        if (interleave)
        {
            int[] cuts = sink is null ? [] : StreamCuts(sched);
            long a = 0, peak = 0;
            for (int i = 0, ci = 0; i < n; i++)
            {
                long need = (sched[i].ArenaElems + 127) & ~127L;
                bool cut = ci < cuts.Length && cuts[ci] == i;
                if (cut) ci++;
                if (i > 0 && (cut || a + need > InterleaveArenaBudget)) { waveStart.Add(i); a = 0; }
                arenaBase[i] = a;
                a += need;
                peak = Math.Max(peak, a);
                maxWave = Math.Max(maxWave, i - waveStart[^1] + 1);
            }
            arena = peak;
        }
        waveStart.Add(n);
        int waves = waveStart.Count - 1;
        EnsureBuffers(arena, inTotal, outTotal);
        long consumed = 0;
        fixed (float* src = input)
            for (int i = 0; i < n; i++)
            {
                long len = n == 1 ? input.Length : sched[i].InElems;
                Buffer.MemoryCopy(src + consumed, _inMap + inBase[i], len * 4, len * 4);
                consumed += len;
            }
        _in!.Flush(0, (ulong)inTotal * 4);
        long t1 = Stopwatch.GetTimestamp();

        EnsurePart(interleave ? maxWave : 1);
        EnsureCommandBuffers(waves);
        _descUsed = 0;
        if (!_dev.PushDescriptors) EnsureDescPool((uint)sched.Sum(x => x.Recs.Length));
        // reused across runs: the caller consumes it before this session runs again
        if (_result.Length < outElems) _result = new float[RoundUpPow2(outElems)];
        float[] result = _result;
        int submitted = 0, drained = 0;
        long t2 = 0, t3;
        try
        {
            for (int w = 0; w < waves; w++)
            {
                IntPtr cmd = _cmds[w];
                BeginRecord(cmd);
                int s0 = waveStart[w], s1 = waveStart[w + 1];
                if (interleave)
                {
                    VkPipeline? bound = null;
                    int levels = 0;
                    for (int i = s0; i < s1; i++) levels = Math.Max(levels, sched[i].Recs.Length);
                    for (int k = 0; k < levels; k++)
                    {
                        for (int i = s0; i < s1; i++)
                            if (k < sched[i].Recs.Length)
                                RecordOne(cmd, sched[i].Recs[k], ref bound, (ulong)inBase[i] * 4,
                                    (ulong)outBase[i] * 4, (ulong)arenaBase[i] * 2,
                                    (ulong)(i - s0) * GpuGraphModel.PartBytes);
                        _dev.CmdComputeBarrier(cmd);
                    }
                }
                else
                    for (int i = s0; i < s1; i++)
                        RecordRecs(cmd, Selected(sched[i]), s_prof && n == 1, !s_noBar,
                            (ulong)inBase[i] * 4, (ulong)outBase[i] * 4);
                Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
                _dev.Submit(cmd, _fences[w]);
                submitted++;
            }
            t2 = Stopwatch.GetTimestamp();
            bool more = true;
            for (int w = 0; w < waves; w++)
            {
                _dev.WaitFence(_fences[w]);
                drained++;
                int s0 = waveStart[w], s1 = waveStart[w + 1];
                ulong lo = (ulong)outBase[s0] * 4;
                _out!.Invalidate(lo, (ulong)Align64(outBase[s1 - 1] + sched[s1 - 1].OutElems) * 4 - lo);
                for (int i = s0; i < s1; i++)
                    new ReadOnlySpan<float>(_outMap + outBase[i], sched[i].OutElems)
                        .CopyTo(result.AsSpan(outOffsets[i]));
                if (sink is not null && more) more = sink(result, outOffsets, s0, s1 - s0);
            }
        }
        finally
        {
            // never leave a submission in flight over buffers the next run reuses
            for (int w = drained; w < submitted; w++)
                try { _dev.WaitFence(_fences[w]); } catch (InvalidOperationException) { }
        }
        t3 = Stopwatch.GetTimestamp();
        if (s_prof && n == 1) AccumulateProfile(sched[0]);
        _last = sched[n - 1];
        if (s_dbgTime)
        {
            double f = Stopwatch.Frequency / 1e3;
            Console.WriteLine($"[t] units={n}{(interleave ? " il" : "")} waves={waves} write={(t1 - t0) / f:F2} record+submit={(t2 - t1) / f:F2} " +
                $"wait+read+sink={(t3 - t2) / f:F2} ms recs={sched.Sum(x => x.Recs.Length)}");
        }
        return result;
    }

    // Unit indices where a streamed run opens a new submission: ~55% of the
    // input volume in the first wave (GPU starts while the rest records; the
    // caller's tail for it overlaps the remainder), a third wave for large
    // images so the last, un-overlapped tail stays short.
    private static int[] StreamCuts(GpuSchedule[] sched)
    {
        int n = sched.Length;
        if (n < 4) return [];
        double[] fractions = n >= 9 ? [0.45, 0.78] : [0.55];
        long total = 0;
        foreach (GpuSchedule s in sched) total += s.InElems;
        var cuts = new List<int>();
        long acc = 0;
        int f = 0;
        for (int i = 0; i < n - 1 && f < fractions.Length; i++)
        {
            acc += sched[i].InElems;
            if (acc >= fractions[f] * total) { cuts.Add(i + 1); f++; }
        }
        return [.. cuts];
    }

    // fp16 elements of concurrently live unit arenas in one interleaved wave (512 MB)
    private const long InterleaveArenaBudget = 256L << 20;

    private float[] _result = [];

    private static long RoundUpPow2(long v)
    {
        long p = 1024;
        while (p < v) p <<= 1;
        return p;
    }

    private int _partUnits;

    private void EnsurePart(int units)
    {
        if (_part is not null && units <= _partUnits) return;
        int alloc = Math.Max(units, _partUnits * 3 / 2);
        _part?.Free();
        _part = null;
        // zeroed once: se_fused ticket counters self-reset after each run
        _part = _dev.NewStorageBuffer((ulong)alloc * GpuGraphModel.PartBytes, hostVisible: false);
        _dev.Zero(_part);
        _partUnits = alloc;
    }

    private void EnsureBuffers(long arenaElems, long needIn, long needOut)
    {
        if (_arena is null || arenaElems > _arenaElems)
        {
            long alloc = Math.Max(arenaElems, _arenaElems * 3 / 2);
            _arena?.Free();
            _arena = null;
            _arena = _dev.NewStorageBuffer((ulong)alloc * 2, hostVisible: false);
            _arenaElems = alloc;
        }
        // host-visible reallocation + map costs ~20 ms: grow in power-of-two
        // steps so a stream of varying DET sizes settles after a few images
        if (_in is null || needIn > _inElems)
        {
            long alloc = RoundUpPow2(needIn);
            FreeMapped(ref _in, ref _inMap);
            _in = _dev.NewStorageBuffer((ulong)alloc * 4, hostVisible: true, preferHost: false);
            _inMap = (float*)_in.Map();
            _inElems = alloc;
        }
        if (_out is null || needOut > _outElems)
        {
            long alloc = RoundUpPow2(needOut);
            FreeMapped(ref _out, ref _outMap);
            _out = _dev.NewStorageBuffer((ulong)alloc * 4, hostVisible: true, preferHost: true);
            _outMap = (float*)_out.Map();
            _outElems = alloc;
        }
    }

    private static void FreeMapped(ref VkBuffer? buf, ref float* map)
    {
        if (buf is null) return;
        buf.Unmap();
        buf.Free();
        buf = null;
        map = null;
    }

    private VkBuffer Resolve(VkBuffer b) =>
        ReferenceEquals(b, GpuGraphModel.RoleArena) ? _arena!
        : ReferenceEquals(b, GpuGraphModel.RoleIn) ? _in!
        : ReferenceEquals(b, GpuGraphModel.RoleOut) ? _out!
        : ReferenceEquals(b, GpuGraphModel.RolePart) ? _part!
        : b;

    private IReadOnlyList<GpuRec> Selected(GpuSchedule s)
    {
        GpuRec[] r = s.Recs;
        if (s_onlyHead) return [r[0]];
        if (s_only >= 0 && s_only < r.Length) return Enumerable.Repeat(r[s_only], 20).ToArray();
        if (s_trunc >= 0 && s_trunc < r.Length) return r.Take(s_trunc).ToArray();
        return r;
    }

    private static void BeginRecord(IntPtr cmd)
    {
        Vk.Check(Vk.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        var begin = new Vk.VkCommandBufferBeginInfo
        {
            SType = VkConst.StCommandBufferBeginInfo, Flags = 1u,   // ONE_TIME_SUBMIT
        };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
    }

    private uint _descUsed;

    private void RecordRecs(IntPtr cmd, IReadOnlyList<GpuRec> recs, bool prof, bool barriers,
        ulong inBase = 0, ulong outBase = 0)
    {
        bool push = _dev.PushDescriptors;
        if (!push) EnsureDescPool(_descUsed + (uint)recs.Count);
        if (prof)
        {
            EnsureQueryPool(recs.Count * 2);
            Vk.vkCmdResetQueryPool(cmd, _queryPool, 0, (uint)(recs.Count * 2));
            _queryCount = recs.Count * 2;
        }
        VkPipeline? bound = null;
        uint q = 0;
        foreach (GpuRec r in recs)
        {
            if (prof) Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, _queryPool, q++);
            RecordOne(cmd, r, ref bound, inBase, outBase, 0, 0);
            if (barriers) _dev.CmdComputeBarrier(cmd);
            if (prof) Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, _queryPool, q++);
        }
    }

    private void RecordOne(IntPtr cmd, GpuRec r, ref VkPipeline? bound, ulong inBase, ulong outBase,
        ulong arenaBase, ulong partBase)
    {
        Vk.VkDescriptorBufferInfo* infos = stackalloc Vk.VkDescriptorBufferInfo[16];
        Vk.VkWriteDescriptorSet* writes = stackalloc Vk.VkWriteDescriptorSet[16];
        bool push = _dev.PushDescriptors;
        if (!ReferenceEquals(bound, r.Pipe))
        {
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, r.Pipe.Pipeline);
            bound = r.Pipe;
        }
        int n = r.Binds.Length;
        IntPtr set = IntPtr.Zero;
        if (!push) { set = _dev.AllocateDescriptorSet(_descPool, r.Pipe.SetLayout); _descUsed++; }
        for (int b = 0; b < n; b++)
        {
            VkBuffer role = r.Binds[b].Buf;
            VkBuffer buf = Resolve(role);
            ulong off = r.Binds[b].ByteOffset
                + (ReferenceEquals(role, GpuGraphModel.RoleIn) ? inBase
                   : ReferenceEquals(role, GpuGraphModel.RoleOut) ? outBase
                   : ReferenceEquals(role, GpuGraphModel.RoleArena) ? arenaBase
                   : ReferenceEquals(role, GpuGraphModel.RolePart) ? partBase : 0);
            infos[b] = new Vk.VkDescriptorBufferInfo
            {
                Buffer = buf.Buffer, Offset = off, Range = buf.Size - off,
            };
            writes[b] = new Vk.VkWriteDescriptorSet
            {
                SType = VkConst.StWriteDescriptorSet, DstSet = set, DstBinding = (uint)b,
                DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                PBufferInfo = &infos[b],
            };
        }
        if (push)
            _dev.CmdPushDescriptors(cmd, r.Pipe.Layout, writes, (uint)n);
        else
        {
            Vk.vkUpdateDescriptorSets(_dev.Device, (uint)n, writes, 0, null);
            Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, r.Pipe.Layout, 0, 1, &set, 0, null);
        }
        fixed (byte* pp = r.Pc)
            Vk.vkCmdPushConstants(cmd, r.Pipe.Layout, VkConst.StageComputeShader,
                0, (uint)r.Pc.Length, pp);
        Vk.vkCmdDispatch(cmd, r.Gx, r.Gy, 1);
    }

    // No-push fallback: one resettable pool per session. A pool that is too
    // small for this recording is replaced before any set of the current
    // recording was taken from it (sizing happens up front per unit).
    private void EnsureDescPool(uint setsNeeded)
    {
        if (_descPool != IntPtr.Zero && setsNeeded <= _descPoolSets)
        {
            if (_descUsed == 0)
                Vk.Check(Vk.vkResetDescriptorPool(_dev.Device, _descPool, 0), "vkResetDescriptorPool");
            return;
        }
        if (_descUsed != 0)
            throw new InvalidOperationException("descriptor pool undersized mid-recording");
        if (_descPool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null);
        _descPoolSets = Math.Max(setsNeeded, 256u) * 4;
        _descPool = _dev.NewDescriptorPool(_descPoolSets);
    }

    private void EnsureQueryPool(int count)
    {
        if (_queryPool != IntPtr.Zero && count <= _queryCap) return;
        if (_queryPool != IntPtr.Zero) Vk.vkDestroyQueryPool(_dev.Device, _queryPool, null);
        var qci = new Vk.VkQueryPoolCreateInfo { SType = 11, QueryType = 2 /* TIMESTAMP */, QueryCount = (uint)count };
        Vk.Check(Vk.vkCreateQueryPool(_dev.Device, &qci, null, out _queryPool), "vkCreateQueryPool");
        _queryCap = count;
    }

    // ---- debug helpers (read arena slots after a Run of the same key) ----

    private GpuSchedule Sched(int[] inputShape, int nodeLimit, int outTensor)
        => _model.TryGetSchedule(GpuGraphModel.KeyOf(inputShape, nodeLimit, outTensor))
           ?? throw new InvalidOperationException("no schedule for this key — Run first");

    private float[] ReadArena(ulong byteOff, int count)
    {
        VkBuffer tmp = _dev.NewStorageBuffer((ulong)count * 4, hostVisible: true, preferHost: true);
        try
        {
            var rec = new GpuRec
            {
                Pipe = _model.OutPipe,
                Binds = [new GpuBind(GpuGraphModel.RoleArena, byteOff), new GpuBind(tmp, 0)],
                Pc = [.. BitConverter.GetBytes((uint)count), .. BitConverter.GetBytes(0u)],
                Gx = (uint)((count + 255) / 256), Gy = 1,
            };
            IntPtr cmd = _cmds[0];
            BeginRecord(cmd);
            _descUsed = 0;
            RecordRecs(cmd, [rec], false, false);
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
            _dev.Submit(cmd, _fences[0]);
            _dev.WaitFence(_fences[0]);
            float* fp = (float*)tmp.Map();
            float[] v = new ReadOnlySpan<float>(fp, count).ToArray();
            tmp.Unmap();
            return v;
        }
        finally { tmp.Free(); }
    }

    private static int PhysOf(GpuSchedule s, int t)
    {
        while (s.Alias[t] != t) t = s.Alias[t];
        return t;
    }

    /// <summary>Debug: raw values of a tensor's arena slot.</summary>
    public float[] DebugValues(int tensorIndex, int[] inputShape, int count = 16,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        GpuSchedule s = Sched(inputShape, nodeLimit, outTensor);
        int t = PhysOf(s, tensorIndex);
        int n = Math.Min(count, (int)s.Numel[tensorIndex]);
        Console.Error.WriteLine($"  [dbg] t={tensorIndex} phys={t} off={s.Off[t]} n={s.Numel[tensorIndex]}");
        if (s.Off[t] < 0) return new float[n];
        return ReadArena((ulong)s.Off[t] * 2, n);
    }

    public long Im2colOffset(int[] inputShape) => Sched(inputShape, int.MaxValue, -1).Im2colOff;

    /// <summary>Debug: arena element offset of a tensor's slot.</summary>
    public long DebugElemOff(int tensorIndex, int[] inputShape,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        GpuSchedule s = Sched(inputShape, nodeLimit, outTensor);
        return s.Off[PhysOf(s, tensorIndex)];
    }

    public float[] DebugValuesRaw(int[] inputShape, long elemOff, int count,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        Sched(inputShape, nodeLimit, outTensor);
        return ReadArena((ulong)elemOff * 2, count);
    }

    /// <summary>Debug: min/max/mean of a tensor's arena slot (NHWC flat) after a Run.</summary>
    public (float Min, float Max, double Mean) DebugStats(int tensorIndex, int[] inputShape,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        GpuSchedule s = Sched(inputShape, nodeLimit, outTensor);
        int t = PhysOf(s, tensorIndex);
        int n = (int)s.Numel[tensorIndex];
        float[] v = s.Off[t] < 0 ? new float[n] : ReadArena((ulong)s.Off[t] * 2, n);
        float mn = float.MaxValue, mx = float.MinValue; double sum = 0;
        foreach (float x in v) { mn = MathF.Min(mn, x); mx = MathF.Max(mx, x); sum += x; }
        return (mn, mx, sum / n);
    }

    public int DispatchCount(int[] inputShape, int nodeLimit = int.MaxValue, int outTensor = -1)
        => _model.TryGetSchedule(GpuGraphModel.KeyOf(inputShape, nodeLimit, outTensor))?.Recs.Length ?? 0;

    // Per-dispatch min over all profiled runs of the same schedule: a shared
    // GPU (other contexts time-slicing in) inflates single samples, the
    // minimum is the dispatch's own cost.
    private GpuSchedule? _profSched;
    private double[] _profMin = [];

    private void AccumulateProfile(GpuSchedule s)
    {
        if (_queryPool == IntPtr.Zero) return;
        ulong[] ts = new ulong[_queryCount];
        fixed (ulong* tp = ts)
            Vk.vkGetQueryPoolResults(_dev.Device, _queryPool, 0, (uint)_queryCount,
                (nuint)(ts.Length * 8), tp, 8, 1 | 2);
        int nr = _queryCount / 2;
        if (!ReferenceEquals(_profSched, s) || _profMin.Length != nr)
        {
            _profSched = s;
            _profMin = Enumerable.Repeat(double.MaxValue, nr).ToArray();
        }
        for (int i = 0; i < nr; i++)
            _profMin[i] = Math.Min(_profMin[i], (ts[2 * i + 1] - ts[2 * i]) * _dev.TimestampPeriodNs / 1e6);
    }

    /// <summary>Per-dispatch GPU time (min over profiled runs; SIMD_OCR_GPU_PROF=1).</summary>
    public void DumpProfile(int[] inputShape, int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        if (_profSched is null)
        { Console.WriteLine("(no profile — set SIMD_OCR_GPU_PROF=1)"); return; }
        IReadOnlyList<GpuRec> recs = Selected(_profSched);
        var agg = new Dictionary<string, double>();
        for (int i = 0; i < _profMin.Length; i++)
        {
            string tag = recs[i].Pipe.Name;
            agg[tag] = agg.TryGetValue(tag, out double v) ? v + _profMin[i] : _profMin[i];
        }
        Console.WriteLine($"--- GPU profile ({recs.Count} dispatches, {_profMin.Sum():F2} ms, min/dispatch) ---");
        foreach (var kv in agg.OrderByDescending(k => k.Value))
            Console.WriteLine($"  {kv.Key,-14} {kv.Value,8:F3} ms");
        if (Environment.GetEnvironmentVariable("SIMD_OCR_GPU_PROF_ALL") == "1")
            for (int i = 0; i < _profMin.Length; i++)
                Console.WriteLine($"  [{i,3}] {_profMin[i],7:F3} {recs[i].Tag}");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (IntPtr f in _fences) _dev.DestroyFence(f);
        _fences.Clear();
        _dev.DestroyCommandPool(_pool);   // frees _cmds with it
        _cmds.Clear();
        _pool = IntPtr.Zero;
        if (_descPool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null);
        if (_queryPool != IntPtr.Zero) Vk.vkDestroyQueryPool(_dev.Device, _queryPool, null);
        _descPool = _queryPool = IntPtr.Zero;
        _arena?.Free(); _arena = null;
        _part?.Free(); _part = null;
        FreeMapped(ref _in, ref _inMap);
        FreeMapped(ref _out, ref _outMap);
        _model.Release();
    }
}
