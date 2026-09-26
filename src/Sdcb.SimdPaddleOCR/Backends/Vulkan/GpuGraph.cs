using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

/// <summary>
/// Whole-graph GPU execution plan for the DET model: translates a
/// <see cref="CompiledModel"/> node list into a recorded command buffer —
/// one dispatch per (fused) node, all activations in a single device-local
/// fp16 NHWC arena, weights uploaded fp16 at build time.
/// A plan is bound to one input shape; plans are cached per shape.
/// </summary>
internal sealed class GpuDetGraph : IDisposable
{
    private readonly VkDevice _dev;
    private readonly Model _model;
    private readonly CompiledModel _compiled;

    private readonly VkPipeline _pSoftmax;
    private bool _sg32; // device cannot run sg16 coopmat pipes (NVIDIA/AMD)
    private readonly VkPipeline _pConv1x1, _pConv1x1N64, _pConv1x1N32, _pDot,
        _pDw, _pDw4, _pDw4T, _pDense, _pConvT, _pConvT4, _pElem, _pElem4,
        _pReduce, _pReduce4, _pReduce4b, _pPool, _pPool4,
        _pResize, _pResize4, _pResize4Add, _pConcat, _pConcat4, _pNchw, _pOut, _pIm2col,
        _pAddPs, _pSeA, _pSeB, _pSeF, _pConvD, _pConvDF32, _pCatRes,
        _pAvg4, _pAffine;
    private VkBuffer? _partBuf;   // reduce_hw4 phase-1 partials (fp32)

    // fp16 copies of constant tensors (weights, biases, scalars), lazy
    private readonly Dictionary<(int, int, int, int), VkBuffer> _constF16 = new();
    private readonly Dictionary<int, VkBuffer> _constF32 = new();
    private readonly Dictionary<int, VkBuffer> _constTap = new();

    // conv weight repack to tap-major [Cout, (dy*kW+dx)*Cin+ci], rows padded to Kp
    private VkBuffer ConstTapMajor(int tensorIndex, int cout, int cin,
        int kH, int kW, int kp, int cinPad = 0)
    {
        if (cinPad == 0) cinPad = cin;
        if (_constTap.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        int kSpan = kH * kW;
        int rowsPad = (cout + 127) / 128 * 128;
        Half[] h = new Half[rowsPad * kp];
        for (int co = 0; co < cout; co++)
            for (int ci = 0; ci < cinPad; ci++)
                for (int tap = 0; tap < kSpan; tap++)
                    if (ci < cin)
                        h[co * kp + tap * cinPad + ci] =
                            (Half)f32[co * cin * kSpan + ci * kSpan + tap];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constTap[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // depthwise weight repack to tap-major [tap*C + c] fp16 (vec4-readable)
    private readonly Dictionary<int, VkBuffer> _constDw = new();
    private VkBuffer ConstDwTap(int tensorIndex, int c, int kSpan)
    {
        if (_constDw.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        Half[] h = new Half[c * kSpan];
        for (int cc = 0; cc < c; cc++)
            for (int tap = 0; tap < kSpan; tap++)
                h[tap * c + cc] = (Half)f32[cc * kSpan + tap];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constDw[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // k-major weight repack for conv1x1_dot: [k*Cout + n], k = tap*cinPad + ci
    private readonly Dictionary<int, VkBuffer> _constKm = new();
    private VkBuffer ConstKMajor(int tensorIndex, int cout, int cin,
        int kSpan, int kp, int cinPad)
    {
        if (_constKm.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        Half[] h = new Half[kp * cout];
        for (int co = 0; co < cout; co++)
            for (int ci = 0; ci < cinPad; ci++)
                for (int tap = 0; tap < kSpan; tap++)
                    if (ci < cin)
                        h[(tap * cinPad + ci) * cout + co] =
                            (Half)f32[co * cin * kSpan + ci * kSpan + tap];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constKm[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // ConvTranspose weight repack [tap*Cin + ci, co] (tap=dy*2+dx) fp16
    private readonly Dictionary<int, VkBuffer> _constCt = new();
    private VkBuffer ConstConvT(int tensorIndex, int cin, int cout)
    {
        if (_constCt.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        Half[] h = new Half[4 * cin * cout];
        for (int tap = 0; tap < 4; tap++)
            for (int ci = 0; ci < cin; ci++)
                for (int co = 0; co < cout; co++)
                    h[(tap * cin + ci) * cout + co] =
                        (Half)f32[(ci * cout + co) * 4 + tap];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constCt[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    private VkBuffer PartBuf(int floats)
    {
        if (_partBuf == null)
        {
            // fixed 64KB covers S≤16 partitions × C≤1024 channels + ctr slots;
            // zeroed once: se_fused counters self-reset to 0 after each run
            _partBuf = _dev.NewStorageBuffer(64 * 1024, hostVisible: false);
            unsafe
            {
                fixed (byte* z = new byte[64 * 1024])
                    _dev.Upload(_partBuf, z, 64 * 1024);
            }
            _allBufs.Add(_partBuf);
        }
        return _partBuf;
    }
    private readonly List<VkBuffer> _allBufs = new();

    public GpuDetGraph(VkDevice dev, CompiledModel compiled)
    {
        _dev = dev;
        _compiled = compiled;
        _model = compiled.Model;
        // sg16-only coopmat shaders: NVIDIA (sg 32-32) and AMD wave64 cannot
        // satisfy requiredSubgroupSize=16 — swap in the sg32 variant.
        _sg32 = dev.SubgroupMin > 16 && dev.SubgroupMax >= 32;
        if (_sg32)
        {
            _pConv1x1 = Pipe("conv1x1_cm_sg32", 6, 16, 32);
            _pConv1x1N64 = _pConv1x1N32 = _pConv1x1;
        }
        else
        {
            _pConv1x1 = Pipe("conv1x1_cm", 6, 16, 16);
            _pConv1x1N64 = Pipe("conv1x1_cm_n64", 6, 16, 16);
            _pConv1x1N32 = Pipe("conv1x1_cm_n32", 6, 16, 16);
        }
        _pDot = Pipe("conv1x1_dot", 8, 28);
        _pDw = Pipe("conv_dw", 4, 48);
        _pDw4 = Pipe("conv_dw4", 4, 48);
        _pDw4T = Pipe("conv_dw4t", 6, 48);
        _pDense = Pipe("conv_dense", 4, 52);
        _pIm2col = Pipe("im2col", 2, 56);
        _pConvT = Pipe("convt2s2", 5, 24);
        _pConvT4 = Pipe("convt4", 4, 24);
        _pReduce4 = Pipe("reduce_hw4", 2, 16);
        _pReduce4b = Pipe("reduce_hw4b", 2, 12);
        _pPool4 = Pipe("maxpool4", 2, 20);
        _pResize4 = Pipe("resize4", 4, 32);
        _pResize4Add = Pipe("resize4add", 5, 24);
        _pConcat4 = Pipe("concat4", 4, 20);
        _pCatRes = Pipe("catresize", 13, 76);
        _pElem = Pipe("elem", 3, 24);
        _pElem4 = Pipe("elem4", 3, 24);
        _pAddPs = Pipe("addps", 4, 12);
        _pSeA = Pipe("se_a", 2, 20);
        _pSeB = Pipe("se_b", 6, 24);
        _pSeF = Pipe("se_fused", 8, 32);
        _pConvD = Pipe("convk_dot", 5, 60);
        _pConvDF32 = Pipe("convk_f32n", 5, 68);
        _pReduce = Pipe("reduce_hw", 2, 8);
        _pPool = Pipe("maxpool2e", 2, 12);
        _pResize = Pipe("resize_nn", 2, 20);
        _pConcat = Pipe("concat_c", 2, 16);
        _pNchw = Pipe("nchw2nhwc", 2, 12);
        _pOut = Pipe("sigmoid_out", 2, 8);
        _pAvg4 = Pipe("avgpool4", 2, 44);
        _pAffine = Pipe("affine4", 4, 8);
        _pSoftmax = Pipe("softmax", 2, 8);
    }

    private VkPipeline Pipe(string name, int bindings, int pcBytes, uint reqSg = 0)
    {
        VkPipeline p = _dev.NewPipeline(
            _dev.NewShaderModule(LoadSpv(name)), bindings, pcBytes, reqSg);
        p.Name = name;
        return p;
    }

    private static byte[] LoadSpv(string name)
    {
        var asm = typeof(GpuDetGraph).Assembly;
        using Stream s = asm.GetManifestResourceStream(
            $"Sdcb.SimdPaddleOCR.Backends.Vulkan.Shaders.{name}.spv")
            ?? throw new FileNotFoundException(name + ".spv");
        byte[] b = new byte[s.Length];
        s.ReadExactly(b);
        return b;
    }

    // fp16 device copy of a constant tensor; padded to padRows*rowElemsPad
    // elements. rowPad>rowElems repacks [rows, rowElems] into [rows, rowPad].
    private VkBuffer ConstF16(int tensorIndex, int padRows = 0, int rowElems = 0,
        int rowPad = 0)
    {
        var key = (tensorIndex, padRows, rowElems, rowPad);
        if (_constF16.TryGetValue(key, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        int n = f32.Length;
        int alloc;
        Half[] h;
        if (rowPad > rowElems && rowElems > 0)
        {
            int rows = Math.Max(n / rowElems, padRows);
            alloc = rows * rowPad;
            h = new Half[alloc];
            for (int r = 0; r * rowElems < n; r++)
                for (int c = 0; c < rowElems && r * rowElems + c < n; c++)
                    h[r * rowPad + c] = (Half)f32[r * rowElems + c];
        }
        else
        {
            alloc = Math.Max(n, padRows * rowElems);
            h = new Half[alloc];
            for (int i = 0; i < n; i++) h[i] = (Half)f32[i];
        }
        VkBuffer buf = _dev.NewStorageBuffer((ulong)alloc * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)alloc * 2); }
        _constF16[key] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    // fp32 device copy of a constant tensor (small tensors needing exact math).
    private VkBuffer ConstF32(int tensorIndex)
    {
        if (_constF32.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        VkBuffer buf = _dev.NewStorageBuffer((ulong)f32.Length * 4, hostVisible: false);
        unsafe { fixed (float* p = f32) _dev.Upload(buf, p, (ulong)f32.Length * 4); }
        _constF32[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    private ReadOnlySpan<float> CstF32(int tensorIndex) =>
        MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);

    // fp16 device copy of a computed vector (e.g. folded BN affine params)
    private readonly List<VkBuffer> _vecF16 = new();
    private VkBuffer VecF16(float[] v)
    {
        Half[] h = new Half[v.Length];
        for (int i = 0; i < v.Length; i++) h[i] = (Half)v[i];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _vecF16.Add(buf);
        _allBufs.Add(buf);
        return buf;
    }

    // MatMul weight repack [K,N] row-major -> [N,K] fp16 (coopmat B layout),
    // N padded to a 128-row tile.
    private readonly Dictionary<int, VkBuffer> _constGw = new();
    private VkBuffer ConstGemmW(int tensorIndex, int K, int N)
    {
        if (_constGw.TryGetValue(tensorIndex, out VkBuffer? cached)) return cached;
        ReadOnlySpan<float> f32 =
            MemoryMarshal.Cast<byte, float>(_compiled.GetTensor(tensorIndex).Constant);
        int nPad = (N + 127) / 128 * 128;
        Half[] h = new Half[nPad * K];
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                h[n * K + k] = (Half)f32[k * N + n];
        VkBuffer buf = _dev.NewStorageBuffer((ulong)h.Length * 2, hostVisible: false);
        unsafe { fixed (Half* p = h) _dev.Upload(buf, p, (ulong)h.Length * 2); }
        _constGw[tensorIndex] = buf;
        _allBufs.Add(buf);
        return buf;
    }

    private sealed class Rec
    {
        public required VkPipeline Pipe;
        public required IntPtr Set;
        public required byte[] Pc;
        public uint Gx, Gy;
        public string Tag = "";
    }

    private sealed class Plan
    {
        public required VkBuffer Arena, InF32, OutF32;
        public required IntPtr Cmd;
        public required List<Rec> Recs;
        public int OutElems;
        public long ArenaBytes;
        public IntPtr QueryPool;
        public int QueryCount;
        public long[] Off = [];
        public int[] Alias = [];
        public long[] Numel = [];
        public int[][] Shapes = [];
        public int[] RecOut = [];   // rec index -> output tensor index
        public long Im2colOff;
    }

    private readonly Dictionary<PlanKey, Plan> _plans = new();
    public readonly record struct PlanKey(int N, int H, int W, int NodeLimit, int OutTensor);

    private static PlanKey KeyOf(int[] inputShape, int nodeLimit, int outTensor)
        => new(inputShape[0], inputShape[2], inputShape[3], nodeLimit, outTensor);
    private IntPtr _fence;
    private readonly bool _dbgTime =
        Environment.GetEnvironmentVariable("SIMD_OCR_GPU_TIME") == "1";

    /// <summary>Run the graph: input fp32 NCHW [n,C,H,W] → fp32 [graph output].
    /// nodeLimit truncates the emit loop (nodes beyond it are not dispatched) and
    /// outTensor overrides the readback tensor — used to stop the REC graph before
    /// the vocab projection (activations readback) instead of the graph output.
    /// </summary>
    public unsafe float[] Run(int[] inputShape, ReadOnlySpan<float> input,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        PlanKey key = KeyOf(inputShape, nodeLimit, outTensor);
        if (!_plans.TryGetValue(key, out Plan? plan))
        {
            plan = BuildPlan(inputShape, nodeLimit, outTensor);
            _plans[key] = plan;
        }
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        void* dst = plan.InF32.Map();
        fixed (float* src = input)
            Buffer.MemoryCopy(src, dst, input.Length * 4, input.Length * 4);
        plan.InF32.Flush(0, (ulong)input.Length * 4);
        plan.InF32.Unmap();
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();

        if (_fence == IntPtr.Zero) _fence = _dev.NewFence();
        _dev.Submit(plan.Cmd, _fence);
        _dev.WaitFence(_fence);
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();

        float[] o = new float[plan.OutElems];
        float* op = (float*)plan.OutF32.Map();
        new ReadOnlySpan<float>(op, plan.OutElems).CopyTo(o);
        plan.OutF32.Unmap();
        long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_dbgTime)
        {
            double f = System.Diagnostics.Stopwatch.Frequency;
            Console.WriteLine($"[t] write={(t1 - t0) / f * 1e3:F2} submit+wait={(t2 - t1) / f * 1e3:F2} read={(t3 - t2) / f * 1e3:F2} ms");
        }
        return o;
    }

    public int DispatchCount(PlanKey key) => _plans.TryGetValue(key, out Plan? p) ? p.Recs.Count : 0;
    public long ArenaBytes(PlanKey key) => _plans.TryGetValue(key, out Plan? p) ? p.ArenaBytes : 0;

    private unsafe Plan BuildPlan(int[] inputShape, int nodeLimit, int outTensor)
    {
        int[][] shapes = _compiled.ResolveShapesFor(inputShape);
        NodeRecord[] nodes = _model.Nodes;
        int nT = shapes.Length;
        int outIdx = outTensor >= 0 ? outTensor
            : checked((int)_model.GraphOutputs[0]);
        int inIdx = checked((int)_model.GraphInputs[0]);
        int emitLimit = Math.Min(nodeLimit, nodes.Length);
        int nb = inputShape[0];   // batch — spatial kernels take it via gy

        var alias = new int[nT];          // metadata-only ops alias producer slot
        var isConst = new bool[nT];
        var numel = new long[nT];
        var used = new bool[nT];
        for (int i = 0; i < nT; i++)
        {
            alias[i] = i;
            numel[i] = 1;
            foreach (int d in shapes[i]) numel[i] *= Math.Max(d, 1);
            isConst[i] = _compiled.GetTensor(i).IsConstant;
        }
        int Phys(int t) { while (alias[t] != t) t = alias[t]; return t; }

        // refcount / producer / consumers for emit-time fusion decisions
        var refCount = new int[nT];
        var prodNode = new int[nT];
        Array.Fill(prodNode, -1);
        var consumers = new List<int>[nT];
        // members of conv-headed fused groups are folded into the conv dispatch;
        // groups headed by other ops emit every member standalone, so a Mul
        // inside e.g. a HardSigmoid+Mul (hardswish) group is still foldable.
        var inConvGroup = new bool[nodes.Length];
        for (int ni = 0; ni < nodes.Length; ni++)
        {
            int skip = _compiled.FusedSkip(ni);
            if (nodes[ni].Operator is OperatorId.Conv or OperatorId.ConvTranspose)
                for (int j = ni + 1; j <= ni + skip && j < nodes.Length; j++)
                    inConvGroup[j] = true;
            foreach (uint inp in nodes[ni].Inputs)
            {
                if (inp == uint.MaxValue) continue;
                refCount[(int)inp]++;
                (consumers[(int)inp] ??= new List<int>()).Add(ni);
            }
            prodNode[checked((int)nodes[ni].Outputs[0])] = ni;
        }

        // ---- pass 1: aliases + which tensors need arena slots ----
        for (int ni = 0; ni < nodes.Length; ni++)
        {
            NodeRecord node = nodes[ni];
            int skip = _compiled.FusedSkip(ni);
            if (node.Operator is OperatorId.LayoutConvert or OperatorId.Squeeze
                or OperatorId.Unsqueeze or OperatorId.Reshape)
            {
                alias[checked((int)node.Outputs[0])] = checked((int)node.Inputs[0]);
                continue;
            }
            // rank-3 [n,c,w] -> [n,w,c] transpose is a no-op under the
            // channel-last physical layout: out flat (n*w'+c') index equals the
            // producer's (n*w+c) index because w'=w rows and c'=c channels swap
            // positions but keep the same flat offset.
            if (node.Operator == OperatorId.Transpose
                && shapes[checked((int)node.Inputs[0])].Length == 3
                && _model.GetParameters(node) is ReadOnlySpan<byte> p2
                && U16(p2, 2) == 3 && I32(p2, 4) == 0 && I32(p2, 8) == 2
                && I32(p2, 12) == 1)
            {
                alias[checked((int)node.Outputs[0])] = checked((int)node.Inputs[0]);
                continue;
            }
            foreach (uint inp in node.Inputs)
                if (inp != uint.MaxValue && !isConst[checked((int)inp)])
                    used[checked((int)inp)] = true;
            // the slot that gets written = the fused group's last node output
            int sink = checked((int)nodes[ni + skip].Outputs[0]);
            used[sink] = true;
            // fused members reading tensors produced BEFORE the group head
            // (residuals, dual-write pre-gelu taps) need real slots too —
            // in-group reads never touch the arena.
            for (int j = ni + 1; j <= ni + skip; j++)
                foreach (uint inp2 in nodes[j].Inputs)
                {
                    if (inp2 == uint.MaxValue) continue;
                    int ip = checked((int)inp2);
                    if (!isConst[ip] && prodNode[ip] >= 0 && prodNode[ip] < ni)
                        used[Phys(ip)] = true;
                }
            // conv+bias+gelu groups dual-write the pre-gelu tensor when it also
            // feeds a residual outside the group — reserve its slot
            if (skip == 6 && node.Operator == OperatorId.Conv
                && nodes[ni + 1].Operator == OperatorId.Add
                && nodes[ni + 5].Operator == OperatorId.Mul)
            {
                // preG has exactly 2 in-group consumers (Div ni+2, Mul ni+5);
                // anything beyond that is an external reader needing a slot
                int preG = checked((int)nodes[ni + 1].Outputs[0]);
                if (refCount[preG] > 2) used[preG] = true;
            }
            ni += skip;
        }
        foreach (uint g in _model.GraphOutputs) used[checked((int)g)] = true;

        // graph input gets physically padded to 4 channels when Cin%4!=0 so the
        // input conv's im2col can use the vector path; weights pack zeros there.
        int inCinPad = shapes[inIdx][1] % 4 != 0 ? 4 : 0;

        var off = new long[nT];
        long cursor = 0;
        long maxIm2col = 0;   // shared scratch: max M*K_pad over dense convs
        for (int i = 0; i < nT; i++)
        {
            off[i] = -1;
            if (!used[i] || isConst[i] || alias[i] != i) continue;
            off[i] = cursor;
            // pad every slab to a 128-row tile multiple: coopmat reads pad rows;
            // slabs are 4-element aligned so packed-uint (fp16 pair) reads stay aligned
            long sz = numel[i];
            if (i == inIdx && inCinPad != 0)
                sz = numel[i] / shapes[i][1] * inCinPad;
            cursor += (sz + 3) / 4 * 4 + 128 * 64;
        }
        for (int ni = 0; ni < nodes.Length; ni++)
        {
            NodeRecord nd = nodes[ni];
            if (nd.Operator != OperatorId.Conv) continue;
            ReadOnlySpan<byte> pp = _model.GetParameters(nd);
            int grp = Math.Max(1, checked((int)U32(pp, 4)));
            int kh = I32(pp, 8), kw = I32(pp, 12);
            int[] os = shapes[nd.Outputs[0]], iss = shapes[nd.Inputs[0]];
            if (grp == 1 && !(kh == 1 && kw == 1 && I32(pp, 16) == 1 && I32(pp, 20) == 1))
            {
                long m = (long)os[2] * os[3];
                int cn = iss[1];
                if (Phys(checked((int)nd.Inputs[0])) == inIdx && inCinPad != 0) cn = inCinPad;
                long kp = ((long)cn * kh * kw + 15) / 16 * 16;
                maxIm2col = Math.Max(maxIm2col, m * kp);
            }
        }
        long im2colOff = (cursor + 63) / 64 * 64;
        // +128*512: coopmat A-tile reads pad rows past M
        cursor = im2colOff + maxIm2col + 128 * 512;
        long arenaElems = cursor + (1L << 20);
        VkBuffer arena = _dev.NewStorageBuffer((ulong)arenaElems * 2, hostVisible: false);
        _allBufs.Add(arena);
        VkBuffer outF32 = _dev.NewStorageBuffer((ulong)numel[outIdx] * 4,
            hostVisible: true, preferHost: true);
        _allBufs.Add(outF32);
        VkBuffer inF32 = _dev.NewStorageBuffer((ulong)numel[inIdx] * 4, hostVisible: true);
        _allBufs.Add(inF32);
        bool inF32Consumed = false;  // stem conv reads fp32 NCHW directly
        bool convTOut = false;   // last convT wrote fp32 out directly


        // ---- pre-emit fusion detection ----
        // Mul(x, seChannelVec) (the SE gate) gets skipped when EVERY consumer can
        // fold the gating: a pointwise Conv applies it as input-prescale, an Add
        // applies it as scaled-residual-add (addps). Any unhandled consumer keeps
        // the Mul emitted.
        var prescale = new Dictionary<int, (uint x, uint se)>();   // convNi -> tensors
        var addScale = new Dictionary<int, (uint x, uint se, uint oth)>();  // addNi -> tensors
        var skipEmit = new HashSet<int>();
        for (int ni = 0; ni < nodes.Length; ni++)
        {
            NodeRecord nd = nodes[ni];
            if (nd.Operator != OperatorId.Mul || _compiled.FusedSkip(ni) != 0 || inConvGroup[ni])
                continue;
            uint a = nd.Inputs[0], b = nd.Inputs[1];
            if (a == uint.MaxValue || b == uint.MaxValue
                || isConst[(int)a] || isConst[(int)b]) continue;
            // se = input whose numel equals the other's channel count
            // (per-image: nb*C for a batch — se tensors are [n,C,1,1])
            long na = numel[(int)a], nelb = numel[(int)b];
            int[] sa = shapes[(int)a], sb = shapes[(int)b];
            uint xT, seT;
            if (na == (sa.Length >= 2 ? (long)sb[1] * nb : -1) && nelb > na) { xT = b; seT = a; }
            else if (nelb == (sb.Length >= 2 ? (long)sa[1] * nb : -1) && na > nelb) { xT = a; seT = b; }
            else continue;
            if (numel[(int)seT] % 4 != 0) continue;

            var pendPrescale = new List<int>();
            var pendAdd = new List<(int c, uint oth)>();
            bool ok = true;
            var stk = new Stack<uint>();
            var seen = new HashSet<uint>();
            stk.Push(nd.Outputs[0]);
            while (stk.Count > 0 && ok)
            {
                uint t = stk.Pop();
                if (!seen.Add(t)) continue;
                List<int>? cs = consumers[(int)t];
                if (cs == null) { ok = false; break; }
                foreach (int c in cs)
                {
                    NodeRecord cn = nodes[c];
                    if (cn.Operator is OperatorId.LayoutConvert or OperatorId.Squeeze
                        or OperatorId.Unsqueeze or OperatorId.Reshape)
                    { stk.Push(cn.Outputs[0]); continue; }
                    if (cn.Operator == OperatorId.Conv && !inConvGroup[c]
                        && Phys((int)cn.Inputs[0]) == Phys((int)nd.Outputs[0]))
                    {
                        ReadOnlySpan<byte> cp = _model.GetParameters(cn);
                        int ccin = shapes[(int)xT][1];
                        int ccout = shapes[(int)cn.Outputs[0]][1];
                        if (U32(cp, 4) == 1 && I32(cp, 8) == 1 && I32(cp, 12) == 1
                            && I32(cp, 16) == 1 && I32(cp, 20) == 1
                            && ccin % 4 == 0 && ccout % 4 == 0 && ccin <= 256
                            && Environment.GetEnvironmentVariable("SIMD_OCR_NOPRESCALE") == null)
                            pendPrescale.Add(c);
                        else ok = false;
                        continue;
                    }
                    if (cn.Operator == OperatorId.Add && !inConvGroup[c]
                        && _compiled.FusedSkip(c) == 0
                        && Environment.GetEnvironmentVariable("SIMD_OCR_NOADDPS") == null)
                    {
                        uint oth = cn.Inputs[0] == t ? cn.Inputs[1]
                                 : cn.Inputs[1] == t ? cn.Inputs[0] : uint.MaxValue;
                        if (oth != uint.MaxValue && !isConst[(int)oth]
                            && numel[(int)oth] == numel[(int)xT]
                            && numel[(int)xT] % 4 == 0)
                        { pendAdd.Add((c, oth)); continue; }
                        ok = false;
                        continue;
                    }
                    ok = false;
                }
            }
            if (!ok)
            {
                foreach (int c in pendPrescale) prescale.Remove(c);
                continue;
            }
            foreach (int c in pendPrescale) prescale[c] = (xT, seT);
            foreach ((int c, uint oth) in pendAdd) addScale[c] = (xT, seT, oth);
            if (Environment.GetEnvironmentVariable("SIMD_OCR_PS_KEEP") != null)
                continue;   // debug: folds registered but mul still emitted
            skipEmit.Add(ni);
        }

        // SE chain fusion: ReduceMean -> Conv1x1(fc1,+bias) -> Conv1x1(fc2,+bias)
        // -> HardSigmoid emits as a single `se` dispatch writing the gate vector
        // directly to the HardSigmoid output slot.
        var seEmit = new Dictionary<int, (int fc1, int fc2, int hs)>();
        var concatAbsorbed = new HashSet<int>();   // phys tensors written directly into a concat slice
        int seCtr = 0;   // fused-SE ticket slot index (counters live at partBuf+32KB, away from partials)
        if (Environment.GetEnvironmentVariable("SIMD_OCR_NOSE") == null)
        {
            bool IsPwConv(int ni)
            {
                if (nodes[ni].Operator != OperatorId.Conv || _compiled.FusedSkip(ni) != 0
                    || inConvGroup[ni]) return false;
                ReadOnlySpan<byte> cp = _model.GetParameters(nodes[ni]);
                return U32(cp, 4) == 1 && I32(cp, 8) == 1 && I32(cp, 12) == 1
                    && I32(cp, 16) == 1 && I32(cp, 20) == 1
                    && nodes[ni].Inputs.Length > 2 && nodes[ni].Inputs[2] != uint.MaxValue;
            }
            bool dbg = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_DUMP") == "1";
            for (int hs = 0; hs < nodes.Length; hs++)
            {
                if (nodes[hs].Operator != OperatorId.HardSigmoid
                    || _compiled.FusedSkip(hs) != 0 || inConvGroup[hs]) continue;
                int fc2 = prodNode[Phys(checked((int)nodes[hs].Inputs[0]))];
                if (dbg)
                    Console.Error.WriteLine($"  se? hs=n{hs} fc2=n{fc2} fc2pw={(fc2 >= 0 && IsPwConv(fc2))} rc2={(fc2 >= 0 ? refCount[Phys(checked((int)nodes[fc2].Outputs[0]))] : -1)}");
                if (fc2 < 0 || !IsPwConv(fc2)
                    || refCount[Phys(checked((int)nodes[fc2].Outputs[0]))] != 1) continue;
                int fc1 = prodNode[Phys(checked((int)nodes[fc2].Inputs[0]))];
                int reluNi = -1;
                if (fc1 >= 0 && nodes[fc1].Operator == OperatorId.Relu
                    && refCount[Phys(checked((int)nodes[fc1].Outputs[0]))] == 1)
                { reluNi = fc1; fc1 = prodNode[Phys(checked((int)nodes[fc1].Inputs[0]))]; }
                // fc1 may be a Conv+Relu compiler group (skip=1 covering reluNi)
                bool fc1ok = fc1 >= 0 && reluNi >= 0
                    && nodes[fc1].Operator == OperatorId.Conv && !inConvGroup[fc1]
                    && _compiled.FusedSkip(fc1) == 1 && fc1 + 1 == reluNi
                    && nodes[fc1].Inputs.Length > 2 && nodes[fc1].Inputs[2] != uint.MaxValue;
                if (fc1ok)
                {
                    ReadOnlySpan<byte> cp = _model.GetParameters(nodes[fc1]);
                    fc1ok = U32(cp, 4) == 1 && I32(cp, 8) == 1 && I32(cp, 12) == 1
                        && I32(cp, 16) == 1 && I32(cp, 20) == 1;
                }
                if (dbg)
                    Console.Error.WriteLine($"    fc1=n{fc1} relu=n{reluNi} fc1ok={fc1ok}");
                if (!fc1ok
                    || refCount[Phys(checked((int)nodes[fc1].Outputs[0]))] != 1) continue;
                int rm = prodNode[Phys(checked((int)nodes[fc1].Inputs[0]))];
                if (dbg)
                    Console.Error.WriteLine($"    rm=n{rm} op={(rm >= 0 ? nodes[rm].Operator.ToString() : "?")} rcIn={(rm >= 0 ? refCount[Phys(checked((int)nodes[fc1].Inputs[0]))] : -1)}");
                if (rm < 0 || nodes[rm].Operator != OperatorId.ReduceMean
                    || _compiled.FusedSkip(rm) != 0 || inConvGroup[rm]
                    || refCount[Phys(checked((int)nodes[fc1].Inputs[0]))] != 1) continue;
                int[] xs = shapes[Phys(checked((int)nodes[rm].Inputs[0]))];
                int[] rs = shapes[Phys(checked((int)nodes[fc1].Outputs[0]))];
                int rDim = rs.Length >= 2 ? rs[1] : 0;
                if (dbg)
                    Console.Error.WriteLine($"    xs=[{string.Join('x', xs)}] rDim={rDim}");
                if (xs.Length != 4 || xs[1] % 4 != 0 || xs[1] > 256
                    || rDim <= 0 || rDim > 64) continue;
                seEmit[rm] = (fc1, fc2, hs);
                for (int m = fc1; m <= fc1 + _compiled.FusedSkip(fc1); m++)
                    skipEmit.Add(m);   // conv16 + its fused relu
                skipEmit.Add(fc2); skipEmit.Add(hs);
            }
        }
        if (Environment.GetEnvironmentVariable("SIMD_OCR_GPU_DUMP") == "1")
        {
            Console.Error.WriteLine($"prescale convs={prescale.Count} addps={addScale.Count} seEmit={seEmit.Count} skipEmit={skipEmit.Count}");
            foreach (int rm in seEmit.Keys.OrderBy(k => k))
                Console.Error.WriteLine($"  se chain rm=n{rm} fc1=n{seEmit[rm].fc1} fc2=n{seEmit[rm].fc2} hs=n{seEmit[rm].hs}");
            foreach (var kv in addScale)
            {
                int px = prodNode[(int)kv.Value.x], ps = prodNode[(int)kv.Value.se], po = prodNode[(int)kv.Value.oth];
                Console.Error.WriteLine($"  addps n{kv.Key}: x=t{kv.Value.x}[{string.Join('x', shapes[(int)kv.Value.x])}] prod={(px < 0 ? "?" : nodes[px].Operator.ToString() + px)} " +
                    $"se=t{kv.Value.se}[{string.Join('x', shapes[(int)kv.Value.se])}] prod={(ps < 0 ? "?" : nodes[ps].Operator.ToString() + ps)} " +
                    $"oth=t{kv.Value.oth} prod={(po < 0 ? "?" : nodes[po].Operator.ToString() + po)}");
            }
            foreach (var kv in prescale)
                Console.Error.WriteLine($"  prescale conv n{kv.Key}: x=t{kv.Value.x} se=t{kv.Value.se}");
        }

        // ---- pass 2: emit ----
        int LastEmitNi = -1;
        List<Rec> recs = new();
        Rec Emit(VkPipeline pipe, string tag, (VkBuffer buf, long elemOff, int elemSize)[] binds,
                 uint[] pc, uint gx, uint gy = 1)
        {
            IntPtr set = _dev.NewDescriptorSet(pipe.SetLayout);
            for (int b = 0; b < binds.Length; b++)
                _dev.BindBuffer(set, (uint)b, binds[b].buf,
                    (ulong)(binds[b].elemOff * binds[b].elemSize));
            Rec r = new() { Pipe = pipe, Set = set, Pc = Pcu(pc), Gx = gx, Gy = gy, Tag = tag };
            recs.Add(r);
            return r;
        }
        uint Div256(long n) => (uint)((n + 255) / 256);
        // coopmat tile: cout<=32 -> 512x32, cout<=64 -> 256x64, else 128x128
        // sg32 variant only ships the 128x128 tile — always use it there.
        (VkPipeline pipe, uint tm, uint tn) CmTile(int c) =>
            _sg32 ? (_pConv1x1, 128u, 128u)
            : c <= 32 ? (_pConv1x1N32, 512u, 32u)
            : c <= 64 ? (_pConv1x1N64, 256u, 64u)
            : (_pConv1x1, 128u, 128u);
        (VkBuffer, long, uint) Operand(int t, int chan, long n)
        {
            long nel = numel[t];
            uint mode = nel == n ? 0u : nel == chan ? 1u : 2u;
            return isConst[t] ? (ConstF16(t), 0, mode) : (arena, off[Phys(t)], mode);
        }

        // ---- addps absorb: the addps rec disappears; its sole consumer reads
        // a + x*se inline. A consumer qualifies only when its emit path is one
        // of the kernels with an absorb flag (dw4t / 1x1-dot / resize4 /
        // resize4add / concat4) AND the gates below exactly match emit — an
        // absorbed tensor read by a non-absorbing kernel would be garbage.
        var addpsSrc = new Dictionary<int, (uint a, uint x, uint se)>();
        if (Environment.GetEnvironmentVariable("SIMD_OCR_NOAPS") == null)
        foreach ((int addNi, (uint x, uint se, uint oth) asc) in addScale)
        {
            int outP = Phys(checked((int)nodes[addNi].Outputs[0]));
            if (refCount[outP] != 1) continue;
            List<int>? cs = consumers[outP];
            if (cs == null || cs.Count != 1 || Absorbable(cs[0], outP) == false)
                continue;
            addpsSrc[outP] = (asc.oth, asc.x, asc.se);
        }

        bool Absorbable(int cn, int srcPhys)
        {
            NodeRecord c = nodes[cn];
            switch (c.Operator)
            {
                case OperatorId.Conv:
                {
                    if (Phys(checked((int)c.Inputs[0])) != srcPhys) return false;
                    ReadOnlySpan<byte> cp = _model.GetParameters(c);
                    int[] ish2 = shapes[checked((int)c.Inputs[0])];
                    int[] osh2 = shapes[checked((int)c.Outputs[0])];
                    int cin2 = ish2[1], cout2 = osh2[1];
                    int kH2 = I32(cp, 8), kW2 = I32(cp, 12),
                        sH2 = I32(cp, 16), sW2 = I32(cp, 20);
                    int grp2 = Math.Max(1, checked((int)U32(cp, 4)));
                    if (grp2 == cin2 && cout2 == cin2)
                        return cin2 % 4 == 0 && kH2 <= 5 && kW2 <= 5
                            && sH2 <= 2 && sW2 <= 2;              // conv_dw4t
                    if (grp2 == 1 && kH2 == 1 && kW2 == 1 && sH2 == 1 && sW2 == 1)
                    {
                        int cinI2 = (srcPhys == inIdx && cin2 % 4 != 0) ? 4 : cin2;
                        if (cinI2 % 4 != 0 || cout2 % 4 != 0 || cinI2 > 128)
                            return false;
                        int cskip = _compiled.FusedSkip(cn);
                        if (cskip == 1 && nodes[cn + 1].Operator == OperatorId.Add)
                        {
                            NodeRecord ad2 = nodes[cn + 1];
                            int bt = ad2.Inputs[0] == c.Outputs[0]
                                ? checked((int)ad2.Inputs[1])
                                : checked((int)ad2.Inputs[0]);
                            if (numel[bt] == 1) return false;   // scalarBias
                        }
                        return !prescale.ContainsKey(cn)
                            && Environment.GetEnvironmentVariable(
                                "SIMD_OCR_NODOT") == null;
                    }
                    return false;
                }
                case OperatorId.Resize:
                    return Phys(checked((int)c.Inputs[0])) == srcPhys
                        && shapes[checked((int)c.Outputs[0])][1] % 4 == 0;
                case OperatorId.Concat:
                {
                    int[] ospc = shapes[checked((int)c.Outputs[0])];
                    if (ospc[1] % 4 != 0) return false;
                    int cOff2 = 0;
                    foreach (uint inp2 in c.Inputs)
                    {
                        int ci2 = shapes[checked((int)inp2)][1];
                        if (Phys(checked((int)inp2)) == srcPhys)
                            return ci2 % 4 == 0 && cOff2 % 4 == 0;
                        cOff2 += ci2;
                    }
                    return false;
                }
                case OperatorId.Add:
                {
                    // consumer Add is the tail of a resize4add fusion: the
                    // other input must come from a Resize that will itself
                    // hit the fused path (sole-consumer, %4, same numel).
                    if (addScale.ContainsKey(cn)) return false;
                    int i0 = Phys(checked((int)c.Inputs[0]));
                    int i1 = Phys(checked((int)c.Inputs[1]));
                    int othP2 = i0 == srcPhys ? i1 : i1 == srcPhys ? i0 : -1;
                    if (othP2 < 0) return false;
                    int prod = prodNode[othP2];
                    if (prod < 0 || nodes[prod].Operator != OperatorId.Resize)
                        return false;
                    int[] osp3 = shapes[checked((int)nodes[prod].Outputs[0])];
                    if (osp3[1] % 4 != 0) return false;
                    if (refCount[othP2] != 1 || consumers[othP2]?.Count != 1
                        || consumers[othP2][0] != cn) return false;
                    return numel[srcPhys] == numel[othP2];
                }
                default: return false;
            }
        }

        // ---- catresize fusion: a Concat whose inputs are sole-consumed
        // Resize nodes (or plain tensors) collapses to ONE dispatch that
        // resamples each source straight into its output slice. The resizes'
        // own recs are skipped (they were the writers, now the kernel writes).
        var catRes = new Dictionary<int,
            (long x, long ra, long se, uint fl, uint inW, uint fh, uint fw,
             uint off4, uint cq4)[]>();
        for (int ci2 = 0; ci2 < nodes.Length; ci2++)
        {
            NodeRecord c = nodes[ci2];
            if (c.Operator != OperatorId.Concat || c.Inputs.Length > 4)
                continue;
            int[] osp = shapes[checked((int)c.Outputs[0])];
            if (osp[1] % 4 != 0) continue;
            var spec = new (long, long, long, uint, uint, uint, uint, uint, uint)
                [c.Inputs.Length];
            int cOff3 = 0; bool okc = true; var skipTmp = new List<int>();
            for (int ii = 0; ii < c.Inputs.Length; ii++)
            {
                uint inp2 = c.Inputs[ii];
                int ip = Phys(checked((int)inp2));
                int ci3 = shapes[checked((int)inp2)][1];
                if (ci3 % 4 != 0 || cOff3 % 4 != 0) { okc = false; break; }
                uint fh3 = 1, fw3 = 1, inW3 = (uint)osp[3];
                int srcP = ip;
                int prod = prodNode[ip];
                if (prod >= 0 && nodes[prod].Operator == OperatorId.Resize
                    && refCount[ip] == 1 && consumers[ip]?.Count == 1
                    && consumers[ip][0] == ci2)
                {
                    int[] rish = shapes[checked((int)nodes[prod].Inputs[0])];
                    if (rish[1] != ci3) { okc = false; break; }
                    fh3 = (uint)(osp[2] / rish[2]);
                    fw3 = (uint)(osp[3] / rish[3]);
                    if (rish[2] * (int)fh3 != osp[2] || rish[3] * (int)fw3 != osp[3])
                    { okc = false; break; }
                    inW3 = (uint)rish[3];
                    srcP = Phys(checked((int)nodes[prod].Inputs[0]));
                    skipTmp.Add(prod);
                }
                else if (refCount[ip] == 1 && consumers[ip]?.Count == 1
                    && consumers[ip][0] == ci2 && prod >= 0
                    && !addpsSrc.ContainsKey(ip))
                {
                    // sole-consumed producer emits a slice-write into this
                    // concat instead of materializing its own tensor —
                    // catRes cannot read it back; keep the old path.
                    okc = false; break;
                }
                uint fl = 0; long xP = off[srcP], raP = 0, seP = 0;
                if (addpsSrc.TryGetValue(srcP, out var ads))
                {
                    xP = off[Phys(checked((int)ads.x))];
                    raP = off[Phys(checked((int)ads.a))];
                    seP = off[Phys(checked((int)ads.se))];
                    fl = 1;
                }
                spec[ii] = (xP, raP, seP, fl, inW3, fh3, fw3,
                            (uint)(cOff3 / 4), (uint)(ci3 / 4));
                cOff3 += ci3;
            }
            if (!okc) continue;
            catRes[ci2] = spec;
            foreach (int p2 in skipTmp) skipEmit.Add(p2);
        }

        for (int ni = 0; ni < emitLimit; ni++)
        {
            NodeRecord node = nodes[ni];
            if (skipEmit.Contains(ni)) continue;
            int skip = _compiled.FusedSkip(ni);
            ReadOnlySpan<byte> p = _model.GetParameters(node);
            int outPhys = Phys(checked((int)nodes[ni + skip].Outputs[0]));
            LastEmitNi = ni;
            long SlotOf(uint t) => off[Phys(checked((int)t))];

            switch (node.Operator)
            {
                case OperatorId.LayoutConvert:
                case OperatorId.Squeeze:
                case OperatorId.Unsqueeze:
                case OperatorId.Reshape:
                    break;

                case OperatorId.Conv:
                {
                    int[] ishp = shapes[node.Inputs[0]];
                    int[] oshp = shapes[node.Outputs[0]];
                    int cin = ishp[1], cout = oshp[1];
                    int inH = ishp[2], inW = ishp[3], outH = oshp[2], outW = oshp[3];
                    int kH = I32(p, 8), kW = I32(p, 12), sH = I32(p, 16), sW = I32(p, 20);
                    int pt = I32(p, 32), pl = I32(p, 36);
                    int group = Math.Max(1, checked((int)U32(p, 4)));

                    int act = 0;
                    VkBuffer biasBuf = arena;
                    uint resT = uint.MaxValue, dualT = uint.MaxValue;
                    uint hasBias = 0, scalarBias = 0, flags16 = 0;
                    bool dualRaw = false;
                    if (node.Inputs.Length > 2 && node.Inputs[2] != uint.MaxValue)
                    { biasBuf = ConstF16(checked((int)node.Inputs[2])); hasBias = 1; }
                    if (skip == 1 && nodes[ni + 1].Operator == OperatorId.Relu)
                        act = 1;
                    else if (skip == 1 && nodes[ni + 1].Operator == OperatorId.Add)
                    {
                        // conv (no bias) + channel-bias add
                        NodeRecord add = nodes[ni + 1];
                        int bt = add.Inputs[0] == node.Outputs[0]
                            ? checked((int)add.Inputs[1]) : checked((int)add.Inputs[0]);
                        biasBuf = ConstF16(bt); hasBias = 1;
                        if (numel[bt] == 1) scalarBias = 8u;
                    }
                    else if (skip == 2 && nodes[ni + 1].Operator == OperatorId.HardSigmoid
                             && nodes[ni + 2].Operator == OperatorId.Mul)
                    {
                        act = 3;
                        ReadOnlySpan<byte> hp = _model.GetParameters(nodes[ni + 1]);
                        float hsA = F32(hp, 4), hsB = F32(hp, 8);
                        if (hsB != 0.5f)
                            throw new NotSupportedException($"hardsigmoid beta={hsB} at node {ni + 1}");
                        flags16 |= (uint)Half2Bits(hsA) << 16;
                    }
                    else if (skip == 6
                             && nodes[ni + 1].Operator == OperatorId.Add
                             && nodes[ni + 2].Operator == OperatorId.Div
                             && nodes[ni + 3].Operator == OperatorId.Erf
                             && nodes[ni + 4].Operator == OperatorId.Add
                             && nodes[ni + 5].Operator == OperatorId.Mul
                             && nodes[ni + 6].Operator == OperatorId.Mul
                             && Phys(checked((int)nodes[ni + 2].Inputs[0]))
                                == Phys(checked((int)nodes[ni + 1].Outputs[0]))
                             && (Phys(checked((int)nodes[ni + 5].Inputs[0]))
                                 == Phys(checked((int)nodes[ni + 1].Outputs[0]))
                              || Phys(checked((int)nodes[ni + 5].Inputs[1]))
                                 == Phys(checked((int)nodes[ni + 1].Outputs[0]))))
                    {
                        // compiler-fused conv + bias-Add + gelu chain:
                        // out = 0.5*x*(1+erf(x/√2)) where x = conv + bias.
                        NodeRecord add = nodes[ni + 1];
                        int bt = add.Inputs[0] == node.Outputs[0]
                            ? checked((int)add.Inputs[1])
                            : checked((int)add.Inputs[0]);
                        biasBuf = ConstF16(bt); hasBias = 1;
                        if (numel[bt] == 1) scalarBias = 8u;
                        act = 2;
                        int preG = Phys(checked((int)nodes[ni + 1].Outputs[0]));
                        if (refCount[preG] > 2)
                        {
                            // conv+bias also feeds a residual outside the
                            // group: sink slot gets gelu(acc) via o; the preG
                            // slot gets raw acc via o2 (flag bit10).
                            dualT = (uint)preG; dualRaw = true;
                        }
                    }
                    else if (skip == 2
                             && nodes[ni + 1].Operator == OperatorId.Add
                             && nodes[ni + 2].Operator == OperatorId.Add)
                    {
                        // conv + channel-bias Add + residual Add
                        NodeRecord add = nodes[ni + 1];
                        int bt = add.Inputs[0] == node.Outputs[0]
                            ? checked((int)add.Inputs[1])
                            : checked((int)add.Inputs[0]);
                        if (!isConst[bt])
                            throw new NotSupportedException(
                                $"conv+add+add mid-add non-const at node {ni + 1}");
                        biasBuf = ConstF16(bt); hasBias = 1;
                        if (numel[bt] == 1) scalarBias = 8u;
                        NodeRecord res = nodes[ni + 2];
                        int mid = Phys(checked((int)add.Outputs[0]));
                        uint oth = Phys(checked((int)res.Inputs[0])) == mid
                            ? res.Inputs[1]
                            : Phys(checked((int)res.Inputs[1])) == mid
                                ? res.Inputs[0] : uint.MaxValue;
                        if (oth == uint.MaxValue || isConst[(int)oth]
                            || numel[(int)oth] != numel[mid])
                            throw new NotSupportedException(
                                $"conv+add+add residual form at node {ni + 2}");
                        resT = oth;
                    }
                    else if (skip != 0)
                        throw new NotSupportedException(
                            $"conv fused skip={skip} at node {ni} (next={nodes[ni + 1].Operator})");

                    // ---- emit-time fusion extensions (compiler didn't group these) ----
                    // pointwise conv + residual Add (+ optional gelu pattern) —
                    // cm/dot kernels implement res binding (flags bit1) and act (bits4-6).
                    if (group == 1 && kH == 1 && kW == 1 && sH == 1 && sW == 1
                        && Environment.GetEnvironmentVariable("SIMD_OCR_NOEXTEND") == null)
                    {
                        int ge = ni + skip;
                        int j = ge + 1;
                        if (j < nodes.Length && nodes[j].Operator == OperatorId.Add
                            && !inConvGroup[j] && _compiled.FusedSkip(j) == 0
                            && !addScale.ContainsKey(j))
                        {
                            NodeRecord ad = nodes[j];
                            int prev = Phys(checked((int)nodes[ge].Outputs[0]));
                            int ia = Phys(checked((int)ad.Inputs[0]));
                            int ib = Phys(checked((int)ad.Inputs[1]));
                            uint other = ia == prev ? ad.Inputs[1]
                                       : ib == prev ? ad.Inputs[0] : uint.MaxValue;
                            if (other != uint.MaxValue && !isConst[(int)other]
                                && numel[(int)other] == numel[prev]
                                && refCount[prev] == 1)
                            { resT = other; ge = j; j++; }
                        }
                        if (j < nodes.Length
                            && nodes[j].Operator == OperatorId.Div
                            && _compiled.FusedSkip(j) == 4
                            && Phys(checked((int)nodes[j].Inputs[0]))
                                == Phys(checked((int)nodes[ge].Outputs[0])))
                        {
                            int preG = Phys(checked((int)nodes[ge].Outputs[0]));
                            if (refCount[preG] == 1 && act == 0)
                            { act = 2; ge = j + 4; }
                            else if (refCount[preG] == 2)
                            {
                                // out also feeds a residual elsewhere — conv dual-
                                // writes plain out (preG) and gelu(out) (gelu sink)
                                dualT = (uint)Phys(checked((int)nodes[j + 4].Outputs[0]));
                                outPhys = preG;
                                ge = j + 4;
                            }
                        }
                        if (ge != ni + skip)
                        {
                            if (dualT == uint.MaxValue)
                                outPhys = Phys(checked((int)nodes[ge].Outputs[0]));
                            skip = ge - ni;
                        }
                    }
                    bool hasPs = prescale.TryGetValue(ni, out (uint x, uint se) psv);
                    uint flags = hasBias | scalarBias | (uint)(act << 4) | flags16
                        | (resT != uint.MaxValue ? 2u : 0u)
                        | (hasPs ? 4u : 0u)
                        | (dualT != uint.MaxValue ? 8u : 0u)
                        | (dualRaw ? 1024u : 0u);

                    if (kH == 1 && kW == 1 && sH == 1 && sW == 1 && group == 1)
                    {
                        int inT = Phys(checked((int)node.Inputs[0]));
                        int cinIn = (inT == inIdx && cin % 4 != 0) ? 4 : cin;
                        if (cinIn % 4 != 0)
                            throw new NotSupportedException($"conv1x1 cin={cin} not %4 at node {ni}");
                        VkBuffer wbuf = ConstF16(checked((int)node.Inputs[1]),
                            padRows: ((cout + 127) / 128) * 128, rowElems: cin,
                            rowPad: cinIn);
                        uint mImg = (uint)(outH * outW);
                        uint M = mImg * (uint)nb;   // flat rows span the batch
                        bool dotOk = cout % 4 == 0 && scalarBias == 0
                            && (cinIn <= 128 || (hasPs && cinIn <= 256))
                            && Environment.GetEnvironmentVariable("SIMD_OCR_NODOT") == null;
                        if (dotOk)
                        {
                            // small-K pointwise conv: direct dot kernel beats coopmat.
                            // prescale: read x (not the mul output) + per-channel se scale.
                            long aOff = hasPs ? SlotOf(psv.x) : SlotOf(node.Inputs[0]);
                            bool aps = addpsSrc.TryGetValue(
                                Phys(checked((int)node.Inputs[0])), out var ads)
                                && !hasPs;
                            if (aps) aOff = SlotOf(ads.x);
                            VkBuffer wk = ConstKMajor(checked((int)node.Inputs[1]),
                                cout, cin, 1, cinIn, cinIn);
                            // se tensors are per-batch [n, C] — mImg lets the
                            // kernel pick the batch's scale vector; 0 = batch-free.
                            uint mIm = hasPs || aps ? mImg : 0u;
                            Emit(_pDot, $"conv1x1d n{ni} {M}x{cinIn}x{cout}",
                                [(arena, aOff, 2), (wk, 0, 2),
                                 (biasBuf, 0, 2),
                                 (arena, resT != uint.MaxValue ? SlotOf(resT) : 0, 2),
                                 (arena, off[outPhys], 2),
                                 (arena, hasPs ? SlotOf(psv.se)
                                              : aps ? SlotOf(ads.se) : 0, 2),
                                 (arena, dualT != uint.MaxValue ? SlotOf(dualT) : 0, 2),
                                 (arena, aps ? SlotOf(ads.a) : 0, 2)],
                                [M, (uint)cout, (uint)cinIn, (uint)(cinIn / 4),
                                 flags | (aps ? 512u : 0u), (uint)(cinIn / 4), mIm],
                                Div256(((M + 3) / 4) * (long)(cout / 4)));
                        }
                        else
                        {
                            var (cp, tm, tn) = CmTile(cout);
                            Emit(cp, $"conv1x1 n{ni} {M}x{cin}x{cout}",
                                [(arena, SlotOf(node.Inputs[0]), 2), (wbuf, 0, 2),
                                 (biasBuf, 0, 2),
                                 (arena, resT != uint.MaxValue ? SlotOf(resT) : 0, 2),
                                 (arena, off[outPhys], 2),
                                 (arena, dualT != uint.MaxValue ? SlotOf(dualT) : 0, 2)],
                                [M, (uint)cout, (uint)cinIn, flags],
                                (M + tm - 1) / tm, (uint)((cout + tn - 1) / tn));
                        }
                    }
                    else if (group == cin && cout == cin)
                    {
                        if (cin % 4 == 0)
                        {
                            if (kH <= 5 && kW <= 5 && sH <= 2 && sW <= 2)
                            {
                                // shared-tile variant: ~kH*kW less global traffic
                                uint tx = (uint)((outW + 15) / 16),
                                     ty = (uint)((outH + 15) / 16);
                                bool aps = addpsSrc.TryGetValue(
                                    Phys(checked((int)node.Inputs[0])), out var ads);
                                Emit(_pDw4T, $"conv_dw4t n{ni} {cout}ch k{kH}s{sH}",
                                    [(arena, aps ? SlotOf(ads.x)
                                                : SlotOf(node.Inputs[0]), 2),
                                     (ConstDwTap(checked((int)node.Inputs[1]), cout, kH * kW), 0, 2),
                                     (biasBuf, 0, 2), (arena, off[outPhys], 2),
                                     (arena, aps ? SlotOf(ads.a) : 0, 2),
                                     (arena, aps ? SlotOf(ads.se) : 0, 2)],
                                    [(uint)outW, (uint)outH, (uint)cout, (uint)kH, (uint)kW,
                                     (uint)sH, (uint)sW, (uint)pt, (uint)pl,
                                     (uint)inW, (uint)inH,
                                     flags | (aps ? 256u : 0u)],
                                    tx * ty * (uint)(cout / 4), (uint)nb);
                            }
                            else
                            Emit(_pDw4, $"conv_dw4 n{ni} {cout}ch k{kH}s{sH}",
                                [(arena, SlotOf(node.Inputs[0]), 2),
                                 (ConstDwTap(checked((int)node.Inputs[1]), cout, kH * kW), 0, 2),
                                 (biasBuf, 0, 2), (arena, off[outPhys], 2)],
                                [(uint)outW, (uint)outH, (uint)cout, (uint)kH, (uint)kW,
                                 (uint)sH, (uint)sW, (uint)pt, (uint)pl,
                                 (uint)inW, (uint)inH, flags],
                                Div256((long)outW * outH * (cout / 4)),
                                (uint)nb);
                        }
                        else
                        {
                            if (nb > 1)
                                throw new NotSupportedException(
                                    $"scalar dw conv lacks batch support (n{ni})");
                            Emit(_pDw, $"conv_dw n{ni} {cout}ch k{kH}s{sH}",
                                [(arena, SlotOf(node.Inputs[0]), 2),
                                 (ConstF16(checked((int)node.Inputs[1])), 0, 2),
                                 (biasBuf, 0, 2), (arena, off[outPhys], 2)],
                                [(uint)outW, (uint)outH, (uint)cout, (uint)kH, (uint)kW,
                                 (uint)sH, (uint)sW, (uint)pt, (uint)pl,
                                 (uint)inW, (uint)inH, flags],
                                Div256((long)outW * outH * cout));
                        }
                    }
                    else if (group == 1)
                    {
                        // dense kxk conv → explicit im2col (tap-major) + coopmat GEMM.
                        // graph-input channels are physically padded to 4 (cinIn).
                        int inT = Phys(checked((int)node.Inputs[0]));
                        int cinIn = (inT == inIdx && cin % 4 != 0) ? 4 : cin;
                        int K = cinIn * kH * kW, Kp = (K + 15) / 16 * 16;
                        uint M = (uint)(outH * outW);
                        // im2col-free direct conv: tap addressing inline in the
                        // dot kernel; skips materializing the K-expanded matrix.
                        if (cout % 4 == 0 && scalarBias == 0 && cinIn % 4 == 0
                            && Kp <= (Environment.GetEnvironmentVariable(
                                "SIMD_OCR_CONVD_KMAX") is string km
                                ? int.Parse(km) : 1024)
                            && Environment.GetEnvironmentVariable("SIMD_OCR_NODCONV") == null)
                        {
                            VkBuffer wk2 = ConstKMajor(checked((int)node.Inputs[1]),
                                cout, cin, kH * kW, Kp, cinIn);
                            // stem conv on the fp32 NCHW input: skip nchw2nhwc
                            // when this is the sole consumer of the graph input
                            bool f32In = inT == inIdx && cinIn == 4 && cin <= 4
                                && Environment.GetEnvironmentVariable(
                                    "SIMD_OCR_NOF32IN") == null;
                            if (f32In && (consumers[inIdx]?.Count ?? 0) == 1)
                                inF32Consumed = true;
                            // sole-consumer concat: write straight into its slice
                            long wO = off[outPhys]; uint dcv = 0, dco = 0;
                            {
                                List<int>? cs = consumers[outPhys];
                                if (refCount[outPhys] == 1 && cs != null
                                    && cs.Count == 1
                                    && nodes[cs[0]].Operator == OperatorId.Concat)
                                {
                                    int cn = cs[0];
                                    int[] ospc = shapes[nodes[cn].Outputs[0]];
                                    int co = 0; bool hit = false;
                                    foreach (uint cin_ in nodes[cn].Inputs)
                                    {
                                        if (Phys(checked((int)cin_)) == outPhys)
                                        { hit = true; break; }
                                        co += shapes[checked((int)cin_)][1];
                                    }
                                    if (hit && co % 4 == 0 && ospc[1] % 4 == 0)
                                    {
                                        wO = off[Phys(checked((int)nodes[cn].Outputs[0]))];
                                        dcv = (uint)(ospc[1] / 4);
                                        dco = (uint)(co / 4);
                                        concatAbsorbed.Add(outPhys);
                                    }
                                }
                            }
                            Emit(f32In ? _pConvDF32 : _pConvD,
                                $"convd n{ni} {M}x{K}x{cout}",
                                [(f32In ? inF32 : arena,
                                  f32In ? 0 : SlotOf(node.Inputs[0]),
                                  f32In ? 4 : 2),
                                 (wk2, 0, 2),
                                 (biasBuf, 0, 2),
                                 (arena, resT != uint.MaxValue ? SlotOf(resT) : 0, 2),
                                 (arena, wO, 2)],
                                f32In
                                ? [M, (uint)cout, (uint)outW, (uint)inW, (uint)inH,
                                   (uint)(cinIn / 4), (uint)(kH * kW), (uint)kW,
                                   (uint)sH, (uint)sW, (uint)pt, (uint)pl, flags,
                                   dcv, dco, (uint)cin, (uint)(inH * inW)]
                                : [M, (uint)cout, (uint)outW, (uint)inW, (uint)inH,
                                   (uint)(cinIn / 4), (uint)(kH * kW), (uint)kW,
                                   (uint)sH, (uint)sW, (uint)pt, (uint)pl, flags,
                                   dcv, dco],
                                Div256(((M + 3) / 4) * (long)(cout / 4)),
                                (uint)nb);
                        }
                        else
                        {
                        if (nb > 1)
                            throw new NotSupportedException(
                                $"im2col conv path lacks batch support (n{ni})");
                        Emit(_pIm2col, $"im2col n{ni} M{M} K{K}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (arena, im2colOff, 2)],
                            [(uint)outW, (uint)outH, (uint)cinIn, (uint)kH, (uint)kW,
                             (uint)sH, (uint)sW, (uint)pt, (uint)pl,
                             (uint)inW, (uint)inH, (uint)K, (uint)Kp,
                             cinIn % 4 != 0 ? 1u : 0u],
                            Div256(M * (long)(Kp / 4)));
                        VkBuffer wbuf = ConstTapMajor(checked((int)node.Inputs[1]),
                            cout, cin, kH, kW, Kp, cinPad: cinIn);
                        if (cout % 4 == 0 && scalarBias == 0 && Kp <= 128)
                        {
                            VkBuffer wk = ConstKMajor(checked((int)node.Inputs[1]),
                                cout, cin, kH * kW, Kp, cinIn);
                            Emit(_pDot, $"im2c_dot n{ni} {M}x{Kp}x{cout}",
                                [(arena, im2colOff, 2), (wk, 0, 2),
                                 (biasBuf, 0, 2), (arena, 0, 2),
                                 (arena, off[outPhys], 2), (arena, 0, 2),
                                 (arena, 0, 2)],
                                [M, (uint)cout, (uint)Kp, (uint)(Kp / 4),
                                 flags, 0u, 0u],
                                Div256(((M + 3) / 4) * (long)(cout / 4)));
                        }
                        else
                        {
                            var (cp, tm, tn) = CmTile(cout);
                            Emit(cp, $"im2colconv n{ni} {M}x{Kp}x{cout}",
                                [(arena, im2colOff, 2), (wbuf, 0, 2),
                                 (biasBuf, 0, 2), (arena, 0, 2), (arena, off[outPhys], 2)],
                                [M, (uint)cout, (uint)Kp, flags],
                                (M + tm - 1) / tm, (uint)((cout + tn - 1) / tn));
                        }
                        }
                    }
                    else
                    {
                        if (nb > 1)
                            throw new NotSupportedException(
                                $"dense conv path lacks batch support (n{ni})");
                        Emit(_pDense, $"conv3x3 n{ni} {cin}->{cout} k{kH}s{sH}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (ConstF16(checked((int)node.Inputs[1])), 0, 2),
                             (biasBuf, 0, 2), (arena, off[outPhys], 2)],
                            [(uint)outW, (uint)outH, (uint)cout, (uint)cin,
                             (uint)kH, (uint)kW, (uint)sH, (uint)sW,
                             (uint)pt, (uint)pl, (uint)inW, (uint)inH, flags],
                            Div256((long)outW * outH * cout));
                    }
                    ni += skip;
                    break;
                }

                case OperatorId.Div when skip == 4:
                {
                    // fused GELU: write to the group sink (last Mul output)
                    long n = numel[Phys(checked((int)nodes[ni + 4].Outputs[0]))];
                    Emit(n % 4 == 0 ? _pElem4 : _pElem, $"gelu n{ni}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (arena, 0, 2), (arena, off[outPhys], 2)],
                        [(uint)n, 1u, 2u, 0u, 0u], Div256(n % 4 == 0 ? n / 4 : n));
                    ni += 4;
                    break;
                }

                case OperatorId.Add or OperatorId.Mul or OperatorId.Sub or OperatorId.Div
                    or OperatorId.Pow:
                {
                    long n = numel[outPhys];
                    if (addScale.TryGetValue(ni, out (uint x, uint se, uint oth) asc))
                    {
                        if (addpsSrc.ContainsKey(outPhys)) break; // consumer absorbs
                        // a + x*se — the SE Mul got skipped, fold it here
                        // per-batch se vector: imgV = vec4 count of one image
                        long seN = numel[(int)asc.se];
                        Emit(_pAddPs, $"addps n{ni}",
                            [(arena, SlotOf(asc.oth), 2), (arena, SlotOf(asc.x), 2),
                             (arena, SlotOf(asc.se), 2), (arena, off[outPhys], 2)],
                            [(uint)n, (uint)(seN / 4 / nb),
                             (uint)(seN == n ? 0 : n / 4 / nb)], Div256(n / 4));
                        break;
                    }
                    int[] osp = shapes[checked((int)node.Outputs[0])];
                    int chan = osp.Length >= 2 ? osp[1] : (int)n;
                    (VkBuffer bA, long oA, uint mA) = Operand(checked((int)node.Inputs[0]), chan, n);
                    (VkBuffer bB, long oB, uint mB) = Operand(checked((int)node.Inputs[1]), chan, n);
                    // vector broadcast along the contiguous (last physical) dim:
                    // rank-3 tails like [n,T,C] + [C] need C as the modulus —
                    // the vector operand's own numel, not osp[1].
                    {
                        long neA = numel[checked((int)node.Inputs[0])];
                        long neB = numel[checked((int)node.Inputs[1])];
                        // per-image channel vector [n,C,1,1] (nel == chan*nb):
                        // mode 3 = kernel picks b[(i/perImg)*C + i%C] using aux;
                        // flat tail vector [C] vs [n,T,C]: mode 1, chan = neB.
                        if (mB == 2 && neB > 1 && neB < n)
                        { if (neB == (long)chan * nb) mB = 3; else { mB = 1; chan = (int)neB; } }
                        else if (mA == 2 && neA > 1 && neA < n)
                        { if (neA == (long)chan * nb) mA = 3; else { mA = 1; chan = (int)neA; } }
                    }
                    uint op2 = node.Operator switch
                    {
                        OperatorId.Add => 8u, OperatorId.Mul => 9u, OperatorId.Sub => 10u,
                        OperatorId.Div => 11u, OperatorId.Pow => 13u, _ => 8u,
                    };
                    bool v4 = n % 4 == 0 && (mA is not (1 or 3) || chan % 4 == 0)
                        && (mB is not (1 or 3) || chan % 4 == 0);
                    Emit(v4 ? _pElem4 : _pElem, $"bin{node.Operator} n{ni}",
                        [(bA, oA, 2), (bB, oB, 2), (arena, off[outPhys], 2)],
                        [(uint)n, (uint)chan, op2, mA, mB, (uint)(n / nb)], Div256(v4 ? n / 4 : n));
                    break;
                }

                case OperatorId.Relu or OperatorId.Sigmoid or OperatorId.HardSigmoid
                    or OperatorId.Erf or OperatorId.Sqrt:
                {
                    // compiler-fused HS+Mul (hardswish, x*hs(x)): emit op 14
                    // against the group's sink instead of a bare hardsigmoid.
                    bool hswish = node.Operator == OperatorId.HardSigmoid
                        && skip == 1
                        && nodes[ni + 1].Operator == OperatorId.Mul
                        && Phys(checked((int)nodes[ni + 1].Inputs[0]))
                           == Phys(checked((int)node.Outputs[0]))
                        && Phys(checked((int)nodes[ni + 1].Inputs[1]))
                           == Phys(checked((int)node.Inputs[0]));
                    if (hswish) ni += 1;
                    long n = numel[outPhys];
                    uint op2 = node.Operator switch
                    {
                        OperatorId.Relu => 1u, OperatorId.HardSigmoid => 3u,
                        OperatorId.Sigmoid => 4u, OperatorId.Erf => 5u,
                        OperatorId.Sqrt => 6u, _ => 0u,
                    };
                    if (hswish) op2 = 14u;
                    uint aux = 0u;
                    if (node.Operator == OperatorId.HardSigmoid)
                        aux = HsAux(_model.GetParameters(node));
                    Emit(n % 4 == 0 ? _pElem4 : _pElem, $"un{node.Operator} n{ni}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (arena, 0, 2), (arena, off[outPhys], 2)],
                        [(uint)n, 1u, op2, 0u, 0u, aux], Div256(n % 4 == 0 ? n / 4 : n));
                    break;
                }

                case OperatorId.ReduceMean:
                {
                    int[] ishp = shapes[node.Inputs[0]];
                    int hw = ishp[2] * ishp[3], c = ishp[1];
                    int cv4 = c / 4;
                    if (seEmit.TryGetValue(ni, out (int fc1, int fc2, int hs) sev))
                    {
                        NodeRecord f1 = nodes[sev.fc1], f2 = nodes[sev.fc2];
                        ReadOnlySpan<byte> hpp = _model.GetParameters(nodes[sev.hs]);
                        int rDim = shapes[Phys(checked((int)f1.Outputs[0]))][1];
                        uint cvP = 1; while (cvP < (uint)(c / 4)) cvP <<= 1;
                        int hsOut = Phys(checked((int)nodes[sev.hs].Outputs[0]));
                        int s = Math.Min((hw + 4095) / 4096, 16);
                        int pp = (hw + s - 1) / s;
                        VkBuffer pb2 = PartBuf(s * c * nb + 4 * nb);
                        Emit(_pSeF, $"se_f n{ni} c{c} s{s} r{rDim}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (pb2, 0, 2),
                             (pb2, 8192 + (long)seCtr, 4),
                             (ConstF32(checked((int)f1.Inputs[1])), 0, 4),
                             (ConstF32(checked((int)f1.Inputs[2])), 0, 4),
                             (ConstF32(checked((int)f2.Inputs[1])), 0, 4),
                             (ConstF32(checked((int)f2.Inputs[2])), 0, 4),
                             (arena, off[hsOut], 2)],
                            [(uint)hw, (uint)c, (uint)s, (uint)pp, cvP,
                             (uint)rDim,
                             BitConverter.SingleToUInt32Bits(F32(hpp, 4)),
                             BitConverter.SingleToUInt32Bits(F32(hpp, 8))],
                            (uint)s, (uint)nb);
                        seCtr += nb;   // one ticket counter per batch
                        break;
                    }
                    if (c % 4 == 0 && 256 % cv4 == 0 && hw >= 4096)
                    {
                        // two-phase: S pixel partitions → fp32 partials → mean
                        int s = Math.Min((hw + 4095) / 4096, 16);
                        int pp = (hw + s - 1) / s;
                        VkBuffer pb = PartBuf(s * c * nb);
                        Emit(_pReduce4, $"reduce4a n{ni} c{c}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (pb, 0, 2)],
                            [(uint)hw, (uint)c, (uint)s, (uint)pp],
                            (uint)s, (uint)nb);
                        Emit(_pReduce4b, $"reduce4b n{ni} c{c}",
                            [(pb, 0, 2), (arena, off[outPhys], 2)],
                            [(uint)s, (uint)c, (uint)hw], (uint)c, (uint)nb);
                    }
                    else
                        Emit(_pReduce, $"reduceHW n{ni} c{c}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (arena, off[outPhys], 2)],
                            [(uint)hw, (uint)c], (uint)c, (uint)nb);
                    break;
                }

                case OperatorId.MaxPool:
                {
                    if (nb > 1)
                        throw new NotSupportedException(
                            $"maxpool lacks batch support (n{ni})");
                    int[] ishp = shapes[node.Inputs[0]];
                    if (ishp[1] % 4 == 0)
                    {
                        long wO = off[outPhys]; uint dcv = 0, dco = 0;
                        {
                            int rp = outPhys;
                            List<int>? cs = consumers[rp];
                            if (refCount[rp] == 1 && cs != null && cs.Count == 1
                                && nodes[cs[0]].Operator == OperatorId.Concat)
                            {
                                int cn = cs[0];
                                int[] ospc = shapes[nodes[cn].Outputs[0]];
                                int co = 0; bool hit = false;
                                foreach (uint cin_ in nodes[cn].Inputs)
                                {
                                    if (Phys(checked((int)cin_)) == rp) { hit = true; break; }
                                    co += shapes[checked((int)cin_)][1];
                                }
                                if (hit && co % 4 == 0 && ospc[1] % 4 == 0)
                                {
                                    wO = off[Phys(checked((int)nodes[cn].Outputs[0]))];
                                    dcv = (uint)(ospc[1] / 4);
                                    dco = (uint)(co / 4);
                                    concatAbsorbed.Add(rp);
                                }
                            }
                        }
                        Emit(_pPool4, $"maxpool4 n{ni}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (arena, wO, 2)],
                            [(uint)ishp[3], (uint)ishp[2], (uint)ishp[1], dcv, dco],
                            Div256(numel[outPhys] / 4));
                    }
                    else
                        Emit(_pPool, $"maxpool n{ni}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (arena, off[outPhys], 2)],
                            [(uint)ishp[3], (uint)ishp[2], (uint)ishp[1]],
                            Div256(numel[outPhys]));
                    break;
                }

                case OperatorId.Resize:
                {
                    if (nb > 1)
                        throw new NotSupportedException(
                            $"resize lacks batch support (n{ni})");
                    int[] ishp = shapes[node.Inputs[0]];
                    int[] osp = shapes[node.Outputs[0]];
                    if (osp[1] % 4 == 0)
                    {
                        // sole-consumer fusion: resize -> Add (FPN merge) or
                        // resize -> Concat (write into the channel slice)
                        int rp = outPhys;
                        List<int>? cs = consumers[rp];
                        int cn = refCount[rp] == 1 && cs != null && cs.Count == 1
                            ? cs[0] : -1;
                        if (cn >= 0 && nodes[cn].Operator == OperatorId.Add
                            && _compiled.FusedSkip(cn) == 0 && !inConvGroup[cn]
                            && !addScale.ContainsKey(cn))
                        {
                            NodeRecord ad = nodes[cn];
                            uint oth = ad.Inputs[0] == node.Outputs[0]
                                ? ad.Inputs[1] : ad.Inputs[0];
                            int othP = Phys(checked((int)oth));
                            if (!isConst[othP] && numel[othP] == numel[rp]
                                && Phys(checked((int)ad.Inputs[0]))
                                   != Phys(checked((int)ad.Inputs[1])))
                            {
                                int ao = Phys(checked((int)ad.Outputs[0]));
                                // addps-absorb on either operand
                                long xOff4 = SlotOf(node.Inputs[0]);
                                long aOff4 = SlotOf(oth), raOff4 = 0, seOff4 = 0;
                                uint raFl = 0;
                                if (addpsSrc.TryGetValue(
                                    Phys(checked((int)node.Inputs[0])), out var adx))
                                { xOff4 = SlotOf(adx.x); raOff4 = SlotOf(adx.a);
                                  seOff4 = SlotOf(adx.se); raFl |= 256u; }
                                if (addpsSrc.TryGetValue(
                                    Phys(checked((int)oth)), out var ada))
                                { aOff4 = SlotOf(ada.x); raOff4 = SlotOf(ada.a);
                                  seOff4 = SlotOf(ada.se); raFl |= 512u; }
                                Emit(_pResize4Add, $"resizeadd n{ni}+{cn}",
                                    [(arena, xOff4, 2),
                                     (arena, aOff4, 2),
                                     (arena, off[ao], 2),
                                     (arena, raOff4, 2), (arena, seOff4, 2)],
                                    [(uint)osp[3], (uint)osp[2], (uint)osp[1],
                                     (uint)(osp[2] / ishp[2]), (uint)(osp[3] / ishp[3]),
                                     raFl],
                                    Div256(numel[rp] / 4));
                                skipEmit.Add(cn);
                                break;
                            }
                        }
                        long wO = off[outPhys]; uint dcv = 0, dco = 0;
                        if (cn >= 0 && nodes[cn].Operator == OperatorId.Concat)
                        {
                            int[] ospc = shapes[nodes[cn].Outputs[0]];
                            int co = 0; bool hit = false;
                            foreach (uint cin_ in nodes[cn].Inputs)
                            {
                                if (Phys(checked((int)cin_)) == rp) { hit = true; break; }
                                co += shapes[checked((int)cin_)][1];
                            }
                            if (hit && co % 4 == 0 && ospc[1] % 4 == 0)
                            {
                                wO = off[Phys(checked((int)nodes[cn].Outputs[0]))];
                                dcv = (uint)(ospc[1] / 4);
                                dco = (uint)(co / 4);
                                concatAbsorbed.Add(rp);
                            }
                        }
                        bool aps = addpsSrc.TryGetValue(
                            Phys(checked((int)node.Inputs[0])), out var ads);
                        Emit(_pResize4, $"resize4 n{ni} x{osp[2] / ishp[2]}",
                            [(arena, aps ? SlotOf(ads.x)
                                        : SlotOf(node.Inputs[0]), 2),
                             (arena, wO, 2),
                             (arena, aps ? SlotOf(ads.a) : 0, 2),
                             (arena, aps ? SlotOf(ads.se) : 0, 2)],
                            [(uint)osp[3], (uint)osp[2], (uint)osp[1],
                             (uint)(osp[2] / ishp[2]), (uint)(osp[3] / ishp[3]),
                             dcv, dco, aps ? 256u : 0u],
                            Div256(numel[outPhys] / 4));
                    }
                    else
                        Emit(_pResize, $"resize n{ni} x{osp[2] / ishp[2]}",
                            [(arena, SlotOf(node.Inputs[0]), 2), (arena, off[outPhys], 2)],
                            [(uint)osp[3], (uint)osp[2], (uint)osp[1],
                             (uint)(osp[2] / ishp[2]), (uint)(osp[3] / ishp[3])],
                            Div256(numel[outPhys]));
                    break;
                }

                case OperatorId.Concat:
                {
                    if (nb > 1)
                        throw new NotSupportedException(
                            $"concat lacks batch support (n{ni})");
                    int[] osp = shapes[node.Outputs[0]];
                    if (catRes.TryGetValue(ni, out var spec))
                    {
                        var bb = new (VkBuffer, long, int)[13];
                        var pc2 = new uint[19];
                        pc2[0] = (uint)osp[3]; pc2[1] = (uint)osp[2];
                        pc2[2] = (uint)osp[1];
                        for (int s = 0; s < 4; s++)
                        {
                            long xP = 0, raP = 0, seP = 0;
                            uint fl = 0, inW3 = 0, fhfw = 0, offc = 0;
                            if (s < spec.Length)
                            {
                                var t = spec[s];
                                xP = t.Item1; raP = t.Item2; seP = t.Item3;
                                fl = t.Item4; inW3 = t.Item5;
                                fhfw = t.Item6 | (t.Item7 << 16);
                                offc = t.Item8 | (t.Item9 << 16);
                            }
                            bb[s * 3] = (arena, xP, 2);
                            bb[s * 3 + 1] = (arena, raP, 2);
                            bb[s * 3 + 2] = (arena, seP, 2);
                            pc2[3 + s * 4] = offc; pc2[4 + s * 4] = inW3;
                            pc2[5 + s * 4] = fhfw; pc2[6 + s * 4] = fl;
                        }
                        bb[12] = (arena, off[outPhys], 2);
                        Emit(_pCatRes, $"catres n{ni}", bb, pc2,
                            Div256(numel[outPhys] / 4));
                        break;
                    }
                    int cOff = 0;
                    foreach (uint inp in node.Inputs)
                    {
                        int it = checked((int)inp);
                        int ci = shapes[it][1];
                        if (concatAbsorbed.Contains(Phys(it)))
                        { cOff += ci; continue; }
                        if (ci % 4 == 0 && cOff % 4 == 0 && osp[1] % 4 == 0)
                        {
                            bool aps = addpsSrc.TryGetValue(Phys(it), out var ads);
                            Emit(_pConcat4, $"concat4 n{ni} c{ci}@{cOff}",
                                [(arena, aps ? SlotOf(ads.x) : SlotOf(inp), 2),
                                 (arena, off[outPhys], 2),
                                 (arena, aps ? SlotOf(ads.a) : 0, 2),
                                 (arena, aps ? SlotOf(ads.se) : 0, 2)],
                                [(uint)(osp[2] * osp[3]), (uint)ci, (uint)osp[1],
                                 (uint)cOff, aps ? 256u : 0u],
                                Div256((long)osp[2] * osp[3] * (ci / 4)));
                        }
                        else
                            Emit(_pConcat, $"concat n{ni} c{ci}@{cOff}",
                                [(arena, SlotOf(inp), 2), (arena, off[outPhys], 2)],
                                [(uint)(osp[2] * osp[3]), (uint)ci, (uint)osp[1], (uint)cOff],
                                Div256((long)osp[2] * osp[3] * ci));
                        cOff += ci;
                    }
                    break;
                }

                case OperatorId.ConvTranspose:
                {
                    if (nb > 1)
                        throw new NotSupportedException(
                            $"convtranspose lacks batch support (n{ni})");
                    int[] ishp = shapes[node.Inputs[0]];
                    int[] oshp = shapes[node.Outputs[0]];
                    if (!(I32(p, 8) == 2 && I32(p, 12) == 2 && I32(p, 16) == 2 && I32(p, 20) == 2
                          && I32(p, 32) == 0 && I32(p, 36) == 0))
                        throw new NotSupportedException($"ConvTranspose shape at node {ni}");
                    int act = 0;
                    VkBuffer biasBuf = arena;
                    uint flags2 = 0;
                    if (skip == 2 && nodes[ni + 1].Operator == OperatorId.Add)
                    {
                        NodeRecord add = nodes[ni + 1];
                        int bt = add.Inputs[0] == node.Outputs[0]
                            ? checked((int)add.Inputs[1]) : checked((int)add.Inputs[0]);
                        biasBuf = ConstF16(bt);
                        flags2 = 1u | (numel[bt] == 1 ? 8u : 0u);
                        OperatorId actOp = nodes[ni + 2].Operator;
                        act = actOp == OperatorId.Relu ? 1 : actOp == OperatorId.Sigmoid ? 4 : 0;
                    }
                    if (ishp[1] % 4 == 0 && oshp[1] % 4 == 0)
                        Emit(_pConvT4, $"convT4 n{ni} {ishp[1]}->{oshp[1]}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (ConstConvT(checked((int)node.Inputs[1]), ishp[1], oshp[1]), 0, 2),
                             (biasBuf, 0, 2), (arena, off[outPhys], 2)],
                            [(uint)oshp[3], (uint)oshp[2], (uint)oshp[1], (uint)ishp[1],
                             (uint)ishp[3], flags2 | (uint)(act << 4)],
                            Div256(numel[outPhys] / 4));
                    else
                    {
                        // scalar-tail convT writing the graph output: emit fp32
                        // straight into the readback buffer (kills the `out` rec)
                        if (outPhys == Phys(outIdx)) { flags2 |= 128u; convTOut = true; }
                        Emit(_pConvT, $"convT n{ni} {ishp[1]}->{oshp[1]}",
                            [(arena, SlotOf(node.Inputs[0]), 2),
                             (ConstF16(checked((int)node.Inputs[1])), 0, 2),
                             (biasBuf, 0, 2), (arena, off[outPhys], 2),
                             (outF32, 0, 4)],
                            [(uint)oshp[3], (uint)oshp[2], (uint)oshp[1], (uint)ishp[1],
                             (uint)ishp[3], flags2 | (uint)(act << 4)],
                            Div256(numel[outPhys]));
                    }
                    ni += skip;
                    break;
                }

                case OperatorId.Transpose:
                    // rank-3 [0,2,1] transposes were folded into aliases in
                    // pass 1 (channel-last physical layout makes them free).
                    if (Phys(checked((int)node.Outputs[0]))
                        != Phys(checked((int)node.Inputs[0])))
                        throw new NotSupportedException(
                            $"non-trivial Transpose at node {ni}");
                    break;

                case OperatorId.AveragePool:
                {
                    int[] ishp = shapes[node.Inputs[0]];
                    int[] osp2 = shapes[node.Outputs[0]];
                    if (ishp[1] % 4 != 0 || ishp.Length != 4)
                        throw new NotSupportedException(
                            $"avgpool C%4 at node {ni}");
                    Emit(_pAvg4, $"avgpool4 n{ni}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (arena, off[outPhys], 2)],
                        [(uint)osp2[3], (uint)osp2[2], (uint)ishp[1],
                         (uint)I32(p, 8), (uint)I32(p, 12),
                         (uint)I32(p, 16), (uint)I32(p, 20),
                         (uint)I32(p, 24), (uint)I32(p, 28),
                         (uint)ishp[3], (uint)ishp[2]],
                        Div256(numel[outPhys] / 4), (uint)nb);
                    break;
                }

                case OperatorId.BatchNormalization:
                {
                    // fold to per-channel affine: o = x*s4 + t4 where
                    // s = scale/sqrt(var+eps), t = bias - mean*s
                    float eps = F32(p, 4);
                    int bt = checked((int)node.Inputs[1]);
                    ReadOnlySpan<float> sc = CstF32(bt),
                        bi = CstF32(checked((int)node.Inputs[2])),
                        mn = CstF32(checked((int)node.Inputs[3])),
                        va = CstF32(checked((int)node.Inputs[4]));
                    int cc = sc.Length;
                    float[] s = new float[cc], t = new float[cc];
                    for (int i = 0; i < cc; i++)
                    {
                        s[i] = sc[i] / MathF.Sqrt(va[i] + eps);
                        t[i] = bi[i] - mn[i] * s[i];
                    }
                    long n = numel[outPhys];
                    if (n % 4 != 0 || cc % 4 != 0)
                        throw new NotSupportedException($"bn vec4 at node {ni}");
                    Emit(_pAffine, $"bn4 n{ni} c{cc}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (VecF16(s), 0, 2), (VecF16(t), 0, 2),
                         (arena, off[outPhys], 2)],
                        [(uint)(n / 4), (uint)(cc / 4)], Div256(n / 4));
                    break;
                }

                case OperatorId.Softmax:
                {
                    // last-axis softmax (cls head [n,2]); one invocation per row.
                    int[] ssp = shapes[node.Inputs[0]];
                    long cols = ssp[^1];
                    long rows = numel[outPhys] / cols;
                    if (cols <= 0 || rows <= 0 || rows * cols != numel[outPhys])
                        throw new NotSupportedException($"softmax at node {ni}");
                    Emit(_pSoftmax, $"softmax n{ni} {rows}x{cols}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (arena, off[outPhys], 2)],
                        [(uint)rows, (uint)cols], Div256(rows));
                    break;
                }

                case OperatorId.MatMul:
                {
                    // [n,M,K] x [K,N]: flat rows over batch feed the coopmat
                    // 1x1-conv GEMM as-is; weights repack to [N,K] fp16.
                    int[] asp = shapes[node.Inputs[0]];
                    int[] bsp = shapes[node.Inputs[1]];
                    int mmK = asp[^1], mmN = bsp[^1];
                    if (bsp.Length != 2 || !isConst[(int)node.Inputs[1]]
                        || mmK % 4 != 0 || mmN % 4 != 0)
                        throw new NotSupportedException($"matmul at node {ni}");
                    uint mmM = (uint)(numel[checked((int)node.Inputs[0])] / mmK);
                    // fold a following bias Add [N] into the GEMM epilogue
                    uint mmFlags = 0; VkBuffer mmBias = arena;
                    if (ni + 1 < emitLimit
                        && nodes[ni + 1].Operator == OperatorId.Add
                        && _compiled.FusedSkip(ni + 1) == 0)
                    {
                        NodeRecord ad = nodes[ni + 1];
                        uint oth = ad.Inputs[0] == node.Outputs[0] ? ad.Inputs[1]
                                 : ad.Inputs[1] == node.Outputs[0] ? ad.Inputs[0]
                                 : uint.MaxValue;
                        if (oth != uint.MaxValue && isConst[(int)oth]
                            && numel[(int)oth] == mmN
                            && refCount[Phys(checked((int)node.Outputs[0]))] == 1)
                        {
                            mmBias = ConstF16(checked((int)oth));
                            mmFlags |= 1u;
                            outPhys = Phys(checked((int)ad.Outputs[0]));
                            ni += 1;
                        }
                    }
                    var (cp2, tm2, tn2) = CmTile(mmN);
                    Emit(cp2, $"matmul n{ni} {mmM}x{mmK}x{mmN}",
                        [(arena, SlotOf(node.Inputs[0]), 2),
                         (ConstGemmW(checked((int)node.Inputs[1]), mmK, mmN), 0, 2),
                         (mmBias, 0, 2), (arena, 0, 2),
                         (arena, off[outPhys], 2),
                         (arena, 0, 2)],
                        [mmM, (uint)mmN, (uint)mmK, mmFlags],
                        (mmM + tm2 - 1) / tm2, (uint)((mmN + tn2 - 1) / tn2));
                    break;
                }

                default:
                    throw new NotSupportedException($"op {node.Operator} at node {ni}");
            }
        }

        // finalize: graph output → fp32 readback.
        // If the last node is a STANDALONE Sigmoid (its own elem dispatch was
        // the last rec), replace it with a sigmoid-ing f32 writeback. If the
        // sigmoid was consumed into a fused group (e.g. convT+bias+sigmoid),
        // the sink slot already holds sigmoid'd values → plain copy.
        int lastNi = recs.Count > 0 ? LastEmitNi : -1;
        NodeRecord last = nodes[nodes.Length - 1];
        bool standaloneSigmoid = lastNi == nodes.Length - 1
            && last.Operator == OperatorId.Sigmoid
            && checked((int)last.Outputs[0]) == outIdx;
        int outSrc;
        uint outAct;
        if (standaloneSigmoid)
        {
            recs.RemoveAt(recs.Count - 1);
            outSrc = Phys(checked((int)last.Inputs[0]));
            outAct = 4u;
        }
        else
        {
            outSrc = Phys(outIdx);
            outAct = 0u;
        }
        if (!convTOut)
            Emit(_pOut, "out", [(arena, off[outSrc], 2), (outF32, 0, 4)],
                [(uint)numel[outIdx], outAct], Div256(numel[outIdx]));

        // head: fp32 NCHW input → fp16 NHWC (channel-padded to inCinPad if set).
        // Skipped entirely when the stem conv read the fp32 input itself.
        if (!inF32Consumed)
        {
            int[] isp = shapes[inIdx];
            int hw = isp[2] * isp[3], c = isp[1];
            uint cout = inCinPad != 0 ? (uint)inCinPad : (uint)c;
            IntPtr set = _dev.NewDescriptorSet(_pNchw.SetLayout);
            _dev.BindBuffer(set, 0, inF32, 0);
            _dev.BindBuffer(set, 1, arena, (ulong)off[inIdx] * 2);
            recs.Insert(0, new Rec
            {
                Pipe = _pNchw, Set = set, Gx = Div256((long)hw * cout),
                Gy = (uint)nb,
                Pc = Pcu((uint)hw, (uint)c, cout), Tag = "nchw2nhwc",
            });
        }

        // debug: SIMD_OCR_GPU_ONLYHEAD records only the input-convert dispatch;
        // SIMD_OCR_GPU_TRUNCATE=N records the first N dispatches.
        List<Rec> recordRecs = recs;
        if (Environment.GetEnvironmentVariable("SIMD_OCR_GPU_ONLYHEAD") == "1")
            recordRecs = [recs[0]];
        else if (int.TryParse(Environment.GetEnvironmentVariable("SIMD_OCR_GPU_ONLY"),
                     out int only) && only < recs.Count)
            recordRecs = Enumerable.Repeat(recs[only], 20).ToList();
        else if (int.TryParse(Environment.GetEnvironmentVariable("SIMD_OCR_GPU_TRUNCATE"),
                     out int trunc) && trunc < recs.Count)
            recordRecs = recs.Take(trunc).ToList();

        if (Environment.GetEnvironmentVariable("SIMD_OCR_GPU_DUMP") == "1")
            for (int ri = 0; ri < recs.Count; ri++)
                Console.Error.WriteLine($"rec[{ri}] {recs[ri].Tag} gx={recs[ri].Gx} gy={recs[ri].Gy}");

        bool prof = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_PROF") == "1";
        bool noBar = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_NOBAR") == "1";
        IntPtr pool = IntPtr.Zero;
        if (prof)
        {
            var qci = new Vk.VkQueryPoolCreateInfo
            { SType = 11, QueryType = 0, QueryCount = (uint)(recordRecs.Count * 2) };
            Vk.Check(Vk.vkCreateQueryPool(_dev.Device, &qci, null, out pool), "vkCreateQueryPool");
        }

        IntPtr cmd = _dev.NewCommandBuffer();
        unsafe
        {
            var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
            Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
            if (prof) Vk.vkCmdResetQueryPool(cmd, pool, 0, (uint)(recordRecs.Count * 2));
            uint q = 0;
            foreach (Rec r in recordRecs)
            {
                if (prof) Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, pool, q++);
                Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, r.Pipe.Pipeline);
                IntPtr s = r.Set;
                Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, r.Pipe.Layout, 0, 1, &s, 0, null);
                fixed (byte* pp = r.Pc)
                    Vk.vkCmdPushConstants(cmd, r.Pipe.Layout, VkConst.StageComputeShader,
                        0, (uint)r.Pc.Length, pp);
                Vk.vkCmdDispatch(cmd, r.Gx, r.Gy, 1);
                if (!noBar) _dev.CmdComputeBarrier(cmd);
                if (prof) Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageBottomOfPipe, pool, q++);
            }
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
        }

        return new Plan
        {
            Arena = arena, InF32 = inF32, OutF32 = outF32, Cmd = cmd, Recs = recs,
            OutElems = (int)numel[outIdx], ArenaBytes = arenaElems * 2,
            Off = off, Alias = alias, Numel = numel, Shapes = shapes,
            QueryPool = pool, QueryCount = recordRecs.Count * 2,
            Im2colOff = im2colOff,
        };
    }

    /// <summary>Debug: raw values of a tensor's arena slot.</summary>
    public unsafe float[] DebugValues(int tensorIndex, int[] inputShape, int count = 16,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        Plan plan = _plans[KeyOf(inputShape, nodeLimit, outTensor)];
        int t = tensorIndex;
        while (plan.Alias[t] != t) t = plan.Alias[t];
        int n = Math.Min(count, (int)plan.Numel[tensorIndex]);
        VkBuffer tmp = _dev.NewStorageBuffer((ulong)n * 4, hostVisible: true, preferHost: true);
        IntPtr set = _dev.NewDescriptorSet(_pOut.SetLayout);
        _dev.BindBuffer(set, 0, plan.Arena, (ulong)plan.Off[t] * 2);
        _dev.BindBuffer(set, 1, tmp, 0);
        IntPtr cmd = _dev.NewCommandBuffer();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, _pOut.Pipeline);
        Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, _pOut.Layout, 0, 1, &set, 0, null);
        byte* pc = stackalloc byte[8];
        *(uint*)pc = (uint)n; *(uint*)(pc + 4) = 0;
        Vk.vkCmdPushConstants(cmd, _pOut.Layout, VkConst.StageComputeShader, 0, 8, pc);
        Vk.vkCmdDispatch(cmd, (uint)((n + 255) / 256), 1, 1);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
        IntPtr f = _dev.NewFence();
        _dev.Submit(cmd, f);
        _dev.WaitFence(f);
        float[] v = new float[n];
        float* fp = (float*)tmp.Map();
        for (int i = 0; i < n; i++) v[i] = fp[i];
        tmp.Unmap();
        Console.Error.WriteLine($"  [dbg] t={tensorIndex} phys={t} off={plan.Off[t]} n={plan.Numel[tensorIndex]}");
        return v;
    }

    public long Im2colOffset(int[] inputShape)
        => _plans[KeyOf(inputShape, int.MaxValue, -1)].Im2colOff;

    /// <summary>Debug: read raw arena elements at element offset after a Run.</summary>
    public long DebugElemOff(int tensorIndex, int[] inputShape,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        Plan plan = _plans[KeyOf(inputShape, nodeLimit, outTensor)];
        int t = tensorIndex;
        while (plan.Alias[t] != t) t = plan.Alias[t];
        return plan.Off[t];
    }

    public unsafe float[] DebugValuesRaw(int[] inputShape, long elemOff, int count,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        Plan plan = _plans[KeyOf(inputShape, nodeLimit, outTensor)];
        VkBuffer tmp = _dev.NewStorageBuffer((ulong)count * 4, hostVisible: true, preferHost: true);
        IntPtr set = _dev.NewDescriptorSet(_pOut.SetLayout);
        _dev.BindBuffer(set, 0, plan.Arena, (ulong)elemOff * 2);
        _dev.BindBuffer(set, 1, tmp, 0);
        IntPtr cmd = _dev.NewCommandBuffer();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, _pOut.Pipeline);
        Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, _pOut.Layout, 0, 1, &set, 0, null);
        byte* pc = stackalloc byte[8];
        *(uint*)pc = (uint)count; *(uint*)(pc + 4) = 0;
        Vk.vkCmdPushConstants(cmd, _pOut.Layout, VkConst.StageComputeShader, 0, 8, pc);
        Vk.vkCmdDispatch(cmd, (uint)((count + 255) / 256), 1, 1);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
        IntPtr f = _dev.NewFence();
        _dev.Submit(cmd, f);
        _dev.WaitFence(f);
        float[] v = new float[count];
        float* fp = (float*)tmp.Map();
        for (int i = 0; i < count; i++) v[i] = fp[i];
        tmp.Unmap();
        return v;
    }

    /// <summary>Debug: min/max/mean of a tensor's arena slot (NHWC flat) after a Run.</summary>
    public unsafe (float Min, float Max, double Mean) DebugStats(int tensorIndex, int[] inputShape,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        Plan plan = _plans[KeyOf(inputShape, nodeLimit, outTensor)];
        int t = tensorIndex;
        while (plan.Alias[t] != t) t = plan.Alias[t];
        int n = (int)plan.Numel[tensorIndex];
        VkBuffer tmp = _dev.NewStorageBuffer((ulong)n * 4, hostVisible: true, preferHost: true);
        IntPtr set = _dev.NewDescriptorSet(_pOut.SetLayout);
        _dev.BindBuffer(set, 0, plan.Arena, (ulong)plan.Off[t] * 2);
        _dev.BindBuffer(set, 1, tmp, 0);
        IntPtr cmd = _dev.NewCommandBuffer();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, _pOut.Pipeline);
        Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, _pOut.Layout, 0, 1, &set, 0, null);
        byte* pc = stackalloc byte[8];
        *(uint*)pc = (uint)n; *(uint*)(pc + 4) = 0;
        Vk.vkCmdPushConstants(cmd, _pOut.Layout, VkConst.StageComputeShader, 0, 8, pc);
        Vk.vkCmdDispatch(cmd, (uint)((n + 255) / 256), 1, 1);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");
        IntPtr f = _dev.NewFence();
        _dev.Submit(cmd, f);
        _dev.WaitFence(f);
        float* fp = (float*)tmp.Map();
        float mn = float.MaxValue, mx = float.MinValue; double s = 0;
        for (int i = 0; i < n; i++) { float v = fp[i]; mn = MathF.Min(mn, v); mx = MathF.Max(mx, v); s += v; }
        tmp.Unmap();
        return (mn, mx, s / n);
    }

    /// <summary>Per-dispatch GPU timestamps (requires SIMD_OCR_GPU_PROF=1 at plan build).</summary>
    public unsafe void DumpProfile(int[] inputShape,
        int nodeLimit = int.MaxValue, int outTensor = -1)
    {
        PlanKey key = KeyOf(inputShape, nodeLimit, outTensor);
        if (!_plans.TryGetValue(key, out Plan? plan) || plan.QueryPool == IntPtr.Zero)
        { Console.WriteLine("(no profile — set SIMD_OCR_GPU_PROF=1)"); return; }
        ulong[] ts = new ulong[plan.QueryCount];
        VkResult qr;
        fixed (ulong* tp = ts)
            qr = Vk.vkGetQueryPoolResults(_dev.Device, plan.QueryPool, 0, (uint)plan.QueryCount,
                (nuint)(ts.Length * 8), tp, 8, 1 | 2);
        Console.WriteLine($"(query result={qr}, ts0={ts[0]}, ts1={ts[1]}, periodNs={_dev.TimestampPeriodNs})");
        if (ts.All(t => t == 0))
        { Console.WriteLine("(timestamps all zero — queue may lack TimestampValidBits)"); return; }
        var agg = new Dictionary<string, double>();
        for (int i = 0; i + 1 < plan.QueryCount; i += 2)
        {
            double ms = (ts[i + 1] - ts[i]) * _dev.TimestampPeriodNs / 1e6;
            string tag = plan.Recs[i / 2].Pipe.Name ?? "?";
            agg[tag] = agg.TryGetValue(tag, out double v) ? v + ms : ms;
        }
        double total = agg.Values.Sum();
        Console.WriteLine($"--- GPU profile ({plan.Recs.Count} dispatches, {total:F2} ms) ---");
        foreach (var kv in agg.OrderByDescending(k => k.Value))
            Console.WriteLine($"  {kv.Key,-14} {kv.Value,8:F3} ms");
    }

    private static byte[] Pcu(params uint[] v)
    {
        byte[] b = new byte[v.Length * 4];
        for (int i = 0; i < v.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(b.AsSpan(i * 4), v[i]);
        return b;
    }

    private static ushort U16(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt16LittleEndian(p[o..]);
    private static int I32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadInt32LittleEndian(p[o..]);
    private static uint U32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt32LittleEndian(p[o..]);
    private static float F32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadSingleLittleEndian(p[o..]);
    // fp16 bits of alpha (lo16) packed for shaders that read aux/flags via unpackHalf2x16
    private static ushort Half2Bits(float v) => BitConverter.HalfToUInt16Bits((Half)v);
    private static uint HsAux(ReadOnlySpan<byte> p) =>
        (uint)Half2Bits(F32(p, 4)) | ((uint)Half2Bits(F32(p, 8)) << 16);

    // VkBuffer lifetime is tied to the device (no per-buffer Dispose in the port).
    public void Dispose() => _allBufs.Clear();
}
