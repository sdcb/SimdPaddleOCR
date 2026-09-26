// M0 POC for the Vulkan backend: WC write-bandwidth probe + conv1x1-as-GEMM
// (coopmat fp16 XMX) vs the CPU Conv1x1 kernel. Usage: [--shape m,k,n]*
using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

// --det <model.onnx> <H> <W>: DET graph GPU vs CPU correctness + latency gate
if (args.Length >= 4 && args[0] == "--det")
{
    string modelPath = args[1];
    int H = int.Parse(args[2]), W = int.Parse(args[3]);
    var mdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(modelPath));
    var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(mdl, intraOpThreads: 4);
    var cpu = new Sdcb.SimdPaddleOCR.OnnxSharp.InferenceSession(compiled);
    cpu.Reshape([1, 3, H, W]);

    var rr = new Random(20260924);
    var inp = new float[3 * H * W];
    for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rr.NextDouble() * 2 - 1);
    int imgIdx = Array.IndexOf(args, "--img");
    if (imgIdx >= 0)
    {
        // real image → resize to H×W (bilinear) → BGR (v/255-mean)/std → NCHW
        using var bmp = new System.Drawing.Bitmap(args[imgIdx + 1]);
        using var rs = new System.Drawing.Bitmap(bmp, W, H);
        var bd = rs.LockBits(new System.Drawing.Rectangle(0, 0, W, H),
            System.Drawing.Imaging.ImageLockMode.ReadOnly,
            System.Drawing.Imaging.PixelFormat.Format24bppRgb);
        double[] mean = [0.485, 0.456, 0.406], istd = [1 / 0.229, 1 / 0.224, 1 / 0.225];
        unsafe
        {
            for (int y = 0; y < H; y++)
            {
                byte* row = (byte*)bd.Scan0 + y * bd.Stride;
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < 3; c++)
                        inp[c * H * W + y * W + x] =
                            (float)((row[x * 3 + c] / 255.0 - mean[c]) * istd[c]);
            }
        }
        rs.UnlockBits(bd);
        Console.WriteLine($"loaded {args[imgIdx + 1]} ({bmp.Width}x{bmp.Height})");
    }

    Console.Write("cpu det... ");
    var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
    float[] probCpu = new float[H * W];
    cpu.Run(inp, probCpu);   // public entry: BindPublicInput does NCHW→NHWC when needed
    double cpuMs1 = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) / (double)System.Diagnostics.Stopwatch.Frequency * 1000;
    Console.WriteLine($"{cpuMs1:F1} ms");

    using var ddev = Sdcb.SimdPaddleOCR.Backends.Vulkan.VkDevice.Create();
    using var gg = new Sdcb.SimdPaddleOCR.Backends.Vulkan.GpuDetGraph(ddev, compiled);
    Console.Write("gpu det... ");
    t0 = System.Diagnostics.Stopwatch.GetTimestamp();
    float[] probGpu = gg.Run([1, 3, H, W], inp);
    double gpuMs1 = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) / (double)System.Diagnostics.Stopwatch.Frequency * 1000;
    Console.WriteLine($"{gpuMs1:F1} ms (first run incl plan build)");

    if (args.Any(a => a == "--probe"))
    {
        foreach (string s in args.SkipWhile(a => a != "--probe").Skip(1).TakeWhile(a => !a.StartsWith("-")))
        {
            int t = int.Parse(s);
            var st = gg.DebugStats(t, [1, 3, H, W]);
            var vv = gg.DebugValues(t, [1, 3, H, W], 8);
            Console.WriteLine($"tensor {t}: gpu[{st.Min:F4},{st.Max:F4},{st.Mean:F5}] vals=[{string.Join(",", vv.Select(v => v.ToString("F3")))}]");
        }
        var (iMin, iMax) = (inp.Min(), inp.Max());
        Console.WriteLine($"input raw: [{iMin:F4},{iMax:F4},{inp.Average():F5}] first8=[{string.Join(",", inp.Take(8).Select(v => v.ToString("F3")))}] nhwc-first8=[{string.Join(",", Enumerable.Range(0, 8).Select(i => inp[(i % 3) * H * W + i / 3].ToString("F3")))}]");
    }
    if (args.Any(a => a == "--c1check"))
    {
        // Recompute node 39 (conv1x1 32->64) on CPU from GPU tensor 202,
        // compare elementwise vs GPU tensor 203.
        var metaIn = compiled.GetTensor(202); var metaOut = compiled.GetTensor(203);
        var metaW = compiled.GetTensor(18); var metaB = compiled.GetTensor(105);
        int cin = metaIn.Shape[1], ih = metaIn.Shape[2], iw = metaIn.Shape[3];
        int cout = metaOut.Shape[1];
        float[] xG = gg.DebugValues(202, [1, 3, H, W], cin * ih * iw);
        float[] oG = gg.DebugValues(203, [1, 3, H, W], cout * ih * iw);
        float[] wf = MemoryMarshal.Cast<byte, float>(metaW.Constant).ToArray();
        float[] bf = MemoryMarshal.Cast<byte, float>(metaB.Constant).ToArray();
        double md = 0; int badIdx = -1;
        for (int px = 0; px < ih * iw; px++)
        for (int co = 0; co < cout; co++)
        {
            double acc = bf[co];
            for (int ci = 0; ci < cin; ci++)
                acc += xG[px * cin + ci] * wf[co * cin + ci];
            double d = Math.Abs(acc - oG[px * cout + co]);
            if (d > md) { md = d; badIdx = px * cout + co; }
        }
        Console.WriteLine($"c1 n39: maxDiff={md:F4} worstIdx={badIdx} (px={badIdx / cout} co={badIdx % cout})");
        // distribution of errors
        int b1 = 0, b2 = 0, b3 = 0;
        for (int px = 0; px < ih * iw; px++)
        for (int co = 0; co < cout; co++)
        {
            double acc = bf[co];
            for (int ci = 0; ci < cin; ci++)
                acc += xG[px * cin + ci] * wf[co * cin + ci];
            double d = Math.Abs(acc - oG[px * cout + co]);
            if (d > 0.01) b1++; if (d > 0.1) b2++; if (d > 0.5) b3++;
        }
        Console.WriteLine($"  err>0.01: {b1}/{ih * iw * cout}  >0.1: {b2}  >0.5: {b3}");
    }
    if (args.Any(a => a == "--n0check"))
    {
    string stage = "init";
    try
    {
        // recompute node0 conv (3->16 k3x3 s2) on CPU from raw input
        var metaOut = compiled.GetTensor(166);
        var n0 = mdl.Nodes[0];
        Console.WriteLine($"  n0: op={n0.OpType} ins=[{string.Join(",", n0.Inputs)}] outs=[{string.Join(",", n0.Outputs)}] " +
            $"graphIn=[{string.Join(",", mdl.GraphInputs)}] t0 const={(compiled.GetTensor(0).Constant.Length > 0)}");
        int wi = checked((int)n0.Inputs[1]), bi = n0.Inputs.Length > 2 ? checked((int)n0.Inputs[2]) : -1;
        var metaW = compiled.GetTensor(wi); var metaB = compiled.GetTensor(bi < 0 ? wi : bi);
        Console.WriteLine($"  n0 w=t{wi} shape=[{string.Join(",", metaW.Shape)}] b=t{bi}");
        var p0 = mdl.GetParameters(mdl.Nodes[0]);
        int kH = BitConverter.ToInt32(p0.Slice(8)), sH = BitConverter.ToInt32(p0.Slice(16));
        int pt = BitConverter.ToInt32(p0.Slice(32)), pl = BitConverter.ToInt32(p0.Slice(36));
        int pb = BitConverter.ToInt32(p0.Slice(40)), pr = BitConverter.ToInt32(p0.Slice(44));
        int cin = 3, cout = 16;
        int oh = (H + pt + pb - kH) / sH + 1, ow = (W + pl + pr - kH) / sH + 1;
        float[] oG = gg.DebugValues(166, [1, 3, H, W], cout * oh * ow);
        Console.WriteLine($"  fetch: cout={cout} oh={oh} ow={ow} ask={cout * oh * ow} oG.Len={oG.Length}");
        // CPU ground truth for tensor 166 (NCHW fp32)
        float[] t0cpu = cpu.TraceNodeValues(inp, 0);
        var tr0 = cpu.Trace(inp)[0];
        Console.WriteLine($"  cpu166: min={t0cpu.Min():F3} max={t0cpu.Max():F3} px0co4={t0cpu[4 * oh * ow]:F4} (NCHW) " +
            $"shape=[{string.Join(",", tr0.Shape)}] nhwc={compiled.IsNhwcNode(0)} len={t0cpu.Length}");
        double mdCpu = 0; int badCpu = -1;
        bool cpuNhwc = compiled.IsNhwcNode(0);
        // cpu stores NHWC[px*cout+co] when nhwc=True else NCHW[co*oh*ow+px]; gpu is NHWC
        for (int px = 0; px < oh * ow; px++)
        for (int co = 0; co < cout; co++)
        { float cv = cpuNhwc ? t0cpu[px * cout + co] : t0cpu[co * oh * ow + px];
          double d = Math.Abs(cv - oG[px * cout + co]); if (d > mdCpu) { mdCpu = d; badCpu = px * cout + co; } }
        Console.WriteLine($"  gpu166 vs cpu166(nhwc={cpuNhwc}): maxDiff={mdCpu:F4} worst(px={badCpu / cout} co={badCpu % cout})");
        float[] wf = MemoryMarshal.Cast<byte, float>(metaW.Constant).ToArray();
        float[] bf = MemoryMarshal.Cast<byte, float>(metaB.Constant).ToArray();
        Console.WriteLine($"n0 params: k={kH} s={sH} pad=T{pt},L{pl},B{pb},R{pr}");
        // dump hardsigmoid params of every HardSigmoid node
        for (int ni2 = 0; ni2 < mdl.Nodes.Length; ni2++)
            if (mdl.Nodes[ni2].OpType == "HardSigmoid")
            {
                var hp = mdl.GetParameters(mdl.Nodes[ni2]).ToArray();
                float a = BitConverter.ToSingle(hp, 4), b2 = BitConverter.ToSingle(hp, 8);
                Console.WriteLine($"  hs n{ni2}: alpha={a:F4} beta={b2:F4} in={mdl.Nodes[ni2].Inputs[0]} out={mdl.Nodes[ni2].Outputs[0]}");
            }
        var p0a = p0.ToArray();
        Console.WriteLine($"  p0 ints: [{string.Join(",", Enumerable.Range(0, p0a.Length / 4).Select(i => BitConverter.ToInt32(p0a, i * 4)))}] len={p0a.Length}");
        double md = 0, mdEdge = 0, mdIn = 0; int badIdx = -1;
        for (int px = 0; px < oh * ow; px++)
        for (int co = 0; co < cout; co++)
        {
            double acc = bf[co];
            int oy = px / ow, ox = px % ow;
            for (int dy = 0; dy < kH; dy++)
            for (int dx = 0; dx < kH; dx++)
            {
                int iy = oy * sH + dy - pt, ix = ox * sH + dx - pl;
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                for (int ci = 0; ci < cin; ci++)
                    acc += inp[ci * H * W + iy * W + ix]
                         * wf[co * cin * kH * kH + ci * kH * kH + dy * kH + dx];
            }
            double d = Math.Abs(acc - oG[px * cout + co]);
            if (d > md) { md = d; badIdx = px * cout + co; }
            bool edge = oy < 2 || ox < 2 || oy >= oh - 2 || ox >= ow - 2;
            if (edge) mdEdge = Math.Max(mdEdge, d); else mdIn = Math.Max(mdIn, d);
        }
        Console.WriteLine($"n0check: maxDiff={md:F4} edge={mdEdge:F4} interior={mdIn:F4} worst(px={badIdx / cout} co={badIdx % cout} oy={badIdx / cout / ow} ox={badIdx / cout % ow})");
        // detail dump of worst pixel + a few edge pixels
        Console.WriteLine($"  oG.Len={oG.Length} wf.Len={wf.Length} bf.Len={bf.Length}");
        foreach (int px2 in new[] { 0, 1, 2, ow, ow + 1, oh * ow - 1, oh * ow - ow })
        {
            stage = $"px{px2} head";
            int co = 4;
            double acc = bf[co], accT = bf[co];
            Console.Write($"  px{px2} co4: ");
            var taps = new List<string>();
            int oy = px2 / ow, ox = px2 % ow;
            try
            {
                for (int dy = 0; dy < kH; dy++)
                for (int dx = 0; dx < kH; dx++)
                {
                    int iy = oy * sH + dy - pt;
                    int ix2 = ox * sH + dx - pl;
                    if (iy < 0 || iy >= H || ix2 < 0 || ix2 >= W) continue;
                    for (int ci = 0; ci < cin; ci++)
                    {
                        acc += inp[ci * H * W + iy * W + ix2]
                             * wf[co * cin * kH * kH + ci * kH * kH + dy * kH + dx];
                        // alternative: OIHW read as [co][dy][dx][ci]
                        accT += inp[ci * H * W + iy * W + ix2]
                              * wf[co * cin * kH * kH + dy * kH * cin + dx * cin + ci];
                    }
                    taps.Add($"({dy},{dx})");
                }
            }
            catch (Exception ex) { Console.WriteLine($"EX {ex.Message} oy={oy} ox={ox}"); }
            stage = $"px{px2} tail";
            Console.WriteLine($"cpu={acc:F4} cpuT={accT:F4} real={t0cpu[co * oh * ow + px2]:F4} gpu={oG[px2 * cout + co]:F4} taps=[{string.Join(" ", taps)}]");
        }
        // dump im2col A rows for key pixels (K=27, Kp=32) — needs TRUNCATE so A is n0's
        stage = "i2off";
        long i2off = gg.Im2colOffset([1, 3, H, W]);
        int K = cin * kH * kH, Kp = (K + 15) / 16 * 16;
        foreach (int px3 in new[] { 0, badIdx / cout, 480, 481 })
        {
            stage = $"arow px{px3}";
            float[] arow = gg.DebugValuesRaw([1, 3, H, W], i2off + (long)px3 * Kp, Kp);
            int oy3 = px3 / ow, ox3 = px3 % ow;
            var exp = new float[Kp];
            for (int dy = 0; dy < kH; dy++)
            for (int dx = 0; dx < kH; dx++)
            {
                int iy = oy3 * sH + dy - pt, ix = ox3 * sH + dx - pl;
                for (int ci = 0; ci < cin; ci++)
                    exp[(dy * kH + dx) * cin + ci] =
                        (iy >= 0 && iy < H && ix >= 0 && ix < W)
                            ? inp[ci * H * W + iy * W + ix] : 0f;
            }
            double amax = 0; int aidx = -1;
            for (int k = 0; k < Kp; k++)
            { double d = Math.Abs(arow[k] - exp[k]); if (d > amax) { amax = d; aidx = k; } }
            Console.WriteLine($"  A px{px3} (oy{oy3} ox{ox3}): maxDiff={amax:F4} k={aidx}");
            Console.WriteLine($"    gpu=[{string.Join(",", arow.Select(v => v.ToString("F2")))}]");
            Console.WriteLine($"    exp=[{string.Join(",", exp.Select(v => v.ToString("F2")))}]");
        }
        stage = "done";
    }
    catch (Exception ex) { Console.WriteLine($"n0check EX at {stage}: {ex}"); }
    }
    if (args.Any(a => a == "--pernode"))
    {
        var traces = cpu.Trace(inp);
        for (int ni = 0; ni < mdl.Nodes.Length; ni++)
        {
            int skip = compiled.FusedSkip(ni);
            int sinkNode = ni + skip;
            int outT = checked((int)mdl.Nodes[sinkNode].Outputs[0]);
            var tc = traces[sinkNode];
            var gs = gg.DebugStats(outT, [1, 3, H, W]);
            double scale = Math.Max(Math.Abs(tc.Maximum), Math.Abs(tc.Minimum));
            bool diverged = scale > 1e-6 &&
                (Math.Abs(gs.Mean - tc.Mean) > 0.02 * scale + 1e-3 ||
                 Math.Abs(gs.Max - tc.Maximum) > 0.1 * scale);
            Console.WriteLine($"n{ni,3} {mdl.Nodes[ni].OpType,-12} skip={skip} out={outT,3} " +
                $"cpu[{tc.Minimum,9:F3},{tc.Maximum,9:F3},{tc.Mean,9:F4}] gpu[{gs.Min,9:F3},{gs.Max,9:F3},{gs.Mean,9:F4}] {(diverged ? "<<< DIVERGED" : "")}");
            ni += skip;
        }
    }

    if (Environment.GetEnvironmentVariable("SIMD_OCR_GPU_DUMP") == "1")
        Console.WriteLine("gpu[0..8]=" + string.Join(' ', probGpu.Take(8).Select(v => v.ToString("F4")))
            + "  cpu[0..8]=" + string.Join(' ', probCpu.Take(8).Select(v => v.ToString("F4"))));

    double maxAbs = 0, meanAbs = 0; int big = 0;
    for (int i = 0; i < probCpu.Length; i++)
    {
        double d = Math.Abs(probGpu[i] - probCpu[i]);
        maxAbs = Math.Max(maxAbs, d); meanAbs += d;
        if (d > 0.02) big++;
    }
    meanAbs /= probCpu.Length;
    // thresholded-map IoU at 0.3 (DET's actual decision boundary)
    double inter = 0, uni = 0;
    for (int i = 0; i < probCpu.Length; i++)
    {
        bool a = probCpu[i] > 0.3f, b = probGpu[i] > 0.3f;
        if (a && b) inter++; if (a || b) uni++;
    }
    Console.WriteLine($"out n={probCpu.Length} maxAbs={maxAbs:F4} meanAbs={meanAbs:E3} big>{0.02}={big} iou@0.3={inter / Math.Max(uni, 1):F4}");

    // timing loop
    const int reps = 10;
    double cpuBest = 1e9, gpuBest = 1e9;
    for (int r = 0; r < reps + 2; r++)
    {
        t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var o1 = new float[H * W];
        cpu.Run(inp, o1);
        double ms = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) / (double)System.Diagnostics.Stopwatch.Frequency * 1000;
        if (r >= 2) cpuBest = Math.Min(cpuBest, ms);

        t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var o2 = gg.Run([1, 3, H, W], inp);
        ms = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) / (double)System.Diagnostics.Stopwatch.Frequency * 1000;
        if (r >= 2) gpuBest = Math.Min(gpuBest, ms);
    }
    Console.WriteLine($"cpu best={cpuBest:F2} ms   gpu best={gpuBest:F2} ms   speedup={cpuBest / gpuBest:F2}x");
    gg.DumpProfile([1, 3, H, W]);
    return 0;
}

// --graph <model.onnx> <H> <W>: dump node list with resolved shapes, then exit
if (args.Length >= 4 && args[0] == "--graph")
{
    var model = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
    var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(model, intraOpThreads: 4);
    int[] gshape = args.Length >= 6
        ? [int.Parse(args[2]), int.Parse(args[3]), int.Parse(args[4]), int.Parse(args[5])]
        : [1, 3, int.Parse(args[2]), int.Parse(args[3])];
    int[][] rshapes = compiled.ResolveShapesFor(gshape);
    Console.WriteLine($"nodes={model.Nodes.Length} tensors={model.Tensors.Length} in={model.GraphInputs[0]} out={model.GraphOutputs[0]}");
    for (int ni = 0; ni < model.Nodes.Length; ni++)
    {
        var n = model.Nodes[ni];
        string ins = string.Join(",", n.Inputs.Select(t => t == uint.MaxValue ? "-" : $"{t}:[{string.Join('x', rshapes[t])}]"));
        string outs = string.Join(",", n.Outputs.Select(t => $"{t}:[{string.Join('x', rshapes[t])}]"));
        string fused = compiled.FusedSkip(ni) > 0 ? $" FUSED_SKIP={compiled.FusedSkip(ni)}" : "";
        string nhwc = compiled.IsNhwcNode(ni) ? " NHWC" : "";
        Console.WriteLine($"[{ni,3}] {n.Operator,-18} {ins} -> {outs}{fused}{nhwc}");
    }
    return 0;
}

// --rec <rec.onnx> <n> <W>: REC graph GPU vs CPU. GPU runs nodes < the vocab
// projection (MatMul) and reads back the [n,T,C] activations; compare vs the
// CPU TryRunUntilCtcProjection activations + per-row argmax after the shared
// CPU projection.
// --pipe <det.onnx> <cls.onnx|-> <rec.onnx> <keys.txt> <img> [cpu|vulkan|auto]:
// full PaddleOcrAll E2E — prints per-line text+score and wall time, so the
// GPU backend path can be diffed against CPU end to end.
if (args.Length >= 5 && args[0] == "--pipe")
{
    string detPath = args[1], clsPath = args[2], recPath = args[3], keysPath = args[4], imgPath = args[5];
    var backend = args.Length > 6 && Enum.TryParse<Sdcb.SimdPaddleOCR.OcrBackend>(args[6], true, out var bb)
        ? bb : Sdcb.SimdPaddleOCR.OcrBackend.Auto;
    var opts = new Sdcb.SimdPaddleOCR.PaddleOcrOptions
    {
        UseDirectionClassification = clsPath != "-",
        Detector = new Sdcb.SimdPaddleOCR.PaddleOcrDetectorOptions { Backend = backend },
        Recognizer = new Sdcb.SimdPaddleOCR.PaddleOcrRecognizerOptions { Backend = backend },
    };
    var detM = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(detPath));
    var recM = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(recPath));
    Sdcb.SimdPaddleOCR.OnnxSharp.Model? clsM = clsPath == "-" ? null
        : Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(clsPath));
    using var ocr = new Sdcb.SimdPaddleOCR.PaddleOcrAll(detM, clsM, recM, File.ReadAllBytes(keysPath), opts);

    using var bmp = new System.Drawing.Bitmap(imgPath);
    var bd = bmp.LockBits(new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height),
        System.Drawing.Imaging.ImageLockMode.ReadOnly,
        System.Drawing.Imaging.PixelFormat.Format24bppRgb);
    byte[] bgr = new byte[bmp.Height * bd.Stride];
    Marshal.Copy(bd.Scan0, bgr, 0, bgr.Length);
    bmp.UnlockBits(bd);

    bool prof = Environment.GetEnvironmentVariable("SIMD_OCR_PROF") == "1";
    Sdcb.SimdPaddleOCR.PipelineProfiler.Enable(prof);
    for (int rep = 0; rep < 3; rep++)
    {
        if (prof) Sdcb.SimdPaddleOCR.PipelineProfiler.Enable(true);   // reset per rep
        var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var r = ocr.Run(bgr, bmp.Width, bmp.Height, bd.Stride,
            Sdcb.SimdPaddleOCR.ImagePixelFormat.Bgr24);
        double ms = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) /
            (double)System.Diagnostics.Stopwatch.Frequency * 1000;
        if (rep == 0)
            foreach (var l in r.Lines) Console.WriteLine($"  [{l.RecognitionScore:F2}] {l.Text}");
        Console.WriteLine($"rep{rep}: {r.Lines.Length} lines {ms:F1}ms");
        if (prof)
            foreach (var (s, i) in Sdcb.SimdPaddleOCR.PipelineProfiler.Snapshot().Select((v, i) => (v, i))
                         .Where(x => x.v.Milliseconds > 0.3))
                Console.WriteLine($"    prof {Sdcb.SimdPaddleOCR.PipelineProfiler.StageNames[i],-16} {s.Milliseconds,8:F1}ms x{s.Calls}");
    }
    return 0;
}

// --recreal <det.onnx> <rec.onnx> <img> [W]: real-image E2E check — CPU det
// boxes → perspective crops → same normalized [n,3,48,W] input to CPU and GPU
// rec graphs → CTC vocab-argmax equality (the actual decode driver).
if (args.Length >= 4 && args[0] == "--recreal")
{
    var detMdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
    var recMdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[2]));
    int W = args.Length > 4 ? int.Parse(args[4]) : 640;
    using var bmp = new System.Drawing.Bitmap(args[3]);
    var bd = bmp.LockBits(new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height),
        System.Drawing.Imaging.ImageLockMode.ReadOnly,
        System.Drawing.Imaging.PixelFormat.Format24bppRgb);
    byte[] bgr = new byte[bmp.Height * bd.Stride];
    Marshal.Copy(bd.Scan0, bgr, 0, bgr.Length);
    bmp.UnlockBits(bd);

    using var det = new Sdcb.SimdPaddleOCR.PaddleOcrDetector(detMdl, intraOpThreads: 8);
    var detRes = det.Detect(bgr, bmp.Width, bmp.Height, bd.Stride,
        Sdcb.SimdPaddleOCR.ImagePixelFormat.Bgr24);
    int n = detRes.Boxes.Length;
    Console.WriteLine($"det: {n} boxes on {bmp.Width}x{bmp.Height}, W={W}");

    float[] inp = new float[n * 3 * 48 * W];
    int sampleLen = 3 * 48 * W;
    for (int k = 0; k < n; k++)
    {
        byte[] crop = Sdcb.SimdPaddleOCR.PPOCRCrop.Extract(bgr, bmp.Width, bmp.Height,
            bd.Stride, detRes.Boxes[k], out int cw, out int ch,
            Sdcb.SimdPaddleOCR.ImagePixelFormat.Bgr24);
        Sdcb.SimdPaddleOCR.PPOCRPreprocess.Rec(crop, cw, ch, cw * 3, W,
            inp.AsSpan(k * sampleLen, sampleLen));
    }

    var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(recMdl, intraOpThreads: 4);
    var cpu = new Sdcb.SimdPaddleOCR.OnnxSharp.InferenceSession(compiled);
    int[] shape = [n, 3, 48, W];
    cpu.Reshape(shape);
    if (!cpu.TryRunUntilCtcProjection(inp,
            out Sdcb.SimdPaddleOCR.OnnxSharp.CtcProjectionOperands ops))
    { Console.Error.WriteLine("cpu: CTC projection tail not found"); return 2; }
    int matMulIdx = ops.MatMulNodeIndex;
    uint actTensor = recMdl.Nodes[matMulIdx].Inputs[0];
    float[] cpuAct = ops.Activations.ToArray();

    using var ddev = Sdcb.SimdPaddleOCR.Backends.Vulkan.VkDevice.Create();
    using var gg = new Sdcb.SimdPaddleOCR.Backends.Vulkan.GpuDetGraph(ddev, compiled);
    float[] gpuAct = gg.Run(shape, inp, nodeLimit: matMulIdx,
        outTensor: checked((int)actTensor));
    double md = 0;
    for (int i = 0; i < cpuAct.Length; i++) md = Math.Max(md, Math.Abs(gpuAct[i] - cpuAct[i]));

    int rowCount = ops.RowCount;
    int[] idxCpu = new int[rowCount], idxGpu = new int[rowCount];
    float[] scCpu = new float[rowCount], scGpu = new float[rowCount];
    Sdcb.SimdPaddleOCR.Kernels.MatMul.TryArgMax(cpuAct, ops.Weights, ops.Bias,
        idxCpu, scCpu, ops.Batch, ops.Rows, ops.Inner, ops.Columns, ops.PackedWeights, 4);
    Sdcb.SimdPaddleOCR.Kernels.MatMul.TryArgMax(gpuAct, ops.Weights, ops.Bias,
        idxGpu, scGpu, ops.Batch, ops.Rows, ops.Inner, ops.Columns, ops.PackedWeights, 4);
    int bad = 0; double worstDelta = 0;
    for (int r = 0; r < rowCount; r++)
        if (idxCpu[r] != idxGpu[r])
        { bad++; worstDelta = Math.Max(worstDelta, Math.Abs(scCpu[r] - scGpu[r])); }
    // non-blank rows only matter for text; count flips where cpu argmax != blank(0)
    int badText = 0;
    for (int r = 0; r < rowCount; r++)
        if (idxCpu[r] != idxGpu[r] && idxCpu[r] != 0) badText++;
    Console.WriteLine($"act n={cpuAct.Length} maxAbs={md:F4}  vocabArgmaxBad={bad}/{rowCount} (non-blank={badText}) worstScoreDelta={worstDelta:F3}");

    // greedy CTC decode both index streams → per-line text equality
    string dictPath = Path.Combine(Path.GetDirectoryName(args[2])!, "rec_keys.txt");
    if (!File.Exists(dictPath)) dictPath = Path.Combine(Path.GetDirectoryName(args[2])!, "ppocr_keys.txt");
    string[] labels = File.ReadAllLines(dictPath);
    int T = ops.Rows;
    int textDiff = 0;
    for (int k = 0; k < n; k++)
    {
        string tc = DecodeGreedy(labels, idxCpu, k * T, T);
        string tg = DecodeGreedy(labels, idxGpu, k * T, T);
        if (tc != tg)
        {
            textDiff++;
            Console.WriteLine($"  line {k}: cpu=\"{tc}\" gpu=\"{tg}\"");
        }
    }
    Console.WriteLine($"textDiff={textDiff}/{n} lines");
    return 0;

    static string DecodeGreedy(string[] labels, int[] idx, int start, int T)
    {
        var sb = new System.Text.StringBuilder();
        int prev = 0;
        for (int t = 0; t < T; t++)
        {
            int b = idx[start + t];
            if (b != 0 && (t == 0 || b != prev))
                sb.Append(b == labels.Length + 1 ? ' ' : labels[b - 1]);
            prev = b;
        }
        return sb.ToString();
    }
}

if (args.Length >= 4 && args[0] == "--rec")
{
    var mdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
    var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(mdl, intraOpThreads: 4);
    int nb = int.Parse(args[2]), W = int.Parse(args[3]);
    int[] shape = [nb, 3, 48, W];

    var rr = new Random(20260924);
    var inp = new float[3 * 48 * W * nb];
    for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rr.NextDouble() * 2 - 1);

    var cpu = new Sdcb.SimdPaddleOCR.OnnxSharp.InferenceSession(compiled);
    cpu.Reshape(shape);
    if (!cpu.TryRunUntilCtcProjection(inp,
            out Sdcb.SimdPaddleOCR.OnnxSharp.CtcProjectionOperands ops))
    { Console.Error.WriteLine("cpu: CTC projection tail not found"); return 2; }
    int matMulIdx = ops.MatMulNodeIndex;
    uint actTensor = mdl.Nodes[matMulIdx].Inputs[0];
    float[] cpuAct = ops.Activations.ToArray();
    Console.WriteLine($"rec: batch={nb} T={ops.Rows} C={ops.Inner} " +
        $"cutoff=node{matMulIdx} actTensor=t{actTensor} actLen={cpuAct.Length}");

    using var ddev = Sdcb.SimdPaddleOCR.Backends.Vulkan.VkDevice.Create();
    using var gg = new Sdcb.SimdPaddleOCR.Backends.Vulkan.GpuDetGraph(ddev, compiled);
    var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
    float[] gpuAct = gg.Run(shape, inp, nodeLimit: matMulIdx,
        outTensor: checked((int)actTensor));
    Console.WriteLine($"gpu first run {(System.Diagnostics.Stopwatch.GetTimestamp() - t0) / (double)System.Diagnostics.Stopwatch.Frequency * 1000:F1} ms");

    if (gpuAct.Length != cpuAct.Length)
    { Console.Error.WriteLine($"len mismatch gpu={gpuAct.Length} cpu={cpuAct.Length}"); return 3; }
    double maxAbs = 0, meanAbs = 0; int big = 0, argmaxBad = 0;
    for (int i = 0; i < cpuAct.Length; i++)
    {
        double d = Math.Abs(gpuAct[i] - cpuAct[i]);
        maxAbs = Math.Max(maxAbs, d); meanAbs += d;
        if (d > 0.05) big++;
    }
    meanAbs /= cpuAct.Length;
    // per-row argmax on activations (pre-projection feature argmax is a proxy
    // for divergence; the projection itself is shared CPU code)
    int T = ops.Rows, C = ops.Inner;
    double worstFlipMargin = 0;
    for (int r = 0; r < nb * T; r++)
    {
        int ca = 0, cb = 0;
        for (int c = 1; c < C; c++)
        {
            if (cpuAct[r * C + c] > cpuAct[r * C + ca]) ca = c;
            if (gpuAct[r * C + c] > gpuAct[r * C + cb]) cb = c;
        }
        if (ca != cb)
        {
            argmaxBad++;
            // margin between cpu's top-2 at this row — flips on huge margins
            // are real bugs; tiny margins are fp16 noise ties
            double top = cpuAct[r * C + ca], second = double.MinValue;
            for (int c = 0; c < C; c++)
                if (c != ca && cpuAct[r * C + c] > second) second = cpuAct[r * C + c];
            worstFlipMargin = Math.Max(worstFlipMargin, top - second);
        }
    }
    Console.WriteLine($"  worstFlipMargin={worstFlipMargin:F4}");
    Console.WriteLine($"act n={cpuAct.Length} maxAbs={maxAbs:F4} meanAbs={meanAbs:E3} " +
        $"big>0.05={big} argmaxBad={argmaxBad}/{nb * T}");

    // real metric: vocab argmax through the CTC projection (the same
    // MatMul.TryArgMax the recognizer uses)
    int rowCount = ops.RowCount;
    int[] idxCpu = new int[rowCount], idxGpu = new int[rowCount];
    float[] scCpu = new float[rowCount], scGpu = new float[rowCount];
    bool okc = Sdcb.SimdPaddleOCR.Kernels.MatMul.TryArgMax(cpuAct, ops.Weights, ops.Bias,
        idxCpu, scCpu, ops.Batch, ops.Rows, ops.Inner, ops.Columns, ops.PackedWeights, 4);
    bool okg = Sdcb.SimdPaddleOCR.Kernels.MatMul.TryArgMax(gpuAct, ops.Weights, ops.Bias,
        idxGpu, scGpu, ops.Batch, ops.Rows, ops.Inner, ops.Columns, ops.PackedWeights, 4);
    if (!okc || !okg) { Console.Error.WriteLine($"TryArgMax failed c={okc} g={okg}"); return 4; }
    int vocabBad = 0; double worstVocabMargin = 0;
    for (int r = 0; r < rowCount; r++)
        if (idxCpu[r] != idxGpu[r])
        {
            vocabBad++;
            worstVocabMargin = Math.Max(worstVocabMargin, Math.Abs(scCpu[r] - scGpu[r]));
        }
    Console.WriteLine($"vocabArgmaxBad={vocabBad}/{rowCount} worstScoreDelta={worstVocabMargin:F3}");

    if (args.Any(a => a == "--pernode"))
    {
        var traces = cpu.Trace(inp);
        for (int ni = 0; ni < matMulIdx; ni++)
        {
            int skip = compiled.FusedSkip(ni);
            int sinkNode = ni + skip;
            int outT = checked((int)mdl.Nodes[sinkNode].Outputs[0]);
            var tc = traces[sinkNode];
            // elementwise: cpu TraceNodeValues for NHWC nodes is NHWC-flat —
            // same order as the gpu arena slot
            float[] cv = cpu.TraceNodeValues(inp, sinkNode);
            bool nhwc = compiled.IsNhwcNode(sinkNode);
            double maxD = 0;
            if (nhwc)
            {
                float[] gv = gg.DebugValues(outT, shape, cv.Length,
                    nodeLimit: matMulIdx, outTensor: checked((int)actTensor));
                if (gv.Length == cv.Length)
                    for (int i = 0; i < cv.Length; i++)
                        maxD = Math.Max(maxD, Math.Abs(gv[i] - cv[i]));
            }
            bool diverged = maxD > 0.05;
            Console.WriteLine($"n{ni,3} {mdl.Nodes[ni].OpType,-18} skip={skip} out={outT,3} " +
                $"cpu[{tc.Minimum,9:F3},{tc.Maximum,9:F3},{tc.Mean,9:F4}] maxAbs={(nhwc ? maxD.ToString("F4") : "  n/a")} {(diverged ? "<<< DIVERGED" : "")}");
            ni += skip;
        }
    }
    if (Array.IndexOf(args, "--tval") is int tvi && tvi >= 0)
        foreach (string s in args.Skip(tvi + 1).TakeWhile(a => !a.StartsWith("-")))
        {
            int nn = int.Parse(s);
            float[] cv2 = cpu.TraceNodeValues(inp, nn);
            int t2 = checked((int)mdl.Nodes[nn].Outputs[0]);
            int perImg = cv2.Length / Math.Max(1, nb);
            for (int b = 0; b < Math.Min(nb, 4); b++)
            {
                float[] gv2 = gg.DebugValuesRaw(shape,
                    gg.DebugElemOff(t2, shape, nodeLimit: matMulIdx,
                        outTensor: checked((int)actTensor)) + (long)b * perImg, 8,
                    nodeLimit: matMulIdx, outTensor: checked((int)actTensor));
                Console.WriteLine($"t{t2} n{nn} b{b}: cpu=[{string.Join(",", cv2.Skip(b * perImg).Take(8).Select(v => v.ToString("F3")))}]");
                Console.WriteLine($"           gpu=[{string.Join(",", gv2.Select(v => v.ToString("F3")))}]");
            }
        }

    const int reps = 10;
    double cpuBest = 1e9, gpuBest = 1e9;
    for (int r = 0; r < reps + 2; r++)
    {
        t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        cpu.TryRunUntilCtcProjection(inp, out _);
        double ms = (System.Diagnostics.Stopwatch.GetTimestamp() - t0)
            / (double)System.Diagnostics.Stopwatch.Frequency * 1000;
        if (r >= 2) cpuBest = Math.Min(cpuBest, ms);

        t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        gg.Run(shape, inp, nodeLimit: matMulIdx, outTensor: checked((int)actTensor));
        ms = (System.Diagnostics.Stopwatch.GetTimestamp() - t0)
            / (double)System.Diagnostics.Stopwatch.Frequency * 1000;
        if (r >= 2) gpuBest = Math.Min(gpuBest, ms);
    }
    Console.WriteLine($"cpu best={cpuBest:F2} ms   gpu best={gpuBest:F2} ms   " +
        $"speedup={cpuBest / gpuBest:F2}x   per-line cpu={cpuBest / nb:F3} gpu={gpuBest / nb:F3}");
    return 0;
}

var asm = typeof(VkDevice).Assembly;
byte[] LoadSpv(string name)
{
    string res = $"Sdcb.SimdPaddleOCR.Backends.Vulkan.Shaders.{name}.spv";
    using Stream s = asm.GetManifestResourceStream(res)
        ?? throw new FileNotFoundException(res);
    byte[] b = new byte[s.Length];
    s.ReadExactly(b);
    return b;
}

using var dev = VkDevice.Create();
Console.WriteLine($"device: {dev.DeviceName} vendor=0x{dev.VendorId:x} sg={dev.SubgroupSize} " +
    $"coopmat={dev.CoopMatrix} push={dev.PushDescriptors} rebar={dev.CoherentDeviceLocal}");
if (!dev.CoopMatrix) { Console.Error.WriteLine("no VK_KHR_cooperative_matrix"); return 2; }

// ---- WC write bandwidth (device-local HOST_VISIBLE on ReBAR) ----
const ulong wcBytes = 64UL << 20;
unsafe
{
    var wc = dev.NewStorageBuffer(wcBytes, hostVisible: true);
    float* p = (float*)wc.Map();
    var wcSpan = new Span<float>(p, (int)(wcBytes / 4));
    // write bandwidth
    long best = long.MaxValue;
    for (int r = 0; r < 3; r++)
    {
        long t0 = Stopwatch.GetTimestamp();
        for (int i = 0; i < wcSpan.Length; i++) wcSpan[i] = i;
        best = Math.Min(best, Stopwatch.GetTimestamp() - t0);
    }
    double gbps = wcBytes / (best / (double)Stopwatch.Frequency) / 1e9;
    Console.WriteLine($"WC write (dev-local host-visible): {gbps:F1} GB/s");

    float[] ram = new float[(int)(wcBytes / 4)];
    best = long.MaxValue;
    for (int r = 0; r < 3; r++)
    {
        long t0 = Stopwatch.GetTimestamp();
        for (int i = 0; i < ram.Length; i++) ram[i] = i;
        best = Math.Min(best, Stopwatch.GetTimestamp() - t0);
    }
    Console.WriteLine($"RAM write baseline:               {wcBytes / (best / (double)Stopwatch.Frequency) / 1e9:F1} GB/s");

    // read bandwidth from mapped VRAM (uncached WC reads are notoriously slow)
    float sink = 0;
    best = long.MaxValue;
    for (int r = 0; r < 3; r++)
    {
        long t0 = Stopwatch.GetTimestamp();
        for (int i = 0; i < wcSpan.Length; i += 16) sink += wcSpan[i];
        best = Math.Min(best, Stopwatch.GetTimestamp() - t0);
    }
    Console.WriteLine($"WC read  (dev-local host-visible): {wcBytes / (best / (double)Stopwatch.Frequency) / 1e9:F1} GB/s (sink={sink:E1})");
}

// ---- conv1x1 GEMM: C[M,N] = X[M,K] * W[N,K]^T, X/W fp16 ----
var pipe = dev.NewPipeline(dev.NewShaderModule(LoadSpv("ocr_gemm_cm")), bindings: 3, pushConstBytes: 16,
    requiredSubgroupSize: 16);
var set = dev.NewDescriptorSet(pipe.SetLayout);

(int M, int K, int N)[] shapes =
{
    (512 * 512, 64, 64),     // tiny det mid-stage
    (256 * 256, 128, 128),   // small/medium mid-stage
    (128 * 128, 256, 256),   // medium deep stage
    (960 * 960, 32, 64),     // wide shallow
};

var rng = new Random(42);
bool allOk = true;
if (args.Contains("--diag"))
{
    // W = identity → C should equal X
    shapes = [(128, 16, 16)];
}
for (int ai = 0; ai < args.Length; ai++)
    if (args[ai] == "--shapes")
        shapes = args[++ai].Split(';')
            .Select(s => s.Split(',').Select(int.Parse).ToArray())
            .Select(a => (a[0], a[1], a[2])).ToArray();
foreach ((int M, int K, int N) in shapes)
{
    int mpad = (M + 127) / 128 * 128;   // X reads pad rows inside edge tiles
    var xF = new float[M * K];
    var wF = new float[N * K];
    var xF16 = new Half[mpad * K];
    var wF16 = new Half[N * K];
    for (int i = 0; i < xF.Length; i++) { xF[i] = (float)(rng.NextDouble() * 2 - 1); xF16[i] = (Half)xF[i]; }
    for (int i = 0; i < wF.Length; i++) { wF[i] = (float)(rng.NextDouble() * 2 - 1) / MathF.Sqrt(K); wF16[i] = (Half)wF[i]; }
    if (args.Contains("--diag"))
    {
        for (int i = 0; i < wF.Length; i++) { wF[i] = 0; wF16[i] = Half.Zero; }
        for (int i = 0; i < N && i < K; i++) { wF[i * K + i] = 1; wF16[i * K + i] = Half.One; }
    }

    // CPU reference (uses the repo's Conv1x1 on NCHW view; GEMM result is
    // layout-equivalent to NHWC, so compare via a plain fp32 reference GEMM
    // instead — keeps the POC independent of conv layout conventions).
    var cRef = new float[M * N];
    CpuRefGemm(xF, wF, cRef, M, K, N);

    var bx = dev.NewStorageBuffer((ulong)(mpad * K) * 2, hostVisible: false);
    var bw = dev.NewStorageBuffer((ulong)(N * K) * 2, hostVisible: false);
    var bc = dev.NewStorageBuffer((ulong)(M * N) * 4, hostVisible: false);            // device-local like real intermediates
    var br = dev.NewStorageBuffer((ulong)(M * N) * 4, hostVisible: true, preferHost: true); // readback only
    unsafe
    {
        fixed (Half* px = xF16) dev.Upload(bx, px, (ulong)xF16.Length * 2);
        fixed (Half* pw = wF16) dev.Upload(bw, pw, (ulong)wF16.Length * 2);
    }
    dev.BindBuffer(set, 0, bx); dev.BindBuffer(set, 1, bw); dev.BindBuffer(set, 2, bc);

    IntPtr cmd = dev.NewCommandBuffer();
    IntPtr cmdRd = dev.NewCommandBuffer();
    IntPtr fence = dev.NewFence();
    uint gx = (uint)(mpad / 128), gy = (uint)((N + 127) / 128);
    uint* pc = stackalloc uint[] { (uint)M, (uint)N, (uint)K, 0u };
    unsafe
    {
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "begin");
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, pipe.Pipeline);
        Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
        Vk.vkCmdPushConstants(cmd, pipe.Layout, VkConst.StageComputeShader, 0, 16, pc);
        Vk.vkCmdDispatch(cmd, gx, gy, 1);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "end");

        Vk.Check(Vk.vkBeginCommandBuffer(cmdRd, &begin), "begin");
        Vk.VkBufferCopy r = new() { Size = (ulong)(M * N) * 4 };
        Vk.vkCmdCopyBuffer(cmdRd, bc.Buffer, br.Buffer, 1, &r);
        Vk.Check(Vk.vkEndCommandBuffer(cmdRd), "end");
    }

    // correctness: run gemm then copy device-local C into the readback buffer
    dev.Submit(cmd, fence); dev.WaitFence(fence);
    dev.Submit(cmdRd, fence); dev.WaitFence(fence);
    var gpu = new float[M * N];
    unsafe
    {
        float* cp = (float*)br.Map();
        new ReadOnlySpan<float>(cp, gpu.Length).CopyTo(gpu);
        br.Unmap();
    }
    double maxRel = 0, maxAbs = 0;
    int nBad = 0;
    for (int i = 0; i < gpu.Length; i++)
    {
        double abs = Math.Abs(gpu[i] - cRef[i]);
        double rel = abs / Math.Max(0.1, Math.Abs(cRef[i]));  // fp16 inputs: tiny denominators are noise
        maxRel = Math.Max(maxRel, rel);
        maxAbs = Math.Max(maxAbs, abs);
        if (abs > 0.02 + 0.02 * Math.Abs(cRef[i])) nBad++;
    }
    if (args.Contains("--diag"))
    {
        for (int i = 0; i < 32; i++)
            Console.WriteLine($"  [{i,3}] gpu={gpu[i],9:F4} ref={cRef[i],9:F4}");
    }
    Console.WriteLine($"  badElems={nBad}/{gpu.Length}");
    if (nBad > 0)
    {
        int shown = 0;
        for (int i = 0; i < gpu.Length && shown < 24; i++)
        {
            double abs = Math.Abs(gpu[i] - cRef[i]);
            if (abs > 0.02 + 0.02 * Math.Abs(cRef[i]))
            {
                Console.WriteLine($"  bad m={i / N,4} n={i % N,4} gpu={gpu[i],9:F4} ref={cRef[i],9:F4}");
                shown++;
            }
        }
    }

    // timing: GPU (submit+wait), CPU reference on 4 threads via repo kernel is
    // layout-bound; use direct Conv1x1.Try on NCHW for the honest comparison.
    const int reps = 30;
    long tBest = long.MaxValue;
    for (int r = 0; r < reps + 5; r++)
    {
        long t0 = Stopwatch.GetTimestamp();
        dev.Submit(cmd, fence); dev.WaitFence(fence);
        if (r >= 5) tBest = Math.Min(tBest, Stopwatch.GetTimestamp() - t0);
    }
    double gpuMs = tBest / (double)Stopwatch.Frequency * 1000;
    double gflops = 2.0 * M * N * K / (gpuMs / 1000) / 1e9;

    Console.WriteLine($"conv1x1 M={M,7} K={K,4} N={N,4}: gpu={gpuMs,7:F3} ms ({gflops,7:F1} GFLOPS)  maxRelErr={maxRel:E2} maxAbsErr={maxAbs:E2}");
    if (nBad > 0) allOk = false;
}

// CPU comparison on the same shapes via the real Conv1x1 kernel
foreach ((int M, int K, int N) in shapes)
{
    int side = (int)Math.Sqrt(M);
    var input = new float[K * M];   // NCHW [1,K,side,side]
    var wNchw = new float[N * K];   // OIHW [N,K,1,1]
    var outp = new float[N * M];
    var rng2 = new Random(42);
    for (int i = 0; i < input.Length; i++) input[i] = (float)(rng2.NextDouble() * 2 - 1);
    for (int i = 0; i < wNchw.Length; i++) wNchw[i] = (float)(rng2.NextDouble() * 2 - 1) / MathF.Sqrt(K);
    if (!Conv1x1.Try(input, wNchw, [], outp, 1, K, side, side, N, 1, intraOpThreads: 4))
    {
        Console.WriteLine($"conv1x1 M={M}: CPU kernel declined this shape");
        continue;
    }
    // warm + timed
    long tBest = long.MaxValue;
    for (int r = 0; r < 12; r++)
    {
        long t0 = Stopwatch.GetTimestamp();
        Conv1x1.Try(input, wNchw, [], outp, 1, K, side, side, N, 1, intraOpThreads: 4);
        if (r >= 2) tBest = Math.Min(tBest, Stopwatch.GetTimestamp() - t0);
    }
    double cpuMs = tBest / (double)Stopwatch.Frequency * 1000;
    Console.WriteLine($"         cpu4t={cpuMs,7:F3} ms  ({2.0 * M * N * K / (cpuMs / 1000) / 1e9,7:F1} GFLOPS)");
}

return allOk ? 0 : 1;

static void CpuRefGemm(float[] x, float[] w, float[] c, int m, int k, int n)
{
    // c[m,n] = sum_k x[m,k] * w[n,k] — multithreaded over m, fp32.
    Parallel.For(0, m, row =>
    {
        for (int j = 0; j < n; j++)
        {
            double acc = 0;
            int wbase = j * k, xbase = row * k;
            for (int kk = 0; kk < k; kk++) acc += (double)x[xbase + kk] * w[wbase + kk];
            c[row * n + j] = (float)acc;
        }
    });
}
