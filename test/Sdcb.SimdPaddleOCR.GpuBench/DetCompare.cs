using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>DET graph GPU vs CPU on one shape.</summary>
static class DetCompare
{
    internal static int Run(string[] args)
    {
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
            float[] probGpu = gg.Run([1, 3, H, W], inp).ToArray();
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
                bool elemwise = args.Any(a => a == "--elemdiff");
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
                    string extra = "";
                    if (elemwise && compiled.IsNhwcNode(sinkNode))
                    {
                        float[] cv = cpu.TraceNodeValues(inp, sinkNode);
                        float[] gv = gg.DebugValues(outT, [1, 3, H, W], cv.Length);
                        if (gv.Length == cv.Length)
                        {
                            double mx = 0; int mi = -1;
                            for (int i = 0; i < cv.Length; i++)
                            { double d = Math.Abs(gv[i] - cv[i]); if (d > mx) { mx = d; mi = i; } }
                            extra = $" elemMax={mx:F4} @{mi}";
                        }
                        else extra = $" elemMax=lenMismatch({gv.Length}vs{cv.Length})";
                    }
                    Console.WriteLine($"n{ni,3} {mdl.Nodes[ni].OpType,-12} skip={skip} out={outT,3} " +
                        $"cpu[{tc.Minimum,9:F3},{tc.Maximum,9:F3},{tc.Mean,9:F4}] gpu[{gs.Min,9:F3},{gs.Max,9:F3},{gs.Mean,9:F4}] {(diverged ? "<<< DIVERGED" : "")}{extra}");
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
        return Harness.Usage(2);
    }
}
