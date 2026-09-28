using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>REC graph GPU vs CPU up to the CTC projection.</summary>
static class RecCompare
{
    internal static int Run(string[] args)
    {
        // --rec <rec.onnx> <n> <W>: REC graph GPU vs CPU. GPU runs nodes < the vocab
        // projection (MatMul) and reads back the [n,T,C] activations; compare vs the
        // CPU TryRunUntilCtcProjection activations + per-row argmax after the shared
        // CPU projection.

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
            float[] gpuAct = gg.Run(shape, inp, nodeLimit: matMulIdx, outTensor: checked((int)actTensor)).ToArray();
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
            Console.WriteLine($"  cpuAct[0..8]=[{string.Join(",", cpuAct.Take(8).Select(v => v.ToString("F3")))}]");
            Console.WriteLine($"  gpuAct[0..8]=[{string.Join(",", gpuAct.Take(8).Select(v => v.ToString("F3")))}]");
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
                    int[] osh = compiled.ResolveShapesFor(shape)[outT];
                    if (!nhwc && osh.Length == 4)
                    {
                        // CPU NCHW -> GPU physical NHWC
                        int N0 = osh[0], C0 = osh[1], HW0 = osh[2] * osh[3];
                        float[] t = new float[cv.Length];
                        for (int b0 = 0; b0 < N0; b0++)
                            for (int c0 = 0; c0 < C0; c0++)
                                for (int q0 = 0; q0 < HW0; q0++)
                                    t[(b0 * HW0 + q0) * C0 + c0] = cv[(b0 * C0 + c0) * HW0 + q0];
                        cv = t;
                        nhwc = true;
                    }
                    else if (!nhwc) nhwc = true;   // rank!=4: GPU slot is row-major
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
        return Harness.Usage(2);
    }
}
