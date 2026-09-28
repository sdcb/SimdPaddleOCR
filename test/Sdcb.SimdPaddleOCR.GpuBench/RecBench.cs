using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>REC profile, session compare, and real-image crop checks.</summary>
static class RecBench
{
    internal static int Run(string[] args)
    {
        // --sessiso <rec.onnx> <W>: same random input through IOcrSession Cpu vs
        // Vulkan via OcrSessionFactory — compares ops.Activations elementwise.
        if (args.Length >= 3 && args[0] == "--sessiso")
        {
            var recMdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
            int W = int.Parse(args[2]);
            var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(recMdl, intraOpThreads: 4);
            int[] shape = [1, 3, 48, W];
            var rngIso = new Random(1234);
            float[] inp = new float[3 * 48 * W];
            for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rngIso.NextDouble() * 2 - 1);

            using var sCpu = Sdcb.SimdPaddleOCR.OnnxSharp.OcrSessionFactory.Create(compiled,
                Sdcb.SimdPaddleOCR.OcrBackend.Cpu);
            using var sGpu = Sdcb.SimdPaddleOCR.OnnxSharp.OcrSessionFactory.Create(compiled,
                Sdcb.SimdPaddleOCR.OcrBackend.Vulkan);
            sCpu.PlanForCtcProjection = true; sGpu.PlanForCtcProjection = true;
            sCpu.Reshape(shape); sGpu.Reshape(shape);
            inp.CopyTo(sCpu.InputData); inp.CopyTo(sGpu.InputData);
            bool okC = sCpu.TryRunUntilCtcProjection(sCpu.InputData, out var oc);
            bool okG = sGpu.TryRunUntilCtcProjection(sGpu.InputData, out var og);
            Console.WriteLine($"cpu ok={okC} act={oc.Activations.Length} gpu ok={okG} act={og.Activations.Length}");
            if (okC && okG)
            {
                double md = 0; int nz = 0;
                var ac = oc.Activations; var ag = og.Activations;
                for (int i = 0; i < Math.Min(ac.Length, ag.Length); i++)
                { md = Math.Max(md, Math.Abs(ac[i] - ag[i])); if (ag[i] != 0) nz++; }
                Console.WriteLine($"batch={oc.Batch}/{og.Batch} rows={oc.Rows}/{og.Rows} inner={oc.Inner}/{og.Inner} cols={oc.Columns}/{og.Columns}");
                Console.WriteLine($"maxAbs={md:F4} gpuNonZero={nz}/{ag.Length}");
            }
            return 0;
        }

        // --reciso <det.onnx> <rec.onnx> <keys> <img>: CPU det boxes → each crop run
        // through PaddleOcrRecognizer with Backend=Cpu then Backend=Vulkan; text diff.
        if (args.Length >= 5 && args[0] == "--reciso")
        {
            var detMdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
            var recMdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[2]));
            byte[] keys = File.ReadAllBytes(args[3]);
            using var bmp = new System.Drawing.Bitmap(args[4]);
            var bd = bmp.LockBits(new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height),
                System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            byte[] bgr = new byte[bmp.Height * bd.Stride];
            Marshal.Copy(bd.Scan0, bgr, 0, bgr.Length);
            bmp.UnlockBits(bd);
            using var det = new Sdcb.SimdPaddleOCR.PaddleOcrDetector(detMdl, intraOpThreads: 8);
            var detRes = det.Detect(bgr, bmp.Width, bmp.Height, bd.Stride,
                Sdcb.SimdPaddleOCR.ImagePixelFormat.Bgr24);
            using var recCpu = new Sdcb.SimdPaddleOCR.PaddleOcrRecognizer(recMdl, keys,
                new Sdcb.SimdPaddleOCR.PaddleOcrRecognizerOptions { Backend = Sdcb.SimdPaddleOCR.OcrBackend.Cpu });
            using var recGpu = new Sdcb.SimdPaddleOCR.PaddleOcrRecognizer(recMdl, keys,
                new Sdcb.SimdPaddleOCR.PaddleOcrRecognizerOptions { Backend = Sdcb.SimdPaddleOCR.OcrBackend.Vulkan });
            int diff = 0;
            for (int k = 0; k < detRes.Boxes.Length; k++)
            {
                byte[] crop = Sdcb.SimdPaddleOCR.PPOCRCrop.Extract(bgr, bmp.Width, bmp.Height,
                    bd.Stride, detRes.Boxes[k], out int cw, out int ch,
                    Sdcb.SimdPaddleOCR.ImagePixelFormat.Bgr24);
                var rc = recCpu.Recognize(crop, cw, ch, cw * 3, Sdcb.SimdPaddleOCR.ImagePixelFormat.Bgr24);
                var rg = recGpu.Recognize(crop, cw, ch, cw * 3, Sdcb.SimdPaddleOCR.ImagePixelFormat.Bgr24);
                string mark = rc.Text == rg.Text ? " " : "!";
                if (rc.Text != rg.Text) diff++;
                Console.WriteLine($"{mark} line{k} w{cw}: cpu=\"{rc.Text}\" gpu=\"{rg.Text}\" | cpu(rw={rc.ResizedWidth} tw={rc.TensorWidth} T={rc.TimeSteps} e={rc.EmittedCount}) gpu(rw={rg.ResizedWidth} tw={rg.TensorWidth} T={rg.TimeSteps} e={rg.EmittedCount})");
            }
            Console.WriteLine($"diff={diff}/{detRes.Boxes.Length}");
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
            float[] gpuAct = gg.Run(shape, inp, nodeLimit: matMulIdx, outTensor: checked((int)actTensor)).ToArray();
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

        // --recprof <rec.onnx> <nb> <W> [reps]: per-dispatch GPU profile of the REC
        // graph up to the CTC projection (set SIMD_OCR_GPU_PROF=1).
        if (args.Length >= 4 && args[0] == "--recprof")
        {
            var mdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
            var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(mdl, intraOpThreads: 8);
            int nb = int.Parse(args[2]), W = int.Parse(args[3]);
            int reps = args.Length >= 5 ? int.Parse(args[4]) : 20;
            int[] shape = [nb, 3, 48, W];
            var inp = new float[3 * 48 * W * nb];
            var rr = new Random(1);
            for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rr.NextDouble() * 2 - 1);
            var cpu = new Sdcb.SimdPaddleOCR.OnnxSharp.InferenceSession(compiled);
            cpu.Reshape(shape);
            cpu.TryRunUntilCtcProjection(inp, out Sdcb.SimdPaddleOCR.OnnxSharp.CtcProjectionOperands ops);
            int matMulIdx = ops.MatMulNodeIndex;
            int actTensor = checked((int)mdl.Nodes[matMulIdx].Inputs[0]);
            Console.WriteLine($"rec: batch={nb} T={ops.Rows} C={ops.Inner} cols={ops.Columns}");
            using var ddev = Sdcb.SimdPaddleOCR.Backends.Vulkan.VkDevice.Create();
            using var gg = new Sdcb.SimdPaddleOCR.Backends.Vulkan.GpuDetGraph(ddev, compiled);
            double best = double.MaxValue;
            for (int r = 0; r < reps; r++)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                gg.Run(shape, inp, nodeLimit: matMulIdx, outTensor: actTensor);
                best = Math.Min(best, System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }
            Console.WriteLine($"best wall {best:F2} ms, dispatches={gg.DispatchCount(shape, matMulIdx, actTensor)}");
            gg.DumpProfile(shape, matMulIdx, actTensor);
            return 0;
        }

        return Harness.Usage(2);
    }
}
