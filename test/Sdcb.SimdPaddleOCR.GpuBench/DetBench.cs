using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>DET idle gap, probability-map debug, and per-dispatch profile.</summary>
static class DetBench
{
    internal static int Run(string[] args)
    {
        // --idle <model.onnx> <H> <W>: run latency after various CPU idle gaps
        if (args.Length >= 4 && args[0] == "--idle")
        {
            var mdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
            var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(mdl, intraOpThreads: 4);
            int H = int.Parse(args[2]), W = int.Parse(args[3]);
            var inp = new float[3 * H * W];
            using var ddev = VkDevice.Create();
            using var gg = new GpuDetGraph(ddev, compiled);
            gg.Run([1, 3, H, W], inp);
            foreach (int gap in new[] { 0, 0, 1, 2, 5, 10, 20, 50, 100, 0, 0 })
            {
                if (gap > 0) Thread.Sleep(gap);
                long t0 = Stopwatch.GetTimestamp();
                gg.Run([1, 3, H, W], inp);
                Console.WriteLine($"gap={gap,4}ms run={Stopwatch.GetElapsedTime(t0).TotalMilliseconds:F2} ms");
            }
            var rnd = new Random(1);
            var lat = new List<double>();
            for (int i = 0; i < 150; i++)
            {
                Thread.Sleep(rnd.Next(0, 12));
                long t0 = Stopwatch.GetTimestamp();
                gg.Run([1, 3, H, W], inp);
                lat.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }
            lat.Sort();
            Console.WriteLine($"stats n=150 p10={lat[15]:F2} p50={lat[75]:F2} p75={lat[112]:F2} p90={lat[135]:F2} mean={lat.Average():F2}");
            // busy-spin gap (thread stays on CPU)
            foreach (int gap in new[] { 5, 20 })
            {
                long s0 = Stopwatch.GetTimestamp();
                while (Stopwatch.GetElapsedTime(s0).TotalMilliseconds < gap) { }
                long t0 = Stopwatch.GetTimestamp();
                gg.Run([1, 3, H, W], inp);
                Console.WriteLine($"spin={gap,4}ms run={Stopwatch.GetElapsedTime(t0).TotalMilliseconds:F2} ms");
            }
            return 0;
        }


        // --detmap <det.onnx> <img>: real-pipeline det debug — ImageSharp decode (same
        // as bench), identical PPOCRPreprocess.Det input into Cpu and Vulkan sessions,
        // prob-map diff + DbPostprocess box lists on both maps.
        if (args.Length >= 3 && args[0] == "--detmap")
        {
            string detPathDm = args[1];
            using var imageDm = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(args[2]);
            int iwDm = imageDm.Width, ihDm = imageDm.Height;
            byte[] bgrDm = new byte[iwDm * ihDm * 3];
            for (int y = 0; y < ihDm; y++)
                for (int x = 0; x < iwDm; x++)
                {
                    var px = imageDm[x, y];
                    int o = (y * iwDm + x) * 3;
                    bgrDm[o] = px.B; bgrDm[o + 1] = px.G; bgrDm[o + 2] = px.R;
                }
            var mdlDm = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(detPathDm));
            var compiledDm = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(mdlDm, intraOpThreads: 4);
            var sizeDm = Sdcb.SimdPaddleOCR.PPOCRPreprocess.ComputeDetSize(iwDm, ihDm, 960);
            int WD = sizeDm.Width, HD = sizeDm.Height;
            Console.WriteLine($"img {iwDm}x{ihDm} -> det {WD}x{HD}");
            var optsDm = new Sdcb.SimdPaddleOCR.PaddleOcrDetectorOptions { BoxThreshold = 0.4f };
            var scratchDm = new Sdcb.SimdPaddleOCR.DbPostprocess.Workspace();
            float[][] mapsDm = new float[2][];
            foreach (var (bi, bk) in new[] { (0, Sdcb.SimdPaddleOCR.OcrBackend.Cpu), (1, Sdcb.SimdPaddleOCR.OcrBackend.Vulkan) })
            {
                using var sessDm = Sdcb.SimdPaddleOCR.OnnxSharp.OcrSessionFactory.Create(compiledDm, bk);
                sessDm.Reshape([1, 3, HD, WD]);
                Sdcb.SimdPaddleOCR.PPOCRPreprocess.Det(bgrDm, iwDm, ihDm, iwDm * 3, WD, HD,
                    sessDm.InputData, sessDm.ResizeWorkspace, 1, sessDm.InputIsNhwc,
                    Sdcb.SimdPaddleOCR.ImagePixelFormat.Bgr24);
                float[] probDm = sessDm.RunInternal(sessDm.InputData).ToArray();
                mapsDm[bi] = probDm;
                var boxesDm = Sdcb.SimdPaddleOCR.DbPostprocess.Run(probDm, WD, HD, optsDm,
                    iwDm, ihDm, sizeDm.WidthRatio, sizeDm.HeightRatio, scratchDm);
                Console.WriteLine($"{bk}: {boxesDm.Length} boxes");
                foreach (var b in boxesDm)
                    Console.WriteLine($"  ({b.X1:F0},{b.Y1:F0})-({b.X3:F0},{b.Y3:F0}) s={b.Score:F3}");
            }
            var pcDm = mapsDm[0]; var pgDm = mapsDm[1];
            int flipUp = 0, flipDn = 0; double mdDm = 0; int mdIdx = -1;
            for (int i = 0; i < pcDm.Length; i++)
            {
                double d = Math.Abs(pcDm[i] - pgDm[i]);
                if (d > mdDm) { mdDm = d; mdIdx = i; }
                if (pcDm[i] < 0.2f && pgDm[i] >= 0.2f) flipUp++;
                if (pcDm[i] >= 0.2f && pgDm[i] < 0.2f) flipDn++;
            }
            Console.WriteLine($"map {WD}x{HD}: maxAbs={mdDm:F4} @({mdIdx % WD},{mdIdx / WD})  0.2-threshold flips: up={flipUp} down={flipDn}");
            // dump raw maps + list every flipped pixel
            var msDm = new MemoryStream(); var bwDm = new BinaryWriter(msDm);
            foreach (var v in pcDm) bwDm.Write(v);
            File.WriteAllBytes("detmap-cpu.bin", msDm.ToArray());
            msDm.SetLength(0);
            foreach (var v in pgDm) bwDm.Write(v);
            File.WriteAllBytes("detmap-gpu.bin", msDm.ToArray());
            Console.WriteLine("wrote detmap-cpu.bin / detmap-gpu.bin");
            for (int i = 0; i < pcDm.Length; i++)
            {
                bool cu = pcDm[i] >= 0.2f, gu = pgDm[i] >= 0.2f;
                if (cu != gu)
                    Console.WriteLine($"  flip ({i % WD},{i / WD}) cpu={pcDm[i]:F4} gpu={pgDm[i]:F4} dir={(gu ? "cpu0->gpu1" : "cpu1->gpu0")}");
            }
            return 0;
        }

        // --detprof <det.onnx> <H> <W> [reps]: per-dispatch GPU profile of a full graph.
        if (args.Length >= 4 && args[0] == "--detprof")
        {
            var mdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
            var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(mdl, intraOpThreads: 8);
            int H = int.Parse(args[2]), W = int.Parse(args[3]);
            int reps = args.Length >= 5 ? int.Parse(args[4]) : 20;
            int[] shape = [1, 3, H, W];
            var inp = new float[3 * H * W];
            var rr = new Random(1);
            for (int i = 0; i < inp.Length; i++) inp[i] = (float)(rr.NextDouble() * 2 - 1);
            using var ddev = Sdcb.SimdPaddleOCR.Backends.Vulkan.VkDevice.Create();
            using var gg = new Sdcb.SimdPaddleOCR.Backends.Vulkan.GpuDetGraph(ddev, compiled);
            double best = double.MaxValue;
            for (int r = 0; r < reps; r++)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                gg.Run(shape, inp);
                best = Math.Min(best, System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }
            Console.WriteLine($"best wall {best:F2} ms, dispatches={gg.DispatchCount(shape)}");
            gg.DumpProfile(shape);
            return 0;
        }
        return Harness.Usage(2);
    }
}
