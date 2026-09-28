using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>Full PaddleOcrAll runs: one image, or concurrent callers.</summary>
static class PipelineBench
{
    internal static int Run(string[] args)
    {
        // --pipe <det.onnx> <cls.onnx|-> <rec.onnx> <keys.txt> <img> [cpu|vulkan|auto]:
        // full PaddleOcrAll E2E — prints per-line text+score and wall time, so the
        // GPU backend path can be diffed against CPU end to end.
        // --conc <det> <cls> <rec> <keys> <imgdir> [threads] [backend]: one shared
        // PaddleOcrAll, serial reference texts vs concurrent callers (thread safety).
        if (args.Length >= 6 && args[0] == "--conc")
        {
            int threads = args.Length > 6 ? int.Parse(args[6]) : 4;
            var backendC = args.Length > 7 && Enum.TryParse<Sdcb.SimdPaddleOCR.OcrBackend>(args[7], true, out var bc)
                ? bc : Sdcb.SimdPaddleOCR.OcrBackend.Vulkan;
            var optsC = new Sdcb.SimdPaddleOCR.PaddleOcrOptions
            {
                Detector = new Sdcb.SimdPaddleOCR.PaddleOcrDetectorOptions { Backend = backendC },
                Recognizer = new Sdcb.SimdPaddleOCR.PaddleOcrRecognizerOptions { Backend = backendC },
                Classifier = new Sdcb.SimdPaddleOCR.PaddleOcrClassifierOptions { Backend = backendC },
            };
            using var ocrC = new Sdcb.SimdPaddleOCR.PaddleOcrAll(
                Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1])),
                Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[2])),
                Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[3])),
                File.ReadAllBytes(args[4]), optsC);
            var imgs = Directory.GetFiles(args[5], "*.jpg").OrderBy(f => f).Take(24).Select(f =>
            {
                using var im = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Bgr24>(f);
                byte[] px = new byte[im.Width * im.Height * 3];
                im.CopyPixelDataTo(px);
                return (px, im.Width, im.Height);
            }).ToArray();
            string Texts(int i) => string.Join("|", ocrC.Run(imgs[i].px, imgs[i].Width, imgs[i].Height).Lines.Select(l => l.Text));
            string[] reference = Enumerable.Range(0, imgs.Length).Select(Texts).ToArray();
            int bad = 0;
            var tc = System.Diagnostics.Stopwatch.StartNew();
            Parallel.For(0, imgs.Length * 3, new ParallelOptions { MaxDegreeOfParallelism = threads }, j =>
            {
                int i = j % imgs.Length;
                if (Texts(i) != reference[i]) { Interlocked.Increment(ref bad); Console.WriteLine($"MISMATCH img{i}"); }
            });
            Console.WriteLine($"conc threads={threads} runs={imgs.Length * 3} mismatches={bad} wall={tc.ElapsedMilliseconds}ms");
            return bad == 0 ? 0 : 5;
        }

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
                    foreach (var l in r.Lines)
                    {
                        var b = l.Box;
                        Console.WriteLine($"  [{l.RecognitionScore:F2}] box=({b.X1:F0},{b.Y1:F0})-({b.X3:F0},{b.Y3:F0}) {l.Text}");
                    }
                Console.WriteLine($"rep{rep}: {r.Lines.Length} lines {ms:F1}ms");
                if (prof)
                    foreach (var (s, i) in Sdcb.SimdPaddleOCR.PipelineProfiler.Snapshot().Select((v, i) => (v, i))
                                 .Where(x => x.v.Milliseconds > 0.3))
                        Console.WriteLine($"    prof {Sdcb.SimdPaddleOCR.PipelineProfiler.StageNames[i],-16} {s.Milliseconds,8:F1}ms x{s.Calls}");
            }
            return 0;
        }
        return Harness.Usage(2);
    }
}
