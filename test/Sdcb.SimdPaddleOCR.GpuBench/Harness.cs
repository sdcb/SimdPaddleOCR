namespace Sdcb.SimdPaddleOCR.GpuBench;

static class DesktopPaths
{
    internal static int Run(Func<string[], int> mode, string[] args)
    {
        string root = AppContext.BaseDirectory;
        while (root.Length > 3 && !File.Exists(Path.Combine(root, "Sdcb.SimdPaddleOCR.slnx")))
            root = Path.GetDirectoryName(root.TrimEnd(Path.DirectorySeparatorChar))!;
        AndroidBench.Paths.Root = root;
        AndroidBench.Paths.ModelsDir = Path.Combine("bench-out", "models");
        return mode(args);
    }
}

static class Harness
{
    internal static int Usage(int code)
    {
        Console.Error.WriteLine("""
            GpuBench — local Vulkan / CTC tuning harness.
            Vulkan modes need a device; --argmax and --argmaxu do not.
            Set SIMD_OCR_GPU_PROF=1 for per-kernel times on --detprof / --recprof.

              --detprof <det.onnx> <H> <W> [reps]
              --resource-perf --backend cpu|vulkan --scope orientation|pipeline [--seconds 20] [--out result.json]
              --orientation [--cpu] [--no-pipeline] [--out result.json]
              --orientation --profile                      SIMD_OCR_GPU_PROF=1 per-dispatch timings
              --recprof <rec.onnx> <batch> <W> [reps]     REC up to the CTC projection
              --conc <det> <cls> <rec> <keys> <imgdir> [threads] [backend]
              --sgtest [sgsize.spv]                       caps + gl_SubgroupSize at required size 0/32/64 (default: Shaders/sgsize.spv)
              --rawbench <spv> <bindings> <gx> <gy> <reps> <bufMB> [pc uints...]
              --argmax <inner> <cols> <rows> <threads> <reps>
              --argmaxu <inner> <cols> <threads> <reps> <TxN,TxN,...>
              --arena <onnx> <n> <C> <H> <W>
              --ops <model.onnx> N C H W
              --idle <model.onnx> <H> <W>
              --det <model.onnx> <H> <W>
              --graph <model.onnx> <H> <W>
              --rec <rec.onnx> <n> <W>
              --pipe <det> <cls|-> <rec> <keys> <img> [cpu|vulkan|auto]
              --sessiso <rec.onnx> <W>
              --reciso <det> <rec> <keys> <img>
              --detmap <det.onnx> <img>
              --detcmp <tiny|small|medium> <img.jpg|all|N> [maxSide]   CPU vs Vulkan DET map + boxes
              --reccmp <tiny|small|medium> <batch> <W>                 CPU vs Vulkan REC activations
              --gemm <shader|file.spv> <M> <N> <K> [flags] [tileM] [tileN] [reqSg] [reps]   GEMM vs fp64 reference
              --layers <onnx> <N> <H> <W> [first] [last]   per-node output stats of the truncated GPU graph
              --recreal <det> <rec> <img> [W]
            """);
        return code;
    }
}
