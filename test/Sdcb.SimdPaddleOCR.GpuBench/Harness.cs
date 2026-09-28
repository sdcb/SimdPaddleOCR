namespace Sdcb.SimdPaddleOCR.GpuBench;

static class Harness
{
    internal static int Usage(int code)
    {
        Console.Error.WriteLine("""
            GpuBench — local Vulkan / CTC tuning harness.
            Vulkan modes need a device; --argmax and --argmaxu do not.
            Set SIMD_OCR_GPU_PROF=1 for per-kernel times on --detprof / --recprof.

              --detprof <det.onnx> <H> <W> [reps]
              --recprof <rec.onnx> <batch> <W> [reps]     REC up to the CTC projection
              --conc <det> <cls> <rec> <keys> <imgdir> [threads] [backend]
              --sgtest <sgsize.spv>                       caps + gl_SubgroupSize at required size 0/32/64
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
              --recreal <det> <rec> <img> [W]
            """);
        return code;
    }
}
