using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>Graph listing and arena size.</summary>
static class InspectBench
{
    internal static int Run(string[] args)
    {
        // --ops <model.onnx> N C H W: node listing with resolved shapes + fused skip
        if (args.Length >= 6 && args[0] == "--ops")
        {
            var mdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
            var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(mdl, intraOpThreads: 4);
            int[] shp = [int.Parse(args[2]), int.Parse(args[3]), int.Parse(args[4]), int.Parse(args[5])];
            int[][] shs = compiled.ResolveShapesFor(shp);
            for (int ni = 0; ni < mdl.Nodes.Length; ni++)
            {
                var nd = mdl.Nodes[ni];
                string ins = string.Join(" ", nd.Inputs.Select(i => i == uint.MaxValue ? "-" :
                    $"t{i}{(compiled.GetTensor((int)i).IsConstant ? "c" : "")}[{string.Join('x', shs[i])}]"));
                string outs = string.Join(" ", nd.Outputs.Select(i => $"t{i}[{string.Join('x', shs[i])}]"));
                var p = mdl.GetParameters(nd);
                string ps = p.Length <= 48 ? Convert.ToHexString(p) : Convert.ToHexString(p[..48]) + "..";
                Console.WriteLine($"n{ni,4} {nd.Operator,-18} skip={compiled.FusedSkip(ni)} nhwc={(compiled.IsNhwcNode(ni) ? 1 : 0)} {ins} -> {outs}  p={ps}");
            }
            return 0;
        }

        // --graph <model.onnx> <H> <W>: dump node list with resolved shapes, then exit
        if (args.Length >= 4 && args[0] == "--graph")
        {
            var model = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
            for (int ti = 0; ti < model.Tensors.Length; ti++)
            {
                var tr = model.Tensors[ti];
                if ((tr.Flags & Sdcb.SimdPaddleOCR.OnnxSharp.Model.TensorInput) != 0)
                    Console.WriteLine($"input tensor {ti}: [{string.Join(',', tr.Dimensions.ToArray().Select(d => d.ToString()))}]");
            }
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

        // --arena <onnx> <n> <C> <H> <W>: arena size of the compiled schedule
        if (args.Length >= 6 && args[0] == "--arena")
        {
            var mdl = Sdcb.SimdPaddleOCR.OnnxSharp.Model.Load(File.ReadAllBytes(args[1]));
            var compiled = new Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel(mdl, intraOpThreads: 4);
            int[] shape = [int.Parse(args[2]), int.Parse(args[3]), int.Parse(args[4]), int.Parse(args[5])];
            using var ddev = Sdcb.SimdPaddleOCR.Backends.Vulkan.VkDevice.Create();
            using var gg = new Sdcb.SimdPaddleOCR.Backends.Vulkan.GpuDetGraph(ddev, compiled);
            var s = gg.Prepare(shape);
            Console.WriteLine($"arena {s.ArenaElems * 2 / 1048576.0:F1} MB, in {s.InElems * 4 / 1048576.0:F1} MB, out {s.OutElems * 4 / 1048576.0:F1} MB");
            return 0;
        }
        return Harness.Usage(2);
    }
}
