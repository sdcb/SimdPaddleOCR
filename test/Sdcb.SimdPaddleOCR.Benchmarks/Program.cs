// metalprof <onnx> <N> <H> <W> [reps]  — one graph, one thread: best/median wall per run and dispatch count;
// SIMD_OCR_GPU_PROF=1 adds the per-kernel serialized profile.
using System.Diagnostics;
using Sdcb.SimdPaddleOCR.OnnxSharp;

#if !V200
if (args[0] == "--plan")
{
    var m = Model.Load(File.ReadAllBytes(args[1]));
    int[] sh = [1, 3, int.Parse(args[2]), int.Parse(args[3])];
    var pl = Sdcb.SimdOnnx.Engine.GraphCompiler.Compile(m.Onnx, sh, CompiledModel.Options with { Vec4Eltwise = true, Target = Sdcb.SimdOnnx.Engine.PlanTarget.Metal, Nhwc = true });
    foreach (var n in pl.Nodes)
        Console.WriteLine($"{n.Op,-16} {n.Name,-28} in=[{string.Join("; ", n.In.Select(v => v < 0 ? "-" : string.Join("x", pl.Values[v].PhysShape) + (pl.Values[v].Floats is null ? "" : "c")))}] out={string.Join("x", pl.Values[n.Out[0]].PhysShape)} {n.P}");
    return;
}
#endif
#if !V200
if (args[0] == "--planmem")
{
    var m = Model.Load(File.ReadAllBytes(args[1]));
    var keep = new List<object>();
    long before = GC.GetTotalMemory(true);
    for (int k = 0; k < 10; k++)
    {
        int[] sh = [1, 3, 640 + 32 * k, 736];
        keep.Add(Sdcb.SimdOnnx.Engine.GraphCompiler.Compile(m.Onnx, sh, CompiledModel.Options with { Vec4Eltwise = true, Target = Sdcb.SimdOnnx.Engine.PlanTarget.Metal, Nhwc = true }));
        Console.WriteLine($"{k}: +{(GC.GetTotalMemory(true) - before) / 1048576.0:F1} MB");
    }
    GC.KeepAlive(keep);
    return;
}
#endif
string path = args[0];
int[] shape = [int.Parse(args[1]), 3, int.Parse(args[2]), int.Parse(args[3])];
int reps = args.Length > 4 ? int.Parse(args[4]) : 30;
bool prof = Environment.GetEnvironmentVariable("SIMD_OCR_GPU_PROF") == "1";
var model = Model.Load(File.ReadAllBytes(path));
var input = new float[shape[0] * 3 * shape[2] * shape[3]];
var rnd = new Random(1);
for (int i = 0; i < input.Length; i++) input[i] = (float)(rnd.NextDouble() * 2 - 1);
var times = new List<double>();
#if V200
var compiled = new CompiledModel(model, 4);
var dev = Sdcb.SimdPaddleOCR.Backends.Metal.MetalBackend.TryGetDevice() ?? throw new Exception("no metal");
using var g = new Sdcb.SimdPaddleOCR.Backends.Metal.MetalDetGraph(dev, compiled);
for (int r = 0; r < reps; r++) { long t0 = Stopwatch.GetTimestamp(); g.Run(shape, input); times.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds); }
int dispatches = g.DispatchCount(shape);
Report("v2.0.0");
if (prof) g.DumpProfile(shape);
#else
var compiled = new CompiledModel(model, 4);
var opt = CompiledModel.Options with { Vec4Eltwise = true };
using var s = Sdcb.SimdOnnx.Backends.Metal.MetalBackend.Default.NewSession(model.Onnx, shape, opt);
for (int r = 0; r < reps; r++) { long t0 = Stopwatch.GetTimestamp(); input.CopyTo(s.Input); s.Run(); times.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds); }
var agg = s.DebugProfile();
int dispatches = s.PerDispatch!.Length;
Report("main");
if (prof)
{
    Console.WriteLine($"--- Metal profile ({dispatches} dispatches, serialized) ---");
    foreach (var kv in agg.OrderByDescending(k => k.Value)) Console.WriteLine($"  {kv.Key,-24} {kv.Value,9:F3} ms");
    foreach (var (n, ms) in s.PerDispatch!.OrderByDescending(x => x.Ms).Take(25)) Console.WriteLine($"    {ms,8:F3} {n}");
}
#endif
void Report(string tag)
{
    var t = times.Skip(3).OrderBy(x => x).ToArray();
    Console.WriteLine($"[{tag}] {Path.GetFileName(path)} {string.Join('x', shape)} best {t[0]:F2} ms median {t[t.Length / 2]:F2} ms dispatches {dispatches}");
}
