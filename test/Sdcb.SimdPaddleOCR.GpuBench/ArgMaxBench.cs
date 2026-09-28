using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>CTC projection plus ArgMax microbenchmarks. No Vulkan device.</summary>
static class ArgMaxBench
{
    internal static int Run(string[] args)
    {
        // --argmax <inner> <cols> <rows> <threads> <reps>: CTC projection+ArgMax kernel timing + checksum
        if (args.Length >= 6 && args[0] == "--argmax")
        {
            int inner = int.Parse(args[1]), cols = int.Parse(args[2]), rows = int.Parse(args[3]);
            int thr = int.Parse(args[4]), reps = int.Parse(args[5]);
            var arng = new Random(7);
            var wgt = new float[inner * cols];
            for (int i = 0; i < wgt.Length; i++) wgt[i] = (float)(arng.NextDouble() * 2 - 1) * 0.1f;
            var bias = new float[cols];
            for (int i = 0; i < cols; i++) bias[i] = (float)(arng.NextDouble() - 0.5);
            var inp = new float[rows * inner];
            for (int i = 0; i < inp.Length; i++) inp[i] = (float)(arng.NextDouble() * 2 - 1);
            int fullTiles = cols / 16;
            var packed = new float[fullTiles * inner * 16];
            for (int tile = 0; tile < fullTiles; tile++)
                for (int k = 0; k < inner; k++)
                    wgt.AsSpan(k * cols + tile * 16, 16).CopyTo(packed.AsSpan((tile * inner + k) * 16, 16));
            var idx = new int[rows];
            var sc = new float[rows];
            var times = new List<double>();
            for (int r = 0; r < reps; r++)
            {
                long t0 = Stopwatch.GetTimestamp();
                MatMul.TryArgMax(inp, wgt, bias, idx, sc, 1, rows, inner, cols, packed, thr);
                times.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }
            times.Sort();
            long h = 17;
            for (int i = 0; i < rows; i++) h = h * 31 + idx[i] * 7 + BitConverter.SingleToInt32Bits(sc[i]);
            Console.WriteLine($"argmax inner={inner} cols={cols} rows={rows} thr={thr}: min={times[0]:F3} med={times[times.Count / 2]:F3} ms hash={h:X}");
            return 0;
        }

        // --argmaxu <inner> <cols> <threads> <reps> <T1xN1,T2xN2,...>: TryArgMaxUnits vs per-unit TryArgMax (bitwise)
        if (args.Length >= 6 && args[0] == "--argmaxu")
        {
            int inner = int.Parse(args[1]), cols = int.Parse(args[2]), thr = int.Parse(args[3]), reps = int.Parse(args[4]);
            var unitsU = args[5].Split(',').Select(s => s.Split('x').Select(int.Parse).ToArray()).ToArray();
            int[] rowsU = unitsU.Select(u => u[0]).ToArray(), nbU = unitsU.Select(u => u[1]).ToArray();
            int total = unitsU.Sum(u => u[0] * u[1]);
            var urng = new Random(11);
            var wgt = new float[inner * cols];
            for (int i = 0; i < wgt.Length; i++) wgt[i] = (float)(urng.NextDouble() * 2 - 1) * 0.1f;
            var bias = new float[cols];
            for (int i = 0; i < cols; i++) bias[i] = (float)(urng.NextDouble() - 0.5);
            var inp = new float[total * inner];
            for (int i = 0; i < inp.Length; i++) inp[i] = (float)(urng.NextDouble() * 2 - 1);
            for (int r = 0; r < 3; r++) { int rr0 = urng.Next(total); for (int k = 0; k < inner; k++) inp[rr0 * inner + k] = 0; } // all-bias rows (ties)
            int fullTiles = cols / 16;
            var packed = new float[fullTiles * inner * 16];
            for (int tile = 0; tile < fullTiles; tile++)
                for (int k = 0; k < inner; k++)
                    wgt.AsSpan(k * cols + tile * 16, 16).CopyTo(packed.AsSpan((tile * inner + k) * 16, 16));
            var idxA = new int[total]; var scA = new float[total];
            var idxB = new int[total]; var scB = new float[total];
            var tA = new List<double>(); var tB = new List<double>();
            for (int r = 0; r < reps; r++)
            {
                long t0 = Stopwatch.GetTimestamp();
                MatMul.TryArgMaxUnits(inp, wgt, bias, idxA, scA, nbU, rowsU, inner, cols, packed, thr);
                tA.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
                t0 = Stopwatch.GetTimestamp();
                int st = 0;
                for (int u = 0; u < unitsU.Length; u++)
                {
                    int n = rowsU[u] * nbU[u];
                    MatMul.TryArgMax(inp.AsSpan(st * inner, n * inner), wgt, bias, idxB.AsSpan(st, n), scB.AsSpan(st, n),
                        nbU[u], rowsU[u], inner, cols, packed, thr);
                    st += n;
                }
                tB.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            }
            tA.Sort(); tB.Sort();
            int diff = 0;
            for (int i = 0; i < total; i++)
                if (idxA[i] != idxB[i] || BitConverter.SingleToInt32Bits(scA[i]) != BitConverter.SingleToInt32Bits(scB[i])) diff++;
            Console.WriteLine($"argmaxu rows={total} units={unitsU.Length} thr={thr}: units med={tA[tA.Count / 2]:F3} min={tA[0]:F3} | per-unit med={tB[tB.Count / 2]:F3} min={tB[0]:F3} ms  bitdiff={diff}");
            return 0;
        }
        return Harness.Usage(2);
    }
}
