namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>Mode dispatch. Relative paths resolve against the app's external files dir.</summary>
static class Bench
{
    internal static string OutDir = "";

    internal static int Run(string[] args, string root, string outDir)
    {
        Paths.Root = root;
        OutDir = outDir;
        if (args.Length == 0) return Usage();
        return args[0] switch
        {
            "--init" => 0,
            "--e2e" => E2e.Run(args),
            _ => Usage(),
        };
    }

    static int Usage()
    {
        Console.WriteLine("""
            --e2e --model tiny|small|medium --backend cpu|vulkan|auto [--workers 4] [--count N] [--warmup N] [--out name.json]
            """);
        return 2;
    }
}
