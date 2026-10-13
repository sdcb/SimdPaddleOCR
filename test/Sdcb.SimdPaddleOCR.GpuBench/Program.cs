using Sdcb.SimdPaddleOCR.GpuBench;

if (args.Length == 0 || args[0] is "-h" or "--help")
    return Harness.Usage(0);

return args[0] switch
{
    "--resource-perf" => ResourceBench.Run(args),
    "--orientation" => OrientationBench.Run(args),
    "--sgtest" or "--rawbench" => DeviceBench.Run(args),
    "--argmax" or "--argmaxu" => ArgMaxBench.Run(args),
    "--ops" or "--graph" or "--arena" => InspectBench.Run(args),
    "--det" => DetCompare.Run(args),
    "--idle" or "--detmap" or "--detprof" => DetBench.Run(args),
    "--rec" => RecCompare.Run(args),
    "--recprof" or "--sessiso" or "--reciso" or "--recreal" => RecBench.Run(args),
    "--conc" or "--pipe" => PipelineBench.Run(args),
    "--gemm" => Sdcb.SimdPaddleOCR.AndroidBench.GemmTest.Run(args),
    // same checks as the phone host; models from bench-out/models, images from dataset/
    "--detcmp" or "--reccmp" => DesktopPaths.Run(Sdcb.SimdPaddleOCR.AndroidBench.Compare.Run, args),
    "--layers" when args.Length >= 5 => Sdcb.SimdPaddleOCR.AndroidBench.LayerProbe.Run(args[1], int.Parse(args[2]),
        int.Parse(args[3]), int.Parse(args[4]), args.Length > 5 ? int.Parse(args[5]) : 0,
        args.Length > 6 ? int.Parse(args[6]) : int.MaxValue),
    _ => Harness.Usage(2),
};
