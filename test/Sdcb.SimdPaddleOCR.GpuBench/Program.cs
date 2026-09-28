using Sdcb.SimdPaddleOCR.GpuBench;

if (args.Length == 0 || args[0] is "-h" or "--help")
    return Harness.Usage(0);

return args[0] switch
{
    "--sgtest" or "--rawbench" => DeviceBench.Run(args),
    "--argmax" or "--argmaxu" => ArgMaxBench.Run(args),
    "--ops" or "--graph" or "--arena" => InspectBench.Run(args),
    "--det" => DetCompare.Run(args),
    "--idle" or "--detmap" or "--detprof" => DetBench.Run(args),
    "--rec" => RecCompare.Run(args),
    "--recprof" or "--sessiso" or "--reciso" or "--recreal" => RecBench.Run(args),
    "--conc" or "--pipe" => PipelineBench.Run(args),
    _ => Harness.Usage(2),
};
