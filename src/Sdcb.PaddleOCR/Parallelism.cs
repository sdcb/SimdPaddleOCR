namespace Sdcb.PaddleOCR;

/// <summary>
/// Line-stage parallelism: workers are CLS/REC sessions, intra-op is leftover
/// cores inside each REC session. Never oversubscribe ProcessorCount.
/// </summary>
internal static class Parallelism
{
    public const int MaxLineWorkers = 16;
    public const int MaxAutoLineWorkers = 4;
    public const int MaxRecognizerIntraOpThreads = 3;

    public static int ResolveLineWorkers(int lineWorkerCount) =>
        ResolveLineWorkers(lineWorkerCount, Environment.ProcessorCount);

    // Positive LineWorkerCount is a maximum, clamped to the CPU. 0 (auto) fills
    // one worker per core up to MaxAutoLineWorkers so 2-core machines use both
    // cores as sessions instead of ProcessorCount/4 → 1.
    public static int ResolveLineWorkers(int lineWorkerCount, int processorCount)
    {
        int cpu = Math.Max(1, processorCount);
        int requested = lineWorkerCount > 0
            ? lineWorkerCount
            : Math.Min(cpu, MaxAutoLineWorkers);
        return Math.Clamp(Math.Min(requested, cpu), 1, MaxLineWorkers);
    }

    public static int ResolveRecognizerIntraOp(int lineWorkers) =>
        ResolveRecognizerIntraOp(lineWorkers, Environment.ProcessorCount);

    // Leftover cores after placing line workers, capped at 3. Two workers on a
    // 2-core machine stay intra-op 1; a single worker gets the second core.
    public static int ResolveRecognizerIntraOp(int lineWorkers, int processorCount)
    {
        int cpu = Math.Max(1, processorCount);
        int workers = Math.Max(1, lineWorkers);
        return Math.Clamp(cpu / workers, 1, MaxRecognizerIntraOpThreads);
    }
}
