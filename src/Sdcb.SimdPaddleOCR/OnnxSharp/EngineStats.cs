using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>Process-wide engine operator profile (benchmarks): cumulative ms / calls per op name and conv class.</summary>
internal static class EngineStats
{
    internal static void EnableProfiling(bool enabled) => EngineProfile.Enabled = enabled;
    internal static Dictionary<string, (double Ms, long Calls)> Operators() => EngineProfile.Operators();
    internal static Dictionary<string, (double Ms, long Calls)> ConvClasses() => EngineProfile.ConvClasses();
}
