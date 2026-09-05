using System.Runtime.CompilerServices;

namespace Sdcb.SimdPaddleOCR;

internal static class MethodImplCompat
{
#if NETSTANDARD2_0
    public const MethodImplOptions AggressiveOptimization = (MethodImplOptions)512;
#else
    public const MethodImplOptions AggressiveOptimization = MethodImplOptions.AggressiveOptimization;
#endif
    internal const MethodImplOptions Hot =
        MethodImplOptions.AggressiveInlining | AggressiveOptimization;
}
