using System.Runtime.CompilerServices;

namespace Sdcb.SimdPaddleOCR;

internal static class MathCompat
{
    [MethodImpl(MethodImplCompat.Hot)]
    public static int Clamp(int value, int min, int max) =>
#if NETSTANDARD2_0
        value < min ? min : value > max ? max : value;
#else
        Math.Clamp(value, min, max);
#endif

    [MethodImpl(MethodImplCompat.Hot)]
    public static long Clamp(long value, long min, long max) =>
#if NETSTANDARD2_0
        value < min ? min : value > max ? max : value;
#else
        Math.Clamp(value, min, max);
#endif

    [MethodImpl(MethodImplCompat.Hot)]
    public static float Clamp(float value, float min, float max) =>
#if NETSTANDARD2_0
        value < min ? min : value > max ? max : value;
#else
        Math.Clamp(value, min, max);
#endif

    [MethodImpl(MethodImplCompat.Hot)]
    public static double Clamp(double value, double min, double max) =>
#if NETSTANDARD2_0
        value < min ? min : value > max ? max : value;
#else
        Math.Clamp(value, min, max);
#endif

    [MethodImpl(MethodImplCompat.Hot)]
    public static bool IsFinite(float value) =>
#if NETSTANDARD2_0
        !float.IsNaN(value) && !float.IsInfinity(value);
#else
        float.IsFinite(value);
#endif

    [MethodImpl(MethodImplCompat.Hot)]
    public static bool IsFinite(double value) =>
#if NETSTANDARD2_0
        !double.IsNaN(value) && !double.IsInfinity(value);
#else
        double.IsFinite(value);
#endif
}
