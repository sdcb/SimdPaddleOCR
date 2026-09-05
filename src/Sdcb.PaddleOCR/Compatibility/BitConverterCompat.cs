using System.Runtime.CompilerServices;

namespace Sdcb.PaddleOCR;

internal static class BitConverterCompat
{
    [MethodImpl(MethodImplCompat.Hot)]
    public static int SingleToInt32Bits(float value)
    {
#if NETSTANDARD2_0
        uint bits = Unsafe.As<float, uint>(ref value);
        return unchecked((int)bits);
#else
        return BitConverter.SingleToInt32Bits(value);
#endif
    }

    [MethodImpl(MethodImplCompat.Hot)]
    public static float Int32BitsToSingle(int value)
    {
#if NETSTANDARD2_0
        uint bits = unchecked((uint)value);
        return Unsafe.As<uint, float>(ref bits);
#else
        return BitConverter.Int32BitsToSingle(value);
#endif
    }

    [MethodImpl(MethodImplCompat.Hot)]
    public static double Int64BitsToDouble(long value)
    {
#if NETSTANDARD2_0
        return Unsafe.As<long, double>(ref value);
#else
        return BitConverter.Int64BitsToDouble(value);
#endif
    }
}
