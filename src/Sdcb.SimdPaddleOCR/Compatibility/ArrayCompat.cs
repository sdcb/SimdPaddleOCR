using System.Runtime.CompilerServices;

namespace Sdcb.SimdPaddleOCR;

internal static class ArrayCompat
{
    [MethodImpl(MethodImplCompat.Hot)]
    public static void Fill<T>(T[] array, T value)
    {
#if NETSTANDARD2_0
        for (int i = 0; i < array.Length; i++) array[i] = value;
#else
        Array.Fill(array, value);
#endif
    }

    [MethodImpl(MethodImplCompat.Hot)]
    public static void Fill<T>(T[] array, T value, int startIndex, int count)
    {
#if NETSTANDARD2_0
        int end = startIndex + count;
        for (int i = startIndex; i < end; i++) array[i] = value;
#else
        Array.Fill(array, value, startIndex, count);
#endif
    }
}
