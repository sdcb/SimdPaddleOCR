#if NETSTANDARD2_0
using System.Numerics;
using System.Runtime.CompilerServices;

namespace System.Numerics
{
    internal static class VectorCompat
    {
        [MethodImpl(MethodImplCompat.Hot)]
        public static T GetElement<T>(this Vector<T> vector, int index) where T : struct =>
            Unsafe.Add(ref Unsafe.As<Vector<T>, T>(ref vector), index);

        [MethodImpl(MethodImplCompat.Hot)]
        public static Vector<T> WithElement<T>(this Vector<T> vector, int index, T value) where T : struct
        {
            Unsafe.Add(ref Unsafe.As<Vector<T>, T>(ref vector), index) = value;
            return vector;
        }
    }
}
#endif
