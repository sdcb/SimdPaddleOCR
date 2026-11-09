// netstandard2.0 only (see the csproj): Vector<T> element access that System.Numerics.Vectors lacks.
#if NETSTANDARD2_0
using System.Numerics;
using System.Runtime.CompilerServices;

// in the library's namespace (not System.Numerics) so it wins over any same-named polyfill in scope
namespace Sdcb.SimdPaddleOCR
{
    internal static class VectorCompat
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T GetElement<T>(this Vector<T> vector, int index) where T : struct =>
            Unsafe.Add(ref Unsafe.As<Vector<T>, T>(ref vector), index);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<T> WithElement<T>(this Vector<T> vector, int index, T value) where T : struct
        {
            Unsafe.Add(ref Unsafe.As<Vector<T>, T>(ref vector), index) = value;
            return vector;
        }
    }
}
#endif