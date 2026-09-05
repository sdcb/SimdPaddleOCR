using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Warp
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void MapRow(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedWidth, int unrotatedHeight,
        bool rotateVertical, double a, double b, double c, double d, double e, double f,
        double g, double h, int y, double v, ref int x)
    {
        #if !NETSTANDARD2_0
        if (Avx512F.IsSupported && unrotatedWidth >= 8)
            MapRowAvx512(sourcePtr, sourceWidth, sourceHeight, sourceStride, cropPtr, outputWidth,
                unrotatedWidth, unrotatedHeight, rotateVertical, a, b, c, d, e, f, g, h, y, v, ref x);
        else if (Avx2.IsSupported && unrotatedWidth >= 4)
            MapRowAvx(sourcePtr, sourceWidth, sourceHeight, sourceStride, cropPtr, outputWidth,
                unrotatedWidth, unrotatedHeight, rotateVertical, a, b, c, d, e, f, g, h, y, v, ref x);
        else
        #endif
        if (Vector.IsHardwareAccelerated && Vector<double>.Count >= 2 &&
            unrotatedWidth >= Vector<double>.Count)
            MapRowVector(sourcePtr, sourceWidth, sourceHeight, sourceStride, cropPtr, outputWidth,
                unrotatedWidth, unrotatedHeight, rotateVertical, a, b, c, d, e, f, g, h, y, v, ref x);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe void SampleCubic(byte* source, int width, int height, int stride,
        double x, double y, byte* destination, int destinationOffset)
    {
        int xBase = (int)Math.Floor(x), yBase = (int)Math.Floor(y);
        if (xBase >= 1 && xBase < width - 3 && yBase >= 1 && yBase < height - 2)
        {
            #if !NETSTANDARD2_0
            if (Avx512F.IsSupported)
            {
                SampleCubicAvx512(source, stride, x, y, xBase, yBase, destination, destinationOffset);
                return;
            }
            else if (Avx.IsSupported && Avx2.IsSupported)
            {
                SampleCubicAvx(source, stride, x, y, xBase, yBase, destination, destinationOffset);
                return;
            }
            else
            #endif
            if (Vector.IsHardwareAccelerated && Vector<double>.Count == 4)
            {
                SampleCubicVector(source, stride, x, y, xBase, yBase, destination, destinationOffset);
                return;
            }
        }
        SampleCubicScalar(source, width, height, stride, x, y, destination, destinationOffset);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SampleMappedPixel(byte* sourcePtr, int sourceWidth, int sourceHeight,
        int sourceStride, byte* cropPtr, int outputWidth, int unrotatedHeight, bool rotateVertical,
        double pixelX, double pixelY, int x, int y)
    {
        int destinationX = rotateVertical ? unrotatedHeight - 1 - y : x;
        int destinationY = rotateVertical ? x : y;
        int destination = checked((destinationY * outputWidth + destinationX) * 3);
        SampleCubic(sourcePtr, sourceWidth, sourceHeight, sourceStride,
            pixelX, pixelY, cropPtr, destination);
    }
}
