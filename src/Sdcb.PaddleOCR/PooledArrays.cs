using System.Buffers;
using System.Runtime.CompilerServices;

namespace Sdcb.PaddleOCR;

/// <summary>
/// ArrayPool wrapper that keeps the small-buffer fast path and refuses to
/// return large buffers to <see cref="ArrayPool{T}.Shared"/>.
/// Unique DET map / crop sizes would otherwise pin one LOH bucket each and
/// the process working set would climb for the lifetime of the Singleton
/// even when later images are smaller.
/// </summary>
internal static class PooledArrays
{
    // Just under the 85,000-byte LOH line. Shared-pool buckets at this size
    // are never given back to the OS; discarding lets GC reclaim them.
    internal const int DiscardThresholdBytes = 65_536;

    public static T[] Rent<T>(int minimumLength)
    {
        if (minimumLength <= 0) return [];
        if (Bytes<T>(minimumLength) >= DiscardThresholdBytes)
            return GC.AllocateUninitializedArray<T>(minimumLength);
        return ArrayPool<T>.Shared.Rent(minimumLength);
    }

    public static void Return<T>(T[]? array, bool clearArray = false)
    {
        if (array is null || array.Length == 0) return;
        if (Bytes<T>(array.Length) >= DiscardThresholdBytes) return;
        ArrayPool<T>.Shared.Return(array, clearArray);
    }

    private static long Bytes<T>(int length) => (long)length * Unsafe.SizeOf<T>();
}
