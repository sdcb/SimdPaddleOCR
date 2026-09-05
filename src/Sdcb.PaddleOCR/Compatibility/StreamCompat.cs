#if NETSTANDARD2_0
using System.Buffers;
using System.IO;
using System.Runtime.CompilerServices;

namespace Sdcb.PaddleOCR;

internal static class StreamCompat
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int Read(this Stream stream, Span<byte> buffer)
    {
        if (buffer.IsEmpty) return 0;
        byte[] rented = ArrayPool<byte>.Shared.Rent(buffer.Length);
        try
        {
            int read = stream.Read(rented, 0, buffer.Length);
            rented.AsSpan(0, read).CopyTo(buffer);
            return read;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }
}
#endif
