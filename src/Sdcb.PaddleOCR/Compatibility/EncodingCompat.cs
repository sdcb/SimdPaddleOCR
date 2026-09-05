using System.Runtime.CompilerServices;
using System.Text;

namespace Sdcb.PaddleOCR;

internal static class EncodingCompat
{
    [MethodImpl(MethodImplCompat.Hot)]
    public static string GetString(Encoding encoding, ReadOnlySpan<byte> bytes)
    {
#if NETSTANDARD2_0
        if (bytes.IsEmpty) return string.Empty;
        return encoding.GetString(bytes.ToArray());
#else
        return encoding.GetString(bytes);
#endif
    }
}
