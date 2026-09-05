#if NETSTANDARD2_0
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Sdcb.PaddleOCR;

internal static class DictionaryCompat
{
    [MethodImpl(MethodImplCompat.Hot)]
    public static TValue GetValueOrDefault<TKey, TValue>(
        this Dictionary<TKey, TValue> dictionary, TKey key)
    {
        return dictionary.TryGetValue(key, out TValue value) ? value : default!;
    }
}
#endif
