using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using System.Threading.Tasks;

namespace Sdcb.SimdPaddleOCR.Kernels;

/// <summary>
/// Shared ISA helpers for the per-operator kernel types (Load/AddMul, intra-op
/// shard policy). Operator entry points live on <c>Conv1x1</c>, <c>Conv3x3</c>,
/// <c>MatMul</c>, and the other types in this folder — not here.
/// </summary>
internal static class SimdOps
{
#if !NETSTANDARD2_0
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> LoadStride2(ReadOnlySpan<float> source, int offset)
    {
        Vector256<float> first = Load(source, offset);
        Vector256<float> second = Load(source, offset + 8);
        Vector128<float> evenLow = Sse.Shuffle(first.GetLower(), first.GetUpper(), 0x88);
        Vector128<float> evenHigh = Sse.Shuffle(second.GetLower(), second.GetUpper(), 0x88);
        return Vector256.Create(evenLow, evenHigh);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe Vector256<float> LoadStride2(float* source)
    {
        Vector256<float> first = Avx.LoadVector256(source);
        Vector256<float> second = Avx.LoadVector256(source + 8);
        Vector128<float> evenLow = Sse.Shuffle(first.GetLower(), first.GetUpper(), 0x88);
        Vector128<float> evenHigh = Sse.Shuffle(second.GetLower(), second.GetUpper(), 0x88);
        return Vector256.Create(evenLow, evenHigh);
    }
#endif

    internal const long IntraOpMinWork = 8_000_000;

    internal const int OutputTile = 4;

    internal static bool CanShardOutputs(int intraOpThreads, int batch, int groups, int outputChannels, long work) =>
        intraOpThreads > 1 && batch == 1 && groups == 1
        && outputChannels >= OutputTile * 2 && work >= IntraOpMinWork;

    internal static bool CanShardChannels(int intraOpThreads, int batch, int channels, long work) =>
        intraOpThreads > 1 && batch == 1 && channels >= 2 && work >= IntraOpMinWork;

    internal static int ShardWorkers(int intraOpThreads, int tiles) =>
        Math.Min(intraOpThreads, Math.Max(1, tiles));

    internal static (int Begin, int End) AlignedOutputShard(int worker, int workers, int outputChannels)
    {
        int begin = (outputChannels * worker / workers) & ~(OutputTile - 1);
        int end = worker == workers - 1 ? outputChannels : (outputChannels * (worker + 1) / workers) & ~(OutputTile - 1);
        return (begin, end);
    }

    internal static (int Begin, int End) BlockShard(int worker, int workers, int blocks) =>
        (blocks * worker / workers, blocks * (worker + 1) / workers);

#if !NETSTANDARD2_0
    // FMA when available (higher throughput and precision); edge pixels use
    // the scalar tap-skipping path, so interior/edge rounding may differ by
    // 1 ulp. Deterministic across runs and thread counts either way.
    // Fma.IsSupported is a JIT-time constant, so the unused arm folds away.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Load(ReadOnlySpan<float> source, int offset) =>
        Vector256.LoadUnsafe(ref Unsafe.Add(ref MemoryMarshal.GetReference(source), offset));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Store(Span<float> destination, int offset, Vector256<float> value) =>
        value.StoreUnsafe(ref Unsafe.Add(ref MemoryMarshal.GetReference(destination), offset));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> AddMul(Vector256<float> accumulator,
        Vector256<float> value, float weight) =>
        Fma.IsSupported
            ? Fma.MultiplyAdd(value, Vector256.Create(weight), accumulator)
            : Avx.Add(accumulator, Avx.Multiply(value, Vector256.Create(weight)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> AddMul(Vector256<float> accumulator,
        Vector256<float> value, Vector256<float> weight) =>
        Fma.IsSupported
            ? Fma.MultiplyAdd(value, weight, accumulator)
            : Avx.Add(accumulator, Avx.Multiply(value, weight));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AddEightPacked(ref Vector256<float> a0, ref Vector256<float> a1,
        ref Vector256<float> a2, ref Vector256<float> a3, ref Vector256<float> a4,
        ref Vector256<float> a5, ref Vector256<float> a6, ref Vector256<float> a7,
        Vector256<float> value, float* weights)
    {
        a0 = AddMul(a0, value, weights[0]); a1 = AddMul(a1, value, weights[1]);
        a2 = AddMul(a2, value, weights[2]); a3 = AddMul(a3, value, weights[3]);
        a4 = AddMul(a4, value, weights[4]); a5 = AddMul(a5, value, weights[5]);
        a6 = AddMul(a6, value, weights[6]); a7 = AddMul(a7, value, weights[7]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AddFourPacked(ref Vector256<float> a0, ref Vector256<float> a1,
        ref Vector256<float> a2, ref Vector256<float> a3, Vector256<float> value, float* weights)
    {
        a0 = AddMul(a0, value, weights[0]); a1 = AddMul(a1, value, weights[1]);
        a2 = AddMul(a2, value, weights[2]); a3 = AddMul(a3, value, weights[3]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<float> Load512(ReadOnlySpan<float> source, int offset) =>
        Vector512.LoadUnsafe(ref Unsafe.Add(ref MemoryMarshal.GetReference(source), offset));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Store512(Span<float> destination, int offset, Vector512<float> value) =>
        value.StoreUnsafe(ref Unsafe.Add(ref MemoryMarshal.GetReference(destination), offset));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe Vector512<float> BroadcastWeight512(float* weight) =>
        Avx512F.BroadcastScalarToVector512(Sse.LoadScalarVector128(weight));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<float> AddMul512(Vector512<float> accumulator,
        Vector512<float> value, float weight) =>
        Avx512F.FusedMultiplyAdd(value, Vector512.Create(weight), accumulator);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<float> AddMul512(Vector512<float> accumulator,
        Vector512<float> value, Vector512<float> weight) =>
        Avx512F.FusedMultiplyAdd(value, weight, accumulator);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AddEightPacked512(ref Vector512<float> a0, ref Vector512<float> a1,
        ref Vector512<float> a2, ref Vector512<float> a3, ref Vector512<float> a4,
        ref Vector512<float> a5, ref Vector512<float> a6, ref Vector512<float> a7,
        Vector512<float> value, float* weights)
    {
        a0 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 0), a0);
        a1 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 1), a1);
        a2 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 2), a2);
        a3 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 3), a3);
        a4 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 4), a4);
        a5 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 5), a5);
        a6 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 6), a6);
        a7 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 7), a7);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void AddFourPacked512(ref Vector512<float> a0, ref Vector512<float> a1,
        ref Vector512<float> a2, ref Vector512<float> a3, Vector512<float> value, float* weights)
    {
        a0 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 0), a0);
        a1 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 1), a1);
        a2 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 2), a2);
        a3 = Avx512F.FusedMultiplyAdd(value, BroadcastWeight512(weights + 3), a3);
    }

    /// <summary>
    /// Packs even lanes from 32 consecutive floats into a 16-wide vector.
    /// Two ZMM loads + one vpermt2ps; indices 0..15 from the first vector,
    /// 16..31 from the second (source[16] is even index 16).
    /// </summary>
    internal static readonly Vector512<int> Stride2EvenIndex512 =
        Vector512.Create(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<float> LoadStride2512(ReadOnlySpan<float> source, int offset)
    {
        ref float origin = ref Unsafe.Add(ref MemoryMarshal.GetReference(source), offset);
        Vector512<float> first = Vector512.LoadUnsafe(ref origin);
        Vector512<float> second = Vector512.LoadUnsafe(ref Unsafe.Add(ref origin, 16));
        return Avx512F.PermuteVar16x32x2(first, Stride2EvenIndex512, second);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe Vector512<float> LoadStride2512(float* source)
    {
        Vector512<float> first = Avx512F.LoadVector512(source);
        Vector512<float> second = Avx512F.LoadVector512(source + 16);
        return Avx512F.PermuteVar16x32x2(first, Stride2EvenIndex512, second);
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> VectorLoad(ReadOnlySpan<float> source, int offset)
    {
#if NETSTANDARD2_0
        return Unsafe.ReadUnaligned<Vector<float>>(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref MemoryMarshal.GetReference(source), offset)));
#else
        return Vector.LoadUnsafe(ref Unsafe.Add(ref MemoryMarshal.GetReference(source), offset));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void VectorStore(Span<float> destination, int offset, Vector<float> value)
    {
#if NETSTANDARD2_0
        Unsafe.WriteUnaligned(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref MemoryMarshal.GetReference(destination), offset)),
            value);
#else
        value.StoreUnsafe(ref Unsafe.Add(ref MemoryMarshal.GetReference(destination), offset));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe Vector<float> VectorLoad(float* source)
    {
#if NETSTANDARD2_0
        return Unsafe.ReadUnaligned<Vector<float>>(source);
#else
        return Vector.LoadUnsafe(ref Unsafe.AsRef<float>(source));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void VectorStore(float* destination, Vector<float> value)
    {
#if NETSTANDARD2_0
        Unsafe.WriteUnaligned(destination, value);
#else
        value.StoreUnsafe(ref Unsafe.AsRef<float>(destination));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<int> VectorNonFiniteMask(Vector<float> value)
    {
        Vector<int> finite = Vector.AsVectorInt32(Vector.Equals(value, value));
        Vector<int> infinity = Vector.AsVectorInt32(
            Vector.GreaterThanOrEqual(Vector.Abs(value), new Vector<float>(float.PositiveInfinity)));
        return (finite ^ new Vector<int>(-1)) | infinity;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool VectorAnyNonZero(Vector<int> mask) =>
        !Vector.EqualsAll(mask, Vector<int>.Zero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> VectorAddMul(Vector<float> accumulator, Vector<float> value, float weight) =>
        accumulator + value * new Vector<float>(weight);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> VectorAddMul(Vector<float> accumulator, Vector<float> value, Vector<float> weight) =>
        accumulator + value * weight;

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static unsafe Vector<float> VectorLoadStride2(float* source) =>
        VectorLoadStride2(ref Unsafe.AsRef<float>(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> VectorLoadStride2(ReadOnlySpan<float> source, int offset) =>
        VectorLoadStride2(ref Unsafe.Add(ref MemoryMarshal.GetReference(source), offset));

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static Vector<float> VectorLoadStride2(ref float source)
    {
#if !NETSTANDARD2_0
        if (Sse.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> first = Vector128.LoadUnsafe(ref source);
            Vector128<float> second = Vector128.LoadUnsafe(ref Unsafe.Add(ref source, 4));
            Vector128<float> even = Sse.Shuffle(first, second, 0x88);
            return Unsafe.BitCast<Vector128<float>, Vector<float>>(even);
        }
#endif
        Vector<float> value = default;
        ref float d = ref Unsafe.As<Vector<float>, float>(ref value);
        int width = Vector<float>.Count;
        if (width == 8)
        {
            Unsafe.Add(ref d, 0) = source;
            Unsafe.Add(ref d, 1) = Unsafe.Add(ref source, 2);
            Unsafe.Add(ref d, 2) = Unsafe.Add(ref source, 4);
            Unsafe.Add(ref d, 3) = Unsafe.Add(ref source, 6);
            Unsafe.Add(ref d, 4) = Unsafe.Add(ref source, 8);
            Unsafe.Add(ref d, 5) = Unsafe.Add(ref source, 10);
            Unsafe.Add(ref d, 6) = Unsafe.Add(ref source, 12);
            Unsafe.Add(ref d, 7) = Unsafe.Add(ref source, 14);
            return value;
        }
        if (width == 4)
        {
            Unsafe.Add(ref d, 0) = source;
            Unsafe.Add(ref d, 1) = Unsafe.Add(ref source, 2);
            Unsafe.Add(ref d, 2) = Unsafe.Add(ref source, 4);
            Unsafe.Add(ref d, 3) = Unsafe.Add(ref source, 6);
            return value;
        }
        for (int lane = 0; lane < width; lane++)
            Unsafe.Add(ref d, lane) = Unsafe.Add(ref source, lane * 2);
        return value;
    }
}
