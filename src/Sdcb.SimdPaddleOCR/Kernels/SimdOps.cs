using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
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
    internal static Vector<float> VectorAddMul(Vector<float> accumulator, Vector<float> value, float weight)
    {
#if !NETSTANDARD2_0
        if (AdvSimd.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> acc = Unsafe.BitCast<Vector<float>, Vector128<float>>(accumulator);
            Vector128<float> val = Unsafe.BitCast<Vector<float>, Vector128<float>>(value);
            return Unsafe.BitCast<Vector128<float>, Vector<float>>(
                AdvSimd.FusedMultiplyAdd(acc, val, AdvSimd.DuplicateToVector128(weight)));
        }
        else if (Fma.IsSupported && Vector<float>.Count == 8)
        {
            Vector256<float> acc = Unsafe.BitCast<Vector<float>, Vector256<float>>(accumulator);
            Vector256<float> val = Unsafe.BitCast<Vector<float>, Vector256<float>>(value);
            return Unsafe.BitCast<Vector256<float>, Vector<float>>(
                Fma.MultiplyAdd(val, Vector256.Create(weight), acc));
        }
        else
#endif
        {
            return accumulator + value * new Vector<float>(weight);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> VectorAddMul(Vector<float> accumulator, Vector<float> value, Vector<float> weight)
    {
#if !NETSTANDARD2_0
        if (AdvSimd.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> acc = Unsafe.BitCast<Vector<float>, Vector128<float>>(accumulator);
            Vector128<float> val = Unsafe.BitCast<Vector<float>, Vector128<float>>(value);
            Vector128<float> w = Unsafe.BitCast<Vector<float>, Vector128<float>>(weight);
            return Unsafe.BitCast<Vector128<float>, Vector<float>>(AdvSimd.FusedMultiplyAdd(acc, val, w));
        }
        else if (Fma.IsSupported && Vector<float>.Count == 8)
        {
            Vector256<float> acc = Unsafe.BitCast<Vector<float>, Vector256<float>>(accumulator);
            Vector256<float> val = Unsafe.BitCast<Vector<float>, Vector256<float>>(value);
            Vector256<float> w = Unsafe.BitCast<Vector<float>, Vector256<float>>(weight);
            return Unsafe.BitCast<Vector256<float>, Vector<float>>(Fma.MultiplyAdd(val, w, acc));
        }
        else
#endif
        {
            return accumulator + value * weight;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe Vector<float> VectorLoadStride2(float* source) =>
        VectorLoadStride2(ref Unsafe.AsRef<float>(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> VectorLoadStride2(ReadOnlySpan<float> source, int offset) =>
        VectorLoadStride2(ref Unsafe.Add(ref MemoryMarshal.GetReference(source), offset));

    // Must stay inlinable: the stride-2 kernels issue this per tap, and the
    // generic scatter body below is too large for the inliner, so it lives apart.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> VectorLoadStride2(ref float source)
    {
#if !NETSTANDARD2_0
        if (AdvSimd.Arm64.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> first = Vector128.LoadUnsafe(ref source);
            Vector128<float> second = Vector128.LoadUnsafe(ref Unsafe.Add(ref source, 4));
            return AdvSimd.Arm64.UnzipEven(first, second).AsVector();
        }
        else if (Sse.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> first = Vector128.LoadUnsafe(ref source);
            Vector128<float> second = Vector128.LoadUnsafe(ref Unsafe.Add(ref source, 4));
            return Sse.Shuffle(first, second, 0x88).AsVector();
        }
        else
#endif
        {
            return VectorLoadStride2Generic(ref source);
        }
    }

    /// <summary>
    /// <c>acc[k] += value * weights[k]</c> for eight consecutive packed weights.
    /// On AdvSimd this is two 128-bit weight loads plus eight <c>FMLA (by element)</c>
    /// instead of a scalar load + broadcast per FMA; every call in the body is a
    /// hardware intrinsic, so it costs the inliner nothing beyond this method.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void VectorAddMulPacked8(
        ref Vector<float> a0, ref Vector<float> a1, ref Vector<float> a2, ref Vector<float> a3,
        ref Vector<float> a4, ref Vector<float> a5, ref Vector<float> a6, ref Vector<float> a7,
        Vector<float> value, float* weights)
    {
#if !NETSTANDARD2_0
        if (AdvSimd.Arm64.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> v = value.AsVector128();
            Vector128<float> low = Vector128.Load(weights), high = Vector128.Load(weights + 4);
            a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0.AsVector128(), v, low, 0).AsVector();
            a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1.AsVector128(), v, low, 1).AsVector();
            a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2.AsVector128(), v, low, 2).AsVector();
            a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3.AsVector128(), v, low, 3).AsVector();
            a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4.AsVector128(), v, high, 0).AsVector();
            a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5.AsVector128(), v, high, 1).AsVector();
            a6 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a6.AsVector128(), v, high, 2).AsVector();
            a7 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a7.AsVector128(), v, high, 3).AsVector();
            return;
        }
#endif
        // Do not hide this behind another 8-ref helper. That generic fallback
        // failed to inline on ns2 and was the 9c54d56 win-x64 Vector regression.
        a0 = VectorAddMul(a0, value, weights[0]); a1 = VectorAddMul(a1, value, weights[1]);
        a2 = VectorAddMul(a2, value, weights[2]); a3 = VectorAddMul(a3, value, weights[3]);
        a4 = VectorAddMul(a4, value, weights[4]); a5 = VectorAddMul(a5, value, weights[5]);
        a6 = VectorAddMul(a6, value, weights[6]); a7 = VectorAddMul(a7, value, weights[7]);
    }

    /// <summary>
    /// <see cref="VectorAddMulPacked8"/> for two spatial tiles (<paramref name="value"/>
    /// and <paramref name="other"/>) sharing one set of eight packed weights.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void VectorAddMulPacked8x2(
        ref Vector<float> a0, ref Vector<float> a1, ref Vector<float> a2, ref Vector<float> a3,
        ref Vector<float> a4, ref Vector<float> a5, ref Vector<float> a6, ref Vector<float> a7,
        ref Vector<float> c0, ref Vector<float> c1, ref Vector<float> c2, ref Vector<float> c3,
        ref Vector<float> c4, ref Vector<float> c5, ref Vector<float> c6, ref Vector<float> c7,
        Vector<float> value, Vector<float> other, float* weights)
    {
#if !NETSTANDARD2_0
        if (AdvSimd.Arm64.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> v = value.AsVector128(), u = other.AsVector128();
            Vector128<float> low = Vector128.Load(weights), high = Vector128.Load(weights + 4);
            a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0.AsVector128(), v, low, 0).AsVector();
            c0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c0.AsVector128(), u, low, 0).AsVector();
            a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1.AsVector128(), v, low, 1).AsVector();
            c1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c1.AsVector128(), u, low, 1).AsVector();
            a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2.AsVector128(), v, low, 2).AsVector();
            c2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c2.AsVector128(), u, low, 2).AsVector();
            a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3.AsVector128(), v, low, 3).AsVector();
            c3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c3.AsVector128(), u, low, 3).AsVector();
            a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4.AsVector128(), v, high, 0).AsVector();
            c4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c4.AsVector128(), u, high, 0).AsVector();
            a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5.AsVector128(), v, high, 1).AsVector();
            c5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c5.AsVector128(), u, high, 1).AsVector();
            a6 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a6.AsVector128(), v, high, 2).AsVector();
            c6 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c6.AsVector128(), u, high, 2).AsVector();
            a7 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a7.AsVector128(), v, high, 3).AsVector();
            c7 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c7.AsVector128(), u, high, 3).AsVector();
            return;
        }
#endif
        a0 = VectorAddMul(a0, value, weights[0]); c0 = VectorAddMul(c0, other, weights[0]);
        a1 = VectorAddMul(a1, value, weights[1]); c1 = VectorAddMul(c1, other, weights[1]);
        a2 = VectorAddMul(a2, value, weights[2]); c2 = VectorAddMul(c2, other, weights[2]);
        a3 = VectorAddMul(a3, value, weights[3]); c3 = VectorAddMul(c3, other, weights[3]);
        a4 = VectorAddMul(a4, value, weights[4]); c4 = VectorAddMul(c4, other, weights[4]);
        a5 = VectorAddMul(a5, value, weights[5]); c5 = VectorAddMul(c5, other, weights[5]);
        a6 = VectorAddMul(a6, value, weights[6]); c6 = VectorAddMul(c6, other, weights[6]);
        a7 = VectorAddMul(a7, value, weights[7]); c7 = VectorAddMul(c7, other, weights[7]);
    }

    /// <summary><c>acc[k] += value * weights[k]</c> for four consecutive packed weights.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void VectorAddMulPacked4(
        ref Vector<float> a0, ref Vector<float> a1, ref Vector<float> a2, ref Vector<float> a3,
        Vector<float> value, float* weights)
    {
#if !NETSTANDARD2_0
        if (AdvSimd.Arm64.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> v = value.AsVector128();
            Vector128<float> w = Vector128.Load(weights);
            a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0.AsVector128(), v, w, 0).AsVector();
            a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1.AsVector128(), v, w, 1).AsVector();
            a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2.AsVector128(), v, w, 2).AsVector();
            a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3.AsVector128(), v, w, 3).AsVector();
            return;
        }
#endif
        a0 = VectorAddMul(a0, value, weights[0]); a1 = VectorAddMul(a1, value, weights[1]);
        a2 = VectorAddMul(a2, value, weights[2]); a3 = VectorAddMul(a3, value, weights[3]);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static Vector<float> VectorLoadStride2Generic(ref float source)
    {
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
        }
        else if (width == 4)
        {
            Unsafe.Add(ref d, 0) = source;
            Unsafe.Add(ref d, 1) = Unsafe.Add(ref source, 2);
            Unsafe.Add(ref d, 2) = Unsafe.Add(ref source, 4);
            Unsafe.Add(ref d, 3) = Unsafe.Add(ref source, 6);
        }
        else
        {
            for (int lane = 0; lane < width; lane++)
                Unsafe.Add(ref d, lane) = Unsafe.Add(ref source, lane * 2);
        }
        return value;
    }
}
