using System.Numerics;
using System.Runtime.CompilerServices;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class SimdKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector<float> VecLoad(float* source)
    {
#if NETSTANDARD2_0
        return Unsafe.ReadUnaligned<Vector<float>>(source);
#else
        return Vector.LoadUnsafe(ref Unsafe.AsRef<float>(source));
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void VecStore(float* destination, Vector<float> value)
    {
#if NETSTANDARD2_0
        Unsafe.WriteUnaligned(destination, value);
#else
        value.StoreUnsafe(ref Unsafe.AsRef<float>(destination));
#endif
    }

    private static readonly Vector<float> VecExpLog2e = new(1.4426950408889634f);
    private static readonly Vector<float> VecExpC1 = new(0.693359375f);
    private static readonly Vector<float> VecExpC2 = new(-2.12194440e-4f);
    private static readonly Vector<float> VecExpMin = new(-88.3762626647949f);
    private static readonly Vector<float> VecExpMax = new(88.3762626647949f);
    private static readonly Vector<float> VecExpHalf = new(0.5f);
    private static readonly Vector<float> VecExpOne = new(1f);
    private static readonly Vector<float> VecExpP0 = new(2.4801587302e-5f);
    private static readonly Vector<float> VecExpP1 = new(1.9841269841e-4f);
    private static readonly Vector<float> VecExpP2 = new(1.3888888889e-3f);
    private static readonly Vector<float> VecExpP3 = new(8.3333333333e-3f);
    private static readonly Vector<float> VecExpP4 = new(4.1666666667e-2f);
    private static readonly Vector<float> VecExpP5 = new(1.6666666667e-1f);
    private static readonly Vector<int> VecExpBias = new(127);
    private static readonly Vector<int> VecExpShift23 = new(1 << 23);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static Vector<float> ExpExactVector(Vector<float> value)
    {
        value = Vector.Max(value, VecExpMin);
        value = Vector.Min(value, VecExpMax);
        Vector<float> scaled = VecMulAdd(value, VecExpLog2e, VecExpHalf);
        Vector<int> exponent = Vector.ConvertToInt32(scaled);
        Vector<float> integerPart = Vector.ConvertToSingle(exponent);
        integerPart -= Vector.ConditionalSelect(
            Vector.GreaterThan(integerPart, scaled), VecExpOne, Vector<float>.Zero);
        Vector<float> reduced = value - integerPart * VecExpC1;
        reduced -= integerPart * VecExpC2;
        Vector<float> square = reduced * reduced;
        Vector<float> p = VecExpP0;
        p = VecMulAdd(reduced, p, VecExpP1);
        p = VecMulAdd(reduced, p, VecExpP2);
        p = VecMulAdd(reduced, p, VecExpP3);
        p = VecMulAdd(reduced, p, VecExpP4);
        p = VecMulAdd(reduced, p, VecExpP5);
        p = VecMulAdd(reduced, p, VecExpHalf);
        p = reduced + square * p;
        p += VecExpOne;
        exponent = Vector.ConvertToInt32(integerPart);
        Vector<int> bits = (exponent + VecExpBias) * VecExpShift23;
        return p * Vector.AsVectorSingle(bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> SigmoidExactVector(Vector<float> value)
    {
        Vector<float> exponent = ExpExactVector(-Vector.Abs(value));
        Vector<float> positiveResult = VecExpOne / (VecExpOne + exponent);
        return Vector.ConditionalSelect(
            Vector.LessThan(value, Vector<float>.Zero),
            VecExpOne - positiveResult,
            positiveResult);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static Vector<float> ErfExactVector(Vector<float> value)
    {
#if !NETSTANDARD2_0
        if (Sse.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> packed = Unsafe.BitCast<Vector<float>, Vector128<float>>(value);
            return Unsafe.BitCast<Vector128<float>, Vector<float>>(ErfVectorSse(packed));
        }
#endif
        return ErfVectorNumerics(value);
    }

    private static readonly Vector<float> VecErfOne = new(1f);
    private static readonly Vector<float> VecErfTwo = new(2f);
    private static readonly Vector<float> VecErfFour = new(4f);
    private static readonly Vector<float> VecErfOnePointFive = new(1.5f);
    private static readonly Vector<float> VecErfThree = new(3f);
    private static readonly Vector<float> VecErfSmall0 = new(1.0590875083315439e-6f);
    private static readonly Vector<float> VecErfSmall1 = new(-1.3906452410711274e-5f);
    private static readonly Vector<float> VecErfSmall2 = new(1.1955437252428243e-4f);
    private static readonly Vector<float> VecErfSmall3 = new(-8.542475960079766e-4f);
    private static readonly Vector<float> VecErfSmall4 = new(5.223771899427153e-3f);
    private static readonly Vector<float> VecErfSmall5 = new(-2.686612888828867e-2f);
    private static readonly Vector<float> VecErfSmall6 = new(1.128379122662546e-1f);
    private static readonly Vector<float> VecErfSmall7 = new(-3.761263888304388e-1f);
    private static readonly Vector<float> VecErfSmall8 = new(1.1283791670929921f);
    private static readonly Vector<float> VecErfMiddle0 = new(-2.400667527836574e-3f);
    private static readonly Vector<float> VecErfMiddle1 = new(-3.8855162788028288e-3f);
    private static readonly Vector<float> VecErfMiddle2 = new(1.9332860298601401e-2f);
    private static readonly Vector<float> VecErfMiddle3 = new(-1.501360723487143e-2f);
    private static readonly Vector<float> VecErfMiddle4 = new(-4.4599369472787913e-2f);
    private static readonly Vector<float> VecErfMiddle5 = new(1.3876136033191724e-1f);
    private static readonly Vector<float> VecErfMiddle6 = new(-1.7839541988688759e-1f);
    private static readonly Vector<float> VecErfMiddle7 = new(1.1893013063163335e-1f);
    private static readonly Vector<float> VecErfMiddle8 = new(9.661051464140682e-1f);
    private static readonly Vector<float> VecErfLarge0 = new(-8.875076532056503e-5f);
    private static readonly Vector<float> VecErfLarge1 = new(3.880528563007222e-4f);
    private static readonly Vector<float> VecErfLarge2 = new(-7.781201071142843e-4f);
    private static readonly Vector<float> VecErfLarge3 = new(1.0255980996254569e-3f);
    private static readonly Vector<float> VecErfLarge4 = new(-1.0307062838910627e-3f);
    private static readonly Vector<float> VecErfLarge5 = new(7.858608012972956e-4f);
    private static readonly Vector<float> VecErfLarge6 = new(-4.193605385522123e-4f);
    private static readonly Vector<float> VecErfLarge7 = new(1.3951109720745927e-4f);
    private static readonly Vector<float> VecErfLarge8 = new(9.999779388683872e-1f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> VecMulAdd(Vector<float> x, Vector<float> y, Vector<float> z) => x * y + z;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> VecCopySign(Vector<float> magnitude, Vector<float> signSource) =>
        Vector.ConditionalSelect(Vector.LessThan(signSource, Vector<float>.Zero), -magnitude, magnitude);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static Vector<float> ErfVectorNumerics(Vector<float> value)
    {
        Vector<float> abs = Vector.Abs(value);
        Vector<float> sq = abs * abs;
        // Feature-map values are spatially correlated, so many vectors stay
        // entirely inside one approximation interval. Mixed vectors still
        // evaluate all three polynomials, matching the AVX/SSE blend.
        if (Vector.LessThanAll(abs, VecErfOne))
            return VecCopySign(ErfPolySmall(abs, sq), value);
        if (Vector.GreaterThanOrEqualAll(abs, VecErfFour))
            return VecCopySign(VecErfOne, value);
        if (Vector.GreaterThanOrEqualAll(abs, VecErfOne) && Vector.LessThanAll(abs, VecErfTwo))
            return VecCopySign(ErfPolyMiddle(abs), value);
        if (Vector.GreaterThanOrEqualAll(abs, VecErfTwo) && Vector.LessThanAll(abs, VecErfFour))
            return VecCopySign(ErfPolyLarge(abs), value);
        Vector<float> small = ErfPolySmall(abs, sq);
        Vector<float> middle = ErfPolyMiddle(abs);
        Vector<float> large = ErfPolyLarge(abs);
        Vector<float> result = Vector.ConditionalSelect(Vector.LessThan(abs, VecErfTwo), middle, large);
        result = Vector.ConditionalSelect(Vector.LessThan(abs, VecErfOne), small, result);
        result = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(abs, VecErfFour), VecErfOne, result);
        return VecCopySign(result, value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ErfPolySmall(Vector<float> a, Vector<float> s)
    {
        Vector<float> p = VecErfSmall0;
        p = VecMulAdd(s, p, VecErfSmall1);
        p = VecMulAdd(s, p, VecErfSmall2);
        p = VecMulAdd(s, p, VecErfSmall3);
        p = VecMulAdd(s, p, VecErfSmall4);
        p = VecMulAdd(s, p, VecErfSmall5);
        p = VecMulAdd(s, p, VecErfSmall6);
        p = VecMulAdd(s, p, VecErfSmall7);
        p = VecMulAdd(s, p, VecErfSmall8);
        return a * p;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ErfPolyMiddle(Vector<float> a)
    {
        Vector<float> z = a - VecErfOnePointFive;
        Vector<float> p = VecErfMiddle0;
        p = VecMulAdd(z, p, VecErfMiddle1);
        p = VecMulAdd(z, p, VecErfMiddle2);
        p = VecMulAdd(z, p, VecErfMiddle3);
        p = VecMulAdd(z, p, VecErfMiddle4);
        p = VecMulAdd(z, p, VecErfMiddle5);
        p = VecMulAdd(z, p, VecErfMiddle6);
        p = VecMulAdd(z, p, VecErfMiddle7);
        return VecMulAdd(z, p, VecErfMiddle8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ErfPolyLarge(Vector<float> a)
    {
        Vector<float> z = a - VecErfThree;
        Vector<float> p = VecErfLarge0;
        p = VecMulAdd(z, p, VecErfLarge1);
        p = VecMulAdd(z, p, VecErfLarge2);
        p = VecMulAdd(z, p, VecErfLarge3);
        p = VecMulAdd(z, p, VecErfLarge4);
        p = VecMulAdd(z, p, VecErfLarge5);
        p = VecMulAdd(z, p, VecErfLarge6);
        p = VecMulAdd(z, p, VecErfLarge7);
        return VecMulAdd(z, p, VecErfLarge8);
    }

#if !NETSTANDARD2_0
    private static readonly Vector128<float> SseAbsMask = Vector128.Create(unchecked((int)0x7fffffff)).AsSingle();

    private static readonly Vector128<float> SseSignMask = Vector128.Create(unchecked((int)0x80000000)).AsSingle();

    private static readonly Vector128<float> SseOne = Vector128.Create(1f);

    private static readonly Vector128<float> SseTwo = Vector128.Create(2f);

    private static readonly Vector128<float> SseFour = Vector128.Create(4f);

    private static readonly Vector128<float> SseOnePointFive = Vector128.Create(1.5f);

    private static readonly Vector128<float> SseThree = Vector128.Create(3f);

    private static readonly Vector128<float> SseSmall0 = Vector128.Create(1.0590875083315439e-6f);

    private static readonly Vector128<float> SseSmall1 = Vector128.Create(-1.3906452410711274e-5f);

    private static readonly Vector128<float> SseSmall2 = Vector128.Create(1.1955437252428243e-4f);

    private static readonly Vector128<float> SseSmall3 = Vector128.Create(-8.542475960079766e-4f);

    private static readonly Vector128<float> SseSmall4 = Vector128.Create(5.223771899427153e-3f);

    private static readonly Vector128<float> SseSmall5 = Vector128.Create(-2.686612888828867e-2f);

    private static readonly Vector128<float> SseSmall6 = Vector128.Create(1.128379122662546e-1f);

    private static readonly Vector128<float> SseSmall7 = Vector128.Create(-3.761263888304388e-1f);

    private static readonly Vector128<float> SseSmall8 = Vector128.Create(1.1283791670929921f);

    private static readonly Vector128<float> SseMiddle0 = Vector128.Create(-2.400667527836574e-3f);

    private static readonly Vector128<float> SseMiddle1 = Vector128.Create(-3.8855162788028288e-3f);

    private static readonly Vector128<float> SseMiddle2 = Vector128.Create(1.9332860298601401e-2f);

    private static readonly Vector128<float> SseMiddle3 = Vector128.Create(-1.501360723487143e-2f);

    private static readonly Vector128<float> SseMiddle4 = Vector128.Create(-4.4599369472787913e-2f);

    private static readonly Vector128<float> SseMiddle5 = Vector128.Create(1.3876136033191724e-1f);

    private static readonly Vector128<float> SseMiddle6 = Vector128.Create(-1.7839541988688759e-1f);

    private static readonly Vector128<float> SseMiddle7 = Vector128.Create(1.1893013063163335e-1f);

    private static readonly Vector128<float> SseMiddle8 = Vector128.Create(9.661051464140682e-1f);

    private static readonly Vector128<float> SseLarge0 = Vector128.Create(-8.875076532056503e-5f);

    private static readonly Vector128<float> SseLarge1 = Vector128.Create(3.880528563007222e-4f);

    private static readonly Vector128<float> SseLarge2 = Vector128.Create(-7.781201071142843e-4f);

    private static readonly Vector128<float> SseLarge3 = Vector128.Create(1.0255980996254569e-3f);

    private static readonly Vector128<float> SseLarge4 = Vector128.Create(-1.0307062838910627e-3f);

    private static readonly Vector128<float> SseLarge5 = Vector128.Create(7.858608012972956e-4f);

    private static readonly Vector128<float> SseLarge6 = Vector128.Create(-4.193605385522123e-4f);

    private static readonly Vector128<float> SseLarge7 = Vector128.Create(1.3951109720745927e-4f);

    private static readonly Vector128<float> SseLarge8 = Vector128.Create(9.999779388683872e-1f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> SseMulAdd(Vector128<float> x, Vector128<float> y, Vector128<float> z) =>
        Sse.Add(Sse.Multiply(x, y), z);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> SseBlend(Vector128<float> falseValue, Vector128<float> trueValue, Vector128<float> mask) =>
        Sse.Or(Sse.And(mask, trueValue), Sse.AndNot(mask, falseValue));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ErfVectorSse(Vector128<float> value)
    {
        Vector128<float> abs = Sse.And(value, SseAbsMask);
        Vector128<float> sign = Sse.And(value, SseSignMask);
        Vector128<float> sq = Sse.Multiply(abs, abs);
        Vector128<float> small = ErfPolySmallSse(abs, sq);
        Vector128<float> middle = ErfPolyMiddleSse(abs);
        Vector128<float> large = ErfPolyLargeSse(abs);
        Vector128<float> result = SseBlend(large, middle, Sse.CompareLessThan(abs, SseTwo));
        result = SseBlend(result, small, Sse.CompareLessThan(abs, SseOne));
        result = SseBlend(result, SseOne, Sse.CompareNotLessThan(abs, SseFour));
        return Sse.Or(result, sign);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ErfPolySmallSse(Vector128<float> a, Vector128<float> s)
    {
        Vector128<float> p = SseSmall0;
        p = SseMulAdd(s, p, SseSmall1);
        p = SseMulAdd(s, p, SseSmall2);
        p = SseMulAdd(s, p, SseSmall3);
        p = SseMulAdd(s, p, SseSmall4);
        p = SseMulAdd(s, p, SseSmall5);
        p = SseMulAdd(s, p, SseSmall6);
        p = SseMulAdd(s, p, SseSmall7);
        p = SseMulAdd(s, p, SseSmall8);
        return Sse.Multiply(a, p);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ErfPolyMiddleSse(Vector128<float> a)
    {
        Vector128<float> z = Sse.Subtract(a, SseOnePointFive);
        Vector128<float> p = SseMiddle0;
        p = SseMulAdd(z, p, SseMiddle1);
        p = SseMulAdd(z, p, SseMiddle2);
        p = SseMulAdd(z, p, SseMiddle3);
        p = SseMulAdd(z, p, SseMiddle4);
        p = SseMulAdd(z, p, SseMiddle5);
        p = SseMulAdd(z, p, SseMiddle6);
        p = SseMulAdd(z, p, SseMiddle7);
        return SseMulAdd(z, p, SseMiddle8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ErfPolyLargeSse(Vector128<float> a)
    {
        Vector128<float> z = Sse.Subtract(a, SseThree);
        Vector128<float> p = SseLarge0;
        p = SseMulAdd(z, p, SseLarge1);
        p = SseMulAdd(z, p, SseLarge2);
        p = SseMulAdd(z, p, SseLarge3);
        p = SseMulAdd(z, p, SseLarge4);
        p = SseMulAdd(z, p, SseLarge5);
        p = SseMulAdd(z, p, SseLarge6);
        p = SseMulAdd(z, p, SseLarge7);
        return SseMulAdd(z, p, SseLarge8);
    }
#endif
}
