using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class SimdKernels
{
    private static readonly Vector256<float> VAbsMask = Vector256.Create(unchecked((int)0x7fffffff)).AsSingle();

    private static readonly Vector256<float> VSignMask = Vector256.Create(unchecked((int)0x80000000)).AsSingle();

    private static readonly Vector256<float> VOne = Vector256.Create(1f);

    private static readonly Vector256<float> VTwo = Vector256.Create(2f);

    private static readonly Vector256<float> VFour = Vector256.Create(4f);

    private static readonly Vector256<float> VOnePointFive = Vector256.Create(1.5f);

    private static readonly Vector256<float> VThree = Vector256.Create(3f);

    private static readonly Vector256<float> VInvSqrtTwo = Vector256.Create(0.70710678118654752f);

    private static readonly Vector256<float> VHalf = Vector256.Create(0.5f);

    private static readonly Vector256<float> VSmall0 = Vector256.Create(1.0590875083315439e-6f);

    private static readonly Vector256<float> VSmall1 = Vector256.Create(-1.3906452410711274e-5f);

    private static readonly Vector256<float> VSmall2 = Vector256.Create(1.1955437252428243e-4f);

    private static readonly Vector256<float> VSmall3 = Vector256.Create(-8.542475960079766e-4f);

    private static readonly Vector256<float> VSmall4 = Vector256.Create(5.223771899427153e-3f);

    private static readonly Vector256<float> VSmall5 = Vector256.Create(-2.686612888828867e-2f);

    private static readonly Vector256<float> VSmall6 = Vector256.Create(1.128379122662546e-1f);

    private static readonly Vector256<float> VSmall7 = Vector256.Create(-3.761263888304388e-1f);

    private static readonly Vector256<float> VSmall8 = Vector256.Create(1.1283791670929921f);

    private static readonly Vector256<float> VMiddle0 = Vector256.Create(-2.400667527836574e-3f);

    private static readonly Vector256<float> VMiddle1 = Vector256.Create(-3.8855162788028288e-3f);

    private static readonly Vector256<float> VMiddle2 = Vector256.Create(1.9332860298601401e-2f);

    private static readonly Vector256<float> VMiddle3 = Vector256.Create(-1.501360723487143e-2f);

    private static readonly Vector256<float> VMiddle4 = Vector256.Create(-4.4599369472787913e-2f);

    private static readonly Vector256<float> VMiddle5 = Vector256.Create(1.3876136033191724e-1f);

    private static readonly Vector256<float> VMiddle6 = Vector256.Create(-1.7839541988688759e-1f);

    private static readonly Vector256<float> VMiddle7 = Vector256.Create(1.1893013063163335e-1f);

    private static readonly Vector256<float> VMiddle8 = Vector256.Create(9.661051464140682e-1f);

    private static readonly Vector256<float> VLarge0 = Vector256.Create(-8.875076532056503e-5f);

    private static readonly Vector256<float> VLarge1 = Vector256.Create(3.880528563007222e-4f);

    private static readonly Vector256<float> VLarge2 = Vector256.Create(-7.781201071142843e-4f);

    private static readonly Vector256<float> VLarge3 = Vector256.Create(1.0255980996254569e-3f);

    private static readonly Vector256<float> VLarge4 = Vector256.Create(-1.0307062838910627e-3f);

    private static readonly Vector256<float> VLarge5 = Vector256.Create(7.858608012972956e-4f);

    private static readonly Vector256<float> VLarge6 = Vector256.Create(-4.193605385522123e-4f);

    private static readonly Vector256<float> VLarge7 = Vector256.Create(1.3951109720745927e-4f);

    private static readonly Vector256<float> VLarge8 = Vector256.Create(9.999779388683872e-1f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ExpApproxVector(Vector256<float> value)
    {
        Vector256<float> log2e = Vector256.Create(1.4426950408889634f);
        Vector256<float> c1 = Vector256.Create(0.693359375f); Vector256<float> c2 = Vector256.Create(-2.12194440e-4f);
        Vector256<float> min = Vector256.Create(-88.3762626647949f); Vector256<float> max = Vector256.Create(88.3762626647949f);
        value = Avx.Max(value, min); value = Avx.Min(value, max);
        Vector256<float> scaled = Avx.Add(Avx.Multiply(value, log2e), Vector256.Create(0.5f));
        Vector256<int> exponent = Avx.ConvertToVector256Int32WithTruncation(scaled);
        Vector256<float> integerPart = Avx.ConvertToVector256Single(exponent);
        Vector256<float> correction = Avx.And(Avx.Compare(integerPart, scaled, FloatComparisonMode.OrderedGreaterThanNonSignaling), Vector256.Create(1f));
        integerPart = Avx.Subtract(integerPart, correction);
        Vector256<float> reduced = Avx.Subtract(value, Avx.Multiply(integerPart, c1));
        reduced = Avx.Subtract(reduced, Avx.Multiply(integerPart, c2));
        Vector256<float> square = Avx.Multiply(reduced, reduced);
        Vector256<float> p = Vector256.Create(2.4801587302e-5f);
        p = Avx.Add(Vector256.Create(1.9841269841e-4f), Avx.Multiply(reduced, p));
        p = Avx.Add(Vector256.Create(1.3888888889e-3f), Avx.Multiply(reduced, p));
        p = Avx.Add(Vector256.Create(8.3333333333e-3f), Avx.Multiply(reduced, p));
        p = Avx.Add(Vector256.Create(4.1666666667e-2f), Avx.Multiply(reduced, p));
        p = Avx.Add(Vector256.Create(1.6666666667e-1f), Avx.Multiply(reduced, p));
        p = Avx.Add(Vector256.Create(5.0e-1f), Avx.Multiply(reduced, p));
        p = Avx.Add(reduced, Avx.Multiply(square, p));
        p = Avx.Add(Vector256.Create(1f), p);
        exponent = Avx.ConvertToVector256Int32WithTruncation(integerPart);
        Vector256<int> bits = Avx2.Add(exponent, Vector256.Create(127));
        Vector256<float> power = Avx2.ShiftLeftLogical(bits, 23).AsSingle();
        return Avx.Multiply(p, power);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ErfVector(Vector256<float> value)
    {
        Vector256<float> abs = Avx.And(value, VAbsMask);
        Vector256<float> sign = Avx.And(value, VSignMask);
        Vector256<float> sq = Avx.Multiply(abs, abs);
        // Feature-map values are spatially correlated, so many vectors stay
        // entirely inside one approximation interval.  Avoid evaluating all
        // three degree-8 polynomials in that common case.  Mixed vectors use
        // the original path below, preserving the exact approximation and
        // its deterministic output.
        const int AllLanes = 0xff;
        if (Avx.MoveMask(Avx.Compare(abs, VOne, FloatComparisonMode.OrderedLessThanNonSignaling)) == AllLanes)
            return Avx.Or(PolySmall(abs, sq), sign);
        if (Avx.MoveMask(Avx.Compare(abs, VFour, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling)) == AllLanes)
            return Avx.Or(VOne, sign);
        int atLeastOne = Avx.MoveMask(Avx.Compare(abs, VOne, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling));
        int belowTwo = Avx.MoveMask(Avx.Compare(abs, VTwo, FloatComparisonMode.OrderedLessThanNonSignaling));
        if ((atLeastOne & belowTwo) == AllLanes)
            return Avx.Or(PolyMiddle(abs), sign);
        int atLeastTwo = Avx.MoveMask(Avx.Compare(abs, VTwo, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling));
        int belowFour = Avx.MoveMask(Avx.Compare(abs, VFour, FloatComparisonMode.OrderedLessThanNonSignaling));
        if ((atLeastTwo & belowFour) == AllLanes)
            return Avx.Or(PolyLarge(abs), sign);
        Vector256<float> small = PolySmall(abs, sq); Vector256<float> middle = PolyMiddle(abs); Vector256<float> large = PolyLarge(abs);
        Vector256<float> result = Avx.BlendVariable(large, middle, Avx.Compare(abs, VTwo, FloatComparisonMode.OrderedLessThanNonSignaling));
        result = Avx.BlendVariable(result, small, Avx.Compare(abs, VOne, FloatComparisonMode.OrderedLessThanNonSignaling));
        result = Avx.BlendVariable(result, VOne, Avx.Compare(abs, VFour, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling));
        result = Avx.Or(result, sign);
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> MulAdd(Vector256<float> x, Vector256<float> y, Vector256<float> z) =>
        Fma.IsSupported ? Fma.MultiplyAdd(x, y, z) : Avx.Add(Avx.Multiply(x, y), z);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> PolySmall(Vector256<float> a, Vector256<float> s) { Vector256<float> p = VSmall0; p = MulAdd(s, p, VSmall1); p = MulAdd(s, p, VSmall2); p = MulAdd(s, p, VSmall3); p = MulAdd(s, p, VSmall4); p = MulAdd(s, p, VSmall5); p = MulAdd(s, p, VSmall6); p = MulAdd(s, p, VSmall7); p = MulAdd(s, p, VSmall8); return Avx.Multiply(a, p); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> PolyMiddle(Vector256<float> a) { Vector256<float> z = Avx.Subtract(a, VOnePointFive); Vector256<float> p = VMiddle0; p = MulAdd(z, p, VMiddle1); p = MulAdd(z, p, VMiddle2); p = MulAdd(z, p, VMiddle3); p = MulAdd(z, p, VMiddle4); p = MulAdd(z, p, VMiddle5); p = MulAdd(z, p, VMiddle6); p = MulAdd(z, p, VMiddle7); return MulAdd(z, p, VMiddle8); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> PolyLarge(Vector256<float> a) { Vector256<float> z = Avx.Subtract(a, VThree); Vector256<float> p = VLarge0; p = MulAdd(z, p, VLarge1); p = MulAdd(z, p, VLarge2); p = MulAdd(z, p, VLarge3); p = MulAdd(z, p, VLarge4); p = MulAdd(z, p, VLarge5); p = MulAdd(z, p, VLarge6); p = MulAdd(z, p, VLarge7); return MulAdd(z, p, VLarge8); }
}
