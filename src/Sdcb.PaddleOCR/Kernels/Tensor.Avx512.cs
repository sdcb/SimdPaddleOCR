using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class SimdKernels
{
    private static readonly Vector512<float> V512AbsMask = Vector512.Create(unchecked((int)0x7fffffff)).AsSingle();

    private static readonly Vector512<float> V512SignMask = Vector512.Create(unchecked((int)0x80000000)).AsSingle();

    private static readonly Vector512<float> V512One = Vector512.Create(1f);

    private static readonly Vector512<float> V512Two = Vector512.Create(2f);

    private static readonly Vector512<float> V512Four = Vector512.Create(4f);

    private static readonly Vector512<float> V512OnePointFive = Vector512.Create(1.5f);

    private static readonly Vector512<float> V512Three = Vector512.Create(3f);

    private static readonly Vector512<float> V512InvSqrtTwo = Vector512.Create(0.70710678118654752f);

    private static readonly Vector512<float> V512Half = Vector512.Create(0.5f);

    private static readonly Vector512<float> V512Small0 = Vector512.Create(1.0590875083315439e-6f);

    private static readonly Vector512<float> V512Small1 = Vector512.Create(-1.3906452410711274e-5f);

    private static readonly Vector512<float> V512Small2 = Vector512.Create(1.1955437252428243e-4f);

    private static readonly Vector512<float> V512Small3 = Vector512.Create(-8.542475960079766e-4f);

    private static readonly Vector512<float> V512Small4 = Vector512.Create(5.223771899427153e-3f);

    private static readonly Vector512<float> V512Small5 = Vector512.Create(-2.686612888828867e-2f);

    private static readonly Vector512<float> V512Small6 = Vector512.Create(1.128379122662546e-1f);

    private static readonly Vector512<float> V512Small7 = Vector512.Create(-3.761263888304388e-1f);

    private static readonly Vector512<float> V512Small8 = Vector512.Create(1.1283791670929921f);

    private static readonly Vector512<float> V512Middle0 = Vector512.Create(-2.400667527836574e-3f);

    private static readonly Vector512<float> V512Middle1 = Vector512.Create(-3.8855162788028288e-3f);

    private static readonly Vector512<float> V512Middle2 = Vector512.Create(1.9332860298601401e-2f);

    private static readonly Vector512<float> V512Middle3 = Vector512.Create(-1.501360723487143e-2f);

    private static readonly Vector512<float> V512Middle4 = Vector512.Create(-4.4599369472787913e-2f);

    private static readonly Vector512<float> V512Middle5 = Vector512.Create(1.3876136033191724e-1f);

    private static readonly Vector512<float> V512Middle6 = Vector512.Create(-1.7839541988688759e-1f);

    private static readonly Vector512<float> V512Middle7 = Vector512.Create(1.1893013063163335e-1f);

    private static readonly Vector512<float> V512Middle8 = Vector512.Create(9.661051464140682e-1f);

    private static readonly Vector512<float> V512Large0 = Vector512.Create(-8.875076532056503e-5f);

    private static readonly Vector512<float> V512Large1 = Vector512.Create(3.880528563007222e-4f);

    private static readonly Vector512<float> V512Large2 = Vector512.Create(-7.781201071142843e-4f);

    private static readonly Vector512<float> V512Large3 = Vector512.Create(1.0255980996254569e-3f);

    private static readonly Vector512<float> V512Large4 = Vector512.Create(-1.0307062838910627e-3f);

    private static readonly Vector512<float> V512Large5 = Vector512.Create(7.858608012972956e-4f);

    private static readonly Vector512<float> V512Large6 = Vector512.Create(-4.193605385522123e-4f);

    private static readonly Vector512<float> V512Large7 = Vector512.Create(1.3951109720745927e-4f);

    private static readonly Vector512<float> V512Large8 = Vector512.Create(9.999779388683872e-1f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> And512(Vector512<float> left, Vector512<float> right) =>
        Avx512F.And(left.AsInt32(), right.AsInt32()).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Or512(Vector512<float> left, Vector512<float> right) =>
        Avx512F.Or(left.AsInt32(), right.AsInt32()).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> ExpApproxVector512(Vector512<float> value)
    {
        Vector512<float> log2e = Vector512.Create(1.4426950408889634f);
        Vector512<float> c1 = Vector512.Create(0.693359375f); Vector512<float> c2 = Vector512.Create(-2.12194440e-4f);
        Vector512<float> min = Vector512.Create(-88.3762626647949f); Vector512<float> max = Vector512.Create(88.3762626647949f);
        value = Avx512F.Max(value, min); value = Avx512F.Min(value, max);
        Vector512<float> scaled = Avx512F.Add(Avx512F.Multiply(value, log2e), Vector512.Create(0.5f));
        Vector512<int> exponent = Avx512F.ConvertToVector512Int32WithTruncation(scaled);
        Vector512<float> integerPart = Avx512F.ConvertToVector512Single(exponent);
        Vector512<float> correction = And512(Avx512F.Compare(integerPart, scaled, FloatComparisonMode.OrderedGreaterThanNonSignaling), Vector512.Create(1f));
        integerPart = Avx512F.Subtract(integerPart, correction);
        Vector512<float> reduced = Avx512F.Subtract(value, Avx512F.Multiply(integerPart, c1));
        reduced = Avx512F.Subtract(reduced, Avx512F.Multiply(integerPart, c2));
        Vector512<float> square = Avx512F.Multiply(reduced, reduced);
        Vector512<float> p = Vector512.Create(2.4801587302e-5f);
        p = Avx512F.Add(Vector512.Create(1.9841269841e-4f), Avx512F.Multiply(reduced, p));
        p = Avx512F.Add(Vector512.Create(1.3888888889e-3f), Avx512F.Multiply(reduced, p));
        p = Avx512F.Add(Vector512.Create(8.3333333333e-3f), Avx512F.Multiply(reduced, p));
        p = Avx512F.Add(Vector512.Create(4.1666666667e-2f), Avx512F.Multiply(reduced, p));
        p = Avx512F.Add(Vector512.Create(1.6666666667e-1f), Avx512F.Multiply(reduced, p));
        p = Avx512F.Add(Vector512.Create(5.0e-1f), Avx512F.Multiply(reduced, p));
        p = Avx512F.Add(reduced, Avx512F.Multiply(square, p));
        p = Avx512F.Add(Vector512.Create(1f), p);
        exponent = Avx512F.ConvertToVector512Int32WithTruncation(integerPart);
        Vector512<int> bits = Avx512F.Add(exponent, Vector512.Create(127));
        Vector512<float> power = Avx512F.ShiftLeftLogical(bits, 23).AsSingle();
        return Avx512F.Multiply(p, power);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> ErfVector512(Vector512<float> value)
    {
        Vector512<float> abs = And512(value, V512AbsMask);
        Vector512<float> sign = And512(value, V512SignMask);
        Vector512<float> sq = Avx512F.Multiply(abs, abs);
        const int AllLanes = unchecked((int)0xffff);
        if (Avx512F.MoveMask(Avx512F.Compare(abs, V512One, FloatComparisonMode.OrderedLessThanNonSignaling)) == AllLanes)
            return Or512(PolySmall512(abs, sq), sign);
        if (Avx512F.MoveMask(Avx512F.Compare(abs, V512Four, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling)) == AllLanes)
            return Or512(V512One, sign);
        int atLeastOne = Avx512F.MoveMask(Avx512F.Compare(abs, V512One, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling));
        int belowTwo = Avx512F.MoveMask(Avx512F.Compare(abs, V512Two, FloatComparisonMode.OrderedLessThanNonSignaling));
        if ((atLeastOne & belowTwo) == AllLanes)
            return Or512(PolyMiddle512(abs), sign);
        int atLeastTwo = Avx512F.MoveMask(Avx512F.Compare(abs, V512Two, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling));
        int belowFour = Avx512F.MoveMask(Avx512F.Compare(abs, V512Four, FloatComparisonMode.OrderedLessThanNonSignaling));
        if ((atLeastTwo & belowFour) == AllLanes)
            return Or512(PolyLarge512(abs), sign);
        Vector512<float> small = PolySmall512(abs, sq); Vector512<float> middle = PolyMiddle512(abs); Vector512<float> large = PolyLarge512(abs);
        Vector512<float> result = Avx512F.BlendVariable(large, middle, Avx512F.Compare(abs, V512Two, FloatComparisonMode.OrderedLessThanNonSignaling));
        result = Avx512F.BlendVariable(result, small, Avx512F.Compare(abs, V512One, FloatComparisonMode.OrderedLessThanNonSignaling));
        result = Avx512F.BlendVariable(result, V512One, Avx512F.Compare(abs, V512Four, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling));
        result = Or512(result, sign);
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> MulAdd512(Vector512<float> x, Vector512<float> y, Vector512<float> z) =>
        Avx512F.FusedMultiplyAdd(x, y, z);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> PolySmall512(Vector512<float> a, Vector512<float> s)
    {
        Vector512<float> p = V512Small0;
        p = MulAdd512(s, p, V512Small1); p = MulAdd512(s, p, V512Small2); p = MulAdd512(s, p, V512Small3);
        p = MulAdd512(s, p, V512Small4); p = MulAdd512(s, p, V512Small5); p = MulAdd512(s, p, V512Small6);
        p = MulAdd512(s, p, V512Small7); p = MulAdd512(s, p, V512Small8);
        return Avx512F.Multiply(a, p);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> PolyMiddle512(Vector512<float> a)
    {
        Vector512<float> z = Avx512F.Subtract(a, V512OnePointFive);
        Vector512<float> p = V512Middle0;
        p = MulAdd512(z, p, V512Middle1); p = MulAdd512(z, p, V512Middle2); p = MulAdd512(z, p, V512Middle3);
        p = MulAdd512(z, p, V512Middle4); p = MulAdd512(z, p, V512Middle5); p = MulAdd512(z, p, V512Middle6);
        p = MulAdd512(z, p, V512Middle7);
        return MulAdd512(z, p, V512Middle8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> PolyLarge512(Vector512<float> a)
    {
        Vector512<float> z = Avx512F.Subtract(a, V512Three);
        Vector512<float> p = V512Large0;
        p = MulAdd512(z, p, V512Large1); p = MulAdd512(z, p, V512Large2); p = MulAdd512(z, p, V512Large3);
        p = MulAdd512(z, p, V512Large4); p = MulAdd512(z, p, V512Large5); p = MulAdd512(z, p, V512Large6);
        p = MulAdd512(z, p, V512Large7);
        return MulAdd512(z, p, V512Large8);
    }
}
