using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.PaddleOCR.Kernels;

internal static partial class SimdKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector<float> VecLoad(float* source) =>
        Vector.LoadUnsafe(ref Unsafe.AsRef<float>(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void VecStore(float* destination, Vector<float> value) =>
        value.StoreUnsafe(ref Unsafe.AsRef<float>(destination));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ExpExactVector(Vector<float> value)
    {
        Vector<float> result = default;
        int width = Vector<float>.Count;
        for (int lane = 0; lane < width; lane++)
            result = result.WithElement(lane, MathF.Exp(value[lane]));
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> SigmoidExactVector(Vector<float> value)
    {
        Vector<float> result = default;
        int width = Vector<float>.Count;
        for (int lane = 0; lane < width; lane++)
            result = result.WithElement(lane, Sigmoid(value[lane]));
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ErfExactVector(Vector<float> value)
    {
        if (Sse.IsSupported && Vector<float>.Count == 4)
        {
            Vector128<float> packed = Unsafe.BitCast<Vector<float>, Vector128<float>>(value);
            return Unsafe.BitCast<Vector128<float>, Vector<float>>(ErfVectorSse(packed));
        }
        Vector<float> result = default;
        int width = Vector<float>.Count;
        for (int lane = 0; lane < width; lane++)
            result = result.WithElement(lane, Erf(value[lane]));
        return result;
    }

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
}
