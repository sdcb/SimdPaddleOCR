using System.Runtime.InteropServices;
using System.Text;

namespace LwPpocrCSharp;

static class OcrRecognitionDefaults
{
    internal const uint LongTextTargetWidth = 960u;
}

[StructLayout(LayoutKind.Sequential)]
struct LwError
{
    public uint StructSize;
    public int Code;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256, ArraySubType = UnmanagedType.I1)]
    public byte[] Message;
}

[StructLayout(LayoutKind.Sequential)]
struct LwPaddleOcrDetectorOptions
{
    public uint StructSize;
    public uint LimitSideLength;
    public uint MaxCandidates;
    public uint UseDilation;
    public float BitmapThreshold;
    public float BoxThreshold;
    public float UnclipRatio;
    public uint Reserved;
    public ulong MaxModelFileSize;
    public ulong MaxWorkspaceSize;
    public ulong MaxTensorSize;
    public ulong MaxImagePixels;
}

[StructLayout(LayoutKind.Sequential)]
struct LwPaddleOcrClassifierOptions
{
    public uint StructSize;
    public uint Reserved;
    public ulong MaxModelFileSize;
    public ulong MaxWorkspaceSize;
    public ulong MaxTensorSize;
    public ulong MaxImagePixels;
}

[StructLayout(LayoutKind.Sequential)]
struct LwPaddleOcrRecognizerOptions
{
    public uint StructSize;
    public uint TargetWidth;
    public uint Reserved0;
    public uint Reserved1;
    public ulong MaxModelFileSize;
    public ulong MaxWorkspaceSize;
    public ulong MaxTensorSize;
    public ulong MaxImagePixels;
}

[StructLayout(LayoutKind.Sequential)]
struct LwPaddleOcrOptions
{
    public uint StructSize;
    public uint UseDirectionClassification;
    public float ClassifierThreshold;
    public uint WorkerCount;
    public ulong MaxCropPixels;
    public LwPaddleOcrDetectorOptions Detector;
    public LwPaddleOcrClassifierOptions Classifier;
    public LwPaddleOcrRecognizerOptions Recognizer;
}

[StructLayout(LayoutKind.Sequential)]
struct LwOcrInfo
{
    public uint StructSize;
    public uint UseDirectionClassification;
    public uint MaxLineCapacity;
    public uint WorkerCount;
    public ulong MaxTextCapacity;
    public ulong MaxTextCapacityPerLine;
    public ulong MaxCropPixels;
}

[StructLayout(LayoutKind.Sequential)]
struct LwPaddleOcrDetectionBox
{
    public float X1;
    public float Y1;
    public float X2;
    public float Y2;
    public float X3;
    public float Y3;
    public float X4;
    public float Y4;
    public float Score;
    public uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
struct LwPaddleOcrLine
{
    public LwPaddleOcrDetectionBox Box;
    public float RecognitionScore;
    public float ClassificationScore;
    public uint ClassificationLabel;
    public uint AppliedRotationDegrees;
    public uint EmittedCount;
    public uint Reserved;
    public ulong TextOffset;
    public ulong TextLength;
}

[StructLayout(LayoutKind.Sequential)]
struct LwPaddleOcrResult
{
    public uint StructSize;
    public uint LineCount;
    public uint RequiredLineCapacity;
    public uint DetectedCount;
    public uint DetectorResizedWidth;
    public uint DetectorResizedHeight;
    public uint Reserved0;
    public uint Reserved1;
    public ulong RequiredTextCapacity;
}

sealed class NativeOcr : IDisposable
{
    private const string DllName = "lw_ppocr_c.dll";
    private readonly object syncRoot = new();
    private IntPtr handle;
    private LwOcrInfo info;
    private bool disposed;

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void lw_error_init(ref LwError error);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void lw_ocr_options_init(ref LwPaddleOcrOptions options);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void lw_ocr_info_init(ref LwOcrInfo value);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void lw_ocr_result_init(ref LwPaddleOcrResult value);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int lw_ocr_create(
        IntPtr detectorPathUtf8,
        IntPtr classifierPathUtf8,
        IntPtr recognizerPathUtf8,
        IntPtr dictionaryPathUtf8,
        ref LwPaddleOcrOptions options,
        out IntPtr ocr,
        ref LwError error);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void lw_ocr_free(IntPtr ocr);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int lw_ocr_get_info(IntPtr ocr, ref LwOcrInfo value);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int lw_ocr_run_bgr_u8(
        IntPtr ocr,
        IntPtr source,
        ulong sourceByteCount,
        uint sourceWidth,
        uint sourceHeight,
        uint sourceStride,
        IntPtr lines,
        uint lineCapacity,
        IntPtr textUtf8,
        ulong textCapacity,
        ref LwPaddleOcrResult result,
        ref LwError error);

    public NativeOcr(
        string detectorPath,
        string classifierPath,
        string recognizerPath,
        string dictionaryPath,
        bool useDirectionClassification,
        uint workerCount)
    {
        if (workerCount > 16u)
            throw new ArgumentOutOfRangeException(nameof(workerCount), "workerCount must be 0..16");
        ValidateAbi();
        ValidateFile(detectorPath, "DET");
        ValidateFile(recognizerPath, "REC");
        ValidateFile(dictionaryPath, "dictionary");
        if (useDirectionClassification) ValidateFile(classifierPath, "CLS");

        IntPtr detector = IntPtr.Zero;
        IntPtr classifier = IntPtr.Zero;
        IntPtr recognizer = IntPtr.Zero;
        IntPtr dictionary = IntPtr.Zero;
        try
        {
            detector = AllocUtf8(detectorPath);
            classifier = useDirectionClassification ? AllocUtf8(classifierPath) : IntPtr.Zero;
            recognizer = AllocUtf8(recognizerPath);
            dictionary = AllocUtf8(dictionaryPath);
            LwPaddleOcrOptions options = new();
            lw_ocr_options_init(ref options);
            options.StructSize = (uint)Marshal.SizeOf<LwPaddleOcrOptions>();
            options.Recognizer.StructSize = (uint)Marshal.SizeOf<LwPaddleOcrRecognizerOptions>();
            options.Recognizer.Reserved0 = 0u;
            options.Recognizer.Reserved1 = 0u;
            options.UseDirectionClassification = useDirectionClassification ? 1u : 0u;
            options.Recognizer.TargetWidth = OcrRecognitionDefaults.LongTextTargetWidth;
            if (workerCount != 0u) options.WorkerCount = workerCount;
            LwError error = CreateError();
            int status = lw_ocr_create(detector, classifier, recognizer, dictionary,
                ref options, out handle, ref error);
            if (status != 0 || handle == IntPtr.Zero)
                throw NativeFailure("OCR init failed", status, error);

            info = new LwOcrInfo();
            lw_ocr_info_init(ref info);
            status = lw_ocr_get_info(handle, ref info);
            if (status != 0 || info.MaxLineCapacity == 0 || info.MaxTextCapacity == 0)
                throw new InvalidOperationException("failed to read OCR output capacity, status=" + status);
            if (info.MaxLineCapacity > int.MaxValue || info.MaxTextCapacity > int.MaxValue)
                throw new InvalidOperationException("OCR output capacity exceeds .NET array limits");
        }
        catch
        {
            if (handle != IntPtr.Zero)
            {
                lw_ocr_free(handle);
                handle = IntPtr.Zero;
            }
            throw;
        }
        finally
        {
            Marshal.FreeHGlobal(detector);
            Marshal.FreeHGlobal(classifier);
            Marshal.FreeHGlobal(recognizer);
            Marshal.FreeHGlobal(dictionary);
        }
    }

    internal OcrResponse RecognizeDecoded(DecodedBgrImage image)
    {
        if (image.Pixels.Length == 0 || image.Width <= 0 || image.Height <= 0 ||
            image.Stride < checked(image.Width * 3) ||
            image.Pixels.LongLength < checked((long)image.Stride * image.Height))
            throw new ArgumentException("invalid BGR image", nameof(image));
        lock (syncRoot)
        {
            ThrowIfDisposed();
            LwPaddleOcrLine[] nativeLines = new LwPaddleOcrLine[(int)info.MaxLineCapacity];
            byte[] text = new byte[(int)info.MaxTextCapacity];
            GCHandle imagePin = default;
            GCHandle linesPin = default;
            GCHandle textPin = default;
            try
            {
                imagePin = GCHandle.Alloc(image.Pixels, GCHandleType.Pinned);
                linesPin = GCHandle.Alloc(nativeLines, GCHandleType.Pinned);
                textPin = GCHandle.Alloc(text, GCHandleType.Pinned);
                LwPaddleOcrResult nativeResult = new();
                lw_ocr_result_init(ref nativeResult);
                LwError error = CreateError();
                int status = lw_ocr_run_bgr_u8(
                    handle, imagePin.AddrOfPinnedObject(), (ulong)image.Pixels.LongLength,
                    (uint)image.Width, (uint)image.Height, (uint)image.Stride,
                    linesPin.AddrOfPinnedObject(), info.MaxLineCapacity,
                    textPin.AddrOfPinnedObject(), info.MaxTextCapacity,
                    ref nativeResult, ref error);
                if (status != 0) throw NativeFailure("OCR failed", status, error);
                if (nativeResult.LineCount > info.MaxLineCapacity ||
                    nativeResult.RequiredTextCapacity > info.MaxTextCapacity)
                    throw new InvalidOperationException("native OCR returned invalid capacity");

                OcrResponse response = new()
                {
                    ok = true,
                    api_version = 1,
                    image_width = image.Width,
                    image_height = image.Height,
                    detected_count = (int)nativeResult.DetectedCount,
                    result = new List<PaddleOcrLine>((int)nativeResult.LineCount),
                };
                for (int index = 0; index < (int)nativeResult.LineCount; ++index)
                {
                    LwPaddleOcrLine line = nativeLines[index];
                    if (line.TextOffset > nativeResult.RequiredTextCapacity ||
                        line.TextLength > nativeResult.RequiredTextCapacity - line.TextOffset ||
                        line.TextOffset + line.TextLength >= (ulong)text.LongLength)
                        throw new InvalidOperationException("native OCR returned invalid text range");
                    int offset = checked((int)line.TextOffset);
                    int length = checked((int)line.TextLength);
                    if (text[offset + length] != 0)
                        throw new InvalidOperationException("native OCR text missing NUL terminator");
                    response.result.Add(ToManagedLine(line, Encoding.UTF8.GetString(text, offset, length)));
                }
                return response;
            }
            finally
            {
                if (textPin.IsAllocated) textPin.Free();
                if (linesPin.IsAllocated) linesPin.Free();
                if (imagePin.IsAllocated) imagePin.Free();
            }
        }
    }

    private static PaddleOcrLine ToManagedLine(LwPaddleOcrLine source, string text) => new()
    {
        text = text,
        score = source.RecognitionScore,
        det_score = source.Box.Score,
        cls_label = (int)source.ClassificationLabel,
        cls_score = source.ClassificationScore,
        rotation = (int)source.AppliedRotationDegrees,
        x1 = source.Box.X1,
        y1 = source.Box.Y1,
        x2 = source.Box.X2,
        y2 = source.Box.Y2,
        x3 = source.Box.X3,
        y3 = source.Box.Y3,
        x4 = source.Box.X4,
        y4 = source.Box.Y4,
    };

    private static void ValidateAbi()
    {
        if (Marshal.SizeOf<LwError>() != 264 ||
            Marshal.SizeOf<LwPaddleOcrOptions>() != 176 ||
            Marshal.SizeOf<LwOcrInfo>() != 40 ||
            Marshal.SizeOf<LwPaddleOcrLine>() != 80 ||
            Marshal.SizeOf<LwPaddleOcrResult>() != 40)
            throw new PlatformNotSupportedException("C# and native OCR ABI struct sizes do not match");
    }

    private static LwError CreateError()
    {
        LwError error = new() { Message = new byte[256] };
        lw_error_init(ref error);
        return error;
    }

    private static Exception NativeFailure(string operation, int status, LwError error)
    {
        int length = 0;
        if (error.Message != null)
            while (length < error.Message.Length && error.Message[length] != 0) ++length;
        string message = error.Message == null ? string.Empty :
            Encoding.UTF8.GetString(error.Message, 0, length);
        return new InvalidOperationException(operation + "(" + status + "): " + message);
    }

    private static IntPtr AllocUtf8(string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value + "\0");
        IntPtr pointer = Marshal.AllocHGlobal(bytes.Length);
        Marshal.Copy(bytes, 0, pointer, bytes.Length);
        return pointer;
    }

    private static void ValidateFile(string path, string name)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
            throw new FileNotFoundException(name + " not found", path);
    }

    private void ThrowIfDisposed()
    {
        if (disposed) throw new ObjectDisposedException(nameof(NativeOcr));
    }

    public void Dispose()
    {
        lock (syncRoot)
        {
            if (disposed) return;
            if (handle != IntPtr.Zero)
            {
                lw_ocr_free(handle);
                handle = IntPtr.Zero;
            }
            disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~NativeOcr()
    {
        Dispose();
    }
}
