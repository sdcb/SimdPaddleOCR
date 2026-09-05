using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR.ModelProvider;

using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR;

/// <summary>Owns the ONNX graphs and dictionary used by a Paddle OCR pipeline.</summary>
public sealed class PaddleOcrModelSet : IDisposable
{
    public Model DetectionModel { get; }
    public Model? ClassificationModel { get; }
    public Model RecognitionModel { get; }
    public ReadOnlyMemory<byte> DictionaryUtf8 { get; }
    private PaddleOcrModelSet(Model det, Model? cls, Model rec, ReadOnlyMemory<byte> dict)
    {
        DetectionModel = det;
        ClassificationModel = cls;
        RecognitionModel = rec;
        DictionaryUtf8 = dict;
    }

    public static PaddleOcrModelSet Load(
        string detectionPath,
        string? classificationPath,
        string recognitionPath,
        string dictionaryPath)
    {
        static FileStream Open(string path, string name)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                throw new FileNotFoundException($"{name} file not found.", path);
            return File.OpenRead(path);
        }

        using FileStream detection = Open(detectionPath, "Detection");
        using FileStream? classification = classificationPath is null ? null : Open(classificationPath, "Classification");
        using FileStream recognition = Open(recognitionPath, "Recognition");
        using FileStream dictionary = Open(dictionaryPath, "Dictionary");
        return Load(detection, classification, recognition, dictionary);
    }

    public static async Task<PaddleOcrModelSet> LoadAsync(
        string detectionPath, 
        string? classificationPath, 
        string recognitionPath,
        string dictionaryPath, 
        CancellationToken cancellationToken = default)
    {
        static FileStream Open(string path, string name)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                throw new FileNotFoundException($"{name} file not found.", path);
            return File.OpenRead(path);
        }
        cancellationToken.ThrowIfCancellationRequested();
        using FileStream detection = Open(detectionPath, "Detection");
        using FileStream? classification = classificationPath is null ? null : Open(classificationPath, "Classification");
        using FileStream recognition = Open(recognitionPath, "Recognition");
        using FileStream dictionary = Open(dictionaryPath, "Dictionary");
        return await LoadAsync(detection, classification, recognition, dictionary, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Loads models directly from streams. Streams are consumed but are not
    /// disposed by this method; the caller retains ownership.
    /// </summary>
    public static PaddleOcrModelSet Load(
        Stream detectionOnnx,
        Stream? classificationOnnx,
        Stream recognitionOnnx,
        Stream dictionaryUtf8)
    {
        if (detectionOnnx is null) throw new ArgumentNullException(nameof(detectionOnnx));
        if (recognitionOnnx is null) throw new ArgumentNullException(nameof(recognitionOnnx));
        if (dictionaryUtf8 is null) throw new ArgumentNullException(nameof(dictionaryUtf8));

        Model? det = null;
        Model? cls = null;
        Model? rec = null;
        try
        {
            det = Model.Load(detectionOnnx);
            cls = classificationOnnx is null ? null : Model.Load(classificationOnnx);
            rec = Model.Load(recognitionOnnx);
            byte[] dictionary = ReadDictionary(dictionaryUtf8);
            ValidateDictionary(dictionary);
            return new PaddleOcrModelSet(det, cls, rec, dictionary);
        }
        catch
        {
            det?.Dispose();
            cls?.Dispose();
            rec?.Dispose();
            throw;
        }
    }

    public static async Task<PaddleOcrModelSet> LoadAsync(
        Stream detectionOnnx,
        Stream? classificationOnnx,
        Stream recognitionOnnx,
        Stream dictionaryUtf8,
        CancellationToken cancellationToken = default)
    {
        if (detectionOnnx is null) throw new ArgumentNullException(nameof(detectionOnnx));
        if (recognitionOnnx is null) throw new ArgumentNullException(nameof(recognitionOnnx));
        if (dictionaryUtf8 is null) throw new ArgumentNullException(nameof(dictionaryUtf8));
        cancellationToken.ThrowIfCancellationRequested();
        Model? det = null;
        Model? cls = null;
        Model? rec = null;
        try
        {
            det = await Model.LoadAsync(detectionOnnx, cancellationToken).ConfigureAwait(false);
            cls = classificationOnnx is null ? null :
                await Model.LoadAsync(classificationOnnx, cancellationToken).ConfigureAwait(false);
            rec = await Model.LoadAsync(recognitionOnnx, cancellationToken).ConfigureAwait(false);
            byte[] dictionary = await ReadDictionaryAsync(dictionaryUtf8, cancellationToken).ConfigureAwait(false);
            ValidateDictionary(dictionary);
            return new PaddleOcrModelSet(det, cls, rec, dictionary);
        }
        catch
        {
            det?.Dispose();
            cls?.Dispose();
            rec?.Dispose();
            throw;
        }
    }

    public static PaddleOcrModelSet Load(
        ReadOnlyMemory<byte> detectionOnnx,
        ReadOnlyMemory<byte>? classificationOnnx,
        ReadOnlyMemory<byte> recognitionOnnx,
        ReadOnlyMemory<byte> dictionaryUtf8)
    {
        if (detectionOnnx.Length == 0 || recognitionOnnx.Length == 0)
            throw new InvalidDataException("ONNX model is empty.");
        if (dictionaryUtf8.Length == 0)
            throw new InvalidDataException("Dictionary is empty.");
        ValidateDictionary(dictionaryUtf8.Span);

        Model? det = null;
        Model? cls = null;
        Model? rec = null;
        try
        {
            det = Model.Load(detectionOnnx);
            cls = classificationOnnx is { } c ? Model.Load(c) : null;
            rec = Model.Load(recognitionOnnx);
            return new PaddleOcrModelSet(det, cls, rec, dictionaryUtf8.ToArray());
        }
        catch
        {
            det?.Dispose();
            cls?.Dispose();
            rec?.Dispose();
            throw;
        }
    }

    public static PaddleOcrModelSet Load(PaddleOcrModelBundle bundle)
    {
        if (bundle is null) throw new ArgumentNullException(nameof(bundle));
        using Stream detection = bundle.Detection.OpenRead();
        using Stream? classification = bundle.Classification?.OpenRead();
        using Stream recognition = bundle.Recognition.OpenRead();
        using Stream dictionary = bundle.Dictionary.OpenRead();
        return Load(detection, classification, recognition, dictionary);
    }

    public static async Task<PaddleOcrModelSet> LoadAsync(PaddleOcrModelBundle bundle,
        CancellationToken cancellationToken = default)
    {
        if (bundle is null) throw new ArgumentNullException(nameof(bundle));
        cancellationToken.ThrowIfCancellationRequested();
        using Stream detection = await bundle.Detection.OpenReadAsync(cancellationToken).ConfigureAwait(false);
        using Stream? classification = bundle.Classification is null ? null :
            await bundle.Classification.OpenReadAsync(cancellationToken).ConfigureAwait(false);
        using Stream recognition = await bundle.Recognition.OpenReadAsync(cancellationToken).ConfigureAwait(false);
        using Stream dictionary = await bundle.Dictionary.OpenReadAsync(cancellationToken).ConfigureAwait(false);
        return await LoadAsync(detection, classification, recognition, dictionary, cancellationToken).ConfigureAwait(false);
    }

    public static Task<PaddleOcrModelSet> LoadAsync(
        ReadOnlyMemory<byte> detectionOnnx, ReadOnlyMemory<byte>? classificationOnnx,
        ReadOnlyMemory<byte> recognitionOnnx, ReadOnlyMemory<byte> dictionaryUtf8,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(Load(detectionOnnx, classificationOnnx, recognitionOnnx, dictionaryUtf8));
    }

    private static byte[] ReadDictionary(Stream source)
    {
        using MemoryStream buffer = new();
        source.CopyTo(buffer);
        if (buffer.Length == 0) throw new InvalidDataException("Dictionary stream is empty.");
        return buffer.ToArray();
    }

    private static async Task<byte[]> ReadDictionaryAsync(Stream source, CancellationToken cancellationToken)
    {
        using MemoryStream buffer = new();
        await source.CopyToAsync(buffer, 81920, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        if (buffer.Length == 0) throw new InvalidDataException("Dictionary stream is empty.");
        return buffer.ToArray();
    }

    private static void ValidateDictionary(ReadOnlySpan<byte> dictionary)
    {
        try { _ = EncodingCompat.GetString(new UTF8Encoding(false, true), dictionary); }
        catch (DecoderFallbackException e) { throw new InvalidDataException("Dictionary must be UTF-8.", e); }
    }

    // Kept separate from model loading because dictionaries are tiny and are
    // retained as an owned byte[] for the recognizer/model-set API.

    public static PaddleOcrModelSet Load(Model detectionModel, Model? classificationModel,
        Model recognitionModel, Stream dictionaryUtf8)
    {
        if (dictionaryUtf8 is null) throw new ArgumentNullException(nameof(dictionaryUtf8));
        if (detectionModel is null) throw new ArgumentNullException(nameof(detectionModel));
        if (recognitionModel is null) throw new ArgumentNullException(nameof(recognitionModel));
        byte[] dictionary = ReadDictionary(dictionaryUtf8);
        ValidateDictionary(dictionary);
        return new PaddleOcrModelSet(detectionModel, classificationModel, recognitionModel, dictionary);
    }

    public static PaddleOcrModelSet Load(Model detectionModel, Model? classificationModel,
        Model recognitionModel, ReadOnlyMemory<byte> dictionaryUtf8)
    {
        if (detectionModel is null) throw new ArgumentNullException(nameof(detectionModel));
        if (recognitionModel is null) throw new ArgumentNullException(nameof(recognitionModel));
        if (dictionaryUtf8.Length == 0)
            throw new InvalidDataException("Dictionary is empty.");
        byte[] dictionary = dictionaryUtf8.ToArray();
        ValidateDictionary(dictionary);
        return new PaddleOcrModelSet(detectionModel, classificationModel,
            recognitionModel, dictionary);
    }
    public void Dispose()
    {
        DetectionModel.Dispose();
        ClassificationModel?.Dispose();
        RecognitionModel.Dispose();
    }
}
