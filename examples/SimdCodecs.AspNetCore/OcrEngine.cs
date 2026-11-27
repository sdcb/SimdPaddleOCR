using System.Collections.Concurrent;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.DocumentOrientation;
using Sdcb.SimdPaddleOCR.Models.TextLineOrientation;

namespace SimdCodecs.AspNetCore;

public sealed class OcrEngine : IDisposable
{
    public const string DefaultModelName = "tiny";
    public static readonly string[] Models = ["tiny", "small", "medium"];

    private readonly OcrServerOptions _serverOptions;
    private readonly PaddleOcrOptions _ocrOptions;
    private readonly PaddleOcrDocumentOrientationOptions _documentOrientationOptions;
    private readonly PaddleOcrClassifierOptions _classifierOptions;
    private readonly ConcurrentDictionary<string, Lazy<Task<PaddleOcrAll>>> _engines = new(StringComparer.Ordinal);
    private readonly Lazy<PaddleOcrDocumentOrientationClassifier> _documentOrientation;
    private readonly Lazy<PaddleOcrClassifier> _standaloneClassifier;

    public OcrEngine(OcrServerOptions serverOptions)
    {
        _serverOptions = serverOptions;
        DefaultModel = serverOptions.GetNormalizedDefaultModel();
        _ocrOptions = serverOptions.ToPaddleOcrOptions();
        _documentOrientationOptions = serverOptions.ToDocumentOrientationOptions();
        _classifierOptions = serverOptions.ToClassifierOptions();
        _documentOrientation = new Lazy<PaddleOcrDocumentOrientationClassifier>(
            LoadDocumentOrientationClassifier,
            LazyThreadSafetyMode.ExecutionAndPublication);
        _standaloneClassifier = new Lazy<PaddleOcrClassifier>(
            LoadStandaloneClassifier,
            LazyThreadSafetyMode.ExecutionAndPublication);
    }

    public string DefaultModel { get; }
    public long MaxImagePixels => _serverOptions.MaxImagePixels;
    public bool IsDocumentOrientationLoaded => _documentOrientation.IsValueCreated;

    public string[] GetLoadedModels() =>
    [
        .. _engines
            .Where(kv => kv.Value.IsValueCreated && kv.Value.Value.Status == TaskStatus.RanToCompletion)
            .Select(kv => kv.Key)
            .OrderBy(k => Array.IndexOf(Models, k))
    ];

    public bool TryNormalizeModel(string? model, out string name) =>
        TryNormalizeModel(model, DefaultModel, out name);

    public static bool TryNormalizeModel(string? model, string defaultModel, out string name)
    {
        name = string.IsNullOrWhiteSpace(model) ? defaultModel : model.Trim().ToLowerInvariant();
        return name is "tiny" or "small" or "medium";
    }

    public Task<PaddleOcrAll> GetAsync(string? model)
    {
        if (!TryNormalizeModel(model, out string name))
            throw new ArgumentOutOfRangeException(nameof(model), "模型必须是 tiny、small 或 medium。");

        return _engines.GetOrAdd(name, key => new Lazy<Task<PaddleOcrAll>>(
            () => LoadModelAsync(key, _ocrOptions),
            LazyThreadSafetyMode.ExecutionAndPublication)).Value;
    }

    public PaddleOcrDocumentOrientationClassifier GetDocumentOrientationClassifier() => _documentOrientation.Value;

    public async Task<PaddleOcrClassifier> GetClassifierAsync()
    {
        if (_ocrOptions.UseDirectionClassification)
        {
            foreach (Lazy<Task<PaddleOcrAll>> lazy in _engines.Values)
            {
                if (lazy.IsValueCreated && lazy.Value.Status == TaskStatus.RanToCompletion && lazy.Value.Result.Classifier is { } loaded)
                    return loaded;
            }

            PaddleOcrAll defaultPipeline = await GetAsync(DefaultModel);
            if (defaultPipeline.Classifier is { } shared)
                return shared;
        }

        return _standaloneClassifier.Value;
    }

    public void Dispose()
    {
        foreach (Lazy<Task<PaddleOcrAll>> lazy in _engines.Values)
        {
            if (!lazy.IsValueCreated)
                continue;

            try
            {
                if (lazy.Value.Status == TaskStatus.RanToCompletion)
                    lazy.Value.Result.Dispose();
            }
            catch
            {
            }
        }

        _engines.Clear();

        if (_documentOrientation.IsValueCreated)
        {
            try { _documentOrientation.Value.Dispose(); }
            catch { }
        }

        if (_standaloneClassifier.IsValueCreated)
        {
            try { _standaloneClassifier.Value.Dispose(); }
            catch { }
        }
    }

    private PaddleOcrDocumentOrientationClassifier LoadDocumentOrientationClassifier()
    {
        using Stream stream = DocumentOrientationModel.OpenRead();
        return new PaddleOcrDocumentOrientationClassifier(stream, _documentOrientationOptions);
    }

    private PaddleOcrClassifier LoadStandaloneClassifier()
    {
        using Stream stream = TextLineOrientationModel.Provider.OpenRead();
        return new PaddleOcrClassifier(stream, _classifierOptions);
    }

    private static Task<PaddleOcrAll> LoadModelAsync(string name, PaddleOcrOptions options) => name switch
    {
        "tiny" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default, options),
        "small" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Small.ChineseV6SmallModels.Default, options),
        "medium" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.ChineseV6MediumModels.Default, options),
        _ => throw new ArgumentOutOfRangeException(nameof(name))
    };
}

internal sealed class OcrPreloadHostedService(
    OcrEngine engine,
    OcrServerOptions options,
    ILogger<OcrPreloadHostedService> logger) : IHostedService
{
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        string[] models = options.ResolvePreloadModels();
        foreach (string model in models)
        {
            cancellationToken.ThrowIfCancellationRequested();
            logger.LogInformation("Preloading OCR model '{Model}'...", model);
            await engine.GetAsync(model).WaitAsync(cancellationToken);
            logger.LogInformation("OCR model '{Model}' loaded.", model);
        }

        if (options.PreloadDocumentOrientation)
        {
            cancellationToken.ThrowIfCancellationRequested();
            logger.LogInformation("Preloading document orientation model...");
            await Task.Run(engine.GetDocumentOrientationClassifier, cancellationToken);
            logger.LogInformation("Document orientation model loaded.");
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
