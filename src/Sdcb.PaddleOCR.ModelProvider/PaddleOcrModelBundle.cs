using System;

namespace Sdcb.PaddleOCR.ModelProvider;

/// <summary>Describes one complete OCR model combination.</summary>
public sealed class PaddleOcrModelBundle
{
    public PaddleOcrModelBundle(string name, string languageCode,
        IPaddleOcrModelProvider detection,
        IPaddleOcrModelProvider recognition,
        IPaddleOcrModelProvider dictionary,
        IPaddleOcrModelProvider? classification = null)
    {
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("A bundle name is required.", nameof(name));
        if (string.IsNullOrWhiteSpace(languageCode)) throw new ArgumentException("A language code is required.", nameof(languageCode));
        Detection = RequireKind(detection, PaddleOcrModelKind.Detection, nameof(detection));
        Recognition = RequireKind(recognition, PaddleOcrModelKind.Recognition, nameof(recognition));
        Dictionary = RequireKind(dictionary, PaddleOcrModelKind.Dictionary, nameof(dictionary));
        Classification = classification is null ? null : RequireKind(classification, PaddleOcrModelKind.Classification, nameof(classification));
        Name = name;
        LanguageCode = languageCode;
    }

    public string Name { get; }
    public string LanguageCode { get; }
    public IPaddleOcrModelProvider Detection { get; }
    public IPaddleOcrModelProvider? Classification { get; }
    public IPaddleOcrModelProvider Recognition { get; }
    public IPaddleOcrModelProvider Dictionary { get; }

    private static IPaddleOcrModelProvider RequireKind(IPaddleOcrModelProvider provider, PaddleOcrModelKind kind, string parameterName)
    {
        if (provider is null) throw new ArgumentNullException(parameterName);
        if (provider.Kind != kind)
            throw new ArgumentException("The provider must expose a " + kind + " model.", parameterName);
        return provider;
    }
}
