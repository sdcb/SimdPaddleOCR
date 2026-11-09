using System.Text.Json.Serialization;

namespace SimdCodecs.AspNetCore;

public sealed record OcrElapsedMs(double Decode, double Ocr, double Total);

public sealed record OcrCharacterDto(string Text, float Score, float[][] Box);

public sealed record OcrLineDto(
    string Text,
    float RecognitionScore,
    float[][] Box,
    float DetectionScore,
    float ClassificationScore,
    int AppliedRotationDegrees)
{
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public OcrCharacterDto[]? Characters { get; init; }
}

public sealed record OcrResponse(
    string Text,
    int DetectedCount,
    string Model,
    string BuildConfiguration,
    OcrElapsedMs ElapsedMs,
    OcrLineDto[] Lines);

public sealed record OcrError(string Error);

public sealed record OcrModelList(string[] Models, string Default);
