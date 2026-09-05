namespace ImageSharp.AspNetCore;

public sealed record OcrElapsedMs(double Decode, double Ocr, double Total);

public sealed record OcrLineDto(
    string Text,
    float RecognitionScore,
    float[][] Box,
    float DetectionScore,
    float ClassificationScore,
    int AppliedRotationDegrees);

public sealed record OcrResponse(
    string Text,
    int DetectedCount,
    string Model,
    string BuildConfiguration,
    OcrElapsedMs ElapsedMs,
    OcrLineDto[] Lines);

public sealed record OcrError(string Error);

public sealed record OcrModelList(string[] Models, string Default);
