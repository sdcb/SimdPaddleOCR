using System.ComponentModel;
using System.Text.Json.Serialization;

namespace SimdCodecs.AspNetCore;

public sealed record OcrElapsedMs(
    [property: Description("图片解码耗时（毫秒）。")] double Decode,
    [property: Description("OCR 推理耗时（毫秒）。")] double Ocr,
    [property: Description("请求总耗时（毫秒）。")] double Total)
{
    [Description("整页方向分类与旋转耗时（毫秒）；仅当启用 documentOrientation=true 时返回。")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Orientation { get; init; }
}

public sealed record StageElapsedMs(
    [property: Description("图片解码耗时（毫秒）。")] double Decode,
    [property: Description("模型推理耗时（毫秒）。")] double Inference,
    [property: Description("请求总耗时（毫秒）。")] double Total);

public sealed record OcrDocumentOrientationDto(
    [property: Description("类别编号：0(0°)、1(90°)、2(180°)、3(270°)。")] uint Label,
    [property: Description("页面顺时针偏转角度（0 / 90 / 180 / 270），同时也是将页面扶正所需的逆时针旋转角度。")] int OrientationDegrees,
    [property: Description("方向分类置信度（0~1）。")] float Score);

public sealed record OcrCharacterDto(
    [property: Description("单字文本。")] string Text,
    [property: Description("单字置信度（0~1）。")] float Score,
    [property: Description("单字四边形 4 个顶点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]。")] float[][] Box);

public sealed record OcrLineDto(
    [property: Description("识别出的整行文本。")] string Text,
    [property: Description("文本识别置信度（0~1）。")] float RecognitionScore,
    [property: Description("文本行检测框 4 个顶点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]（始终在原图坐标系中）。")] float[][] Box,
    [property: Description("文本检测置信度（0~1）。")] float DetectionScore,
    [property: Description("文本行方向分类置信度（0~1）。")] float ClassificationScore,
    [property: Description("文本行方向分类应用的旋转角度（0 或 180）。")] int AppliedRotationDegrees)
{
    [Description("逐字检测框列表；仅当请求字段 characters=true 时返回。")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public OcrCharacterDto[]? Characters { get; init; }
}

public sealed record OcrResponse(
    [property: Description("按行拼接的完整识别文本。")] string Text,
    [property: Description("检测到的文本框数量。")] int DetectedCount,
    [property: Description("实际使用的模型名称（tiny / small / medium）。")] string Model,
    [property: Description("服务编译配置（Release / Debug）。")] string BuildConfiguration,
    [property: Description("各阶段耗时统计（毫秒）。")] OcrElapsedMs ElapsedMs,
    [property: Description("逐行识别结果列表。")] OcrLineDto[] Lines)
{
    [Description("整页方向分类结果；仅当启用 documentOrientation=true 时返回。")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public OcrDocumentOrientationDto? DocumentOrientation { get; init; }
}

public sealed record DocumentOrientationResponse(
    [property: Description("类别编号：0(0°)、1(90°)、2(180°)、3(270°)。")] uint Label,
    [property: Description("页面顺时针偏转角度（0 / 90 / 180 / 270），同时也是将页面扶正所需的逆时针旋转角度。")] int OrientationDegrees,
    [property: Description("方向分类置信度（0~1）。")] float Score,
    [property: Description("输入图片宽度（应用 EXIF 方向后）。")] int Width,
    [property: Description("输入图片高度（应用 EXIF 方向后）。")] int Height,
    [property: Description("服务编译配置（Release / Debug）。")] string BuildConfiguration,
    [property: Description("各阶段耗时统计（毫秒）。")] StageElapsedMs ElapsedMs);

public sealed record OcrDetectionBoxDto(
    [property: Description("文本框 4 个顶点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]。")] float[][] Box,
    [property: Description("文本检测置信度（0~1）。")] float Score);

public sealed record OcrDetectResponse(
    [property: Description("检测到的文本框数量。")] int DetectedCount,
    [property: Description("检测器缩放后的输入宽度。")] int ResizedWidth,
    [property: Description("检测器缩放后的输入高度。")] int ResizedHeight,
    [property: Description("实际使用的模型名称（tiny / small / medium）。")] string Model,
    [property: Description("服务编译配置（Release / Debug）。")] string BuildConfiguration,
    [property: Description("各阶段耗时统计（毫秒）。")] StageElapsedMs ElapsedMs,
    [property: Description("检测到的文本框列表。")] OcrDetectionBoxDto[] Boxes);

public sealed record OcrClassifyResponse(
    [property: Description("类别编号：0(0°)、1(180°)。")] uint Label,
    [property: Description("文本行方向角度（0 或 180）。")] int OrientationDegrees,
    [property: Description("方向分类置信度（0~1）。")] float Score,
    [property: Description("分类器缩放后的输入宽度。")] int ResizedWidth,
    [property: Description("服务编译配置（Release / Debug）。")] string BuildConfiguration,
    [property: Description("各阶段耗时统计（毫秒）。")] StageElapsedMs ElapsedMs);

public sealed record OcrRecognizeResponse(
    [property: Description("识别出的文本行内容。")] string Text,
    [property: Description("文本识别置信度（0~1）。")] float Score,
    [property: Description("CTC 解码输出的 token 数量。")] int EmittedCount,
    [property: Description("识别器缩放后的内容宽度。")] int ResizedWidth,
    [property: Description("CTC 时间步数量。")] int TimeSteps,
    [property: Description("实际使用的模型名称（tiny / small / medium）。")] string Model,
    [property: Description("服务编译配置（Release / Debug）。")] string BuildConfiguration,
    [property: Description("各阶段耗时统计（毫秒）。")] StageElapsedMs ElapsedMs)
{
    [Description("逐字检测框列表；仅当请求字段 characters=true 时返回。")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public OcrCharacterDto[]? Characters { get; init; }
}

public sealed record OcrError(
    [property: Description("错误说明。")] string Error);

public sealed record OcrModelList(
    [property: Description("支持的模型列表。")] string[] Models,
    [property: Description("未指定 model 时的默认模型。")] string Default);

public sealed record HealthResponse(
    [property: Description("服务健康状态。")] string Status,
    [property: Description("当前默认模型。")] string DefaultModel,
    [property: Description("已加载到内存的模型列表。")] string[] LoadedModels,
    [property: Description("整页方向分类模型是否已加载到内存。")] bool DocumentOrientationLoaded,
    [property: Description("服务编译配置（Release / Debug）。")] string BuildConfiguration);
