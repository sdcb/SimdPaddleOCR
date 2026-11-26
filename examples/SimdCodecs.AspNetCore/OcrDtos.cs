using System.ComponentModel;
using System.Text.Json.Serialization;

namespace SimdCodecs.AspNetCore;

public sealed record OcrElapsedMs(
    [property: Description("图片解码耗时（毫秒）。")] double Decode,
    [property: Description("OCR 推理耗时（毫秒）。")] double Ocr,
    [property: Description("请求总耗时（毫秒）。")] double Total);

public sealed record OcrCharacterDto(
    [property: Description("单字文本。")] string Text,
    [property: Description("单字置信度（0~1）。")] float Score,
    [property: Description("单字四边形 4 个顶点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]。")] float[][] Box);

public sealed record OcrLineDto(
    [property: Description("识别出的整行文本。")] string Text,
    [property: Description("文本识别置信度（0~1）。")] float RecognitionScore,
    [property: Description("文本行检测框 4 个顶点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]。")] float[][] Box,
    [property: Description("文本检测置信度（0~1）。")] float DetectionScore,
    [property: Description("文本行方向分类置信度（0~1）。")] float ClassificationScore,
    [property: Description("方向分类应用的旋转角度（0 或 180）。")] int AppliedRotationDegrees)
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
    [property: Description("逐行识别结果列表。")] OcrLineDto[] Lines);

public sealed record OcrError(
    [property: Description("错误说明。")] string Error);

public sealed record OcrModelList(
    [property: Description("支持的模型列表。")] string[] Models,
    [property: Description("未指定 model 时的默认模型。")] string Default);

public sealed record HealthResponse(
    [property: Description("服务健康状态。")] string Status,
    [property: Description("当前默认模型。")] string DefaultModel,
    [property: Description("已加载到内存的模型列表。")] string[] LoadedModels,
    [property: Description("服务编译配置（Release / Debug）。")] string BuildConfiguration);
