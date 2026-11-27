using System.Buffers;
using System.ComponentModel;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc;
using Microsoft.OpenApi;
using Scalar.AspNetCore;
using Sdcb.SimdCodecs;
using Sdcb.SimdCodecs.Abstractions;
using Sdcb.SimdPaddleOCR;
using SimdCodecs.AspNetCore;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

OcrServerOptions serverOptions = builder.Configuration
    .GetSection(OcrServerOptions.SectionName)
    .Get<OcrServerOptions>() ?? new OcrServerOptions();

if (serverOptions.MaxUploadBytes <= 0)
    throw new InvalidOperationException("OCR:MaxUploadBytes must be greater than 0.");
if (serverOptions.MaxImagePixels <= 0)
    throw new InvalidOperationException("OCR:MaxImagePixels must be greater than 0.");
_ = serverOptions.GetNormalizedDefaultModel();
_ = serverOptions.ResolvePreloadModels();

builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = serverOptions.MaxUploadBytes);
builder.Services.Configure<FormOptions>(options => options.MultipartBodyLengthLimit = serverOptions.MaxUploadBytes);
builder.Services.AddSingleton(serverOptions);
builder.Services.AddSingleton<OcrEngine>();
builder.Services.AddHostedService<OcrPreloadHostedService>();
builder.Services.AddOpenApi(options =>
{
    options.AddDocumentTransformer((document, _, _) =>
    {
        document.Info = new OpenApiInfo
        {
            Title = "Sdcb.SimdPaddleOCR API",
            Version = typeof(PaddleOcrAll).Assembly.GetName().Version?.ToString(3) ?? "2.1.0",
            Description = "纯 C# PP-OCRv6 HTTP 识别服务（基于 Sdcb.SimdPaddleOCR + Sdcb.SimdCodecs）。" +
                "内置 tiny / small / medium 三个中文模型与整页四向文档方向分类模型（PP-LCNet_x1_0_doc_ori），" +
                "提供完整 OCR、整页方向分类，以及共享模型权重的独立检测 / 文本行分类 / 单行识别接口。" +
                (string.IsNullOrWhiteSpace(serverOptions.ApiKey)
                    ? string.Empty
                    : " 当前服务已启用 API Key 鉴权，请在请求头传入 X-API-Key 或 Authorization: Bearer <key>。")
        };
        return Task.CompletedTask;
    });
});
builder.Services.AddCors(options =>
    options.AddDefaultPolicy(policy => policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));

WebApplication app = builder.Build();
app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();

if (!string.IsNullOrWhiteSpace(serverOptions.ApiKey))
{
    byte[] expectedKeyBytes = Encoding.UTF8.GetBytes(serverOptions.ApiKey);
    app.Use(async (context, next) =>
    {
        if (context.Request.Path.StartsWithSegments("/api", StringComparison.OrdinalIgnoreCase))
        {
            string? provided = ExtractApiKey(context.Request);
            if (provided is null || !FixedTimeEquals(provided, expectedKeyBytes))
            {
                context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                await context.Response.WriteAsJsonAsync(new OcrError("无效或缺少 API Key（请传入 X-API-Key 或 Authorization: Bearer <key>）。"));
                return;
            }
        }
        await next();
    });
}

app.MapOpenApi();
app.MapScalarApiReference(options =>
{
    options.Title = "Sdcb.SimdPaddleOCR API";
});

app.MapGet("/health", (OcrEngine engine) =>
    Results.Ok(new HealthResponse(
        "ok",
        engine.DefaultModel,
        engine.GetLoadedModels(),
        engine.IsDocumentOrientationLoaded,
        GetBuildConfiguration())))
    .WithName("HealthCheck")
    .WithTags("System")
    .WithSummary("健康检查")
    .WithDescription("返回服务状态、默认模型与当前已加载到内存的模型列表。")
    .Produces<HealthResponse>();

app.MapGet("/api/ocr/models", (OcrEngine engine) =>
    Results.Ok(new OcrModelList(OcrEngine.Models, engine.DefaultModel)))
    .WithName("ListOcrModels")
    .WithTags("OCR")
    .WithSummary("列出可用 OCR 模型")
    .Produces<OcrModelList>()
    .Produces<OcrError>(StatusCodes.Status401Unauthorized);

app.MapGet("/sample.jpg", (IWebHostEnvironment env) =>
{
    string? path = ResolveSampleImage(env);
    return path is null ? Results.NotFound() : Results.File(path, "image/jpeg");
}).ExcludeFromDescription();

app.MapPost("/api/ocr", RunFullOcrAsync)
    .DisableAntiforgery()
    .WithName("RecognizeImage")
    .WithTags("OCR")
    .WithSummary("识别图片中的文字（完整流水线）")
    .WithDescription("上传图片并选择 tiny、small 或 medium 模型，返回识别文本、检测框和耗时。" +
        "表单字段 documentOrientation=true 时，先执行 0°/90°/180°/270° 整页方向检测并自动旋转扶正后再识别，检测框自动映射回原图坐标；" +
        "表单字段 characters=true 时，每一行额外返回单字框。")
    .Produces<OcrResponse>()
    .Produces<OcrError>(StatusCodes.Status400BadRequest)
    .Produces<OcrError>(StatusCodes.Status401Unauthorized)
    .Produces<OcrError>(StatusCodes.Status500InternalServerError)
    .Accepts<OcrUpload>("multipart/form-data");

app.MapPost("/api/ocr/orientation", ClassifyDocumentOrientationAsync)
    .DisableAntiforgery()
    .WithName("ClassifyDocumentOrientation")
    .WithTags("OCR Stages")
    .WithSummary("整页文档方向分类（0° / 90° / 180° / 270°）")
    .WithDescription("使用内置 PP-LCNet_x1_0_doc_ori 模型判断整页文档的旋转方向，返回顺时针偏转角度（同时也是扶正所需的逆时针旋转角度）和置信度。")
    .Produces<DocumentOrientationResponse>()
    .Produces<OcrError>(StatusCodes.Status400BadRequest)
    .Produces<OcrError>(StatusCodes.Status401Unauthorized)
    .Produces<OcrError>(StatusCodes.Status500InternalServerError)
    .Accepts<ImageUpload>("multipart/form-data");

app.MapPost("/api/ocr/detect", DetectTextAsync)
    .DisableAntiforgery()
    .WithName("DetectText")
    .WithTags("OCR Stages")
    .WithSummary("仅执行文本检测（DET）")
    .WithDescription("复用共享 PaddleOcrAll 实例中的检测模型，仅返回文本框坐标和检测置信度，不执行分类与识别。")
    .Produces<OcrDetectResponse>()
    .Produces<OcrError>(StatusCodes.Status400BadRequest)
    .Produces<OcrError>(StatusCodes.Status401Unauthorized)
    .Produces<OcrError>(StatusCodes.Status500InternalServerError)
    .Accepts<DetectUpload>("multipart/form-data");

app.MapPost("/api/ocr/classify", ClassifyTextLineAsync)
    .DisableAntiforgery()
    .WithName("ClassifyTextLine")
    .WithTags("OCR Stages")
    .WithSummary("仅执行文本行方向分类（CLS，0° / 180°）")
    .WithDescription("复用共享 PaddleOcrAll 实例中的文本行方向分类模型，对单行裁剪图片判断 0° 或 180° 方向。")
    .Produces<OcrClassifyResponse>()
    .Produces<OcrError>(StatusCodes.Status400BadRequest)
    .Produces<OcrError>(StatusCodes.Status401Unauthorized)
    .Produces<OcrError>(StatusCodes.Status500InternalServerError)
    .Accepts<ImageUpload>("multipart/form-data");

app.MapPost("/api/ocr/recognize", RecognizeTextLineAsync)
    .DisableAntiforgery()
    .WithName("RecognizeTextLine")
    .WithTags("OCR Stages")
    .WithSummary("仅执行单行文本识别（REC）")
    .WithDescription("复用共享 PaddleOcrAll 实例中的识别模型，对单行裁剪图片直接运行 CTC 识别；characters=true 时额外返回逐字框。")
    .Produces<OcrRecognizeResponse>()
    .Produces<OcrError>(StatusCodes.Status400BadRequest)
    .Produces<OcrError>(StatusCodes.Status401Unauthorized)
    .Produces<OcrError>(StatusCodes.Status500InternalServerError)
    .Accepts<RecognizeUpload>("multipart/form-data");

app.Run();

static async Task<IResult> RunFullOcrAsync(
    [FromForm] OcrUpload upload,
    OcrEngine engine,
    OcrServerOptions options,
    CancellationToken cancellationToken)
{
    if (!engine.TryNormalizeModel(upload.Model, out string model))
        return Results.BadRequest(new OcrError("模型必须是 tiny、small 或 medium。"));

    Stopwatch total = Stopwatch.StartNew();
    (PixelBuffer? image, double decodeMs, IResult? error) = await DecodeUploadedImageAsync(upload.File, engine, options, cancellationToken);
    if (error is not null || image is null)
        return error!;

    using (image)
    {
        try
        {
            bool useDocOrientation = upload.DocumentOrientation ?? options.DefaultDocumentOrientation;
            OcrDocumentOrientationDto? docOrientationDto = null;
            double? orientationMs = null;
            int appliedDocDegrees = 0;

            byte[]? rotatedRent = null;
            int ocrWidth = image.Width;
            int ocrHeight = image.Height;
            int ocrStride = image.Stride;

            try
            {
                if (useDocOrientation)
                {
                    Stopwatch oriWatch = Stopwatch.StartNew();
                    PaddleOcrDocumentOrientationClassifier docOri = engine.GetDocumentOrientationClassifier();
                    PaddleOcrDocumentOrientationResult ori = docOri.Classify(
                        image.Pixels, image.Width, image.Height, image.Stride, ImagePixelFormat.Bgr24);
                    appliedDocDegrees = ori.OrientationDegrees;
                    docOrientationDto = new OcrDocumentOrientationDto(ori.Label, ori.OrientationDegrees, ori.Score);

                    if (appliedDocDegrees != 0)
                    {
                        (ocrWidth, ocrHeight) = PaddleOcrDocumentRotation.RotatedSize(image.Width, image.Height, appliedDocDegrees);
                        ocrStride = checked(ocrWidth * 3);
                        int byteCount = checked(ocrStride * ocrHeight);
                        rotatedRent = ArrayPool<byte>.Shared.Rent(byteCount);
                        PaddleOcrDocumentRotation.Rotate(
                            image.Pixels, image.Width, image.Height, image.Stride,
                            ImagePixelFormat.Bgr24, appliedDocDegrees, rotatedRent.AsSpan(0, byteCount));
                    }

                    orientationMs = Math.Round(oriWatch.Elapsed.TotalMilliseconds, 1);
                }

                Stopwatch ocrWatch = Stopwatch.StartNew();
                PaddleOcrAll ocr = await engine.GetAsync(model);
                ReadOnlySpan<byte> ocrPixels = rotatedRent is not null
                    ? rotatedRent.AsSpan(0, checked(ocrStride * ocrHeight))
                    : image.Pixels;
                PaddleOcrResult result = upload.Characters
                    ? ocr.Run(ocrPixels, ocrWidth, ocrHeight, returnCtcAlignment: true, ocrStride, ImagePixelFormat.Bgr24)
                    : ocr.Run(ocrPixels, ocrWidth, ocrHeight, ocrStride, ImagePixelFormat.Bgr24);
                double ocrMs = ocrWatch.Elapsed.TotalMilliseconds;
                total.Stop();

                OcrLineDto[] lines =
                [
                    .. result.Lines.Select(line =>
                    {
                        PaddleOcrLine mapped = appliedDocDegrees == 0
                            ? line
                            : PaddleOcrDocumentRotation.ToSource(line, image.Width, image.Height, appliedDocDegrees);
                        return ToLineDto(mapped, upload.Characters);
                    })
                ];

                return Results.Ok(new OcrResponse(
                    result.Text,
                    result.DetectedCount,
                    model,
                    GetBuildConfiguration(),
                    new OcrElapsedMs(
                        Math.Round(decodeMs, 1),
                        Math.Round(ocrMs, 1),
                        Math.Round(total.Elapsed.TotalMilliseconds, 1))
                    {
                        Orientation = orientationMs
                    },
                    lines)
                {
                    DocumentOrientation = docOrientationDto
                });
            }
            finally
            {
                if (rotatedRent is not null)
                    ArrayPool<byte>.Shared.Return(rotatedRent);
            }
        }
        catch (Exception ex)
        {
            return Results.Json(new OcrError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
        }
    }
}

static async Task<IResult> ClassifyDocumentOrientationAsync(
    [FromForm] ImageUpload upload,
    OcrEngine engine,
    OcrServerOptions options,
    CancellationToken cancellationToken)
{
    Stopwatch total = Stopwatch.StartNew();
    (PixelBuffer? image, double decodeMs, IResult? error) = await DecodeUploadedImageAsync(upload.File, engine, options, cancellationToken);
    if (error is not null || image is null)
        return error!;

    using (image)
    {
        try
        {
            Stopwatch stage = Stopwatch.StartNew();
            PaddleOcrDocumentOrientationClassifier classifier = engine.GetDocumentOrientationClassifier();
            PaddleOcrDocumentOrientationResult result = classifier.Classify(
                image.Pixels, image.Width, image.Height, image.Stride, ImagePixelFormat.Bgr24);
            double inferenceMs = stage.Elapsed.TotalMilliseconds;
            total.Stop();

            return Results.Ok(new DocumentOrientationResponse(
                result.Label,
                result.OrientationDegrees,
                result.Score,
                image.Width,
                image.Height,
                GetBuildConfiguration(),
                new StageElapsedMs(
                    Math.Round(decodeMs, 1),
                    Math.Round(inferenceMs, 1),
                    Math.Round(total.Elapsed.TotalMilliseconds, 1))));
        }
        catch (Exception ex)
        {
            return Results.Json(new OcrError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
        }
    }
}

static async Task<IResult> DetectTextAsync(
    [FromForm] DetectUpload upload,
    OcrEngine engine,
    OcrServerOptions options,
    CancellationToken cancellationToken)
{
    if (!engine.TryNormalizeModel(upload.Model, out string model))
        return Results.BadRequest(new OcrError("模型必须是 tiny、small 或 medium。"));

    Stopwatch total = Stopwatch.StartNew();
    (PixelBuffer? image, double decodeMs, IResult? error) = await DecodeUploadedImageAsync(upload.File, engine, options, cancellationToken);
    if (error is not null || image is null)
        return error!;

    using (image)
    {
        try
        {
            Stopwatch stage = Stopwatch.StartNew();
            PaddleOcrAll ocr = await engine.GetAsync(model);
            PaddleOcrDetectionResult result = ocr.Detector.Detect(
                image.Pixels, image.Width, image.Height, image.Stride, ImagePixelFormat.Bgr24);
            double inferenceMs = stage.Elapsed.TotalMilliseconds;
            total.Stop();

            return Results.Ok(new OcrDetectResponse(
                result.Boxes.Length,
                result.ResizedWidth,
                result.ResizedHeight,
                model,
                GetBuildConfiguration(),
                new StageElapsedMs(
                    Math.Round(decodeMs, 1),
                    Math.Round(inferenceMs, 1),
                    Math.Round(total.Elapsed.TotalMilliseconds, 1)),
                [.. result.Boxes.Select(ToDetectionBoxDto)]));
        }
        catch (Exception ex)
        {
            return Results.Json(new OcrError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
        }
    }
}

static async Task<IResult> ClassifyTextLineAsync(
    [FromForm] ImageUpload upload,
    OcrEngine engine,
    OcrServerOptions options,
    CancellationToken cancellationToken)
{
    Stopwatch total = Stopwatch.StartNew();
    (PixelBuffer? image, double decodeMs, IResult? error) = await DecodeUploadedImageAsync(upload.File, engine, options, cancellationToken);
    if (error is not null || image is null)
        return error!;

    using (image)
    {
        try
        {
            Stopwatch stage = Stopwatch.StartNew();
            PaddleOcrClassifier classifier = await engine.GetClassifierAsync();
            PaddleOcrClassificationResult result = classifier.Classify(
                image.Pixels, image.Width, image.Height, image.Stride, ImagePixelFormat.Bgr24);
            double inferenceMs = stage.Elapsed.TotalMilliseconds;
            total.Stop();

            return Results.Ok(new OcrClassifyResponse(
                result.Label,
                result.OrientationDegrees,
                result.Score,
                result.ResizedWidth,
                GetBuildConfiguration(),
                new StageElapsedMs(
                    Math.Round(decodeMs, 1),
                    Math.Round(inferenceMs, 1),
                    Math.Round(total.Elapsed.TotalMilliseconds, 1))));
        }
        catch (Exception ex)
        {
            return Results.Json(new OcrError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
        }
    }
}

static async Task<IResult> RecognizeTextLineAsync(
    [FromForm] RecognizeUpload upload,
    OcrEngine engine,
    OcrServerOptions options,
    CancellationToken cancellationToken)
{
    if (!engine.TryNormalizeModel(upload.Model, out string model))
        return Results.BadRequest(new OcrError("模型必须是 tiny、small 或 medium。"));

    Stopwatch total = Stopwatch.StartNew();
    (PixelBuffer? image, double decodeMs, IResult? error) = await DecodeUploadedImageAsync(upload.File, engine, options, cancellationToken);
    if (error is not null || image is null)
        return error!;

    using (image)
    {
        try
        {
            Stopwatch stage = Stopwatch.StartNew();
            PaddleOcrAll ocr = await engine.GetAsync(model);
            PaddleOcrRecognitionResult result = upload.Characters
                ? ocr.Recognizer.Recognize(image.Pixels, image.Width, image.Height, returnCtcAlignment: true, image.Stride, ImagePixelFormat.Bgr24)
                : ocr.Recognizer.Recognize(image.Pixels, image.Width, image.Height, image.Stride, ImagePixelFormat.Bgr24);
            double inferenceMs = stage.Elapsed.TotalMilliseconds;
            total.Stop();

            OcrCharacterDto[]? characters = null;
            if (upload.Characters)
            {
                PaddleOcrDetectionBox fullCrop = new(0, 0, image.Width, 0, image.Width, image.Height, 0, image.Height, 1f);
                characters = [.. PaddleOcrCharacterBoxes.Estimate(fullCrop, result).Select(ToCharacterDto)];
            }

            return Results.Ok(new OcrRecognizeResponse(
                result.Text,
                result.Score,
                result.EmittedCount,
                result.ResizedWidth,
                result.TimeSteps,
                model,
                GetBuildConfiguration(),
                new StageElapsedMs(
                    Math.Round(decodeMs, 1),
                    Math.Round(inferenceMs, 1),
                    Math.Round(total.Elapsed.TotalMilliseconds, 1)))
            {
                Characters = characters
            });
        }
        catch (Exception ex)
        {
            return Results.Json(new OcrError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
        }
    }
}

static async Task<(PixelBuffer? Image, double DecodeMs, IResult? Error)> DecodeUploadedImageAsync(
    IFormFile? file,
    OcrEngine engine,
    OcrServerOptions options,
    CancellationToken cancellationToken)
{
    if (file is null || file.Length == 0)
        return (null, 0, Results.BadRequest(new OcrError("请上传图片文件（表单字段 file）。")));
    if (file.Length > options.MaxUploadBytes)
        return (null, 0, Results.BadRequest(new OcrError($"图片超过 {options.MaxUploadBytes / (1024 * 1024)} MB 上限。")));

    Stopwatch stage = Stopwatch.StartNew();
    try
    {
        byte[] data = new byte[file.Length];
        await using (Stream stream = file.OpenReadStream())
            await stream.ReadExactlyAsync(data, cancellationToken);

        if (ImageCodecs.TryReadInfo(data, out ImageInfo info) && (long)info.Width * info.Height > engine.MaxImagePixels)
            return (null, 0, Results.BadRequest(new OcrError($"图片像素超过上限（{engine.MaxImagePixels:N0}）。")));

        PixelBuffer image = ImageCodecs.Decode(data, PixelFormat.Bgr24,
            new DecodeOptions { ApplyOrientation = true }, cancellationToken);
        return (image, stage.Elapsed.TotalMilliseconds, null);
    }
    catch (Exception ex) when (ex is NotSupportedException or InvalidDataException)
    {
        return (null, 0, Results.BadRequest(new OcrError(ex.Message)));
    }
    catch (Exception ex)
    {
        return (null, 0, Results.Json(new OcrError(ex.Message), statusCode: StatusCodes.Status500InternalServerError));
    }
}

static string? ExtractApiKey(HttpRequest request)
{
    if (request.Headers.TryGetValue("X-API-Key", out var apiKeyHeader) && !string.IsNullOrWhiteSpace(apiKeyHeader))
        return apiKeyHeader.ToString().Trim();

    if (request.Headers.TryGetValue("Authorization", out var authHeader))
    {
        string value = authHeader.ToString().Trim();
        if (value.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            return value["Bearer ".Length..].Trim();
    }

    return null;
}

static bool FixedTimeEquals(string provided, byte[] expectedBytes)
{
    byte[] providedBytes = Encoding.UTF8.GetBytes(provided);
    return providedBytes.Length == expectedBytes.Length &&
        CryptographicOperations.FixedTimeEquals(providedBytes, expectedBytes);
}

static OcrDetectionBoxDto ToDetectionBoxDto(PaddleOcrDetectionBox box) => new(
    [
        [box.X1, box.Y1],
        [box.X2, box.Y2],
        [box.X3, box.Y3],
        [box.X4, box.Y4]
    ],
    box.Score);

static OcrLineDto ToLineDto(PaddleOcrLine line, bool characters)
{
    OcrLineDto dto = new(
        line.Text,
        line.RecognitionScore,
        [
            [line.Box.X1, line.Box.Y1],
            [line.Box.X2, line.Box.Y2],
            [line.Box.X3, line.Box.Y3],
            [line.Box.X4, line.Box.Y4]
        ],
        line.Box.Score,
        line.ClassificationScore,
        line.AppliedRotationDegrees);
    if (!characters) return dto;
    return dto with
    {
        Characters = [.. line.EstimateCharacterBoxes().Select(ToCharacterDto)]
    };
}

static OcrCharacterDto ToCharacterDto(PaddleOcrCharacterBox box) => new(
    box.Text,
    box.Score,
    [
        [box.X1, box.Y1],
        [box.X2, box.Y2],
        [box.X3, box.Y3],
        [box.X4, box.Y4]
    ]);

static string? ResolveSampleImage(IWebHostEnvironment env)
{
    string[] candidates =
    [
        Path.Combine(env.WebRootPath, "sample.jpg"),
        Path.GetFullPath(Path.Combine(env.ContentRootPath, "..", "sample.jpg"))
    ];
    return candidates.FirstOrDefault(File.Exists);
}

static string GetBuildConfiguration()
{
#if DEBUG
    return "Debug";
#else
    return "Release";
#endif
}

internal sealed class ImageUpload
{
    [Description("待处理的图片文件（支持 JPEG / PNG / WebP / BMP / GIF / TIFF 等）。")]
    public IFormFile? File { get; init; }
}

internal sealed class DetectUpload
{
    [Description("待检测的图片文件（支持 JPEG / PNG / WebP / BMP / GIF / TIFF 等）。")]
    public IFormFile? File { get; init; }

    [Description("模型名称：tiny（默认）、small 或 medium。不传则使用服务端配置的默认模型。")]
    public string? Model { get; init; }
}

internal sealed class RecognizeUpload
{
    [Description("待识别的单行裁剪图片文件（支持 JPEG / PNG / WebP / BMP / GIF / TIFF 等）。")]
    public IFormFile? File { get; init; }

    [Description("模型名称：tiny（默认）、small 或 medium。不传则使用服务端配置的默认模型。")]
    public string? Model { get; init; }

    [Description("为 true 时返回单字框。不传则响应中不含 characters。")]
    public bool Characters { get; init; }
}

internal sealed class OcrUpload
{
    [Description("待识别的图片文件（支持 JPEG / PNG / WebP / BMP / GIF / TIFF 等）。")]
    public IFormFile? File { get; init; }

    [Description("模型名称：tiny（默认）、small 或 medium。不传则使用服务端配置的默认模型。")]
    public string? Model { get; init; }

    [Description("为 true 时每一行返回单字框。不传则响应中不含 characters。")]
    public bool Characters { get; init; }

    [Description("为 true 时先执行 0°/90°/180°/270° 整页方向检测并自动旋转扶正后再识别；不传则使用服务端默认配置（默认 false）。")]
    public bool? DocumentOrientation { get; init; }
}
