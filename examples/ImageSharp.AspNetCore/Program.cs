using System.Diagnostics;
using ImageSharp.AspNetCore;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc;
using Scalar.AspNetCore;
using Sdcb.PaddleOCR;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

const long maxUploadBytes = 20 * 1024 * 1024;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = maxUploadBytes);
builder.Services.Configure<FormOptions>(options => options.MultipartBodyLengthLimit = maxUploadBytes);
builder.Services.AddSingleton<OcrEngine>();
builder.Services.AddOpenApi();
builder.Services.AddCors(options =>
    options.AddDefaultPolicy(policy => policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));

WebApplication app = builder.Build();
app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapOpenApi();
app.MapScalarApiReference();

app.MapGet("/api/ocr/models", (OcrEngine _) =>
    Results.Ok(new OcrModelList(OcrEngine.Models, OcrEngine.DefaultModel)))
    .WithName("ListOcrModels")
    .WithTags("OCR")
    .WithSummary("列出可用 OCR 模型")
    .Produces<OcrModelList>();

app.MapGet("/sample.jpg", (IWebHostEnvironment env) =>
{
    string? path = ResolveSampleImage(env);
    return path is null ? Results.NotFound() : Results.File(path, "image/jpeg");
});

app.MapPost("/api/ocr", RecognizeAsync)
    .DisableAntiforgery()
    .WithName("RecognizeImage")
    .WithTags("OCR")
    .WithSummary("识别图片中的文字")
    .WithDescription("上传图片并选择 tiny、small 或 medium 模型，返回识别文本、检测框和耗时。")
    .Produces<OcrResponse>()
    .Produces<OcrError>(StatusCodes.Status400BadRequest)
    .Produces<OcrError>(StatusCodes.Status500InternalServerError)
    .Accepts<OcrUpload>("multipart/form-data");

app.Run();

static async Task<IResult> RecognizeAsync(
    [FromForm] OcrUpload upload,
    OcrEngine engine,
    CancellationToken cancellationToken)
{
    IFormFile? file = upload.File;
    if (file is null || file.Length == 0)
        return Results.BadRequest(new OcrError("请上传图片文件（表单字段 file）。"));
    if (file.Length > maxUploadBytes)
        return Results.BadRequest(new OcrError("图片超过 20 MB 上限。"));
    if (!OcrEngine.TryNormalizeModel(upload.Model, out string model))
        return Results.BadRequest(new OcrError("模型必须是 tiny、small 或 medium。"));

    Stopwatch total = Stopwatch.StartNew();
    Stopwatch stage = Stopwatch.StartNew();
    try
    {
        await using Stream stream = file.OpenReadStream();
        using Image<Bgr24> image = await Image.LoadAsync<Bgr24>(stream, cancellationToken);
        if ((long)image.Width * image.Height > OcrEngine.MaxImagePixels)
            return Results.BadRequest(new OcrError("图片像素超过上限（4000 万）。"));

        byte[] bgr = new byte[checked(image.Width * image.Height * 3)];
        image.CopyPixelDataTo(bgr);
        double decodeMs = stage.Elapsed.TotalMilliseconds;

        stage.Restart();
        PaddleOcrAll ocr = await engine.GetAsync(model);
        PaddleOcrResult result = ocr.Run(bgr, image.Width, image.Height);
        double ocrMs = stage.Elapsed.TotalMilliseconds;
        total.Stop();

        return Results.Ok(new OcrResponse(
            result.Text,
            result.DetectedCount,
            model,
            GetBuildConfiguration(),
            new OcrElapsedMs(
                Math.Round(decodeMs, 1),
                Math.Round(ocrMs, 1),
                Math.Round(total.Elapsed.TotalMilliseconds, 1)),
            [.. result.Lines.Select(ToLineDto)]));
    }
    catch (UnknownImageFormatException)
    {
        return Results.BadRequest(new OcrError("无法解析图片，请上传 JPEG、PNG、WebP、BMP 或 GIF。"));
    }
    catch (InvalidImageContentException ex)
    {
        return Results.BadRequest(new OcrError($"图片内容无效：{ex.Message}"));
    }
    catch (InvalidOperationException ex) when (ex.Message.Contains("MaxImagePixels", StringComparison.Ordinal))
    {
        return Results.BadRequest(new OcrError("图片像素超过上限（4000 万）。"));
    }
    catch (Exception ex)
    {
        return Results.Json(new OcrError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
    }
}

static OcrLineDto ToLineDto(PaddleOcrLine line) => new(
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

internal sealed class OcrUpload
{
    public IFormFile? File { get; init; }
    public string? Model { get; init; }
}
