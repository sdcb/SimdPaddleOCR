using System.ComponentModel;
using System.Diagnostics;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc;
using Scalar.AspNetCore;
using Sdcb.SimdCodecs;
using Sdcb.SimdCodecs.Abstractions;
using Sdcb.SimdPaddleOCR;
using SimdCodecs.AspNetCore;

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
    .WithDescription("上传图片并选择 tiny、small 或 medium 模型，返回识别文本、检测框和耗时。表单字段 characters=true 时，每一行额外返回单字框（text、score、四个点）；不传该字段则响应中不含 characters。")
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
        // The facade decodes from memory: read the upload (at most 20 MB) into one array.
        byte[] data = new byte[file.Length];
        await using (Stream stream = file.OpenReadStream())
            await stream.ReadExactlyAsync(data, cancellationToken);

        // The header gives the size, so an oversized image is refused before any pixel is allocated.
        // A header that cannot be read is left to Decode below, which says why.
        if (ImageCodecs.TryReadInfo(data, out ImageInfo info) && (long)info.Width * info.Height > OcrEngine.MaxImagePixels)
            return Results.BadRequest(new OcrError("图片像素超过上限（4000 万）。"));

        // One call: packed BGR (the OCR default), EXIF orientation applied so phone photos arrive upright.
        using PixelBuffer image = ImageCodecs.Decode(data, PixelFormat.Bgr24,
            new DecodeOptions { ApplyOrientation = true }, cancellationToken);

        double decodeMs = stage.Elapsed.TotalMilliseconds;

        stage.Restart();
        PaddleOcrAll ocr = await engine.GetAsync(model);
        PaddleOcrResult result = upload.Characters
            ? ocr.Run(image.Pixels, image.Width, image.Height, returnCtcAlignment: true, image.Stride, ImagePixelFormat.Bgr24)
            : ocr.Run(image.Pixels, image.Width, image.Height, image.Stride, ImagePixelFormat.Bgr24);
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
            [.. result.Lines.Select(line => ToLineDto(line, upload.Characters))]));
    }
    catch (Exception ex) when (ex is NotSupportedException or InvalidDataException)
    {
        // The codec's own reason: the detected format that cannot be decoded, or what is wrong with the file.
        return Results.BadRequest(new OcrError(ex.Message));
    }
    catch (Exception ex)
    {
        return Results.Json(new OcrError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
    }
}

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

internal sealed class OcrUpload
{
    public IFormFile? File { get; init; }
    public string? Model { get; init; }

    [Description("为 true 时每一行返回单字框。不传则响应中不含 characters。")]
    public bool Characters { get; init; }
}
