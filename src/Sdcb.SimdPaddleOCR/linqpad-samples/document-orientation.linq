<Query Kind="Program">
  <NuGetReference>Sdcb.SimdPaddleOCR</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.DocumentOrientation</NuGetReference>
  <NuGetReference Version="2.1.0">Sdcb.SimdCodecs</NuGetReference>
  <Namespace>Sdcb.SimdCodecs</Namespace>
  <Namespace>Sdcb.SimdCodecs.Abstractions</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.DocumentOrientation</Namespace>
  <Namespace>System.Linq</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
</Query>

async Task Main()
{
	using HttpClient http = new();
	byte[] jpeg = await http.GetByteArrayAsync("https://cv-public.sdcb.ai/2026/sample.jpg");

	using PixelBuffer sample = ImageCodecs.Decode(jpeg, PixelFormat.Bgr24, new DecodeOptions { ApplyOrientation = true });

	// Simulate a sideways (90° clockwise) scanned document: rotate 270° counterclockwise.
	(int srcW, int srcH) = PaddleOcrDocumentRotation.RotatedSize(sample.Width, sample.Height, 270);
	byte[] sideways = new byte[srcW * srcH * 3];
	PaddleOcrDocumentRotation.Rotate(sample.Pixels, sample.Width, sample.Height, sample.Stride, ImagePixelFormat.Bgr24, 270, sideways);

	using Stream oriStream = DocumentOrientationModel.OpenRead();
	using PaddleOcrDocumentOrientationClassifier docOri = new(oriStream);
	PaddleOcrDocumentOrientationResult ori = docOri.Classify(sideways, srcW, srcH);
	Console.WriteLine($"Detected page orientation: {ori.OrientationDegrees}° (label={ori.Label}, score={ori.Score:P2})");

	// Rotate counterclockwise by OrientationDegrees to make the page upright before OCR.
	(int uprightW, int uprightH) = PaddleOcrDocumentRotation.RotatedSize(srcW, srcH, ori.OrientationDegrees);
	byte[] upright = new byte[uprightW * uprightH * 3];
	PaddleOcrDocumentRotation.Rotate(sideways, srcW, srcH, 0, ImagePixelFormat.Bgr24, ori.OrientationDegrees, upright);

	using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
	PaddleOcrResult result = ocr.Run(upright, uprightW, uprightH);
	Console.WriteLine(result.Text);

	// Optional: map recognized boxes back into the original (sideways) source image coordinates.
	PaddleOcrLine[] sourceLines = result.Lines
		.Select(line => PaddleOcrDocumentRotation.ToSource(line, srcW, srcH, ori.OrientationDegrees))
		.ToArray();
	Console.WriteLine($"\nMapped {sourceLines.Length} lines back to source ({srcW}x{srcH}) coordinates.");
}
