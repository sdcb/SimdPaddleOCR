<Query Kind="Program">
  <NuGetReference>Sdcb.SimdPaddleOCR</NuGetReference>
  <NuGetReference>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</NuGetReference>
  <NuGetReference Version="2.0.0-preview.20261109.2" Prerelease="true">Sdcb.SimdCodecs</NuGetReference>
  <Namespace>Sdcb.SimdCodecs</Namespace>
  <Namespace>Sdcb.SimdCodecs.Abstractions</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR</Namespace>
  <Namespace>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
</Query>

async Task Main()
{
	using HttpClient http = new();
	byte[] jpeg = await http.GetByteArrayAsync("https://cv-public.sdcb.ai/2026/sample.jpg");

	using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);
	// Bgr24 is the OCR default layout; ApplyOrientation makes phone photos (EXIF Orientation) upright.
	using PixelBuffer image = ImageCodecs.Decode(jpeg, PixelFormat.Bgr24, new DecodeOptions { ApplyOrientation = true });
	PaddleOcrResult result = ocr.Run(image.Pixels, image.Width, image.Height, image.Stride);
	Console.WriteLine(result.Text);
}
