using System;
using System.Diagnostics;
using DrawingBitmap = System.Drawing.Bitmap;
using DrawingGraphics = System.Drawing.Graphics;
using DrawingPen = System.Drawing.Pen;
using DrawingBrush = System.Drawing.Brush;
using DrawingColor = System.Drawing.Color;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using OpenCvSharp;
using Sdcb.PaddleOCR;

namespace OpenCvSharp5.Wpf;

internal sealed class MainWindow : System.Windows.Window
{
    private readonly System.Windows.Controls.Image _preview = new();
    private readonly TextBox _imagePathText = new();
    private readonly TextBox _output = new();
    private readonly TextBlock _status = new();
    private readonly Button _runButton = new() { Content = "运行 OCR", IsEnabled = false };
    private readonly Button _selectImageButton = new() { Content = "选择图片" };
    private readonly RadioButton _tiny = new() { Content = "tiny", GroupName = "model" };
    private readonly RadioButton _small = new() { Content = "small", GroupName = "model", IsChecked = true };
    private readonly RadioButton _medium = new() { Content = "medium", GroupName = "model" };
    private readonly StackPanel _modelPanel = new();
    private PaddleOcrAll? _ocr;
    private string _imagePath;

    public MainWindow(string imagePath, string modelDirectory)
    {
        _ = modelDirectory; _imagePath = imagePath;
        Title = $"Sdcb.PaddleOCR - OpenCvSharp5 WPF [{BuildConfiguration}]";
        Width = 1200; Height = 760; MinWidth = 900; MinHeight = 600; WindowStartupLocation = WindowStartupLocation.CenterScreen;
        _imagePathText.Text = imagePath; _imagePathText.IsReadOnly = true;
        _output.AcceptsReturn = true; _output.IsReadOnly = true; _output.TextWrapping = TextWrapping.Wrap; _output.FontSize = 14; ScrollViewer.SetVerticalScrollBarVisibility(_output, ScrollBarVisibility.Auto);
        _status.Text = $"正在加载模型…    配置：{BuildConfiguration}";
        _selectImageButton.Click += (_, _) => SelectImage(); _runButton.Click += async (_, _) => await RunOcrAsync();
        _tiny.Checked += async (_, _) => await ModelSelectionChangedAsync(); _small.Checked += async (_, _) => await ModelSelectionChangedAsync(); _medium.Checked += async (_, _) => await ModelSelectionChangedAsync();
        BuildLayout(); Loaded += async (_, _) => { LoadPreview(); await InitializeModelAsync(); }; Closed += (_, _) => _ocr?.Dispose();
    }

    private void BuildLayout()
    {
        StackPanel toolbar = new() { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 8) };
        toolbar.Children.Add(_selectImageButton); _imagePathText.Margin = new Thickness(8, 0, 8, 0); _imagePathText.Width = 700; toolbar.Children.Add(_imagePathText); toolbar.Children.Add(_runButton);
        _modelPanel.Orientation = Orientation.Horizontal; _modelPanel.Margin = new Thickness(0, 0, 0, 8); _modelPanel.Children.Add(new TextBlock { Text = "模型：", VerticalAlignment = VerticalAlignment.Center });
        foreach (RadioButton button in new[] { _tiny, _small, _medium }) { button.Margin = new Thickness(8, 0, 8, 0); _modelPanel.Children.Add(button); }
        Border imageBorder = new() { BorderBrush = Brushes.Gray, BorderThickness = new Thickness(1), Padding = new Thickness(4), Child = _preview };
        Border outputBorder = new() { BorderBrush = Brushes.Gray, BorderThickness = new Thickness(1), Padding = new Thickness(8), Child = _output };
        Grid result = new(); result.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); result.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) }); result.Children.Add(new TextBlock { Text = "识别文本", FontWeight = FontWeights.Bold, Margin = new Thickness(0, 0, 0, 6) }); result.Children.Add(outputBorder); Grid.SetRow(outputBorder, 1);
        Grid content = new(); content.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) }); content.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) }); content.Children.Add(imageBorder); content.Children.Add(result); Grid.SetColumn(result, 1);
        Border status = new() { BorderBrush = Brushes.LightGray, BorderThickness = new Thickness(1), Padding = new Thickness(8, 5, 8, 5), Margin = new Thickness(0, 8, 0, 0), Child = _status };
        Grid root = new() { Margin = new Thickness(10) }; root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) }); root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); root.Children.Add(toolbar); root.Children.Add(_modelPanel); Grid.SetRow(_modelPanel, 1); root.Children.Add(content); Grid.SetRow(content, 2); root.Children.Add(status); Grid.SetRow(status, 3); Content = root;
    }

    private void SelectImage() { OpenFileDialog dialog = new() { Title = "选择 OCR 图片", Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp;*.webp|所有文件|*.*", FileName = _imagePath }; if (dialog.ShowDialog(this) != true) return; _imagePath = dialog.FileName; _imagePathText.Text = _imagePath; _output.Clear(); LoadPreview(); }
    private async Task ModelSelectionChangedAsync() { if (IsLoaded) await InitializeModelAsync(); }
    private string SelectedModelName => _tiny.IsChecked == true ? "tiny" : _medium.IsChecked == true ? "medium" : "small";
    private static Task<PaddleOcrAll> LoadModelAsync(string name) => name switch { "tiny" => PaddleOcrAll.LoadAsync(Sdcb.PaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default), "medium" => PaddleOcrAll.LoadAsync(Sdcb.PaddleOCR.Models.ChineseV6Medium.ChineseV6MediumModels.Default), _ => PaddleOcrAll.LoadAsync(Sdcb.PaddleOCR.Models.ChineseV6Small.ChineseV6SmallModels.Default) };

    private async Task InitializeModelAsync()
    {
        _runButton.IsEnabled = false; _modelPanel.IsEnabled = false; _selectImageButton.IsEnabled = false;
        string modelName = SelectedModelName;
        try { _status.Text = $"正在加载 {modelName} 模型…"; PaddleOcrAll loaded = await Task.Run(() => LoadModelAsync(modelName)); PaddleOcrAll? old = _ocr; _ocr = loaded; old?.Dispose(); _status.Text = $"模型已加载（{modelName}）    图片：{_imagePath}    配置：{BuildConfiguration}"; _runButton.IsEnabled = true; }
        catch (Exception ex) { _status.Text = $"模型加载失败：{ex.Message}"; }
        finally { _modelPanel.IsEnabled = true; _selectImageButton.IsEnabled = true; }
    }

    private void LoadPreview() { try { using Mat image = Cv2.ImRead(_imagePath, ImreadModes.Color); if (image.Empty()) throw new InvalidDataException($"无法读取图片：{_imagePath}"); _preview.Source = ToBitmapSource(image); _status.Text = $"图片：{_imagePath}    配置：{BuildConfiguration}"; } catch (Exception ex) { _status.Text = $"图片加载失败：{ex.Message}"; } }

    private async Task RunOcrAsync()
    {
        PaddleOcrAll? ocr = _ocr; string path = _imagePath; if (ocr is null || string.IsNullOrWhiteSpace(path)) return; _runButton.IsEnabled = false; _modelPanel.IsEnabled = false; _selectImageButton.IsEnabled = false; Stopwatch total = Stopwatch.StartNew();
        try { OcrRunResult run = await Task.Run(() => RunPipeline(path, ocr)); total.Stop(); _preview.Source = ToBitmapSource(run.Annotated); _output.Text = string.Join(Environment.NewLine, run.Result.Lines.Select(line => line.Text)); _status.Text = $"完成：{run.Result.DetectedCount} 行，总耗时 {total.Elapsed.TotalMilliseconds:F1} ms。当前为 {BuildConfiguration}，可再次点击“运行 OCR”。"; run.Annotated.Dispose(); }
        catch (Exception ex) { _output.Text = ex.ToString(); _status.Text = "OCR 执行失败"; }
        finally { _runButton.IsEnabled = _ocr is not null; _modelPanel.IsEnabled = true; _selectImageButton.IsEnabled = true; }
    }

    private static OcrRunResult RunPipeline(string path, PaddleOcrAll ocr) { using Mat image = Cv2.ImRead(path, ImreadModes.Color); if (image.Empty()) throw new InvalidDataException($"无法读取图片：{path}"); PaddleOcrResult result = ocr.Run(CopyBgr(image), image.Width, image.Height); return new OcrRunResult(result, Annotate(image, result)); }
    private static BitmapSource ToBitmapSource(Mat image) { Cv2.ImEncode(".png", image, out byte[] png); return ToBitmapSource(png); }
    private static BitmapSource ToBitmapSource(DrawingBitmap image) { using MemoryStream stream = new(); image.Save(stream, System.Drawing.Imaging.ImageFormat.Png); return ToBitmapSource(stream.ToArray()); }
    private static BitmapSource ToBitmapSource(byte[] png) { using MemoryStream stream = new(png, false); BitmapImage bitmap = new(); bitmap.BeginInit(); bitmap.CacheOption = BitmapCacheOption.OnLoad; bitmap.StreamSource = stream; bitmap.EndInit(); bitmap.Freeze(); return bitmap; }
    private static DrawingBitmap Annotate(Mat image, PaddleOcrResult result) { Cv2.ImEncode(".png", image, out byte[] png); using MemoryStream stream = new(png, false); using System.Drawing.Image source = System.Drawing.Image.FromStream(stream); DrawingBitmap bitmap = new(source); using DrawingGraphics graphics = DrawingGraphics.FromImage(bitmap); using DrawingPen pen = new(DrawingColor.LimeGreen, 3); using DrawingBrush brush = new System.Drawing.SolidBrush(DrawingColor.Red); foreach (PaddleOcrLine line in result.Lines) { System.Drawing.PointF[] points = [new(line.Box.X1, line.Box.Y1), new(line.Box.X2, line.Box.Y2), new(line.Box.X3, line.Box.Y3), new(line.Box.X4, line.Box.Y4)]; graphics.DrawPolygon(pen, points); graphics.DrawString(line.Text, System.Drawing.SystemFonts.DefaultFont, brush, points[0]); } return bitmap; }
    private static byte[] CopyBgr(Mat image) { int rowBytes = checked(image.Width * image.Channels()); int height = image.Height; byte[] bgr = new byte[checked(rowBytes * height)]; for (int y = 0; y < height; y++) Marshal.Copy(IntPtr.Add(image.Data, checked((int)(y * image.Step()))), bgr, y * rowBytes, rowBytes); return bgr; }
    private static string BuildConfiguration
    {
        get
        {
#if DEBUG
            return "Debug";
#else
            return "Release";
#endif
        }
    }
    private sealed record OcrRunResult(PaddleOcrResult Result, DrawingBitmap Annotated);
}
