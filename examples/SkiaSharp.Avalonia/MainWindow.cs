using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using Sdcb.SimdPaddleOCR;
using SkiaSharp;
using AvaloniaBitmap = Avalonia.Media.Imaging.Bitmap;

namespace SkiaSharp.Avalonia;

internal sealed class MainWindow : Window
{
    private readonly Image _preview = new();
    private readonly TextBox _imagePathText = new();
    private readonly TextBox _output = new();
    private readonly TextBlock _status = new();
    private readonly Button _runButton = new();
    private readonly Button _selectImageButton = new();
    private readonly RadioButton _tiny = new() { Content = "tiny", GroupName = "model" };
    private readonly RadioButton _small = new() { Content = "small", GroupName = "model" };
    private readonly RadioButton _medium = new() { Content = "medium", GroupName = "model", IsChecked = true };
    private readonly StackPanel _modelPanel = new();
    private AvaloniaBitmap? _previewBitmap;
    private PaddleOcrAll? _ocr;
    private string _imagePath;

    public MainWindow(string imagePath)
    {
        _imagePath = imagePath;

        Title = $"Sdcb.SimdPaddleOCR - SkiaSharp/Avalonia [{BuildConfiguration}]";
        Width = 1100;
        Height = 700;
        MinWidth = 820;
        MinHeight = 560;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;

        _selectImageButton.Content = "选择图片";
        _selectImageButton.Click += async (_, _) => await SelectImageAsync();

        _imagePathText.Text = _imagePath;
        _imagePathText.IsReadOnly = true;
        _imagePathText.TextWrapping = TextWrapping.NoWrap;
        _imagePathText.HorizontalAlignment = HorizontalAlignment.Stretch;

        _runButton.Content = "运行 OCR";
        _runButton.Background = Brushes.SteelBlue;
        _runButton.Foreground = Brushes.White;
        _runButton.BorderBrush = Brushes.DodgerBlue;
        _runButton.BorderThickness = new Thickness(1);
        _runButton.Padding = new Thickness(18, 8);
        _runButton.FontWeight = FontWeight.Bold;
        _runButton.Click += async (_, _) => await RunOcrAsync();
        _tiny.IsCheckedChanged += async (_, _) => await ModelSelectionChangedAsync();
        _small.IsCheckedChanged += async (_, _) => await ModelSelectionChangedAsync();
        _medium.IsCheckedChanged += async (_, _) => await ModelSelectionChangedAsync();

        _preview.Stretch = Stretch.Uniform;
        _preview.HorizontalAlignment = HorizontalAlignment.Stretch;
        _preview.VerticalAlignment = VerticalAlignment.Stretch;

        _output.AcceptsReturn = true;
        _output.IsReadOnly = true;
        _output.TextWrapping = TextWrapping.Wrap;
        _output.FontSize = 16;
        _output.MinHeight = 180;
        ScrollViewer.SetVerticalScrollBarVisibility(_output, ScrollBarVisibility.Auto);

        _status.VerticalAlignment = VerticalAlignment.Center;
        _status.TextTrimming = TextTrimming.CharacterEllipsis;
        _status.Text = $"请选择图片后运行 OCR    配置：{BuildConfiguration}";

        Grid toolbar = new()
        {
            ColumnDefinitions = new ColumnDefinitions("Auto, *, Auto"),
            ColumnSpacing = 8,
            Margin = new Thickness(0, 0, 0, 10)
        };
        toolbar.Children.Add(_selectImageButton);
        toolbar.Children.Add(_imagePathText);
        Grid.SetColumn(_imagePathText, 1);
        toolbar.Children.Add(_runButton);
        Grid.SetColumn(_runButton, 2);

        _modelPanel.Orientation = Orientation.Horizontal;
        _modelPanel.Margin = new Thickness(0, 0, 0, 8);
        _modelPanel.Children.Add(new TextBlock { Text = "模型：", VerticalAlignment = VerticalAlignment.Center });
        foreach (RadioButton button in new[] { _tiny, _small, _medium })
        {
            button.Margin = new Thickness(8, 0, 8, 0);
            _modelPanel.Children.Add(button);
        }

        Border imageBorder = new()
        {
            BorderBrush = Brushes.Gray,
            BorderThickness = new Thickness(1),
            Padding = new Thickness(4),
            Child = _preview
        };

        Border outputBorder = new()
        {
            BorderBrush = Brushes.Gray,
            BorderThickness = new Thickness(1),
            Padding = new Thickness(8),
            Child = _output
        };

        Grid resultPanel = new() { RowDefinitions = new RowDefinitions("Auto, *"), RowSpacing = 6 };
        resultPanel.Children.Add(new TextBlock { Text = "识别文本", FontWeight = FontWeight.Bold });
        resultPanel.Children.Add(outputBorder);
        Grid.SetRow(outputBorder, 1);
        Grid content = new()
        {
            ColumnDefinitions = new ColumnDefinitions("1*, 1*")
        };
        content.Children.Add(imageBorder);
        content.Children.Add(resultPanel);
        Grid.SetColumn(resultPanel, 1);

        Border statusBorder = new()
        {
            BorderBrush = Brushes.LightGray,
            BorderThickness = new Thickness(1),
            Padding = new Thickness(8, 5),
            Margin = new Thickness(0, 10, 0, 0),
            Child = _status
        };

        Grid root = new()
        {
            RowDefinitions = new RowDefinitions("Auto, Auto, *, Auto"),
            Margin = new Thickness(12)
        };
        root.Children.Add(toolbar);
        root.Children.Add(_modelPanel);
        Grid.SetRow(_modelPanel, 1);
        root.Children.Add(content);
        Grid.SetRow(content, 2);
        root.Children.Add(statusBorder);
        Grid.SetRow(statusBorder, 3);
        Content = root;

        Opened += async (_, _) =>
        {
            LoadPreview();
            await InitializeModelAsync();
        };
        Closed += (_, _) => DisposeResources();
    }

    private async Task SelectImageAsync()
    {
        IStorageProvider? provider = GetTopLevel(this)?.StorageProvider;
        if (provider is null || !provider.CanOpen)
            return;

        IReadOnlyList<IStorageFile> files = await provider.OpenFilePickerAsync(
            new FilePickerOpenOptions
            {
                AllowMultiple = false,
                Title = "选择 OCR 图片",
                FileTypeFilter =
                [
                    new FilePickerFileType("图片")
                    {
                        Patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
                    }
                ]
            });

        if (files.Count == 0)
            return;

        string? path = files[0].TryGetLocalPath();
        if (string.IsNullOrWhiteSpace(path))
        {
            _status.Text = "当前平台无法取得所选文件的本地路径";
            return;
        }

        _imagePath = path;
        _imagePathText.Text = path;
        _output.Text = string.Empty;
        LoadPreview();
    }

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

    private void LoadPreview()
    {
        try
        {
            if (!File.Exists(_imagePath))
                throw new FileNotFoundException("图片不存在", _imagePath);

            SetPreviewBitmap(new AvaloniaBitmap(_imagePath));
            _status.Text = $"图片：{_imagePath}    配置：{BuildConfiguration}";
        }
        catch (Exception ex)
        {
            _status.Text = $"图片加载失败：{ex.Message}";
        }
    }

    private async Task InitializeModelAsync()
    {
        _runButton.IsEnabled = false;
        _selectImageButton.IsEnabled = false;
        try
        {
            string modelName = SelectedModelName;
            _status.Text = $"正在加载 {modelName} 模型…";
            PaddleOcrAll loaded = await Task.Run(() => LoadModelAsync(modelName));
            _ocr?.Dispose();
            _ocr = loaded;
            _status.Text = $"模型已加载（{modelName}）    图片：{_imagePath}    配置：{BuildConfiguration}";
            _runButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            _status.Text = $"模型加载失败：{ex.Message}";
        }
        finally
        {
            _selectImageButton.IsEnabled = true;
        }
    }

    private string SelectedModelName => _tiny.IsChecked == true ? "tiny" : _small.IsChecked == true ? "small" : "medium";
    private static Task<PaddleOcrAll> LoadModelAsync(string name) => name switch { "tiny" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.ChineseV6TinyModels.Default), "small" => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Small.ChineseV6SmallModels.Default), _ => PaddleOcrAll.LoadAsync(Sdcb.SimdPaddleOCR.Models.ChineseV6Medium.ChineseV6MediumModels.Default) };
    private async Task ModelSelectionChangedAsync() { if (IsVisible) await InitializeModelAsync(); }

    private async Task RunOcrAsync()
    {
        if (string.IsNullOrWhiteSpace(_imagePath))
            return;

        Stopwatch total = Stopwatch.StartNew();
        double decodeMs = 0;
        double ocrMs = 0;

        try
        {
            _runButton.IsEnabled = false;
            _selectImageButton.IsEnabled = false;
            _status.Text = "正在解码图片…";

            Stopwatch stage = Stopwatch.StartNew();
            DecodedImage decoded = await Task.Run(() => DecodeBgr(_imagePath));
            decodeMs = stage.Elapsed.TotalMilliseconds;

            try
            {
                if (_ocr is null)
                    throw new InvalidOperationException("模型尚未加载");

                _status.Text = "正在运行 OCR…";
                stage.Restart();
                PaddleOcrResult result = await Task.Run(() =>
                    _ocr.Run(decoded.Bgr, decoded.Width, decoded.Height, decoded.Stride));
                ocrMs = stage.Elapsed.TotalMilliseconds;

                _status.Text = "正在绘制检测框和识别文本…";
                stage.Restart();
                byte[] annotatedPng = await Task.Run(() => AnnotateAndEncode(decoded.Bitmap, result));

                total.Stop();
                await Dispatcher.UIThread.InvokeAsync(() =>
                {
                    SetPreviewBitmap(LoadBitmap(annotatedPng));
                    _output.Text = string.Join(Environment.NewLine, result.Lines.Select(line => line.Text));
                    _status.Text =
                        $"完成：{result.DetectedCount} 行，总耗时 {total.Elapsed.TotalMilliseconds:F1} ms。当前为 {BuildConfiguration}，" +
                        "再次点击“运行 OCR”可重复运行。";
                });
            }
            finally
            {
                decoded.Dispose();
            }
        }
        catch (Exception ex)
        {
            total.Stop();
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                _output.Text = ex.ToString();
                _status.Text = "OCR 执行失败";
            });
        }
        finally
        {
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                _runButton.IsEnabled = true;
                _selectImageButton.IsEnabled = true;
            });
        }
    }

    private void SetPreviewBitmap(AvaloniaBitmap bitmap)
    {
        AvaloniaBitmap? old = _previewBitmap;
        _previewBitmap = bitmap;
        _preview.Source = bitmap;
        old?.Dispose();
    }

    private static AvaloniaBitmap LoadBitmap(byte[] png)
    {
        using MemoryStream stream = new(png, writable: false);
        return new AvaloniaBitmap(stream);
    }

    private static DecodedImage DecodeBgr(string path)
    {
        SKBitmap bitmap = SKBitmap.Decode(path)
            ?? throw new InvalidDataException($"无法读取图片：{path}");

        try
        {
            int stride = checked(bitmap.Width * 3);
            byte[] bgr = new byte[checked(stride * bitmap.Height)];

            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    SKColor color = bitmap.GetPixel(x, y);
                    int offset = y * stride + x * 3;
                    bgr[offset] = color.Blue;
                    bgr[offset + 1] = color.Green;
                    bgr[offset + 2] = color.Red;
                }
            }

            return new DecodedImage(bitmap, bgr, bitmap.Width, bitmap.Height, stride);
        }
        catch
        {
            bitmap.Dispose();
            throw;
        }
    }

    private static byte[] AnnotateAndEncode(SKBitmap bitmap, PaddleOcrResult result)
    {
        using SKCanvas canvas = new(bitmap);
        float fontSize = Math.Clamp(Math.Min(bitmap.Width, bitmap.Height) / 45f, 14f, 40f);
        using SKPaint boxPaint = new()
        {
            IsAntialias = true,
            Color = SKColors.LimeGreen,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = Math.Max(2f, fontSize / 8f)
        };
        using SKPaint textPaint = new()
        {
            IsAntialias = true,
            Color = SKColors.Red
        };
        using SKTypeface typeface = SKTypeface.FromFamilyName("Microsoft YaHei UI", SKFontStyle.Bold);
        using SKFont font = new(typeface, fontSize, 1, 0);
        foreach (PaddleOcrLine line in result.Lines)
        {
            SKPoint[] points =
            [
                new(line.Box.X1, line.Box.Y1),
                new(line.Box.X2, line.Box.Y2),
                new(line.Box.X3, line.Box.Y3),
                new(line.Box.X4, line.Box.Y4)
            ];

            for (int index = 0; index < points.Length; index++)
                canvas.DrawLine(points[index], points[(index + 1) % points.Length], boxPaint);

            string text = string.IsNullOrWhiteSpace(line.Text) ? "(空)" : line.Text;
            float minX = Math.Clamp(Math.Min(Math.Min(points[0].X, points[1].X),
                Math.Min(points[2].X, points[3].X)), 0, bitmap.Width - 1);
            float minY = Math.Min(Math.Min(points[0].Y, points[1].Y),
                Math.Min(points[2].Y, points[3].Y));
            float textX = minX;
            float textY = Math.Clamp(minY - 6, fontSize, bitmap.Height - 2);
            canvas.DrawText(text, textX, textY, SKTextAlign.Left, font, textPaint);
        }

        using SKImage image = SKImage.FromBitmap(bitmap);
        using SKData data = image.Encode(SKEncodedImageFormat.Png, 100)
            ?? throw new InvalidOperationException("无法编码标注图片");
        return data.ToArray();
    }

    private void DisposeResources()
    {
        _ocr?.Dispose();
        _ocr = null;
        _previewBitmap?.Dispose();
        _previewBitmap = null;
    }

    private sealed class DecodedImage : IDisposable
    {
        public DecodedImage(SKBitmap bitmap, byte[] bgr, int width, int height, int stride)
        {
            Bitmap = bitmap;
            Bgr = bgr;
            Width = width;
            Height = height;
            Stride = stride;
        }

        public SKBitmap Bitmap { get; }
        public byte[] Bgr { get; }
        public int Width { get; }
        public int Height { get; }
        public int Stride { get; }

        public void Dispose() => Bitmap.Dispose();
    }
}
