using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;
using Sdcb.PaddleOCR;

namespace SystemDrawing.WinForms;

internal sealed class MainForm : Form
{
    private readonly PictureBox _preview = new() { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom, BackColor = Color.FromArgb(32, 32, 32) };
    private readonly TextBox _imagePath = new();
    private readonly TextBox _detPath = new();
    private readonly TextBox _clsPath = new();
    private readonly TextBox _recPath = new();
    private readonly TextBox _dictPath = new();
    private readonly TextBox _result = new() { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Both };
    private readonly StatusStrip _statusBar = new()
    {
        Dock = DockStyle.Fill,
        AutoSize = false,
        SizingGrip = true,
        LayoutStyle = ToolStripLayoutStyle.HorizontalStackWithOverflow,
        BackColor = SystemColors.Control,
        ForeColor = Color.Black
    };
    private readonly ToolStripStatusLabel _status = new()
    {
        Spring = false,
        AutoSize = true,
        Overflow = ToolStripItemOverflow.Never,
        TextAlign = ContentAlignment.MiddleLeft,
        ForeColor = Color.Black
    };
    private readonly Button _run = new() { Text = "运行 OCR", AutoSize = true, Enabled = false };
    private readonly TableLayoutPanel _paths = new() { Dock = DockStyle.Fill, AutoSize = true, ColumnCount = 4 };
    private Bitmap? _bitmap;
    private PaddleOcrAll? _ocr;

    public MainForm()
    {
        Text = $"Sdcb.PaddleOCR - System.Drawing WinForms [{TargetFrameworkDisplayName}]";
        AutoScaleMode = AutoScaleMode.Dpi;
        AutoScaleDimensions = new SizeF(96F, 96F);
        Width = 1200; Height = 760; MinimumSize = new Size(900, 600);
        string root = FindRepositoryRoot();
        _imagePath.Text = Path.Combine(root, "examples", "sample.jpg");
        _detPath.Text = Path.Combine(root, "models", "det.onnx");
        _clsPath.Text = Path.Combine(root, "models", "cls.onnx");
        _recPath.Text = Path.Combine(root, "models", "rec.onnx");
        _dictPath.Text = Path.Combine(root, "models", "ppocr_keys.txt");
        _statusBar.Items.Add(_status);
        BuildLayout();
        Shown += async (_, _) => { LoadPreview(); await InitializeModelAsync(); };
        FormClosed += (_, _) => { _ocr?.Dispose(); _bitmap?.Dispose(); };
    }

    private void BuildLayout()
    {
        _paths.Padding = new Padding(8, 8, 8, 4);
        _paths.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 86)); _paths.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        _paths.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 70)); _paths.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 90));
        AddPathRow(0, "图片", _imagePath, "图片|*.jpg;*.jpeg;*.png;*.bmp;*.webp|所有文件|*.*", true);
        AddPathRow(1, "DET 模型", _detPath, "模型|*.onnx|所有文件|*.*", false);
        AddPathRow(2, "CLS 模型", _clsPath, "模型|*.onnx|所有文件|*.*", false);
        AddPathRow(3, "REC 模型", _recPath, "模型|*.onnx|所有文件|*.*", false);
        AddPathRow(4, "字典", _dictPath, "字典|*.txt|所有文件|*.*", false);

        TableLayoutPanel root = new() { Dock = DockStyle.Fill, Padding = new Padding(12), RowCount = 3, ColumnCount = 1 };
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize)); root.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); root.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
        root.Controls.Add(_paths, 0, 0);
        SplitContainer split = new() { Dock = DockStyle.Fill, Orientation = Orientation.Vertical };
        split.SizeChanged += (_, _) =>
        {
            int target = split.ClientSize.Width / 2;
            if (target > 0 && split.SplitterDistance != target)
                split.SplitterDistance = target;
        };
        split.Panel1.Controls.Add(_preview);
        TableLayoutPanel resultPanel = new() { Dock = DockStyle.Fill, Padding = new Padding(10, 0, 0, 0), RowCount = 2, ColumnCount = 1 };
        resultPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 28)); resultPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        resultPanel.Controls.Add(new Label { Text = "识别文本", Dock = DockStyle.Fill, Font = new Font("Microsoft YaHei UI", 10F, FontStyle.Bold) }, 0, 0);
        resultPanel.Controls.Add(_result, 0, 1);
        split.Panel2.Controls.Add(resultPanel);
        root.Controls.Add(split, 0, 1); root.Controls.Add(_statusBar, 0, 2); Controls.Add(root);
    }

    private void AddPathRow(int row, string label, TextBox textBox, string filter, bool image)
    {
        Label name = new() { Text = label, AutoSize = true, Anchor = AnchorStyles.Left, Margin = new Padding(0, 7, 6, 4) };
        textBox.Dock = DockStyle.Fill; textBox.ReadOnly = true; textBox.Margin = new Padding(0, 2, 6, 2);
        Button browse = new() { Text = "选择", Dock = DockStyle.Fill, Margin = new Padding(0, 2, 0, 2) };
        browse.Click += (_, _) => SelectFile(textBox, filter, image);
        _paths.Controls.Add(name, 0, row); _paths.Controls.Add(textBox, 1, row); _paths.Controls.Add(browse, 2, row);
        if (image) { _paths.Controls.Add(_run, 3, row); _run.Click += async (_, _) => await RunOcrAsync(); }
        else _paths.SetColumnSpan(browse, 2);
    }

    private void SelectFile(TextBox target, string filter, bool image)
    {
        using OpenFileDialog dialog = new() { Filter = filter, FileName = target.Text };
        if (dialog.ShowDialog(this) != DialogResult.OK) return;
        target.Text = dialog.FileName;
        if (image) { _result.Clear(); LoadPreview(); } else _ = InitializeModelAsync();
    }

    private async Task InitializeModelAsync()
    {
        _run.Enabled = false; _paths.Enabled = false; _status.Text = "正在加载模型…";
        string detPath = _detPath.Text;
        string clsPath = _clsPath.Text;
        string recPath = _recPath.Text;
        string dictPath = _dictPath.Text;
        try
        {
            PaddleOcrAll loaded = await Task.Run(() => PaddleOcrAll.LoadAsync(detPath, clsPath, recPath, dictPath));
            _ocr?.Dispose(); _ocr = loaded; _status.Text = $"模型已加载    图片：{_imagePath.Text}    配置：{BuildConfiguration}"; _run.Enabled = true;
        }
        catch (Exception ex) { _status.Text = $"模型加载失败：{ex.Message}"; }
        finally { _paths.Enabled = true; }
    }

    private void LoadPreview()
    {
        try { using Bitmap loaded = new(_imagePath.Text); SetBitmap(new Bitmap(loaded)); _status.Text = $"图片：{_imagePath.Text}    配置：{BuildConfiguration}"; }
        catch (Exception ex) { _status.Text = $"图片加载失败：{ex.Message}"; }
    }

    private async Task RunOcrAsync()
    {
        if (_ocr is null || _bitmap is null) return;
        _run.Enabled = false; _paths.Enabled = false; Stopwatch stopwatch = Stopwatch.StartNew();
        try
        {
            string imagePath = _imagePath.Text;
            OcrResult result = await Task.Run(() => RunPipeline(imagePath)); stopwatch.Stop(); _result.Text = result.Text; SetBitmap(result.Annotated);
            _status.Text = $"完成：{result.Count} 行，总耗时 {stopwatch.Elapsed.TotalMilliseconds:F1} ms。当前为 {BuildConfiguration}，可再次点击“运行 OCR”。";
        }
        catch (Exception ex) { _status.Text = $"OCR 执行失败：{ex.Message}"; }
        finally { _run.Enabled = _ocr is not null; _paths.Enabled = true; }
    }

    private OcrResult RunPipeline(string imagePath)
    {
        using Bitmap bitmap = new(imagePath);
        byte[] bgr = ToBgr(bitmap, out int stride); PaddleOcrResult result = _ocr!.Run(bgr, bitmap.Width, bitmap.Height, stride); Bitmap annotated = new(bitmap);
        using Graphics graphics = Graphics.FromImage(annotated); using Pen pen = new(Color.LimeGreen, 3); using Brush brush = new SolidBrush(Color.Red);
        foreach (PaddleOcrLine line in result.Lines)
        {
            Point[] points = [new((int)line.Box.X1, (int)line.Box.Y1), new((int)line.Box.X2, (int)line.Box.Y2), new((int)line.Box.X3, (int)line.Box.Y3), new((int)line.Box.X4, (int)line.Box.Y4)];
            graphics.DrawPolygon(pen, points); graphics.DrawString(line.Text, Font, brush, points[0]);
        }
        return new OcrResult(string.Join(Environment.NewLine, result.Lines.Select(line => line.Text)), result.DetectedCount, annotated);
    }

    private void SetBitmap(Bitmap bitmap) { Bitmap? old = _bitmap; _bitmap = bitmap; _preview.Image = bitmap; old?.Dispose(); }

    private static byte[] ToBgr(Bitmap bitmap, out int stride)
    {
        stride = checked(bitmap.Width * 3); byte[] bgr = new byte[checked(stride * bitmap.Height)]; Rectangle rectangle = new(0, 0, bitmap.Width, bitmap.Height);
        BitmapData data = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        try { for (int y = 0; y < bitmap.Height; y++) Marshal.Copy(new IntPtr(data.Scan0.ToInt64() + y * (long)data.Stride), bgr, y * stride, stride); }
        finally { bitmap.UnlockBits(data); }
        return bgr;
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

    private static string TargetFrameworkDisplayName
    {
#if NET48
        get => ".NET Framework 4.8";
#elif NET10_0_OR_GREATER
        get => ".NET 10 Windows";
#else
        get => ".NET";
#endif
    }

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory); while (directory is not null) { if (File.Exists(Path.Combine(directory.FullName, "Sdcb.PaddleOCR.slnx"))) return directory.FullName; directory = directory.Parent; }
        return Directory.GetCurrentDirectory();
    }

    private sealed class OcrResult
    {
        public OcrResult(string text, int count, Bitmap annotated) { Text = text; Count = count; Annotated = annotated; }
        public string Text { get; } public int Count { get; } public Bitmap Annotated { get; }
    }
}
