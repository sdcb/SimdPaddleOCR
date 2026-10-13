using System.Collections.Concurrent;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR;

/// <summary>在文本检测之前识别整页纸张的 0°、90°、180°、270° 顺时针方向。</summary>
/// <remarks>
/// 支持并发调用 <see cref="Classify"/>：共享只读编译模型，每个请求独立租用推理会话和预处理缓冲区。
/// 本类型只返回方向与置信度；整页旋转由 <see cref="PaddleOcrAll"/> 的第一阶段负责。
/// </remarks>
public sealed class PaddleOcrDocumentOrientationClassifier : IDisposable
{
    private readonly CompiledModel _compiled; // 共享编译结果，避免每次调用重新解析模型或打包权重。
    private readonly ConcurrentBag<IOcrSession> _sessions = []; // 会话独占输入与中间张量，可供并发请求分别租用。
    private readonly PaddleOcrDocumentOrientationOptions _options;
    private readonly bool _ownsModel; // 从流加载时拥有模型；直接传入 Model 时由调用方管理模型。
    private int _pooledCount; // 原子维护池中数量，避免在热路径遍历 ConcurrentBag。
    private bool _disposed;

    /// <summary>使用已加载的四分类 ONNX 模型创建纸张方向分类器。</summary>
    /// <param name="model">输入为 1×3×224×224、输出含四个方向分数的模型；调用方负责释放模型。</param>
    /// <param name="options">后端、线程与缓冲池配置；为 null 时使用默认配置。</param>
    public PaddleOcrDocumentOrientationClassifier(Model model, PaddleOcrDocumentOrientationOptions? options = null)
        : this(model, options, false) { }

    /// <summary>从 ONNX 流加载纸张方向模型，并在分类器释放时释放所加载的模型。</summary>
    /// <param name="model">从当前位置读取的模型流；流本身仍由调用方负责释放。</param>
    /// <param name="options">后端、线程与缓冲池配置；为 null 时使用默认配置。</param>
    public PaddleOcrDocumentOrientationClassifier(Stream model, PaddleOcrDocumentOrientationOptions? options = null)
        : this(Model.Load(model ?? throw new ArgumentNullException(nameof(model))), options, true) { }

    /// <summary>统一编译四方向模型、验证输出类别数，并记录模型的释放责任。</summary>
    /// <param name="model">已加载的四方向模型。</param>
    /// <param name="options">可选配置，为 null 时创建默认配置。</param>
    /// <param name="ownsModel">是否由分类器在构造失败或释放时释放模型。</param>
    private PaddleOcrDocumentOrientationClassifier(Model model, PaddleOcrDocumentOrientationOptions? options, bool ownsModel)
    {
        _options = options ?? new();
        _ownsModel = ownsModel;
        try
        {
            if (_options.IntraOpThreads is < 0 or > 16 || _options.MaxPooledSessions < 0 || _options.MaxImagePixels < 1)
                throw new ArgumentOutOfRangeException(nameof(options));
            // 默认最多使用 4 个 CPU 算子线程；GPU 后端也保留该预算供必要的 CPU 路径使用。
            int threads = _options.IntraOpThreads == 0 ? Math.Min(Environment.ProcessorCount, 4) : _options.IntraOpThreads;
            _compiled = new CompiledModel(model, [1, 3, 224, 224], threads);
            // 两分类 textline_ori 不能用于四方向纸张校正，在实际运行前检查并拒绝这类模型。
            using InferenceSession request = _compiled.CreateRequest();
            if (request.OutputShape.ElementCount != 4)
            {
                _compiled.Dispose();
                throw new ArgumentException("Document orientation requires four output classes (0, 90, 180, 270).", nameof(model));
            }
        }
        catch
        {
            if (ownsModel) model.Dispose();
            throw;
        }
    }

    /// <summary>对原始像素执行缩放、中心裁剪、RGB 归一化和四方向推理，返回分数最高的方向。</summary>
    /// <param name="source">按行排列的原始像素，可包含行尾填充；调用期间不会修改此缓冲区。</param>
    /// <param name="width">输入图像宽度，单位为像素，必须为正数。</param>
    /// <param name="height">输入图像高度，单位为像素，必须为正数。</param>
    /// <param name="stride">相邻两行首字节的间隔；0 表示按宽度和像素格式计算紧密排列的行跨度。</param>
    /// <param name="format">像素格式，支持 BGR24、RGB24、BGRA32、RGBA32，默认 BGR24。</param>
    /// <returns>原图顺时针方向及对应置信度；方向类别 0、1、2、3 对应 0°、90°、180°、270°。</returns>
    public PaddleOcrDocumentOrientationResult Classify(ReadOnlySpan<byte> source, int width, int height,
        int stride = 0, ImagePixelFormat format = ImagePixelFormat.Bgr24)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PaddleOcrDocumentOrientationClassifier));
        stride = ImagePixels.ResolveStride(width, stride, format);
        if ((long)width * height > _options.MaxImagePixels)
            throw new InvalidOperationException("Source image exceeds MaxImagePixels.");
        IOcrSession session;
        if (_sessions.TryTake(out IOcrSession? pooled))
        {
            Interlocked.Decrement(ref _pooledCount);
            session = pooled;
        }
        else session = OcrSessionFactory.Create(_compiled, _options.Backend);
        try
        {
            // GPU 会话需要显式确定形状；后端可自行选择 NCHW 或 NHWC 的内部输入布局。
            session.Reshape([1, 3, 224, 224]);
            bool profile = PipelineProfiler.Enabled;
            long started = profile ? PipelineProfiler.Now() : 0;
            PPOCRPreprocess.DocumentOrientation(source, width, height, stride, session.InputData,
                session.ResizeWorkspace, session.InputIsNhwc, format);
            if (profile) PipelineProfiler.Add(PipelineProfiler.DocumentPreprocess, started);
            started = profile ? PipelineProfiler.Now() : 0;
            ReadOnlySpan<float> probabilities = session.RunInternal(session.InputData);
            if (profile) PipelineProfiler.Add(PipelineProfiler.DocumentGraph, started);
            if (probabilities.Length != 4) throw new InvalidDataException("Expected four document orientation scores.");
            // 检查全部输出均为有限值；相同分数保留先出现的类别，与通常的 argmax 行为一致。
            int label = 0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                if (!MathCompat.IsFinite(probabilities[i])) throw new InvalidDataException("Non-finite document orientation score.");
                if (probabilities[i] > probabilities[label]) label = i;
            }
            return new((uint)label, probabilities[label]);
        }
        finally
        {
            // 即使预处理或推理抛出异常，也归还会话；超出池容量的会话立即释放。
            if (_disposed || Interlocked.Increment(ref _pooledCount) > _options.MaxPooledSessions)
            {
                if (!_disposed) Interlocked.Decrement(ref _pooledCount);
                session.Dispose();
            }
            else _sessions.Add(session);
        }
    }

    /// <summary>释放池中的推理会话和编译结果；从流创建时同时释放所拥有的模型。</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        while (_sessions.TryTake(out IOcrSession? session)) session.Dispose();
        _compiled.Dispose();
        if (_ownsModel) _compiled.Model.Dispose();
    }
}
