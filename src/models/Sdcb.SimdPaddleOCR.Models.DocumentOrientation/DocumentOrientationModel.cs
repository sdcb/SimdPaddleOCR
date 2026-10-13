using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Sdcb.SimdPaddleOCR.ModelProvider;

namespace Sdcb.SimdPaddleOCR.Models.DocumentOrientation;

/// <summary>独立的四方向纸张模型包，通过 Provider 显式传给 PaddleOcrAll.Load 或 LoadAsync。</summary>
public static class DocumentOrientationModel
{
    /// <summary>提供内嵌的 PP-LCNet_x1_0_doc_ori ONNX 模型及版本信息，不依赖其他 OCR 模型包。</summary>
    public static IPaddleOcrModelProvider Provider { get; } = new ProviderImpl();
    /// <summary>打开内嵌纸张方向模型的只读流。</summary>
    /// <returns>新的 ONNX 资源流，由调用方负责释放。</returns>
    public static Stream OpenRead() => Provider.OpenRead();

    /// <summary>把程序集内嵌资源适配为通用模型提供方。</summary>
    private sealed class ProviderImpl : IPaddleOcrModelProvider
    {
        /// <summary>官方纸张方向模型名称。</summary>
        public string Name => "PP-LCNet_x1_0_doc_ori";
        /// <summary>模型类型为整页方向分类，用于区别文本行两分类。</summary>
        public PaddleOcrModelKind Kind => PaddleOcrModelKind.DocumentOrientation;
        /// <summary>模型文件格式；资源已完成独立 ONNX 简化及数值验证。</summary>
        public string Format => "onnx";
        /// <summary>方向分类不绑定特定识别语言，因此返回 null。</summary>
        public string? LanguageCode => null;
        /// <summary>固定的 Hugging Face 源模型提交哈希，用于追溯转换来源。</summary>
        public string? Version => "d3b95a6dff5fe8a94f2748e12b61cb26818a0df8";
        /// <summary>打开当前程序集内嵌的四分类 ONNX 模型。</summary>
        /// <returns>独立的只读资源流，由调用方负责释放。</returns>
        public Stream OpenRead() => typeof(DocumentOrientationModel).Assembly.GetManifestResourceStream(
            "Sdcb.SimdPaddleOCR.Models.DocumentOrientation.doc_ori.onnx")
            ?? throw new InvalidOperationException("Embedded document orientation model was not found.");
        /// <summary>以异步接口打开内嵌模型；本地资源无网络等待，验证取消后直接返回已完成任务。</summary>
        /// <param name="cancellationToken">已取消时立即抛出取消异常，不打开资源流。</param>
        /// <returns>包含新资源流的已完成任务；流由调用方释放。</returns>
        public Task<Stream> OpenReadAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(OpenRead());
        }
    }
}
