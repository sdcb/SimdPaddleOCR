namespace Sdcb.SimdPaddleOCR.ModelProvider;

public enum PaddleOcrModelKind
{
    Detection,
    Classification,
    Recognition,
    Dictionary,
    /// <summary>整页纸张的 0°、90°、180°、270° 方向分类模型，作为完整 OCR 的第一阶段。</summary>
    DocumentOrientation,
}
