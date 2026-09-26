namespace Sdcb.SimdPaddleOCR;

/// <summary>Compute backend for the OCR graph stages.</summary>
public enum OcrBackend
{
    /// <summary>Prefer Vulkan when a usable device exists, else CPU.</summary>
    Auto,
    /// <summary>Always run the CPU interpreter.</summary>
    Cpu,
    /// <summary>Run the graph on Vulkan; falls back to CPU when no usable device exists.</summary>
    Vulkan,
}
