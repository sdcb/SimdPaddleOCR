namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>Backend-neutral session factory.</summary>
internal static class OcrSessionFactory
{
    internal static IOcrSession Create(CompiledModel compiled, OcrBackend backend)
    {
        _ = backend;
        return new CpuOcrSession(compiled);
    }

    /// <summary>True when the option resolves to a GPU session on the probed device.</summary>
    internal static bool IsGpuBackend(OcrBackend backend)
    {
        _ = backend;
        return false;
    }
}
