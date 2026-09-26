using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

/// <summary>
/// Process-wide Vulkan device for OCR graphs plus the backend-resolution
/// rules (option → SIMD_OCR_BACKEND env → probe). Sessions serialize their
/// GPU submissions on <see cref="VkDevice.Sync"/>.
/// </summary>
internal static class GpuBackend
{
    private static readonly object s_probeLock = new();
    private static VkDevice? s_device;
    private static bool s_probed;

    /// <summary>The shared device, or null when no usable Vulkan GPU exists.</summary>
    internal static VkDevice? TryGetDevice()
    {
        if (s_probed) return s_device;
        lock (s_probeLock)
        {
            if (s_probed) return s_device;
            try { s_device = VkDevice.Create(); }
            catch { s_device = null; }
            s_probed = true;
            return s_device;
        }
    }

    /// <summary>Whether the given option resolves to the Vulkan path.</summary>
    internal static bool IsVulkanSelected(OcrBackend backend) => backend switch
    {
        OcrBackend.Vulkan => true,
        OcrBackend.Cpu => false,
        _ => Environment.GetEnvironmentVariable("SIMD_OCR_BACKEND") switch
        {
            { } s when s.Equals("cpu", StringComparison.OrdinalIgnoreCase) => false,
            { } s when s.Equals("vulkan", StringComparison.OrdinalIgnoreCase) => true,
            _ => true, // Auto: prefer GPU when the probe succeeds
        },
    };

    /// <summary>Creates a session on the resolved backend; CPU on any GPU failure.</summary>
    internal static IOcrSession CreateSession(CompiledModel compiled, OcrBackend backend)
    {
        if (IsVulkanSelected(backend) && TryGetDevice() is { } dev)
        {
            try { return new GpuSession(dev, compiled); }
            catch { /* fall through to CPU */ }
        }
        return compiled.CreateRequest();
    }
}
