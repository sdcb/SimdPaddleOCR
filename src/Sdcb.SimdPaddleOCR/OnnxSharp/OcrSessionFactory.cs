namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Backend-neutral session factory. On net10.0+ it can hand out a GPU-backed
/// session; on netstandard2.0 (or when the Vulkan device is absent/unusable)
/// it always yields the CPU interpreter.
/// </summary>
internal static class OcrSessionFactory
{
    internal static IOcrSession Create(CompiledModel compiled, OcrBackend backend)
    {
#if NET10_0_OR_GREATER
        return Backends.Vulkan.GpuBackend.CreateSession(compiled, backend);
#else
        _ = backend;
        return compiled.CreateRequest();
#endif
    }

    /// <summary>True when the option resolves to GPU AND a device probed OK.</summary>
    internal static bool IsGpuBackend(OcrBackend backend)
    {
#if NET10_0_OR_GREATER
        return Backends.Vulkan.GpuBackend.IsVulkanSelected(backend) &&
            Backends.Vulkan.GpuBackend.TryGetDevice() is not null;
#else
        _ = backend;
        return false;
#endif
    }
}
