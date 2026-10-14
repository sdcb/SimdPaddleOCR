#if NET10_0_OR_GREATER
using System.Runtime.InteropServices;
using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
#endif

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// Backend-neutral session factory. On net10.0+ it can hand out a GPU-backed session over
/// Sdcb.SimdOnnx — Metal on Apple Silicon (unless overridden), Vulkan elsewhere — resolved as
/// option → SIMD_OCR_BACKEND env → device probe; on netstandard2.0, or when no usable GPU exists,
/// it yields the CPU session.
/// </summary>
internal static class OcrSessionFactory
{
    internal static IOcrSession Create(CompiledModel compiled, OcrBackend backend)
    {
#if NET10_0_OR_GREATER
        if (IsMetalSelected(backend))
        {
            if (MetalUsable(backend)) return new GpuOcrSession(compiled, metal: true);
            return new CpuOcrSession(compiled);
        }
        if (VulkanUsable(backend)) return new GpuOcrSession(compiled, metal: false);
#endif
        return new CpuOcrSession(compiled);
    }

    /// <summary>True when the option resolves to a GPU session on the probed device.</summary>
    internal static bool IsGpuBackend(OcrBackend backend)
    {
#if NET10_0_OR_GREATER
        return IsMetalSelected(backend) ? MetalUsable(backend) : VulkanUsable(backend);
#else
        _ = backend;
        return false;
#endif
    }

#if NET10_0_OR_GREATER
    private static string? EnvBackend => Environment.GetEnvironmentVariable("SIMD_OCR_BACKEND");

    private static bool EnvIs(string name) => string.Equals(EnvBackend, name, StringComparison.OrdinalIgnoreCase);

    private static bool IsMetalSelected(OcrBackend backend) => backend switch
    {
        OcrBackend.Metal => true,
        OcrBackend.Cpu or OcrBackend.Vulkan => false,
        _ when EnvIs("metal") => true,
        _ when EnvIs("cpu") || EnvIs("vulkan") => false,
        // Auto: Apple Silicon only (the Metal kernels assume 32-lane simdgroups)
        _ => OperatingSystem.IsMacOS() && RuntimeInformation.ProcessArchitecture == Architecture.Arm64,
    };

    private static bool IsVulkanSelected(OcrBackend backend) => backend switch
    {
        _ when OperatingSystem.IsBrowser() || OperatingSystem.IsWasi() => false,
        OcrBackend.Vulkan => true,
        OcrBackend.Cpu or OcrBackend.Metal => false,
        _ when EnvIs("cpu") || EnvIs("metal") => false,
        _ => true,   // Auto: prefer the GPU when the probe succeeds
    };

    private static readonly Lazy<VkBackend?> s_vk = new(() =>
    {
        try
        {
            if (OperatingSystem.IsBrowser() || OperatingSystem.IsWasi()) return null;
            VkBackend b = VkBackend.Default;
            _ = b.Dev.DeviceName;
            return b;
        }
        catch { return null; }
    });

    private static readonly Lazy<MetalBackend?> s_mtl = new(() =>
    {
        try { return OperatingSystem.IsMacOS() ? MetalBackend.Default : null; }
        catch { return null; }
    });

    /// <summary>Auto takes Vulkan on tensor-core devices (fp16 cooperative matrix) and on wide-subgroup
    /// mobile GPUs (Adreno: subgroups never below 64 lanes); other devices only for an explicit choice —
    /// there the CPU kernels win (as before the engine switch).</summary>
    private static bool VulkanUsable(OcrBackend backend)
    {
        if (!IsVulkanSelected(backend) || s_vk.Value is not { } vk) return false;
        bool explicitChoice = backend == OcrBackend.Vulkan || backend == OcrBackend.Auto && EnvIs("vulkan");
        return vk.Dev.CoopMat16 || vk.Dev.SubgroupMin >= 64 || explicitChoice;
    }

    private static bool MetalUsable(OcrBackend backend) => IsMetalSelected(backend) && s_mtl.Value is not null;
#endif
}
