// Minimal Vulkan 1.1 P/Invoke for compute-only use (SimdPaddleOCR Vulkan backend).
// Loads vulkan-1.dll (Windows) / libvulkan.so.1 (Linux) via SetDllImportResolver.
// Only the entry points and structs we need; all structs are blittable.
using System.Runtime.InteropServices;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

internal static unsafe partial class Vk
{
    public const string LibName = "vulkan-1";

    private static int _resolverRegistered;

    public static void RegisterResolver()
    {
        if (Interlocked.Exchange(ref _resolverRegistered, 1) != 0)
            return;
        NativeLibrary.SetDllImportResolver(typeof(Vk).Assembly, (name, asm, path) =>
        {
            if (name != LibName)
                return IntPtr.Zero;
            string candidate = OperatingSystem.IsWindows() ? "vulkan-1.dll" : "libvulkan.so.1";
            return NativeLibrary.TryLoad(candidate, out IntPtr h) ? h : IntPtr.Zero;
        });
    }

    public const ulong WholeSize = ~0UL;
    public const uint QueueFamilyIgnored = ~0u;

    public static void Check(VkResult r, string what)
    {
        if (r != VkResult.Success)
            throw new InvalidOperationException($"Vulkan {what} failed: {r}");
    }

    // ---- structs (field order must match the Vulkan spec) ----

    [StructLayout(LayoutKind.Sequential)]
    public struct VkApplicationInfo
    {
        public uint SType; public void* PNext; public void* PAppName;
        public uint AppVersion; public void* PEngineName; public uint EngineVersion; public uint ApiVersion;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkInstanceCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public VkApplicationInfo* PAppInfo;
        public uint EnabledLayerCount; public void* PpEnabledLayerNames;
        public uint EnabledExtensionCount; public void* PpEnabledExtensionNames;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceProperties
    {
        public uint ApiVersion, DriverVersion, VendorID, DeviceID, DeviceType;
        public fixed byte DeviceName[256];
        public fixed byte PipelineCacheUUID[16];
        public fixed byte LimitsAndSparse[1024]; // VkPhysicalDeviceLimits then Sparse; timestampPeriod float @428
        public float TimestampPeriodNs
        {
            get { fixed (byte* p = LimitsAndSparse) return *(float*)(p + 428); }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceProperties2
    {
        public uint SType; public void* PNext;
        public VkPhysicalDeviceProperties Properties;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceSubgroupSizeControlProperties
    {
        public uint SType; public void* PNext;
        public uint MinSubgroupSize, MaxSubgroupSize;
        public uint MaxComputeWorkgroupSubgroups, RequiredSubgroupSizeStages;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceSubgroupProperties
    {
        public uint SType; public void* PNext;
        public uint SubgroupSize;
        public uint SupportedStages;       // VkShaderStageFlags
        public uint SupportedOperations;   // VkSubgroupFeatureFlags
        public uint QuadOperationsInAllStages;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryType { public uint PropertyFlags; public uint HeapIndex; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryHeap { public ulong Size; public uint Flags; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceMemoryProperties
    {
        public uint MemoryTypeCount;
        public fixed byte MemoryTypes[32 * 8]; // VkMemoryType[32]
        public uint MemoryHeapCount;
        public fixed byte MemoryHeaps[16 * 16]; // VkMemoryHeap is 16 bytes (8-aligned), not 12

        public VkMemoryType TypeAt(int i)
        {
            fixed (byte* p = MemoryTypes) return ((VkMemoryType*)p)[i];
        }
        public VkMemoryHeap HeapAt(int i)
        {
            fixed (byte* p = MemoryHeaps) return ((VkMemoryHeapNative*)p)[i].ToManaged();
        }
        private struct VkMemoryHeapNative { public ulong Size; public uint Flags; public uint Pad; public VkMemoryHeap ToManaged() => new() { Size = Size, Flags = Flags }; }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkQueueFamilyProperties
    {
        public uint QueueFlags, QueueCount, TimestampValidBits;
        public uint MinImageTransferGranularityX, MinImageTransferGranularityY, MinImageTransferGranularityZ;
    }

    /// <summary>VkPhysicalDeviceFeatures as raw uint[55]; shaderInt16 is index 41.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceFeatures
    {
        public fixed uint F[55];
        public uint ShaderInt16 { get => F[41]; set => F[41] = value; }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceFeatures2
    {
        public uint SType; public void* PNext;
        public VkPhysicalDeviceFeatures Features;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDevice16BitStorageFeatures
    {
        public uint SType; public void* PNext;
        public uint StorageBuffer16BitAccess, UniformAndStorageBuffer16BitAccess, StoragePushConstant16, StorageInputOutput16;
    }

    // VK_KHR_shader_float16_int8 (Vulkan 1.2: shaderFloat16/shaderInt8)
    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceShaderFloat16Int8Features
    {
        public uint SType; public void* PNext;
        public uint ShaderFloat16, ShaderInt8;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDeviceQueueCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public uint QueueFamilyIndex, QueueCount; public float* PQueuePriorities;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDeviceCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public uint QueueCreateInfoCount; public VkDeviceQueueCreateInfo* PQueueCreateInfos;
        public uint EnabledLayerCount; public void* PpEnabledLayerNames;
        public uint EnabledExtensionCount; public void* PpEnabledExtensionNames;
        public void* PEnabledFeatures;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkBufferCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public ulong Size; public uint Usage; public uint SharingMode;
        public uint QueueFamilyIndexCount; public uint* PQueueFamilyIndices;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryRequirements { public ulong Size; public ulong Alignment; public uint MemoryTypeBits; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryAllocateInfo
    {
        public uint SType; public void* PNext;
        public ulong AllocationSize; public uint MemoryTypeIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMappedMemoryRange
    {
        public uint SType; public void* PNext;
        public IntPtr Memory; public ulong Offset; public ulong Size;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkShaderModuleCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public nuint CodeSize; public uint* PCode;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPushConstantRange { public uint StageFlags; public uint Offset; public uint Size; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPipelineLayoutCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public uint SetLayoutCount; public IntPtr* PSetLayouts;
        public uint PushConstantRangeCount; public VkPushConstantRange* PPushConstantRanges;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPipelineShaderStageCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public uint Stage; public IntPtr Module; public byte* PName; public void* PSpecializationInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR
    {
        public uint SType; public void* PNext;
        public uint CooperativeMatrix; public uint CooperativeMatrixRobustBufferAccess;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkCooperativeMatrixPropertiesKHR
    {
        public uint SType; public void* PNext;
        public uint MSize, NSize, KSize;
        public uint AType, BType, CType, ResultType;
        public uint SaturatingAccumulation, Scope;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceSubgroupSizeControlFeaturesEXT
    {
        public uint SType; public void* PNext;
        public uint SubgroupSizeControl; public uint ComputeFullSubgroups;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
    {
        public uint SType; public void* PNext; public uint RequiredSubgroupSize;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkComputePipelineCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public VkPipelineShaderStageCreateInfo Stage;
        public IntPtr Layout; public IntPtr BasePipelineHandle; public int BasePipelineIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorSetLayoutBinding
    {
        public uint Binding; public uint DescriptorType; public uint DescriptorCount;
        public uint StageFlags; public IntPtr* PImmutableSamplers;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorSetLayoutCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public uint BindingCount; public VkDescriptorSetLayoutBinding* PBindings;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorPoolSize { public uint Type; public uint DescriptorCount; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorPoolCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags;
        public uint MaxSets; public uint PoolSizeCount; public VkDescriptorPoolSize* PPoolSizes;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorSetAllocateInfo
    {
        public uint SType; public void* PNext; public IntPtr DescriptorPool;
        public uint DescriptorSetCount; public IntPtr* PSetLayouts;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorBufferInfo { public IntPtr Buffer; public ulong Offset; public ulong Range; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkExtensionProperties
    {
        public unsafe fixed byte ExtensionName[256];
        public uint SpecVersion;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkWriteDescriptorSet
    {
        public uint SType; public void* PNext; public IntPtr DstSet;
        public uint DstBinding; public uint DstArrayElement; public uint DescriptorCount;
        public uint DescriptorType; public void* PImageInfo; public VkDescriptorBufferInfo* PBufferInfo;
        public void* PTexelBufferView;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkCommandPoolCreateInfo
    {
        public uint SType; public void* PNext; public uint Flags; public uint QueueFamilyIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkCommandBufferAllocateInfo
    {
        public uint SType; public void* PNext; public IntPtr CommandPool;
        public uint Level; public uint CommandBufferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkCommandBufferBeginInfo
    {
        public uint SType; public void* PNext; public uint Flags; public void* PInheritanceInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkSubmitInfo
    {
        public uint SType; public void* PNext;
        public uint WaitSemaphoreCount; public IntPtr* PWaitSemaphores; public uint* PWaitDstStageMask;
        public uint CommandBufferCount; public IntPtr* PCommandBuffers;
        public uint SignalSemaphoreCount; public IntPtr* PSignalSemaphores;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkFenceCreateInfo { public uint SType; public void* PNext; public uint Flags; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryBarrier
    {
        public uint SType; public void* PNext; public uint SrcAccessMask; public uint DstAccessMask;
    }

    // ---- entry points ----

    [LibraryImport(LibName)] public static partial VkResult vkCreateInstance(VkInstanceCreateInfo* ci, void* alloc, out IntPtr instance);
    [LibraryImport(LibName)] public static partial void vkDestroyInstance(IntPtr instance, void* alloc);
    [LibraryImport(LibName)] public static partial VkResult vkEnumeratePhysicalDevices(IntPtr instance, uint* count, IntPtr* devices);
    [LibraryImport(LibName)] public static partial void vkGetPhysicalDeviceProperties(IntPtr device, VkPhysicalDeviceProperties* props);
    [LibraryImport(LibName)] public static partial void vkGetPhysicalDeviceProperties2(IntPtr device, VkPhysicalDeviceProperties2* props);
    [LibraryImport(LibName)] public static partial void vkGetPhysicalDeviceMemoryProperties(IntPtr device, VkPhysicalDeviceMemoryProperties* props);
    [LibraryImport(LibName)] public static partial void vkGetPhysicalDeviceQueueFamilyProperties(IntPtr device, uint* count, VkQueueFamilyProperties* props);
    [LibraryImport(LibName)] public static partial void vkGetPhysicalDeviceFeatures(IntPtr device, VkPhysicalDeviceFeatures* features);
    [LibraryImport(LibName)] public static partial void vkGetPhysicalDeviceFeatures2(IntPtr device, VkPhysicalDeviceFeatures2* features);
    [LibraryImport(LibName)] public static partial VkResult vkCreateDevice(IntPtr physicalDevice, VkDeviceCreateInfo* ci, void* alloc, out IntPtr device);
    [LibraryImport(LibName)] public static partial void vkDestroyDevice(IntPtr device, void* alloc);
    [LibraryImport(LibName)] public static partial void vkGetDeviceQueue(IntPtr device, uint family, uint index, out IntPtr queue);

    [LibraryImport(LibName)] public static partial VkResult vkCreateBuffer(IntPtr device, VkBufferCreateInfo* ci, void* alloc, out IntPtr buffer);
    [LibraryImport(LibName)] public static partial void vkDestroyBuffer(IntPtr device, IntPtr buffer, void* alloc);
    [LibraryImport(LibName)] public static partial void vkGetBufferMemoryRequirements(IntPtr device, IntPtr buffer, VkMemoryRequirements* req);
    [LibraryImport(LibName)] public static partial VkResult vkAllocateMemory(IntPtr device, VkMemoryAllocateInfo* ai, void* alloc, out IntPtr memory);
    [LibraryImport(LibName)] public static partial void vkFreeMemory(IntPtr device, IntPtr memory, void* alloc);
    [LibraryImport(LibName)] public static partial VkResult vkBindBufferMemory(IntPtr device, IntPtr buffer, IntPtr memory, ulong offset);
    [LibraryImport(LibName)] public static partial VkResult vkMapMemory(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, void** data);
    [LibraryImport(LibName)] public static partial void vkUnmapMemory(IntPtr device, IntPtr memory);
    [LibraryImport(LibName)] public static partial VkResult vkFlushMappedMemoryRanges(IntPtr device, uint count, VkMappedMemoryRange* ranges);
    [LibraryImport(LibName)] public static partial VkResult vkInvalidateMappedMemoryRanges(IntPtr device, uint count, VkMappedMemoryRange* ranges);

    [LibraryImport(LibName)] public static partial VkResult vkCreateShaderModule(IntPtr device, VkShaderModuleCreateInfo* ci, void* alloc, out IntPtr module);
    [LibraryImport(LibName)] public static partial void vkDestroyShaderModule(IntPtr device, IntPtr module, void* alloc);
    [LibraryImport(LibName)] public static partial VkResult vkCreatePipelineLayout(IntPtr device, VkPipelineLayoutCreateInfo* ci, void* alloc, out IntPtr layout);
    [LibraryImport(LibName)] public static partial VkResult vkCreateComputePipelines(IntPtr device, IntPtr cache, uint count, VkComputePipelineCreateInfo* ci, void* alloc, IntPtr* pipelines);
    [LibraryImport(LibName)] public static partial void vkDestroyPipeline(IntPtr device, IntPtr pipeline, void* alloc);

    [LibraryImport(LibName)] public static partial VkResult vkCreateDescriptorSetLayout(IntPtr device, VkDescriptorSetLayoutCreateInfo* ci, void* alloc, out IntPtr setLayout);
    [LibraryImport(LibName)] public static partial VkResult vkCreateDescriptorPool(IntPtr device, VkDescriptorPoolCreateInfo* ci, void* alloc, out IntPtr pool);
    [LibraryImport(LibName)] public static partial VkResult vkAllocateDescriptorSets(IntPtr device, VkDescriptorSetAllocateInfo* ai, IntPtr* sets);
    [LibraryImport(LibName)] public static partial void vkUpdateDescriptorSets(IntPtr device, uint writeCount, VkWriteDescriptorSet* writes, uint copyCount, void* copies);
    [LibraryImport(LibName)] public static partial VkResult vkEnumerateDeviceExtensionProperties(IntPtr physicalDevice, byte* layerName, uint* count, VkExtensionProperties* props);
    [LibraryImport(LibName)] public static partial IntPtr vkGetDeviceProcAddr(IntPtr device, byte* name);
    [LibraryImport(LibName)] public static partial IntPtr vkGetInstanceProcAddr(IntPtr instance, byte* name);

    [LibraryImport(LibName)] public static partial VkResult vkCreateCommandPool(IntPtr device, VkCommandPoolCreateInfo* ci, void* alloc, out IntPtr pool);
    [LibraryImport(LibName)] public static partial VkResult vkAllocateCommandBuffers(IntPtr device, VkCommandBufferAllocateInfo* ai, IntPtr* buffers);
    [LibraryImport(LibName)] public static partial VkResult vkResetCommandBuffer(IntPtr cmd, uint flags);
    [LibraryImport(LibName)] public static partial VkResult vkBeginCommandBuffer(IntPtr cmd, VkCommandBufferBeginInfo* bi);
    [LibraryImport(LibName)] public static partial VkResult vkEndCommandBuffer(IntPtr cmd);
    [LibraryImport(LibName)] public static partial void vkCmdBindPipeline(IntPtr cmd, uint bindPoint, IntPtr pipeline);
    [LibraryImport(LibName)] public static partial void vkCmdBindDescriptorSets(IntPtr cmd, uint bindPoint, IntPtr layout, uint firstSet, uint count, IntPtr* sets, uint dynCount, uint* dynOffsets);
    [LibraryImport(LibName)] public static partial void vkCmdPushConstants(IntPtr cmd, IntPtr layout, uint stageFlags, uint offset, uint size, void* values);
    [LibraryImport(LibName)] public static partial void vkCmdDispatch(IntPtr cmd, uint x, uint y, uint z);
    [LibraryImport(LibName)] public static partial void vkCmdPipelineBarrier(IntPtr cmd, uint srcStage, uint dstStage, uint depFlags,
        uint memBarrierCount, VkMemoryBarrier* memBarriers, uint bufBarrierCount, void* bufBarriers, uint imgBarrierCount, void* imgBarriers);
    [LibraryImport(LibName)] public static partial void vkCmdCopyBuffer(IntPtr cmd, IntPtr src, IntPtr dst, uint count, VkBufferCopy* regions);

    [StructLayout(LayoutKind.Sequential)]
    public struct VkQueryPoolCreateInfo { public uint SType; public void* PNext; public uint Flags; public uint QueryType; public uint QueryCount; public uint PipelineStatistics; }
    [LibraryImport(LibName)] public static partial VkResult vkCreateQueryPool(IntPtr device, VkQueryPoolCreateInfo* ci, void* alloc, out IntPtr pool);
    [LibraryImport(LibName)] public static partial void vkDestroyQueryPool(IntPtr device, IntPtr pool, void* alloc);
    [LibraryImport(LibName)] public static partial void vkCmdResetQueryPool(IntPtr cmd, IntPtr pool, uint first, uint count);
    [LibraryImport(LibName)] public static partial void vkCmdWriteTimestamp(IntPtr cmd, uint stage, IntPtr pool, uint query);
    [LibraryImport(LibName)] public static partial VkResult vkGetQueryPoolResults(IntPtr device, IntPtr pool, uint first, uint count, nuint dataSize, void* data, ulong stride, uint flags);

    [StructLayout(LayoutKind.Sequential)]
    public struct VkBufferCopy { public ulong SrcOffset, DstOffset, Size; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkBufferMemoryBarrier
    {
        public uint SType; public void* PNext;
        public uint SrcAccessMask, DstAccessMask;
        public uint SrcQueueFamilyIndex, DstQueueFamilyIndex;
        public IntPtr Buffer; public ulong Offset, Size;
    }

    [LibraryImport(LibName)] public static partial VkResult vkQueueSubmit(IntPtr queue, uint submitCount, VkSubmitInfo* submits, IntPtr fence);
    [LibraryImport(LibName)] public static partial VkResult vkCreateFence(IntPtr device, VkFenceCreateInfo* ci, void* alloc, out IntPtr fence);
    [LibraryImport(LibName)] public static partial VkResult vkWaitForFences(IntPtr device, uint count, IntPtr* fences, uint waitAll, ulong timeout);
    [LibraryImport(LibName)] public static partial VkResult vkResetFences(IntPtr device, uint count, IntPtr* fences);
    [LibraryImport(LibName)] public static partial VkResult vkQueueWaitIdle(IntPtr queue);
    [LibraryImport(LibName)] public static partial VkResult vkDeviceWaitIdle(IntPtr device);
}

internal enum VkResult : int
{
    Success = 0,
    NotReady = 1,
    Timeout = 2,
    ErrorOutOfHostMemory = -1,
    ErrorOutOfDeviceMemory = -2,
    ErrorInitializationFailed = -3,
    ErrorDeviceLost = -4,
    ErrorExtensionNotPresent = -7,
    ErrorFeatureNotPresent = -8,
    ErrorIncompatibleDriver = -9,
}

internal static class VkConst
{
    // VkStructureType
    public const uint StApplicationInfo = 0;
    public const uint StInstanceCreateInfo = 1;
    public const uint StDeviceQueueCreateInfo = 2;
    public const uint StDeviceCreateInfo = 3;
    public const uint StSubmitInfo = 4;
    public const uint StMemoryAllocateInfo = 5;
    public const uint StMappedMemoryRange = 6;
    public const uint StFenceCreateInfo = 8;
    public const uint StBufferCreateInfo = 12;
    public const uint StShaderModuleCreateInfo = 16;
    public const uint StPipelineShaderStageCreateInfo = 18;
    public const uint StComputePipelineCreateInfo = 29;
    public const uint StPipelineLayoutCreateInfo = 30;
    public const uint StDescriptorSetLayoutCreateInfo = 32;
    public const uint StDescriptorPoolCreateInfo = 33;
    public const uint StDescriptorSetAllocateInfo = 34;
    public const uint StWriteDescriptorSet = 35;
    public const uint StCommandPoolCreateInfo = 39;
    public const uint StCommandBufferAllocateInfo = 40;
    public const uint StCommandBufferBeginInfo = 42;
    public const uint StMemoryBarrier = 46;
    public const uint StBufferMemoryBarrier = 44;
    public const uint QueueFamilyIgnored = 0xFFFFFFFFu;
    public const uint StPhysicalDeviceFeatures2 = 49;
    public const uint StPhysicalDevice16BitStorageFeatures = 1000083000u;
    public const uint StPhysicalDeviceShaderFloat16Int8Features = 1000082000u;
    public const uint StPhysicalDeviceCooperativeMatrixFeaturesKHR = 1000506000u;
    public const uint StCooperativeMatrixPropertiesKHR = 1000506001u;
    public const uint StPhysicalDeviceSubgroupProperties = 1000094000u;
    public const uint StPhysicalDeviceSubgroupSizeControlPropertiesEXT = 1000225000u;
    public const uint StPhysicalDeviceSubgroupSizeControlFeaturesEXT = 1000225002u;
    public const uint StPhysicalDeviceProperties2 = 1000059000u;
    public const uint StPipelineShaderStageRequiredSubgroupSizeCreateInfo = 1000225001u;

    // VkPhysicalDeviceType
    public const uint PhysDeviceIntegrated = 1;
    public const uint PhysDeviceDiscrete = 2;
    public const uint PhysDeviceVirtualGpu = 3;
    public const uint PhysDeviceCpu = 4;

    // VkQueueFlagBits
    public const uint QueueGraphics = 0x1;
    public const uint QueueCompute = 0x2;
    public const uint QueueTransfer = 0x4;

    // VkBufferUsageFlagBits
    public const uint BufferUsageTransferSrc = 0x1;
    public const uint BufferUsageTransferDst = 0x2;
    public const uint BufferUsageStorageBuffer = 0x20;

    // VkMemoryPropertyFlagBits
    public const uint MemDeviceLocal = 0x1;
    public const uint MemHostVisible = 0x2;
    public const uint MemHostCoherent = 0x4;
    public const uint MemHostCached = 0x8;

    // VkDescriptorType
    public const uint DescStorageBuffer = 7;
    public const uint DescUniformBuffer = 6;

    // VkShaderStageFlagBits / VkPipelineStageFlagBits
    public const uint StageComputeShader = 0x20;
    public const uint PipelineStageTransfer = 0x1000;
    public const uint PipelineStageComputeShader = 0x800;
    public const uint PipelineStageHost = 0x4000;
    public const uint PipelineStageTopOfPipe = 0x1;
    public const uint PipelineStageBottomOfPipe = 0x2000;

    // VkAccessFlagBits
    public const uint AccessShaderRead = 0x20;
    public const uint AccessShaderWrite = 0x40;
    public const uint AccessTransferRead = 0x800;
    public const uint AccessTransferWrite = 0x1000;
    public const uint AccessHostRead = 0x2000;
    public const uint AccessHostWrite = 0x4000;
    public const uint StageTopOfPipe = 0x1;
    public const uint StageAllCommands = 0x10000;

    // VkPipelineBindPoint
    public const uint BindPointCompute = 1;

    // VkCommandBufferLevel
    public const uint CmdBufferLevelPrimary = 0;

    // VkCommandPoolCreateFlagBits
    public const uint CmdPoolResetCommandBuffer = 0x2;

    // VkSharingMode
    public const uint SharingExclusive = 0;
}
