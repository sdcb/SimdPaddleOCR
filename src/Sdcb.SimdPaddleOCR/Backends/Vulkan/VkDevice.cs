// Thin object wrappers over Vk.cs: instance/device/queue, buffers with
// memory-type selection (ReBAR fast path vs host-visible staging), shader
// modules from embedded SPIR-V, compute pipelines, descriptor sets, and
// one-shot command submission.
using System.Runtime.InteropServices;

namespace Sdcb.SimdPaddleOCR.Backends.Vulkan;

internal unsafe sealed class VkDevice : IDisposable
{
    public IntPtr Instance, PhysDevice, Device, Queue;
    public string DeviceName = "";
    public uint VendorId;
    public uint SubgroupSize;         // actual subgroup size (lanes)
    public uint SubgroupOps;          // VkSubgroupFeatureFlags (SHUFFLE = 0x10)
    public uint SubgroupStages;       // VkShaderStageFlags (COMPUTE = 0x20)
    public uint QueueFamily;
    public Vk.VkPhysicalDeviceMemoryProperties MemProps;
    public bool CoherentDeviceLocal;   // DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT exists (ReBAR)
    public bool CoopMatrix;            // VK_KHR_cooperative_matrix (feature + ext enabled)
    public bool SubgroupSizeControl;   // VK_EXT_subgroup_size_control
    public uint SubgroupMin = 1, SubgroupMax = 128;
    public int CoopM, CoopN, CoopK;    // best fp16->fp32 subgroup coopmat config
    public bool Storage16Bit;          // storageBuffer16BitAccess + shaderInt16 enabled
    public bool ShaderFloat16;         // shaderFloat16 (VK_KHR_shader_float16_int8) enabled — required by fp16 shaders
    public uint DeviceLocalHostVisibleType = uint.MaxValue;
    public uint HostVisibleCoherentType = uint.MaxValue;
    public bool PushDescriptors;
    private unsafe delegate* unmanaged[Cdecl]<IntPtr, uint, IntPtr, uint, uint, Vk.VkWriteDescriptorSet*, void> _pushDesc;
    public double TimestampPeriodNs = 1;
    private IntPtr _commandPool;
    private IntPtr _descPool;
    /// <summary>Serializes all queue submission / command-pool use across sessions.</summary>
    public readonly object Sync = new();

    public static VkDevice Create(uint deviceIndex = 0)
    {
        Vk.RegisterResolver();
        var d = new VkDevice();

        Vk.VkApplicationInfo app = new()
        {
            SType = VkConst.StApplicationInfo,
            AppVersion = 1, EngineVersion = 1, ApiVersion = (1u << 22) | (1u << 12), // Vulkan 1.1
        };
        Vk.VkInstanceCreateInfo ici = new() { SType = VkConst.StInstanceCreateInfo, PAppInfo = &app };
        Vk.Check(Vk.vkCreateInstance(&ici, null, out d.Instance), "vkCreateInstance");

        uint ndev = 0;
        Vk.Check(Vk.vkEnumeratePhysicalDevices(d.Instance, &ndev, null), "enum physical devices");
        if (ndev == 0) throw new PlatformNotSupportedException("Vulkan: no physical devices (no Vulkan-capable GPU/driver)");
        IntPtr* devs = stackalloc IntPtr[(int)ndev];
        Vk.Check(Vk.vkEnumeratePhysicalDevices(d.Instance, &ndev, devs), "enum physical devices");

        // Prefer discrete GPU; honor deviceIndex as "the nth discrete first, else nth overall".
        int chosen = -1;
        for (int pass = 0; pass < 2 && chosen < 0; pass++)
        {
            uint seen = 0;
            for (int i = 0; i < (int)ndev; i++)
            {
                Vk.VkPhysicalDeviceProperties p;
                Vk.vkGetPhysicalDeviceProperties(devs[i], &p);
                bool want = pass == 0 ? p.DeviceType == VkConst.PhysDeviceDiscrete : p.DeviceType != VkConst.PhysDeviceCpu;
                if (!want) continue;
                if (seen++ == deviceIndex) { chosen = i; break; }
            }
        }
        if (chosen < 0) throw new PlatformNotSupportedException($"Vulkan: device index {deviceIndex} out of range ({ndev} devices)");
        d.PhysDevice = devs[chosen];

        Vk.VkPhysicalDeviceProperties props;
        Vk.vkGetPhysicalDeviceProperties(d.PhysDevice, &props);
        d.DeviceName = Marshal.PtrToStringAnsi((IntPtr)props.DeviceName) ?? "?";
        d.VendorId = props.VendorID;
        d.TimestampPeriodNs = props.TimestampPeriodNs;

        uint nqf = 0;
        Vk.vkGetPhysicalDeviceQueueFamilyProperties(d.PhysDevice, &nqf, null);
        Vk.VkQueueFamilyProperties* qf = stackalloc Vk.VkQueueFamilyProperties[(int)nqf];
        Vk.vkGetPhysicalDeviceQueueFamilyProperties(d.PhysDevice, &nqf, qf);
        d.QueueFamily = uint.MaxValue;
        for (uint i = 0; i < nqf; i++)
            if ((qf[i].QueueFlags & VkConst.QueueCompute) != 0) { d.QueueFamily = i; break; }
        if (d.QueueFamily == uint.MaxValue)
            throw new PlatformNotSupportedException("Vulkan: no compute queue family");

        // Feature probe: 16-bit storage (u16 buffer views for non-4B-aligned blocks like Q6_K) + shaderInt16.
        Vk.VkPhysicalDevice16BitStorageFeatures s16query = new() { SType = VkConst.StPhysicalDevice16BitStorageFeatures };
        Vk.VkPhysicalDeviceShaderFloat16Int8Features f16query = new() { SType = VkConst.StPhysicalDeviceShaderFloat16Int8Features };
        s16query.PNext = &f16query;
        Vk.VkPhysicalDeviceFeatures2 f2 = new() { SType = VkConst.StPhysicalDeviceFeatures2, PNext = &s16query };
        Vk.vkGetPhysicalDeviceFeatures2(d.PhysDevice, &f2);
        Vk.VkPhysicalDeviceFeatures coreF;
        Vk.vkGetPhysicalDeviceFeatures(d.PhysDevice, &coreF);
        d.Storage16Bit = s16query.StorageBuffer16BitAccess != 0 && coreF.ShaderInt16 != 0;
        d.ShaderFloat16 = f16query.ShaderFloat16 != 0;
        if (Environment.GetEnvironmentVariable("HYMT_VK_DEBUG") == "1")
            Console.Error.WriteLine($"vk-dbg device={d.DeviceName} api={props.ApiVersion:x8} s16={s16query.StorageBuffer16BitAccess} int16(core)={coreF.ShaderInt16} int16(f2)={f2.Features.ShaderInt16} f36core={coreF.F[36]}");

        // Extension probe: VK_KHR_push_descriptor (llama.cpp-style inline descriptor writes,
        // skips per-dispatch descriptor-set binds).
        uint next = 0;
        Vk.vkEnumerateDeviceExtensionProperties(d.PhysDevice, null, &next, null);
        Vk.VkExtensionProperties* exts = stackalloc Vk.VkExtensionProperties[(int)next];
        Vk.vkEnumerateDeviceExtensionProperties(d.PhysDevice, null, &next, exts);
        byte* wantPush = stackalloc byte[] { (byte)'V', (byte)'K', (byte)'_', (byte)'K', (byte)'H', (byte)'R',
            (byte)'_', (byte)'p', (byte)'u', (byte)'s', (byte)'h', (byte)'_', (byte)'d', (byte)'e', (byte)'s', (byte)'c',
            (byte)'r', (byte)'i', (byte)'p', (byte)'t', (byte)'o', (byte)'r', 0 };
        bool hasPush = false, hasCoop = false, hasSgc = false, hasF16Int8 = false;
        for (int i = 0; i < (int)next; i++)
        {
            string en = new((sbyte*)exts[i].ExtensionName);
            if (en == "VK_KHR_push_descriptor") hasPush = true;
            else if (en == "VK_KHR_cooperative_matrix") hasCoop = true;
            else if (en == "VK_EXT_subgroup_size_control") hasSgc = true;
            else if (en == "VK_KHR_shader_float16_int8") hasF16Int8 = true;
        }

        Vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopQ = new() { SType = VkConst.StPhysicalDeviceCooperativeMatrixFeaturesKHR };
        if (hasCoop)
        {
            Vk.VkPhysicalDeviceFeatures2 fq = new() { SType = VkConst.StPhysicalDeviceFeatures2, PNext = &coopQ };
            Vk.vkGetPhysicalDeviceFeatures2(d.PhysDevice, &fq);
            d.CoopMatrix = coopQ.CooperativeMatrix != 0;
        }

        float prio = 1f;
        Vk.VkDeviceQueueCreateInfo qci = new()
        {
            SType = VkConst.StDeviceQueueCreateInfo,
            QueueFamilyIndex = d.QueueFamily, QueueCount = 1, PQueuePriorities = &prio,
        };
        Vk.VkPhysicalDevice16BitStorageFeatures s16en = new()
        {
            SType = VkConst.StPhysicalDevice16BitStorageFeatures,
            StorageBuffer16BitAccess = d.Storage16Bit ? 1u : 0u,
        };
        Vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopEn = new()
        {
            SType = VkConst.StPhysicalDeviceCooperativeMatrixFeaturesKHR,
            CooperativeMatrix = d.CoopMatrix ? 1u : 0u,
        };
        {
            Vk.VkPhysicalDeviceSubgroupProperties sgP = new() { SType = VkConst.StPhysicalDeviceSubgroupProperties };
            Vk.VkPhysicalDeviceSubgroupSizeControlProperties sgcP = new() { SType = VkConst.StPhysicalDeviceSubgroupSizeControlPropertiesEXT };
            if (hasSgc) sgP.PNext = &sgcP;
            Vk.VkPhysicalDeviceProperties2 p2 = new() { SType = VkConst.StPhysicalDeviceProperties2, PNext = &sgP };
            Vk.vkGetPhysicalDeviceProperties2(d.PhysDevice, &p2);
            d.SubgroupSize = sgP.SubgroupSize;
            d.SubgroupOps = sgP.SupportedOperations;
            d.SubgroupStages = sgP.SupportedStages;
            if (hasSgc) { d.SubgroupMin = sgcP.MinSubgroupSize; d.SubgroupMax = sgcP.MaxSubgroupSize; }
        }

        Vk.VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgcEn = new()
        {
            SType = VkConst.StPhysicalDeviceSubgroupSizeControlFeaturesEXT,
            SubgroupSizeControl = hasSgc ? 1u : 0u,
        };
        d.SubgroupSizeControl = hasSgc;
        Vk.VkPhysicalDeviceShaderFloat16Int8Features f16en = new()
        {
            SType = VkConst.StPhysicalDeviceShaderFloat16Int8Features,
            ShaderFloat16 = d.ShaderFloat16 ? 1u : 0u,
        };
        if (Environment.GetEnvironmentVariable("HYMT_VK_DEBUG") == "1")
            Console.Error.WriteLine($"vk-dbg coop={d.CoopMatrix} sgc={hasSgc} sgMin={d.SubgroupMin} sgMax={d.SubgroupMax} push={hasPush} f16={d.ShaderFloat16}");
        f16en.PNext = d.CoopMatrix ? &coopEn : hasSgc ? &sgcEn : null;
        s16en.PNext = hasF16Int8 && d.ShaderFloat16 ? &f16en : f16en.PNext;
        coopEn.PNext = hasSgc ? &sgcEn : null;
        Vk.VkPhysicalDeviceFeatures feats = new();
        if (d.Storage16Bit) feats.ShaderInt16 = 1;
        byte* wantCoop = stackalloc byte[] { (byte)'V', (byte)'K', (byte)'_', (byte)'K', (byte)'H', (byte)'R',
            (byte)'_', (byte)'c', (byte)'o', (byte)'o', (byte)'p', (byte)'e', (byte)'r', (byte)'a', (byte)'t',
            (byte)'i', (byte)'v', (byte)'e', (byte)'_', (byte)'m', (byte)'a', (byte)'t', (byte)'r', (byte)'i',
            (byte)'x', 0 };
        byte* wantSgc = stackalloc byte[] { (byte)'V', (byte)'K', (byte)'_', (byte)'E', (byte)'X', (byte)'T',
            (byte)'_', (byte)'s', (byte)'u', (byte)'b', (byte)'g', (byte)'r', (byte)'o', (byte)'u', (byte)'p',
            (byte)'_', (byte)'s', (byte)'i', (byte)'z', (byte)'e', (byte)'_', (byte)'c', (byte)'o', (byte)'n',
            (byte)'t', (byte)'r', (byte)'o', (byte)'l', 0 };
        byte* wantF16 = stackalloc byte[] { (byte)'V', (byte)'K', (byte)'_', (byte)'K', (byte)'H', (byte)'R',
            (byte)'_', (byte)'s', (byte)'h', (byte)'a', (byte)'d', (byte)'e', (byte)'r', (byte)'_', (byte)'f',
            (byte)'l', (byte)'o', (byte)'a', (byte)'t', (byte)'1', (byte)'6', (byte)'_', (byte)'i', (byte)'n',
            (byte)'t', (byte)'8', 0 };
        byte** extsToEnable = stackalloc byte*[4];
        uint nExt = 0;
        if (hasPush) extsToEnable[nExt++] = wantPush;
        if (d.CoopMatrix) extsToEnable[nExt++] = wantCoop;
        if (hasSgc) extsToEnable[nExt++] = wantSgc;
        if (hasF16Int8 && d.ShaderFloat16) extsToEnable[nExt++] = wantF16;
        Vk.VkDeviceCreateInfo dci = new()
        {
            SType = VkConst.StDeviceCreateInfo,
            QueueCreateInfoCount = 1, PQueueCreateInfos = &qci,
            PEnabledFeatures = &feats,
            PNext = &s16en,
            EnabledExtensionCount = nExt,
            PpEnabledExtensionNames = nExt > 0 ? extsToEnable : null,
        };
        if (Environment.GetEnvironmentVariable("HYMT_VK_VERBOSE") == "1")
            Console.Error.WriteLine($"[vk] devext push={hasPush} coop={d.CoopMatrix} sgc={hasSgc} sgRange={d.SubgroupMin}-{d.SubgroupMax}");
        Vk.Check(Vk.vkCreateDevice(d.PhysDevice, &dci, null, out d.Device), "vkCreateDevice");
        Vk.vkGetDeviceQueue(d.Device, d.QueueFamily, 0, out d.Queue);
        if (d.CoopMatrix)
        {
            byte* fnCm = stackalloc byte[] { (byte)'v', (byte)'k', (byte)'G', (byte)'e', (byte)'t',
                (byte)'P', (byte)'h', (byte)'y', (byte)'s', (byte)'i', (byte)'c', (byte)'a', (byte)'l',
                (byte)'D', (byte)'e', (byte)'v', (byte)'i', (byte)'c', (byte)'e', (byte)'C', (byte)'o',
                (byte)'o', (byte)'p', (byte)'e', (byte)'r', (byte)'a', (byte)'t', (byte)'i', (byte)'v',
                (byte)'e', (byte)'M', (byte)'a', (byte)'t', (byte)'r', (byte)'i', (byte)'x', (byte)'P',
                (byte)'r', (byte)'o', (byte)'p', (byte)'e', (byte)'r', (byte)'t', (byte)'i', (byte)'e',
                (byte)'s', (byte)'K', (byte)'H', (byte)'R', 0 };
            IntPtr fncm = Vk.vkGetInstanceProcAddr(d.Instance, fnCm);
            if (fncm != IntPtr.Zero)
            {
                var qprops = (delegate* unmanaged[Cdecl]<IntPtr, uint*, Vk.VkCooperativeMatrixPropertiesKHR*, VkResult>)fncm;
                uint np = 0;
                qprops(d.PhysDevice, &np, null);
                var cmprops = stackalloc Vk.VkCooperativeMatrixPropertiesKHR[(int)np];
                for (int i = 0; i < (int)np; i++) cmprops[i].SType = VkConst.StCooperativeMatrixPropertiesKHR;
                qprops(d.PhysDevice, &np, cmprops);
                for (int i = 0; i < (int)np; i++)
                {
                    // AType/BType 0 = float16 KHR, CType/ResultType 1 = float32, scope 3 = subgroup
                    if (cmprops[i].AType == 0 && cmprops[i].BType == 0 && cmprops[i].ResultType == 1 && cmprops[i].Scope == 3
                        && cmprops[i].MSize > (uint)d.CoopM)
                    {
                        d.CoopM = (int)cmprops[i].MSize; d.CoopN = (int)cmprops[i].NSize; d.CoopK = (int)cmprops[i].KSize;
                    }
                    if (Environment.GetEnvironmentVariable("HYMT_VK_VERBOSE") == "1")
                        Console.Error.WriteLine($"[vk] coopmat {cmprops[i].MSize}x{cmprops[i].NSize}x{cmprops[i].KSize} at={cmprops[i].AType} bt={cmprops[i].BType} ct={cmprops[i].CType} rt={cmprops[i].ResultType} scope={cmprops[i].Scope}");
                }
            }
        }
        if (hasPush)
        {
            byte* fnName = stackalloc byte[] { (byte)'v', (byte)'k', (byte)'C', (byte)'m', (byte)'d',
                (byte)'P', (byte)'u', (byte)'s', (byte)'h', (byte)'D', (byte)'e', (byte)'s', (byte)'c',
                (byte)'r', (byte)'i', (byte)'p', (byte)'t', (byte)'o', (byte)'r', (byte)'S', (byte)'e',
                (byte)'t', (byte)'K', (byte)'H', (byte)'R', 0 };
            IntPtr fn = Vk.vkGetDeviceProcAddr(d.Device, fnName);
            if (Environment.GetEnvironmentVariable("HYMT_VK_VERBOSE") == "1")
                Console.Error.WriteLine($"[vk] push_descriptor ext={hasPush} fn=0x{fn:X}");
            if (fn != IntPtr.Zero)
            {
                d.PushDescriptors = true;
                d._pushDesc = (delegate* unmanaged[Cdecl]<IntPtr, uint, IntPtr, uint, uint, Vk.VkWriteDescriptorSet*, void>)fn;
            }
        }

        fixed (Vk.VkPhysicalDeviceMemoryProperties* mp = &d.MemProps)
            Vk.vkGetPhysicalDeviceMemoryProperties(d.PhysDevice, mp);
        for (int i = 0; i < (int)d.MemProps.MemoryTypeCount; i++)
        {
            uint f = d.MemProps.TypeAt(i).PropertyFlags;
            if ((f & (VkConst.MemDeviceLocal | VkConst.MemHostVisible | VkConst.MemHostCoherent)) == (VkConst.MemDeviceLocal | VkConst.MemHostVisible | VkConst.MemHostCoherent)
                && d.DeviceLocalHostVisibleType == uint.MaxValue)
                d.DeviceLocalHostVisibleType = (uint)i;
            if ((f & (VkConst.MemHostVisible | VkConst.MemHostCoherent)) == (VkConst.MemHostVisible | VkConst.MemHostCoherent)
                && d.HostVisibleCoherentType == uint.MaxValue)
                d.HostVisibleCoherentType = (uint)i;
        }
        d.CoherentDeviceLocal = d.DeviceLocalHostVisibleType != uint.MaxValue;

        Vk.VkCommandPoolCreateInfo cpci = new()
        {
            SType = VkConst.StCommandPoolCreateInfo,
            Flags = VkConst.CmdPoolResetCommandBuffer,
            QueueFamilyIndex = d.QueueFamily,
        };
        Vk.Check(Vk.vkCreateCommandPool(d.Device, &cpci, null, out d._commandPool), "vkCreateCommandPool");

        Vk.VkDescriptorPoolSize psz = new() { Type = VkConst.DescStorageBuffer, DescriptorCount = 1 << 15 };
        Vk.VkDescriptorPoolCreateInfo dpci = new()
        {
            SType = VkConst.StDescriptorPoolCreateInfo,
            // FREE_DESCRIPTOR_SET_BIT: the GPU graph evicts cached plans
            // (variable-shape runs) and releases their descriptor sets.
            Flags = 1, MaxSets = 4096, PoolSizeCount = 1, PPoolSizes = &psz,
        };
        Vk.Check(Vk.vkCreateDescriptorPool(d.Device, &dpci, null, out d._descPool), "vkCreateDescriptorPool");
        return d;
    }

    /// <summary>Allocate a storage buffer. hostVisible=false prefers DEVICE_LOCAL;
    /// hostVisible=true prefers DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT (ReBAR zero-copy), falling
    /// back to HOST_VISIBLE|HOST_COHERENT. preferHost skips DEVICE_LOCAL types — for buffers the
    /// CPU reads (uncached WC reads from VRAM are ~15MB/s, catastrophic).</summary>
    public VkBuffer NewStorageBuffer(ulong bytes, bool hostVisible, bool preferHost = false)
    {
        var b = new VkBuffer { Dev = Device, Size = bytes };
        Vk.VkBufferCreateInfo bci = new()
        {
            SType = VkConst.StBufferCreateInfo,
            Size = bytes,
            Usage = VkConst.BufferUsageStorageBuffer | VkConst.BufferUsageTransferSrc | VkConst.BufferUsageTransferDst,
            SharingMode = VkConst.SharingExclusive,
        };
        Vk.Check(Vk.vkCreateBuffer(Device, &bci, null, out b.Buffer), "vkCreateBuffer");
        Vk.VkMemoryRequirements req;
        Vk.vkGetBufferMemoryRequirements(Device, b.Buffer, &req);

        uint memType = uint.MaxValue;
        for (int i = 0; i < (int)MemProps.MemoryTypeCount; i++)
        {
            if ((req.MemoryTypeBits & (1u << i)) == 0) continue;
            uint f = MemProps.TypeAt(i).PropertyFlags;
            bool ok;
            if (!hostVisible)
                ok = (f & VkConst.MemDeviceLocal) != 0;
            else if (preferHost)
                ok = (f & (VkConst.MemHostVisible | VkConst.MemHostCoherent)) == (VkConst.MemHostVisible | VkConst.MemHostCoherent)
                     && (f & VkConst.MemDeviceLocal) == 0
                     && (f & VkConst.MemHostCached) != 0;
            else
                ok = (f & VkConst.MemHostVisible) != 0 && (f & VkConst.MemHostCoherent) != 0;
            if (ok && (memType == uint.MaxValue || (!preferHost && (f & VkConst.MemDeviceLocal) != 0)))
            {
                memType = (uint)i;
                if (preferHost || (f & VkConst.MemDeviceLocal) != 0) break;
            }
        }
        if (memType == uint.MaxValue && preferHost)
        {
            // relax: some drivers expose host-visible non-local types without CACHED
            for (int i = 0; i < (int)MemProps.MemoryTypeCount && memType == uint.MaxValue; i++)
            {
                if ((req.MemoryTypeBits & (1u << i)) == 0) continue;
                uint f = MemProps.TypeAt(i).PropertyFlags;
                if ((f & (VkConst.MemHostVisible | VkConst.MemHostCoherent)) == (VkConst.MemHostVisible | VkConst.MemHostCoherent)
                    && (f & VkConst.MemDeviceLocal) == 0)
                    memType = (uint)i;
            }
        }
        if (memType == uint.MaxValue)
            throw new PlatformNotSupportedException($"Vulkan: no suitable memory type for {bytes}-byte buffer (hostVisible={hostVisible})");

        Vk.VkMemoryAllocateInfo mai = new()
        {
            SType = VkConst.StMemoryAllocateInfo,
            AllocationSize = req.Size, MemoryTypeIndex = memType,
        };
        Vk.Check(Vk.vkAllocateMemory(Device, &mai, null, out b.Memory), "vkAllocateMemory");
        Vk.Check(Vk.vkBindBufferMemory(Device, b.Buffer, b.Memory, 0), "vkBindBufferMemory");
        b.Flags = MemProps.TypeAt((int)memType).PropertyFlags;
        return b;
    }

    public IntPtr NewShaderModule(byte[] spirv)
    {
        if ((spirv.Length & 3) != 0) throw new ArgumentException("SPIR-V size must be a multiple of 4");
        fixed (byte* p = spirv)
        {
            Vk.VkShaderModuleCreateInfo ci = new()
            {
                SType = VkConst.StShaderModuleCreateInfo,
                CodeSize = (nuint)spirv.Length, PCode = (uint*)p,
            };
            Vk.Check(Vk.vkCreateShaderModule(Device, &ci, null, out IntPtr m), "vkCreateShaderModule");
            return m;
        }
    }

    /// <summary>One set layout with `bindings` storage buffers (0..bindings-1) + push constants.</summary>
    public VkPipeline NewPipeline(IntPtr shaderModule, int bindings, int pushConstBytes, uint requiredSubgroupSize = 0)
    {
        var dslb = stackalloc Vk.VkDescriptorSetLayoutBinding[bindings];
        for (int i = 0; i < bindings; i++)
            dslb[i] = new Vk.VkDescriptorSetLayoutBinding
            {
                Binding = (uint)i, DescriptorType = VkConst.DescStorageBuffer,
                DescriptorCount = 1, StageFlags = VkConst.StageComputeShader,
            };
        IntPtr setLayout;
        Vk.VkDescriptorSetLayoutCreateInfo dslci = new()
        {
            SType = VkConst.StDescriptorSetLayoutCreateInfo,
            BindingCount = (uint)bindings, PBindings = dslb,
            // VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR — required when the
            // layout is written via vkCmdPushDescriptorSetKHR.
            Flags = PushDescriptors ? 1u : 0u,
        };
        Vk.Check(Vk.vkCreateDescriptorSetLayout(Device, &dslci, null, out setLayout), "vkCreateDescriptorSetLayout");

        Vk.VkPushConstantRange pcr = new()
        {
            StageFlags = VkConst.StageComputeShader, Offset = 0, Size = (uint)pushConstBytes,
        };
        IntPtr layout;
        Vk.VkPipelineLayoutCreateInfo plci = new()
        {
            SType = VkConst.StPipelineLayoutCreateInfo,
            SetLayoutCount = 1, PSetLayouts = &setLayout,
            PushConstantRangeCount = 1, PPushConstantRanges = &pcr,
        };
        Vk.Check(Vk.vkCreatePipelineLayout(Device, &plci, null, out layout), "vkCreatePipelineLayout");

        byte* name = stackalloc byte[5] { (byte)'m', (byte)'a', (byte)'i', (byte)'n', 0 };
        Vk.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sgc = new()
        {
            SType = VkConst.StPipelineShaderStageRequiredSubgroupSizeCreateInfo,
            RequiredSubgroupSize = requiredSubgroupSize,
        };
        Vk.VkComputePipelineCreateInfo cpci = new()
        {
            SType = VkConst.StComputePipelineCreateInfo,
            Stage = new Vk.VkPipelineShaderStageCreateInfo
            {
                SType = VkConst.StPipelineShaderStageCreateInfo,
                Stage = VkConst.StageComputeShader, Module = shaderModule, PName = name,
                PNext = requiredSubgroupSize != 0 && SubgroupSizeControl ? &sgc : null,
            },
            Layout = layout,
        };
        IntPtr pipeline;
        Vk.Check(Vk.vkCreateComputePipelines(Device, IntPtr.Zero, 1, &cpci, null, &pipeline), "vkCreateComputePipelines");
        return new VkPipeline { Pipeline = pipeline, Layout = layout, SetLayout = setLayout };
    }

    public IntPtr NewDescriptorSet(IntPtr setLayout)
    {
        IntPtr set;
        Vk.VkDescriptorSetAllocateInfo ai = new()
        {
            SType = VkConst.StDescriptorSetAllocateInfo,
            DescriptorPool = _descPool, DescriptorSetCount = 1, PSetLayouts = &setLayout,
        };
        Vk.Check(Vk.vkAllocateDescriptorSets(Device, &ai, &set), "vkAllocateDescriptorSets");
        return set;
    }

    public void FreeDescriptorSet(IntPtr set)
    {
        if (set != IntPtr.Zero)
            Vk.Check(Vk.vkFreeDescriptorSets(Device, _descPool, 1, &set), "vkFreeDescriptorSets");
    }

    public void FreeCommandBuffer(IntPtr cmd)
    {
        if (cmd != IntPtr.Zero)
            Vk.vkFreeCommandBuffers(Device, _commandPool, 1, &cmd);
    }

    /// <summary>VK_KHR_push_descriptor: write bindings inline into the command buffer
    /// (DstSet in each write is ignored).</summary>
    public void CmdPushDescriptors(IntPtr cmd, IntPtr layout, Vk.VkWriteDescriptorSet* writes, uint count)
        => _pushDesc(cmd, VkConst.BindPointCompute, layout, 0, count, writes);

    /// <summary>Record the standard write->read barrier between two compute dispatches.</summary>
    public void CmdComputeBarrier(IntPtr cmd)
    {
        Vk.VkMemoryBarrier mb = new()
        {
            SType = VkConst.StMemoryBarrier,
            SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessShaderRead,
        };
        Vk.vkCmdPipelineBarrier(cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageComputeShader, 0,
            1, &mb, 0, null, 0, null);
    }

    /// <summary>write->read barrier scoped to one buffer.</summary>
    public void CmdBufferBarrier(IntPtr cmd, VkBuffer buf)
    {
        Vk.VkBufferMemoryBarrier bb = new()
        {
            SType = VkConst.StBufferMemoryBarrier,
            SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessShaderRead,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored, DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = buf.Buffer, Offset = 0, Size = buf.Size,
        };
        Vk.vkCmdPipelineBarrier(cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageComputeShader, 0,
            0, null, 1, &bb, 0, null);
    }

    /// <summary>write->read barrier over several buffers in a single call.</summary>
    public void CmdBufferBarrier(IntPtr cmd, params VkBuffer[] bufs)
    {
        if (bufs.Length == 1) { CmdBufferBarrier(cmd, bufs[0]); return; }
        Vk.VkBufferMemoryBarrier* bb = stackalloc Vk.VkBufferMemoryBarrier[bufs.Length];
        for (int i = 0; i < bufs.Length; i++)
            bb[i] = new Vk.VkBufferMemoryBarrier
            {
                SType = VkConst.StBufferMemoryBarrier,
                SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessShaderRead,
                SrcQueueFamilyIndex = Vk.QueueFamilyIgnored, DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                Buffer = bufs[i].Buffer, Offset = 0, Size = bufs[i].Size,
            };
        Vk.vkCmdPipelineBarrier(cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageComputeShader, 0,
            0, null, (uint)bufs.Length, bb, 0, null);
    }

    /// <summary>Upload raw bytes into a buffer (host memcpy when mappable, else staging + copy cmd).</summary>
    public void Upload(VkBuffer buf, void* src, ulong bytes)
    {
        if ((buf.Flags & VkConst.MemHostVisible) != 0)
        {
            void* p = buf.Map();
            Buffer.MemoryCopy(src, p, bytes, bytes);
            buf.Flush(0, bytes);
            buf.Unmap();
            return;
        }
        VkBuffer staging = NewStorageBuffer(bytes, hostVisible: true);
        void* sp = staging.Map();
        Buffer.MemoryCopy(src, sp, bytes, bytes);
        staging.Flush(0, bytes);
        staging.Unmap();
        IntPtr cmd = NewCommandBuffer();
        IntPtr fence = NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        Vk.VkBufferCopy r = new() { Size = bytes };
        Vk.vkCmdCopyBuffer(cmd, staging.Buffer, buf.Buffer, 1, &r);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        Submit(cmd, fence);
        WaitFence(fence);
    }

    public void BindBuffer(IntPtr set, uint binding, VkBuffer buf, ulong offset = 0)
    {
        Vk.VkDescriptorBufferInfo bi = new() { Buffer = buf.Buffer, Offset = offset, Range = buf.Size - offset };
        Vk.VkWriteDescriptorSet w = new()
        {
            SType = VkConst.StWriteDescriptorSet, DstSet = set, DstBinding = binding,
            DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer, PBufferInfo = &bi,
        };
        Vk.vkUpdateDescriptorSets(Device, 1, &w, 0, null);
    }

    public IntPtr NewCommandBuffer()
    {
        Vk.VkCommandBufferAllocateInfo ai = new()
        {
            SType = VkConst.StCommandBufferAllocateInfo,
            CommandPool = _commandPool, Level = VkConst.CmdBufferLevelPrimary, CommandBufferCount = 1,
        };
        IntPtr cmd;
        Vk.Check(Vk.vkAllocateCommandBuffers(Device, &ai, &cmd), "vkAllocateCommandBuffers");
        return cmd;
    }

    public IntPtr NewFence(bool signaled = false)
    {
        Vk.VkFenceCreateInfo ci = new() { SType = VkConst.StFenceCreateInfo, Flags = signaled ? 1u : 0u };
        Vk.Check(Vk.vkCreateFence(Device, &ci, null, out IntPtr f), "vkCreateFence");
        return f;
    }

    public void Submit(IntPtr cmd, IntPtr fence)
    {
        Vk.VkSubmitInfo si = new()
        {
            SType = VkConst.StSubmitInfo,
            CommandBufferCount = 1, PCommandBuffers = &cmd,
        };
        Vk.Check(Vk.vkQueueSubmit(Queue, 1, &si, fence), "vkQueueSubmit");
    }

    public void WaitFence(IntPtr fence)
    {
        Vk.Check(Vk.vkWaitForFences(Device, 1, &fence, 1, ulong.MaxValue), "vkWaitForFences");
        // auto-reset: a fence must be unsignaled before reuse in vkQueueSubmit
        Vk.Check(Vk.vkResetFences(Device, 1, &fence), "vkResetFences");
    }

    public void Dispose()
    {
        if (Device != IntPtr.Zero) { Vk.vkDeviceWaitIdle(Device); Vk.vkDestroyDevice(Device, null); Device = IntPtr.Zero; }
        if (Instance != IntPtr.Zero) { Vk.vkDestroyInstance(Instance, null); Instance = IntPtr.Zero; }
    }
}

internal sealed class VkPipeline
{
    public IntPtr Pipeline, Layout, SetLayout;
    public string Name = "";
}

internal unsafe sealed class VkBuffer
{
    public IntPtr Dev, Buffer, Memory;
    public ulong Size;
    public uint Flags;

    public bool HostCoherent => (Flags & VkConst.MemHostCoherent) != 0;

    public void* Map()
    {
        void* p;
        Vk.Check(Vk.vkMapMemory(Dev, Memory, 0, Size, 0, &p), "vkMapMemory");
        return p;
    }
    public void Unmap() => Vk.vkUnmapMemory(Dev, Memory);

    public void Flush(ulong offset, ulong size)
    {
        if (HostCoherent) return;
        Vk.VkMappedMemoryRange r = new() { SType = VkConst.StMappedMemoryRange, Memory = Memory, Offset = offset, Size = size };
        Vk.vkFlushMappedMemoryRanges(Dev, 1, &r);
    }
    public void Invalidate(ulong offset, ulong size)
    {
        if (HostCoherent) return;
        Vk.VkMappedMemoryRange r = new() { SType = VkConst.StMappedMemoryRange, Memory = Memory, Offset = offset, Size = size };
        Vk.vkInvalidateMappedMemoryRanges(Dev, 1, &r);
    }

    public void Free()
    {
        if (Buffer != IntPtr.Zero) { Vk.vkDestroyBuffer(Dev, Buffer, null); Buffer = IntPtr.Zero; }
        if (Memory != IntPtr.Zero) { Vk.vkFreeMemory(Dev, Memory, null); Memory = IntPtr.Zero; }
    }
}
