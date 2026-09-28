using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.GpuBench;

/// <summary>Device caps and single-dispatch timing.</summary>
static class DeviceBench
{
    internal static int Run(string[] args)
    {
        // --sgtest [sgsize.spv]: device caps + actual subgroup size under requiredSubgroupSize
        if (args.Length >= 1 && args[0] == "--sgtest")
        {
            using var sdev = VkDevice.Create();
            unsafe
            {
                Vk.VkPhysicalDeviceSubgroupSizeControlProperties sgcP = new() { SType = VkConst.StPhysicalDeviceSubgroupSizeControlPropertiesEXT };
                Vk.VkPhysicalDeviceSubgroupProperties sgP = new() { SType = VkConst.StPhysicalDeviceSubgroupProperties, PNext = &sgcP };
                Vk.VkPhysicalDeviceProperties2 p2 = new() { SType = VkConst.StPhysicalDeviceProperties2, PNext = &sgP };
                Vk.vkGetPhysicalDeviceProperties2(sdev.PhysDevice, &p2);
                byte* lim = p2.Properties.LimitsAndSparse + 4; // native VkPhysicalDeviceLimits is 8-aligned
                Console.WriteLine($"device={sdev.DeviceName} vendor=0x{sdev.VendorId:x} sgDefault={sgP.SubgroupSize} sgStages=0x{sgP.SupportedStages:x} sgOps=0x{sgP.SupportedOperations:x}");
                Console.WriteLine($"sgc min={sgcP.MinSubgroupSize} max={sgcP.MaxSubgroupSize} maxWgSubgroups={sgcP.MaxComputeWorkgroupSubgroups} requiredStages=0x{sgcP.RequiredSubgroupSizeStages:x}");
                Console.WriteLine($"maxComputeSharedMemorySize={*(uint*)(lim + 216)} maxWgInvocations={*(uint*)(lim + 232)} tsPeriodLim={*(float*)(lim + 424)} tsPeriod={sdev.TimestampPeriodNs}ns");
                for (int i = 0; i < (int)sdev.MemProps.MemoryHeapCount; i++)
                    Console.WriteLine($"heap {i}: {sdev.MemProps.HeapAt(i).Size / (1 << 20)} MB flags=0x{sdev.MemProps.HeapAt(i).Flags:x}");
                for (int i = 0; i < (int)sdev.MemProps.MemoryTypeCount; i++)
                    Console.WriteLine($"type {i}: flags=0x{sdev.MemProps.TypeAt(i).PropertyFlags:x} heap={sdev.MemProps.TypeAt(i).HeapIndex}");
                Console.WriteLine($"DeviceLocalHostVisibleType={sdev.DeviceLocalHostVisibleType} HostVisibleCoherentType={sdev.HostVisibleCoherentType}");
            }
            byte[] spv = File.ReadAllBytes(args.Length >= 2 ? args[1]
                : Path.Combine(AppContext.BaseDirectory, "Shaders", "sgsize.spv"));
            foreach (uint req in new uint[] { 0, 32, 64 })
            {
                var sp = sdev.NewPipeline(sdev.NewShaderModule(spv), 1, 4, req);
                var sset = sdev.NewDescriptorSet(sp.SetLayout);
                const int groups = 64;
                var ob = sdev.NewStorageBuffer(groups * 16, hostVisible: true, preferHost: true);
                sdev.BindBuffer(sset, 0, ob);
                IntPtr scmd = sdev.NewCommandBuffer();
                IntPtr sf = sdev.NewFence();
                unsafe
                {
                    var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
                    Vk.Check(Vk.vkBeginCommandBuffer(scmd, &begin), "begin");
                    Vk.vkCmdBindPipeline(scmd, VkConst.BindPointCompute, sp.Pipeline);
                    Vk.vkCmdBindDescriptorSets(scmd, VkConst.BindPointCompute, sp.Layout, 0, 1, &sset, 0, null);
                    Vk.vkCmdDispatch(scmd, groups, 1, 1);
                    Vk.Check(Vk.vkEndCommandBuffer(scmd), "end");
                    sdev.Submit(scmd, sf); sdev.WaitFence(sf);
                    uint* o = (uint*)ob.Map();
                    var sizes = new HashSet<string>();
                    for (int g = 0; g < groups; g++) sizes.Add($"sg={o[g * 4]} n={o[g * 4 + 1]} lastId={o[g * 4 + 2]} lastLane={o[g * 4 + 3]}");
                    ob.Unmap();
                    Console.WriteLine($"requiredSubgroupSize={req}: {string.Join(" | ", sizes)}");
                }
            }
            return 0;
        }

        // --rawbench <spv> <bindings> <gx> <gy> <reps> <bufMB> [pc uints...]: min GPU time of one dispatch
        if (args.Length >= 7 && args[0] == "--rawbench")
        {
            using var rdev = VkDevice.Create();
            int nbind = int.Parse(args[2]);
            uint rgx = uint.Parse(args[3]), rgy = uint.Parse(args[4]);
            int rreps = int.Parse(args[5]);
            ulong bytes = ulong.Parse(args[6]) << 20;
            uint[] pcs = args.Skip(7).Select(uint.Parse).ToArray();
            int pcBytes = Math.Max(16, pcs.Length * 4);
            var rp = rdev.NewPipeline(rdev.NewShaderModule(File.ReadAllBytes(args[1])), nbind, pcBytes, 32);
            var rset = rdev.NewDescriptorSet(rp.SetLayout);
            for (int b = 0; b < nbind; b++)
            {
                var buf = rdev.NewStorageBuffer(bytes, hostVisible: false);
                rdev.Zero(buf);
                rdev.BindBuffer(rset, (uint)b, buf);
            }
            IntPtr rcmd = rdev.NewCommandBuffer();
            IntPtr rf = rdev.NewFence();
            unsafe
            {
                var qci = new Vk.VkQueryPoolCreateInfo { SType = 11, QueryType = 2, QueryCount = 2 };
                Vk.Check(Vk.vkCreateQueryPool(rdev.Device, &qci, null, out IntPtr qp), "qp");
                uint[] pcv = new uint[pcBytes / 4];
                pcs.CopyTo(pcv, 0);
                double best = double.MaxValue;
                for (int r = 0; r < rreps; r++)
                {
                    var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
                    Vk.Check(Vk.vkResetCommandBuffer(rcmd, 0), "reset");
                    Vk.Check(Vk.vkBeginCommandBuffer(rcmd, &begin), "begin");
                    Vk.vkCmdResetQueryPool(rcmd, qp, 0, 2);
                    Vk.vkCmdBindPipeline(rcmd, VkConst.BindPointCompute, rp.Pipeline);
                    Vk.vkCmdBindDescriptorSets(rcmd, VkConst.BindPointCompute, rp.Layout, 0, 1, &rset, 0, null);
                    fixed (uint* pp = pcv) Vk.vkCmdPushConstants(rcmd, rp.Layout, VkConst.StageComputeShader, 0, (uint)pcBytes, pp);
                    Vk.vkCmdWriteTimestamp(rcmd, VkConst.PipelineStageBottomOfPipe, qp, 0);
                    Vk.vkCmdDispatch(rcmd, rgx, rgy, 1);
                    Vk.vkCmdWriteTimestamp(rcmd, VkConst.PipelineStageBottomOfPipe, qp, 1);
                    Vk.Check(Vk.vkEndCommandBuffer(rcmd), "end");
                    rdev.Submit(rcmd, rf); rdev.WaitFence(rf);
                    ulong* ts = stackalloc ulong[2];
                    Vk.vkGetQueryPoolResults(rdev.Device, qp, 0, 2, 16, ts, 8, 1 | 2);
                    best = Math.Min(best, (ts[1] - ts[0]) * rdev.TimestampPeriodNs / 1e6);
                }
                Console.WriteLine($"rawbench best={best:F4} ms");
            }
            return 0;
        }
        return Harness.Usage(2);
    }
}
