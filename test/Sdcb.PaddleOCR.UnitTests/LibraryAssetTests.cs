using System.Reflection;
using System.Runtime.Versioning;

namespace Sdcb.PaddleOCR.UnitTests;

public class LibraryAssetTests
{
    [Fact]
    public void ConsumesExpectedLibraryTfm()
    {
        string? tfm = typeof(PaddleOcrAll).Assembly
            .GetCustomAttribute<TargetFrameworkAttribute>()?.FrameworkName;
        Assert.NotNull(tfm);
#if USE_NS20_LIBRARY
        Assert.Contains("NETStandard", tfm, StringComparison.OrdinalIgnoreCase);
#else
        Assert.Contains("NETCoreApp", tfm, StringComparison.OrdinalIgnoreCase);
#endif
    }
}
