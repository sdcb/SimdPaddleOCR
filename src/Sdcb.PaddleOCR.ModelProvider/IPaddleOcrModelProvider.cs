using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Sdcb.PaddleOCR.ModelProvider;

public interface IPaddleOcrModelProvider
{
    string Name { get; }
    PaddleOcrModelKind Kind { get; }
    string Format { get; }
    string? LanguageCode { get; }
    string? Version { get; }
    Stream OpenRead();
    Task<Stream> OpenReadAsync(CancellationToken cancellationToken = default);
}
