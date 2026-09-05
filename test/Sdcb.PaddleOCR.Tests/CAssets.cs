using Sdcb.PaddleOCR.Models.ChineseV6Tiny;

namespace Sdcb.PaddleOCR.Tests;

static class CAssets
{
    public const string BaseUrl = "https://cv-public.sdcb.ai/2026/";
    public const string DllName = "lw_ppocr_c.dll";
    public const string DetName = "det.lwm";
    public const string ClsName = "cls.lwm";
    public const string RecName = "rec.lwm";
    public const string DictName = "ppocr_keys.txt";

    public static readonly string[] Files = [DllName, DetName, ClsName, RecName];

    public static async Task EnsureAsync(string dir, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(dir);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        await Task.WhenAll(Files.Select(name => DownloadIfMissing(http, dir, name, cancellationToken)));
    }

    public static string WriteDictionary(string dir)
    {
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, DictName);
        using Stream src = ChineseV6TinyModel.Dictionary.OpenRead();
        using FileStream dst = File.Create(path);
        src.CopyTo(dst);
        return path;
    }

    public static void CopyDll(string dir)
    {
        string src = Path.Combine(dir, DllName);
        if (!File.Exists(src))
            throw new FileNotFoundException("lw_ppocr_c.dll was not downloaded", src);
        File.Copy(src, Path.Combine(AppContext.BaseDirectory, DllName), overwrite: true);
    }

    public static string DetPath(string dir) => Path.Combine(dir, DetName);
    public static string ClsPath(string dir) => Path.Combine(dir, ClsName);
    public static string RecPath(string dir) => Path.Combine(dir, RecName);
    public static string DictPath(string dir) => Path.Combine(dir, DictName);

    static async Task DownloadIfMissing(HttpClient http, string dir, string name, CancellationToken cancellationToken)
    {
        string path = Path.Combine(dir, name);
        if (File.Exists(path) && new FileInfo(path).Length > 0)
        {
            Console.WriteLine($"skip {name} (cached, {new FileInfo(path).Length} bytes)");
            return;
        }

        string url = BaseUrl + name;
        Console.WriteLine($"download {url}");
        using HttpResponseMessage response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        string tmp = path + ".tmp";
        await using (FileStream fs = File.Create(tmp))
            await response.Content.CopyToAsync(fs, cancellationToken);
        File.Move(tmp, path, overwrite: true);
        Console.WriteLine($"saved {name} ({new FileInfo(path).Length} bytes)");
    }
}
