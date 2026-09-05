using System;
using System.IO;
using Avalonia;

namespace SkiaSharp.Avalonia;

internal static class Program
{
    [STAThread]
    private static void Main(string[] args)
    {
        string repositoryRoot = FindRepositoryRoot();
        string imagePath = args.Length > 0
            ? Path.GetFullPath(args[0])
            : Path.Combine(repositoryRoot, "examples", "sample.jpg");

        BuildAvaloniaApp(imagePath).StartWithClassicDesktopLifetime(args);
    }

    private static AppBuilder BuildAvaloniaApp(string imagePath)
    {
        return AppBuilder.Configure(() => new App(imagePath))
            .UsePlatformDetect()
            .LogToTrace();
    }

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Sdcb.SimdPaddleOCR.slnx")))
                return directory.FullName;
            directory = directory.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}
