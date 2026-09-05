using System;
using System.IO;
using System.Windows;

namespace OpenCvSharp5.Wpf;

internal static class Program
{
    [STAThread]
    private static void Main(string[] args)
    {
        string repositoryRoot = FindRepositoryRoot();
        string imagePath = args.Length > 0
            ? Path.GetFullPath(args[0])
            : Path.Combine(repositoryRoot, "examples", "sample.jpg");
        string modelDirectory = args.Length > 1
            ? Path.GetFullPath(args[1])
            : Path.Combine(repositoryRoot, "models");

        Application application = new();
        application.Run(new MainWindow(imagePath, modelDirectory));
    }

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Sdcb.PaddleOCR.slnx")))
                return directory.FullName;
            directory = directory.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}
