using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Themes.Fluent;

namespace SkiaSharp.Avalonia;

internal sealed class App : Application
{
    private readonly string _imagePath;

    public App(string imagePath)
    {
        _imagePath = imagePath;
        Styles.Add(new FluentTheme());
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            desktop.MainWindow = new MainWindow(_imagePath);

        base.OnFrameworkInitializationCompleted();
    }
}
