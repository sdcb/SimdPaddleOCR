using System;
using System.Windows.Forms;

namespace SystemDrawing.WinForms;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
#if NET6_0_OR_GREATER
        // Applies ApplicationHighDpiMode / ApplicationDefaultFont from the csproj so
        // .NET 10 control metrics match .NET Framework 4.8 (Segoe UI 9pt is taller).
        ApplicationConfiguration.Initialize();
#else
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
#endif
        Application.Run(new MainForm());
    }
}
