using Android.App;
using Android.OS;
using Android.Views;

namespace Sdcb.SimdPaddleOCR.AndroidBench;

/// <summary>
/// adb entry point. Extras: <c>args</c> (space-separated bench arguments),
/// <c>env</c> (<c>K=V;K=V</c>, applied before the library is touched so its
/// static env switches see them) and <c>run</c> (output id). Everything the
/// bench prints goes to logcat (tag SimdOcrBench) and to
/// <c>files/out/&lt;run&gt;.log</c>; <c>&lt;run&gt;.done</c> holds the exit code.
/// The process kills itself afterwards so every run starts cold.
/// </summary>
[Activity(Label = "SimdOcrBench", MainLauncher = true, Exported = true,
    Name = "com.sdcb.simdocr.bench.MainActivity")]
public class MainActivity : Activity
{
    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);
        Window?.AddFlags(WindowManagerFlags.KeepScreenOn);
        string args = Intent?.GetStringExtra("args") ?? "--e2e";
        string env = Intent?.GetStringExtra("env") ?? "";
        string run = Intent?.GetStringExtra("run") ?? "run";
        string root = GetExternalFilesDir(null)!.AbsolutePath;
        string outDir = Path.Combine(root, "out");
        // app-created, so the app can read what adb pushes into them later
        // (directories made by the shell user are not readable by the app)
        foreach (string d in new[] { outDir, Path.Combine(root, "models"), Path.Combine(root, "dataset") })
            Directory.CreateDirectory(d);
        File.Delete(Path.Combine(outDir, run + ".done"));

        foreach (string kv in env.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            int eq = kv.IndexOf('=');
            if (eq > 0) System.Environment.SetEnvironmentVariable(kv[..eq], kv[(eq + 1)..]);
        }

        var worker = new Thread(() =>
        {
            int rc;
            using (var log = new TeeWriter(Path.Combine(outDir, run + ".log")))
            {
                Console.SetOut(log);
                Console.SetError(log);
                Console.WriteLine($"args: {args}");
                if (env.Length > 0) Console.WriteLine($"env: {env}");
                try
                {
                    Directory.SetCurrentDirectory(root);
                    rc = Bench.Run(args.Split(' ', StringSplitOptions.RemoveEmptyEntries), root, outDir);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"FATAL {ex}");
                    rc = 99;
                }
                Console.WriteLine($"exit {rc}");
            }
            File.WriteAllText(Path.Combine(outDir, run + ".done"), rc.ToString());
            Process.KillProcess(Process.MyPid());
        }, 64 << 20);
        worker.Start();
    }
}

/// <summary>Line-buffered tee to logcat and a file.</summary>
sealed class TeeWriter(string path) : TextWriter
{
    private readonly StreamWriter _file = new(path, append: false) { AutoFlush = true };
    private readonly System.Text.StringBuilder _line = new();
    private readonly object _sync = new();
    public override System.Text.Encoding Encoding => System.Text.Encoding.UTF8;

    public override void Write(char value)
    {
        lock (_sync)
        {
            if (value == '\n') { FlushLine(); return; }
            if (value != '\r') _line.Append(value);
        }
    }

    public override void Write(string? value)
    {
        if (value is null) return;
        lock (_sync)
            foreach (char c in value)
            {
                if (c == '\n') FlushLine();
                else if (c != '\r') _line.Append(c);
            }
    }

    public override void WriteLine(string? value)
    {
        lock (_sync) { Write(value); FlushLine(); }
    }

    private void FlushLine()
    {
        string s = _line.ToString();
        _line.Clear();
        Android.Util.Log.Info("SimdOcrBench", s);
        _file.WriteLine(s);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            lock (_sync) if (_line.Length > 0) FlushLine();
            _file.Dispose();
        }
        base.Dispose(disposing);
    }
}
