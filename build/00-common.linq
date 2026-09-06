<Query Kind="Program">
  <Namespace>System.Threading.Tasks</Namespace>
  <Namespace>System.Net.Http</Namespace>
  <IncludeUncapsulator>false</IncludeUncapsulator>
</Query>

async Task Main()
{
	await SetupAsync(QueryCancelToken);
}

async Task SetupAsync(CancellationToken cancellationToken = default)
{
	Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
	Environment.CurrentDirectory = Util.CurrentQuery.Location;
	Directory.CreateDirectory("nupkgs");
	await EnsureNugetExe(cancellationToken);
}

static void NuGetRun(string args) => Run(@".\nuget.exe", args, Encoding.GetEncoding("gb2312"));
static void DotNetRun(string args) => Run("dotnet", args.Dump(), Encoding.GetEncoding("utf-8"));
static void Run(string exe, string args, Encoding encoding) => Util.Cmd(exe, args, encoding);

// Independent package versions. Sdcb.SimdPaddleOCR is expected to ship often;
// ModelProvider and the four model packages should stay stable.
// Keep Directory.Build.props defaults in sync.
// Do not pack with -p:Version=; pack passes these properties instead so a 1.0.1
// SimdPaddleOCR package can depend on ModelProvider 1.0.0.
static ProjectVersion[] Projects =
{
	new("Sdcb.SimdPaddleOCR", "1.1.0"),
	new("Sdcb.SimdPaddleOCR.ModelProvider", "1.0.0"),
	new("Sdcb.SimdPaddleOCR.Models.TextLineOrientation", "1.0.0"),
	new("Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny", "1.0.0"),
	new("Sdcb.SimdPaddleOCR.Models.ChineseV6Small", "1.0.0"),
	new("Sdcb.SimdPaddleOCR.Models.ChineseV6Medium", "1.0.0"),
};

static string VersionPropertyName(string projectName) => projectName.Replace(".", "") + "Version";

static string AllVersionMsBuildArgs() =>
	string.Join(" ", Projects.Select(p => $"-p:{VersionPropertyName(p.name)}={p.version}"));

static string FindProjectPath(string projectName)
{
	string repoRoot = Path.GetFullPath(Path.Combine(Util.CurrentQuery.Location, ".."));
	string[] matches = Directory.GetFiles(repoRoot, projectName + ".csproj", SearchOption.AllDirectories)
		.Where(p =>
		{
			string n = p.Replace('/', Path.DirectorySeparatorChar);
			return !n.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}")
				&& !n.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}");
		})
		.ToArray();
	if (matches.Length == 0)
		throw new Exception($"Project {projectName}.csproj was not found.");
	if (matches.Length > 1)
		throw new Exception($"Multiple {projectName}.csproj files found:{Environment.NewLine}{string.Join(Environment.NewLine, matches)}");
	return matches[0];
}

static async Task DownloadFile(Uri uri, string localFile, CancellationToken cancellationToken = default)
{
	if (uri.Scheme == "https" || uri.Scheme == "http")
	{
		using HttpClient http = new();

		HttpResponseMessage resp = await http.GetAsync(uri, cancellationToken);
		if (!resp.IsSuccessStatusCode)
		{
			throw new Exception($"Failed to download: {uri}, status code: {(int)resp.StatusCode}({resp.StatusCode})");
		}

		using (FileStream file = File.OpenWrite(localFile))
		{
			await resp.Content.CopyToAsync(file, cancellationToken);
		}
	}
	else if (uri.Scheme == "file")
	{
		File.Copy(uri.ToString()[8..], localFile, overwrite: true);
	}
	else
	{
		throw new Exception($"Uri scheme: {uri.Scheme} not supported.");
	}
}

static async Task<string> EnsureNugetExe(CancellationToken cancellationToken = default)
{
	Uri uri = new Uri(@"https://dist.nuget.org/win-x86-commandline/latest/nuget.exe");
	string localPath = @".\nuget.exe";
	if (!File.Exists(localPath))
	{
		await DownloadFile(uri, localPath, cancellationToken);
	}
	return localPath;
}

record ProjectVersion(string name, string version);
