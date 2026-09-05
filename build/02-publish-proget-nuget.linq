<Query Kind="Program">
  <Namespace>System.Threading.Tasks</Namespace>
  <Namespace>LINQPad.Controls</Namespace>
  <IncludeUncapsulator>false</IncludeUncapsulator>
</Query>

#load ".\00-common"

DumpContainer dc = new DumpContainer().Dump();

async Task Main()
{
	await SetupAsync(QueryCancelToken);
	Refresh();
}

void PublishProGet(string path)
{
	QueryCancelToken.ThrowIfCancellationRequested();
	string nugetApiKey = Util.GetPassword("proget-api-key");
	string nugetApiUrl = Util.GetPassword("proget-api-test");
	DotNetRun($@"nuget push ""{path}"" -s {nugetApiUrl} -k {nugetApiKey}");
}

void PublishNuGet(string path)
{
	QueryCancelToken.ThrowIfCancellationRequested();
	string nugetApiUrl = "nuget.org";
	string nugetApiKey = Util.GetPassword("nuget-api-key");
	DotNetRun($@"nuget push ""{path}"" -k {nugetApiKey} -s {nugetApiUrl}");
}

void Refresh()
{
	string dir = Path.Combine(Util.CurrentQuery.Location, "nupkgs");
	Directory.CreateDirectory(dir);
	IEnumerable<string> pkgs = Directory.EnumerateFiles(dir, "*.nupkg")
		.Where(x => !x.EndsWith(".symbols.nupkg", StringComparison.OrdinalIgnoreCase))
		.OrderBy(x => x, StringComparer.OrdinalIgnoreCase)
		.ToList();
	dc.Content = new
	{
		Functions = Util.HorizontalRun(true,
			new Button("✅Publish All", _ => pkgs.ToList().ForEach(PublishProGet)),
			new Button("⚠Publish All to NuGet", _ => pkgs.ToList().ForEach(PublishNuGet)),
			new Button("📂Open Folder", _ => Process.Start("explorer", @$"/select, ""{dir}"""))
			),
		Table = BuildTable()
	};

	object BuildTable()
	{
		return pkgs
			.Select(x => new
			{
				Package = Path.GetFileNameWithoutExtension(x),
				Size = $"{new FileInfo(x).Length / 1024.0 / 1024:N2}MB",
				Functions = Util.HorizontalRun(true,
					new Button("✅Publish", o => PublishProGet(x)),
					new Button("⚠Publish NuGet", o => PublishNuGet(x)),
					new Button("📁Open Folder", o => Process.Start("explorer", @$"/select, ""{x}""")),
					new Button("❌Delete", o =>
					{
						File.Delete(x);
						string snupkg = Path.ChangeExtension(x, ".snupkg");
						if (File.Exists(snupkg)) File.Delete(snupkg);
						Refresh();
					})),
			});
	}
}
