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

void Refresh()
{
	dc.Content = LoadTable();
}

object LoadTable()
{
	return new
	{
		Functions = Util.HorizontalRun(true,
			new Button("Build All", _ => Projects.ToList().ForEach(Build)),
			new Button("Clear Cache", _ => ClearNuGetCache()),
			new Button("📂Open nupkgs", _ => Process.Start("explorer", Path.GetFullPath(@".\nupkgs")))),
		Table = Projects
			.Select(x => new
			{
				Project = x.name,
				Version = x.version,
				Build = new Button("Build", o => Build(x))
			})
	};
}

void ClearNuGetCache()
{
	DotNetRun("nuget locals http-cache --clear");
	DotNetRun("nuget locals temp --clear");
}

void Build(ProjectVersion p)
{
	QueryCancelToken.ThrowIfCancellationRequested();
	string projPosition = FindProjectPath(p.name);
	// Do not pass -p:Version=; it is global and NuGet would copy it onto ProjectReference
	// dependencies. Each project's version comes from AllVersionMsBuildArgs() via Directory.Build.props.
	DotNetRun($@"pack ""{projPosition}"" -c Release -o .\nupkgs {AllVersionMsBuildArgs()}");
	Refresh();
}
