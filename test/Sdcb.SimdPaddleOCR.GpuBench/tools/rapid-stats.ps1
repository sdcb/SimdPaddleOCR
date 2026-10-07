# Rolls the harness JSONs of one experiment up to per-(label, engine, model) rows:
# three round medians plus their median, and the accuracy / stage means behind them.
# usage: tools/rapid-stats.ps1 [-Dir bench-out\rapid-3080ti] [-Filter *.json]
param([string]$Dir = "bench-out\rapid-3080ti", [string]$Filter = "*.json")
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$files = Get-ChildItem (Join-Path $root $Dir) -Filter $Filter | Where-Object { $_.Name -notmatch "summarize" }
$rows = @()
foreach ($f in $files) {
    $j = Get-Content $f.FullName -Raw | ConvertFrom-Json
    $m = $j.meta
    $a = $m.accuracy
    if ($null -eq $a) { continue }
    $stages = $j.summary.stage_ms_mean
    $label = if ($f.Name -match '^rapid-3080ti-([a-z0-9]+)-') { $Matches[1] } else { 'default' }
    $rows += [pscustomobject]@{
        Label = $label; Engine = $m.mode; Model = $m.model; Round = $m.replica
        Mean = [math]::Round($j.summary.total_ms.mean, 1)
        Median = [math]::Round($j.summary.total_ms.median, 1)
        P95 = [math]::Round($j.summary.total_ms.p95, 1)
        WsLoaded = [math]::Round($m.working_set_mb_loaded, 0)
        WsPeak = [math]::Round($m.working_set_mb_peak, 0)
        Exact = "$($a.exact_lines)/$($a.total_lines)"
        Cer = [math]::Round(100 * $a.cer, 2)
        Cls = "$($a.cls_correct)/$($a.cls_total)"
        Det = [math]::Round($stages.det_graph, 1)
        Rec = [math]::Round($stages.rec_graph, 1)
        ClsMs = [math]::Round($stages.cls_graph, 1)
    }
}
$rows | Sort-Object Label, Model, Engine, Round | Format-Table -AutoSize
foreach ($g in ($rows | Group-Object Label, Model, Engine | Sort-Object Name)) {
    $med = @($g.Group.Median | Sort-Object)
    $mid = if ($med.Count % 2) { $med[[int]($med.Count / 2)] } else { ($med[$med.Count / 2 - 1] + $med[$med.Count / 2]) / 2 }
    $p95 = @($g.Group.P95 | Sort-Object)
    "{0,-22} medians={1,-24} -> {2,7}  mean={3,7:F1} p95={4,7:F1} exact={5,-11} CER={6,5:F2}% cls={7,-11} det={8,7:F1} rec={9,7:F1} cls_ms={10,6:F1} ws={11}/{12}MB" -f `
        $g.Name, ($med -join "/"), $mid, ($g.Group.Mean | Measure-Object -Average).Average,
        $p95[[int]($p95.Count / 2)], $g.Group[0].Exact, $g.Group[0].Cer, $g.Group[0].Cls,
        ($g.Group.Det | Measure-Object -Average).Average, ($g.Group.Rec | Measure-Object -Average).Average,
        ($g.Group.ClsMs | Measure-Object -Average).Average, $g.Group[0].WsLoaded, $g.Group[0].WsPeak
}
