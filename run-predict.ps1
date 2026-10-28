# 批量预测 + 指标计算（不依赖 HDFS）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path "out")) { & "$root\build.ps1" }
$cp = "out;" + ((Get-ChildItem "lib" -Filter *.jar | ForEach-Object { $_.FullName }) -join ";")

& java "-Dfile.encoding=UTF-8" -Xmx2g -cp $cp com.movie.sentiment.Main predict `
    --model "output/model.dat" `
    --input "data/movies-to-be-predicted.txt" `
    --out "output/predictResult.txt" `
    --metrics "output/metrics.json" `
    --z-thresholds "output/z-thresholds.json" `
    --z-threshold 1.25
