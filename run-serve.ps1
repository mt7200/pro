# 启动 HTTP 预测服务（默认 8080）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path "out")) { & "$root\build.ps1" }
$cp = "out;" + ((Get-ChildItem "lib" -Filter *.jar | ForEach-Object { $_.FullName }) -join ";")

& java "-Dfile.encoding=UTF-8" -cp $cp com.movie.sentiment.Main serve `
    --model "output/model.dat" `
    --port 8080 `
    --z-threshold 1.25 `
    --z-thresholds "output/z-thresholds.json"
