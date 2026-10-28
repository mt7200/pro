# 本地训练 + 阈值校准（不依赖 HDFS）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path "out")) { & "$root\build.ps1" }
$cp = "out;" + ((Get-ChildItem "lib" -Filter *.jar | ForEach-Object { $_.FullName }) -join ";")

& java "-Dfile.encoding=UTF-8" -Xmx2g -cp $cp com.movie.sentiment.Main train `
    --input "data/movies-from-kggle.txt" `
    --model "output/model.dat" `
    --calibrate-samples 20000
