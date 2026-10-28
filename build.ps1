# 编译算法模块（JDK 8+，不需要 Maven，也不需要 Hadoop 集群）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (Test-Path "out") { Remove-Item "out" -Recurse -Force }
New-Item -ItemType Directory "out" | Out-Null

$cp = (Get-ChildItem "lib" -Filter *.jar | ForEach-Object { $_.FullName }) -join ";"
$src = @(Get-ChildItem "com" -Recurse -Filter *.java | ForEach-Object { $_.FullName })
Write-Host "源文件 $($src.Count) 个，依赖 jar $((Get-ChildItem 'lib' -Filter *.jar).Count) 个"

& javac -encoding UTF-8 -cp $cp -d out $src
if ($LASTEXITCODE -ne 0) { throw "javac 编译失败，请确认已安装 JDK 8 或更高版本并在 PATH 中" }
Write-Host "编译完成：$root\out"
