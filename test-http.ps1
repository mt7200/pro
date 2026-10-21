# HTTP 接口端到端测试脚本（第二阶段）
# 覆盖：成员4 labels 格式 / 第二阶段 full schema / tags 与 text 双入参 /
#       GET 与 POST 双路由 / CORS 预检 / 400 与 405 / 全 OOV 输入
#
# 用 .NET HttpClient 发真实请求并取【响应原始字节】做 UTF-8 解码校验，
# 避免 PowerShell > 重定向产生 UTF-16 的假象。
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$port = if ($args.Count -gt 0) { $args[0] } else { "8090" }
$base = "http://localhost:$port"

Add-Type -AssemblyName System.Net.Http
$client = New-Object System.Net.Http.HttpClient
$client.Timeout = [TimeSpan]::FromSeconds(30)

$pass = 0; $fail = 0

function Send-Req {
    param([string]$Method, [string]$Url, [string]$Json)
    $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::$Method, $Url)
    if ($Json) {
        $req.Content = New-Object System.Net.Http.StringContent($Json, [System.Text.Encoding]::UTF8, "application/json")
    }
    $resp = $client.SendAsync($req).Result
    $bytes = $resp.Content.ReadAsByteArrayAsync().Result
    $body = [System.Text.Encoding]::UTF8.GetString($bytes)
    return @{ Status = [int]$resp.StatusCode; Body = $body; Headers = $resp.Headers }
}

function Check {
    param([string]$Name, [scriptblock]$Assert)
    try {
        $r = & $Assert
        if ($r -eq $true) {
            Write-Host "  [PASS] $Name" -ForegroundColor Green
            $script:pass++
        } else {
            Write-Host "  [FAIL] $Name -> $r" -ForegroundColor Red
            $script:fail++
        }
    } catch {
        Write-Host "  [FAIL] $Name -> 异常: $_" -ForegroundColor Red
        $script:fail++
    }
}

Write-Host ""
Write-Host "════════ 接口测试目标: $base ════════"

# ── 0) 服务健康 ──
Write-Host "`n【0】健康检查 GET /health"
$r = Send-Req GET "$base/health" $null
Check "返回 200" { $r.Status -eq 200 }
Check "含模型规模字段 N/V/G" { ($r.Body -match '"N":\d+') -and ($r.Body -match '"V":\d+') -and ($r.Body -match '"G":\d+') }
Check "带 CORS Allow-Origin 头" { $r.Headers.Contains("Access-Control-Allow-Origin") }
Write-Host "      响应: $($r.Body)"

# ── 1) 标签列表 ──
Write-Host "`n【1】标签列表 GET /genres"
$r = Send-Req GET "$base/genres" $null
Check "返回 200" { $r.Status -eq 200 }
Check "genres 数组非空" { $r.Body -match '"genres":\[".+"\]' }
Write-Host "      响应: $($r.Body.Substring(0, [Math]::Min(120, $r.Body.Length)))..."

# ── 2) POST + tags 入参（默认 labels schema）──
Write-Host "`n【2】POST /predict  tags 入参，默认 labels schema"
$r = Send-Req POST "$base/predict" '{"tags":["spacewar","alien","marine","soldier","battle"]}'
Check "返回 200" { $r.Status -eq 200 }
Check "含 success 字段" { $r.Body -match '"success":true' }
Check "含 labels 数组" { $r.Body -match '"labels":\[' }
Check "label 含 name 与 confidence" { $r.Body -match '"name":"[A-Za-z]+"' -and $r.Body -match '"confidence":[0-9.]+' }
Check "confidence 是数字而非百分比字符串" { $r.Body -notmatch '"confidence":"' }
Check "不含 full schema 专有字段 predGenres" { $r.Body -notmatch '"predGenres"' }
Write-Host "      响应: $($r.Body)"

# ── 3) POST + text 入参（自然语言剧情）──
Write-Host "`n【3】POST /predict  text 入参（自然语言）"
$r = Send-Req POST "$base/predict" '{"text":"A space marine battles aliens on a distant planet in a futuristic war."}'
Check "返回 200" { $r.Status -eq 200 }
Check "labels 非空" { $r.Body -match '"labels":\[\{' }
$o = $r.Body | ConvertFrom-Json
Write-Host "      命中标签数: $($o.labels.Count)"
Write-Host "      响应: $($r.Body)"

# ── 4) POST + tags + schema=full ──
Write-Host "`n【4】POST /predict?schema=full  完整 schema（= predictResult.txt 单行）"
$r = Send-Req POST "$base/predict?schema=full" '{"tags":["spacewar","alien","marine"]}'
Check "返回 200" { $r.Status -eq 200 }
$o = $r.Body | ConvertFrom-Json
$need = @("idx","tags","trueGenres","predGenres","scores","probs","confidence","maxProb")
foreach ($f in $need) {
    Check "含字段 $f" { $null -ne $o.PSObject.Properties[$f] }
}
Check "trueGenres 为 null（实时预测无 ground-truth）" { $null -eq $o.trueGenres }
Check "scores 为对象且非空" { ($null -ne $o.scores) -and ($o.scores.PSObject.Properties.Name.Count -gt 0) }

# ── 5) GET + query 传参 ──
Write-Host "`n【5】GET /predict?text=...  query 传参"
$encoded = [System.Uri]::EscapeDataString("action,adventure,spacewar")
$r = Send-Req GET "$base/predict?text=$encoded" $null
Check "返回 200" { $r.Status -eq 200 }
Check "labels 格式" { $r.Body -match '"labels":\[' }
Write-Host "      响应: $($r.Body.Substring(0, [Math]::Min(160, $r.Body.Length)))..."

# ── 6) CORS 预检 ──
Write-Host "`n【6】CORS 预检 OPTIONS /predict"
$r = Send-Req OPTIONS "$base/predict" $null
Check "预检返回 204" { $r.Status -eq 204 }
Check "Allow-Origin 为 *" { $r.Headers.GetValues("Access-Control-Allow-Origin") -contains "*" }
Check "Allow-Methods 含 GET,POST,OPTIONS" { ($r.Headers.GetValues("Access-Control-Allow-Methods") -join "") -match "POST" }
Check "Allow-Headers 含 Content-Type" { ($r.Headers.GetValues("Access-Control-Allow-Headers") -join "") -match "Content-Type" }

# ── 7) 空输入 → 400（成员4 规范第 3 节第 4 条）──
Write-Host "`n【7】空输入 POST /predict {}"
$r = Send-Req POST "$base/predict" '{}'
Check "返回 400" { $r.Status -eq 400 }
Check "success 为 false" { $r.Body -match '"success":false' }
Check "labels 为空数组" { $r.Body -match '"labels":\[\]' }
Check "含中文提示" { $r.Body -match '"message":"[^"]+"' }
Write-Host "      响应: $($r.Body)"

# ── 8) 空输入 + full schema → 200 + error 字段（第二阶段规范）──
Write-Host "`n【8】空输入 + schema=full（应返 200 + error 字段，便于前端统一 parse）"
$r = Send-Req POST "$base/predict?schema=full" '{}'
Check "返回 200（full schema 一律 200）" { $r.Status -eq 200 }
Check "含 error 字段" { $r.Body -match '"error":"' }
Check "predGenres 为空数组" { $r.Body -match '"predGenres":\[\]' }
Write-Host "      响应: $($r.Body)"

# ── 9) 全 OOV 输入（不崩溃）──
Write-Host "`n【9】全 OOV 输入（词汇表外的词）"
$r = Send-Req POST "$base/predict" '{"text":"zzzzqqqq notaword atall xyzzy"}'
Check "返回 200 而非 500" { $r.Status -eq 200 }
Check "labels 为空数组" { $r.Body -match '"labels":\[\]' }
Check "给出结构化提示 message" { $r.Body -match '"message":"[^"]+"' }
Write-Host "      响应: $($r.Body)"

# ── 10) 错误路由 → 404 ──
Write-Host "`n【10】不存在的路由 GET /nonexistent"
$r = Send-Req GET "$base/nonexistent" $null
Check "返回 404" { $r.Status -eq 404 }

# ── 11) 错误方法 → 405 ──
Write-Host "`n【11】错误方法 DELETE /predict"
$r = Send-Req DELETE "$base/predict" $null
Check "返回 405" { $r.Status -eq 405 }

# ── 12) UTF-8 中文校验（字节层）──
Write-Host "`n【12】中文字段 UTF-8 字节校验"
$r = Send-Req POST "$base/predict" '{"tags":["spacewar","alien"]}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($r.Body)
$redecoded = [System.Text.Encoding]::UTF8.GetString($bytes)
Check "UTF-8 往返一致（无乱码）" { $redecoded -eq $r.Body }
$hasChinese = $r.Body -match '[\u4e00-\u9fa5]'
if ($hasChinese) {
    Write-Host "      响应含中文，已确认 UTF-8 编码正确" -ForegroundColor Green
} else {
    Write-Host "      本次响应不含中文（labels 为空时为正常情况）"
}

# ── 13) 并发请求（验证模型单例线程安全）──
Write-Host "`n【13】并发 8 个请求（验证单例 predictor 线程安全）"
$tasks = @()
for ($i = 0; $i -lt 8; $i++) {
    $t = $client.PostAsync("$base/predict",
        (New-Object System.Net.Http.StringContent('{"tags":["spacewar","alien","marine"]}',
            [System.Text.Encoding]::UTF8, "application/json")))
    $tasks += $t
}
[System.Threading.Tasks.Task]::WaitAll($tasks)
$okCount = 0
foreach ($t in $tasks) {
    $resp = $t.Result
    if ([int]$resp.StatusCode -eq 200) { $okCount++ }
}
Check "8 个并发请求全部 200" { $okCount -eq 8 }
# 校验所有并发响应内容一致（无状态 => 同输入必同输出）
$bodies = @()
foreach ($t in $tasks) { $bodies += $t.Result.Content.ReadAsStringAsync().Result }
$unique = $bodies | Sort-Object -Unique
Check "同输入并发响应完全一致（无竞态）" { $unique.Count -eq 1 }

# ── 14) Top-N 截断（应对常见 tag 过度激活，仅作用于 labels 格式）──
Write-Host "`n【14】Top-N 截断 ?topN=（常见 tag 'murder' 原本激活多个）"
$r0 = Send-Req POST "$base/predict" '{"text":"murder"}'
$full = ($r0.Body | ConvertFrom-Json).labels.Count
$rN = Send-Req POST "$base/predict?topN=3" '{"text":"murder"}'
$o3 = $rN.Body | ConvertFrom-Json
Check "不截断时返回 >3 个标签（证明存在过度激活场景）" { $full -gt 3 }
Check "topN=3 时恰好返回 3 个" { $o3.labels.Count -eq 3 }
Check "截断保留的是置信度最高的前 3 个" {
    $allConf = ($r0.Body | ConvertFrom-Json).labels | ForEach-Object { $_.confidence }
    $topConf = $o3.labels | ForEach-Object { $_.confidence }
    ($topConf -join ',') -eq (($allConf | Select-Object -First 3) -join ',')
}
Write-Host "      不截断=$full 个 → topN=3 后=$($o3.labels.Count) 个"

# ── 15) topN 对 full schema 无效（三处同源不可破）──
Write-Host "`n【15】topN 不得影响 full schema 的 predGenres"
$text = '{"text":"A space marine battles aliens on a distant planet."}'
$rfNo = Send-Req POST "$base/predict?schema=full" $text
$rfN = Send-Req POST "$base/predict?schema=full&topN=1" $text
$ofNo = $rfNo.Body | ConvertFrom-Json
$ofN = $rfN.Body | ConvertFrom-Json
Check "full schema 仍返回 predGenres 字段" { $null -ne $ofN.PSObject.Properties["predGenres"] }
Check "full schema 的 predGenres 未被 topN=1 截断（与不带 topN 完全一致）" {
    ($ofN.predGenres -join ',') -eq ($ofNo.predGenres -join ',')
}
Check "该样本 predGenres 多于 1 个（截断若生效会被压到 1）" { $ofNo.predGenres.Count -gt 1 }
Write-Host "      带 topN=1 与不带的 predGenres 数均为 $($ofN.predGenres.Count)"

Write-Host "`n════════ 测试结果: PASS=$pass  FAIL=$fail ════════"
if ($fail -eq 0) { Write-Host "全部通过" -ForegroundColor Green }
else { Write-Host "有 $fail 项失败，请检查上方 [FAIL]" -ForegroundColor Red }
$client.Dispose()
exit $fail
