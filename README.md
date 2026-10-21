# 《大数据技术》课程期末项目 — 成员3 算法模块（第二阶段）

**算法**：TF-IDF 加权 + 多标签朴素贝叶斯（one-vs-rest + 阈值激活）
**任务**：用电影的 tag 列表预测 genre 集合（一部电影可属多个类型）
**技术栈**：JDK 8 标准库 + Hadoop client，**无任何第三方 ML / 分词 / JSON / Web 框架 / CLI 库**

---

## 一、第二阶段做了什么

第一阶段已在本地跑通算法。第二阶段**不重写**解析与算法逻辑，只在其之上叠加四块：

| 模块 | 内容 | 状态 |
|---|---|---|
| Module-HDFS | `HdfsIO` 统一读写层，hdfs:// 与 file:// 透明切换 + 失败自动降级 | ✅ 已实测 |
| Module-BATCH | train/predict 走 HDFS，产出标准化 `predictResult.txt` + `metrics.json` | ✅ 已实测 |
| Module-SERVER | HTTP 后端（GET/POST、双入参、双 schema、CORS） | ✅ 51 项测试通过 |
| Module-MR | MapReduce 两作业并行训练（可选加分项） | ⚠️ 编译通过，未在真实集群实跑 |

新增/改动的文件：

```
com/movie/sentiment/
├── HdfsIO.java          【新增】HDFS 统一读写 + 降级
├── ModelSerializer.java 【改造】走 HdfsIO 字节流，新增 dualWrite 双写
├── DataParser.java      【改造】读取走 HdfsIO（parseLine 仍为无状态纯函数）
├── Main.java            【改造】HDFS 参数、双写、predictResult 标准产物、三处同源
├── JsonUtil.java        【改造】新增 buildView / 双 schema 序列化 / query 解析
├── PredictServer.java   【改造】GET+POST 路由、tags/text 双入参、schema 切换
├── NbTrainer.java       【改造】抽出 assembleModel 供 MR 复用（公式只有一份）
├── VocabIdfBuilder.java 【改造】新增 --raw-tf 消融开关
└── mr/                  【新增】Module-MR（独立子包，主流程不强依赖）
    ├── DfCountJob.java      作业1：DF(t) / freq(g) / N
    ├── TfIdfSumJob.java     作业2：S(g,t) = Σ TFIDF(t,d)
    └── MrTrainDriver.java   driver：编排两作业 + 调 assembleModel 收尾
```

---

## 二、编译

### 方式 A：无 Maven（推荐，本项目已实测）

项目 `lib/` 已备好 27 个 Hadoop 相关 jar，直接编译：

**Windows PowerShell：**
```powershell
cd "java代码"
$cp = (Get-ChildItem lib\*.jar | ForEach-Object {$_.FullName}) -join ';'
javac --release 8 -encoding UTF-8 -Xlint:-options -cp $cp -d out (Get-ChildItem -Recurse -Filter *.java com | % {$_.FullName})
```

**Linux / macOS：**
```bash
cd java代码
HADOOP_CP=$(ls lib/*.jar | tr '\n' ':')
javac -encoding UTF-8 -d out $(find com -name '*.java')
```

> `--release 8` 用于在高版本 JDK 上编译出 JDK 8 字节码。若本机就是 JDK 8，
> 直接 `javac -encoding UTF-8 -d out ...` 即可（去掉 `--release 8`）。
> `-Xlint:-options` 只是关掉"源值 8 已过时"的警告，不影响产物。

### 方式 B：Maven

```bash
cd java代码
mvn -q package          # 产出 target/sentiment-mr.jar
```

`pom.xml` 里的 `HADOOP_VERSION` 当前为 **3.3.6**，
**必须向成员2/组长确认集群实际版本后修改**，否则可能出现
`NoSuchMethodError` / 序列化不兼容（Hadoop 跨大版本 API 有变动）。
查询集群版本：`hadoop version`。

> 说明：本机未安装 Maven，故 `mvn package` 未经实跑验证；
> 已验证的是方式 A 的 javac 直接编译（16 个源文件全部通过 JDK 8 目标编译）。

---

## 三、运行期占位符（全部可命令行/环境变量覆盖，无硬编码绝对路径）

| 占位符 | 命令行参数 | 环境变量 | 默认值 |
|---|---|---|---|
| `${NAMENODE_URI}` | `--namenode` | `NAMENODE_URI` | `hdfs://localhost:9000` |
| `${HDFS_TRAIN}` | `--input`(train) | `HDFS_TRAIN` | `/input/movies-from-kggle.txt` |
| `${HDFS_PREDICT}` | `--input`(predict) | `HDFS_PREDICT` | `/input/movies-to-be-predicted.txt` |
| `${HDFS_MODEL}` | `--model-hdfs` | `HDFS_MODEL` | `/output/model.dat` |
| `${LOCAL_MODEL}` | `--model` | `LOCAL_MODEL` | `output/model.dat` |
| `${PREDICT_RESULT}` | `--out` | `PREDICT_RESULT` | `output/predictResult.txt` |
| `${METRICS_OUT}` | `--metrics` | `METRICS_OUT` | `output/metrics.json` |
| `${SERVER_PORT}` | `--port` | `SERVER_PORT` | `8080` |
| `${HADOOP_USER}` | `--hadoop-user` | `HADOOP_USER` | 空（简单认证） |

**⚠️ 向成员2确认**：NameNode 实际地址与端口（常见 `8020` 或 `9000`）。

---

## 四、完整运行流程

以下 `$CP` 代表编译好的 classpath（Windows 见上，Linux 用 `out:$HADOOP_CP`）。

### 1) 训练（读 HDFS 数据集，模型双写 HDFS + 本地）

```bash
java -cp "$CP" com.movie.sentiment.Main train \
     --input /input/movies-from-kggle.txt \
     --model-hdfs /output/model.dat \
     --model-local output/model.dat
```

**双写策略**（Module-HDFS 规范）：模型同时写 `${HDFS_MODEL}` 与 `${LOCAL_MODEL}`。
数据集读取走 HDFS 是硬性要求；模型落本地副本不违反该要求（模型是产物而非输入数据集），
且让后端默认读本地更稳，降低联调变量。

### 2) 批量预测（产出兜底文件 + 指标）

```bash
java -cp "$CP" com.movie.sentiment.Main predict \
     --model output/model.dat \
     --input /input/movies-to-be-predicted.txt \
     --out output/predictResult.txt \
     --metrics output/metrics.json \
     --format jsonl
```

`--format` 默认 `jsonl`（成员4 前端可直接 parse 兜底），可选 `tsv`（人类可读，给老师/组长看）。

### 3) 启动 HTTP 后端

```bash
java -cp "$CP" com.movie.sentiment.Main serve --model output/model.dat --port 8080
```

模型启动时**一次性**加载到内存，所有请求复用，不会每请求读盘/读 HDFS。
如需从集群读模型：`--model-hdfs /output/model.dat`。

### 4) 单条预测（联调自测）

```bash
java -cp "$CP" com.movie.sentiment.Main predict-single \
     --model output/model.dat \
     --text "A space marine battles aliens on a distant planet." \
     --format api
```

### 5) curl 自测命令（服务启动后一键验证）

```bash
# POST + tags 数组（默认返回成员4 labels 格式）
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"tags":["action","adventure","spacewar"]}'

# POST + text 自然语言
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"A hero travels through a mysterious world to save his family."}'

# GET + query 传参（浏览器可直接打开）
curl "http://localhost:8080/predict?text=action,adventure"

# GET + 完整 schema（= predictResult.txt 单行字段）
curl "http://localhost:8080/predict?text=action,adventure&schema=full"

# 辅助接口
curl http://localhost:8080/genres
curl http://localhost:8080/health
```

---

## 五、HTTP 接口说明（对接成员4）

### 路由

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/predict` | 核心预测，body 支持 `{"tags":[...]}` 与 `{"text":"..."}` |
| GET | `/predict?text=` | query 传参，便于 curl/浏览器自测 |
| GET | `/genres` | 全部 20 个标签显示名（前端做下拉/图例） |
| GET | `/health` | 模型规模与校准状态 |

已配置 CORS（`Allow-Origin: *`、`Allow-Methods: GET, POST, OPTIONS`、
`Allow-Headers: Content-Type`），OPTIONS 预检直接返回 204。
成员4 前端跑在 `file://` 或异端口都不会被跨域拦下。

### 两套响应 schema（并存，`?schema=` 切换）

**默认 `labels`**（成员4《前后端预测接口对接规范》）：
```json
{"success":true,"message":"预测成功",
 "labels":[{"name":"Action","confidence":0.66},{"name":"Adventure","confidence":0.65}]}
```

**`?schema=full`**（与 `predictResult.txt` 单行字段完全一致）：
```json
{"idx":0,"tags":["spacewar","alien"],"trueGenres":null,
 "predGenres":["sciencefiction","action"],"scores":{...},"probs":{...},
 "confidence":0.0,"maxProb":1.0}
```

> **为何两套并存**：成员4 的规范是真实前端契约（她的前端已按 labels 写好），
> 第二阶段要求的 full schema 是批量产物格式。两者都实现、默认 labels，
> 前端无需改动，需要详细字段时加 `?schema=full` 即可。

### 错误处理（两套 schema 各自口径）

| 情况 | labels schema | full schema |
|---|---|---|
| 空输入 | **400** `{"success":false,"message":"请输入电影描述或 tag 列表","labels":[]}` | **200** + `{"error":"...","predGenres":[]}` |
| 全 OOV | **200** + 空 labels + 引导提示 | **200** + 空 predGenres |
| 方法不允许 | 405 | 405 |
| 路由不存在 | 404 | 404 |
| 内部异常 | 500（统一 JSON，不外泄异常栈） | 500 |

### 置信度语义

模型是**逐标签独立预测**（one-vs-rest），`confidence` 是该标签独立的
sigmoid 校准概率，取值 0~1，**多个标签之间不相加为 1**——
这正是成员4 规范第 4 节要求说明的形式。`labels` 已按 confidence 降序排列，
直接乘 100 即可显示百分比。

### 三处同源

批量 `predictResult.txt` 单行 == 后端 `/predict?schema=full` 响应 == `predict-single` 输出，
三者一律经 `JsonUtil.buildView()` 构建视图后序列化，公式与字段只有一份实现，不会漂移。

---

## 六、HDFS 降级机制（对齐组长"降级方案"哲学）

`HdfsIO` 的路由规则（对齐 Hadoop 自身的路径约定）：

| 路径形式 | 处理方式 |
|---|---|
| `hdfs://nn:9000/...` | 走 Hadoop FileSystem |
| `file:///...` | 走 Hadoop FileSystem |
| `/input/movies.txt`（无 scheme 的绝对路径） | 走 Hadoop FileSystem，由 `fs.defaultFS` 决定集群 |
| `output/model.dat`（相对路径） | java.io 直连本地，**不依赖 Hadoop 运行时** |

**降级行为**：HDFS 读写失败时打印明确 WARN 并自动回退到**同名本地相对路径**，
保证联调 deadline 不因集群抖动卡死。映射规则：去掉 scheme+authority 与前导斜杠。

```
hdfs://localhost:9000/output/model.dat  →  output/model.dat
/input/movies-from-kggle.txt            →  input/movies-from-kggle.txt
```

用 `--no-fallback` 可关闭降级（用于严格验证集群连通性，此时失败直接报错并给出排错建议）。

> 相对路径走 java.io 是有意设计：后端服务只读本地模型副本，
> 不应被迫依赖 Hadoop 运行时，也避免集群抖动影响纯本地操作。

---

## 七、Module-MR（可选加分项）

### 并行化设计（对应 PPT 第05页四步）

| 步骤 | 实现位置 |
|---|---|
| 1. 分词/解析 | 各 Mapper，复用 `DataParser.parseLine`（无状态纯函数） |
| 2. 统计 DF(t)/freq(g)/N | `DfCountJob`，Combiner + Reducer 求和 |
| 3. 累加 S(g,t)=Σ TFIDF(t,d) | `TfIdfSumJob`，Mapper 算局部贡献、Reducer 求和 |
| 4. 全局归一 + 平滑 | driver 端调 `NbTrainer.assembleModel`（需 N、\|V\|、totalLabelOcc 全局量） |

**为何需要两个作业**：TFIDF 依赖 IDF，而 IDF=log(N/(DF+1)) 是全局量，
必须先扫一遍全量数据拿到 DF 才能算。第 2 步的 DF 通过 DistributedCache
分发给第 3 步的 Mapper。

**为何第 4 步在 driver**：P(t\|g)=(S(g,t)+1)/(Z(g)+\|V\|) 需要 Z(g)=Σ_t S(g,t)
覆盖全部 tag，而单个 Reducer 只看到某个 key 分组。

**一致性保证**：MR 模块**不重写任何算法公式**，第 4 步直接调用
`NbTrainer.assembleModel()`，与单机训练共用同一段归一+平滑代码。
genre 顺序固定为字典序（与并行度无关，保证可复现）。

### 提交命令

```bash
hadoop jar target/sentiment-mr.jar com.movie.sentiment.mr.MrTrainDriver \
    -input  /input/movies-from-kggle.txt \
    -model  /output/model-mr.dat \
    -model-local output/model-mr.dat \
    -tmp    /tmp/sentiment-mr
```

`-D` 参数示例（调并行度与内存）：
```bash
hadoop jar target/sentiment-mr.jar \
    -D mapreduce.job.reduces=8 \
    -D mapreduce.map.memory.mb=4096 \
    -D mapreduce.reduce.memory.mb=8192 \
    com.movie.sentiment.mr.MrTrainDriver -input /input/movies-from-kggle.txt -model /output/model-mr.dat
```

### ⚠️ 两个必须知道的限制

1. **未在真实集群实跑**。开发机无 Hadoop 集群，本模块只做到
   **编译通过 + 逻辑自检**。联调 deadline 前若跑不通，
   **以 Module-HDFS 的单机读 HDFS 为准**（那条路径已实测跑通）。

2. **MR 产出的模型不含阈值校准**。校准需在留出集上反复打分搜最优 F1，
   属单机内存操作，分布式化收益低。故 MR 模型的 `tauPerGenre` 为空，
   `NbPredictor` 会自动回退全局 tau=0.0 —— 而 log 域分数是较大负数，
   **tau=0.0 不会激活任何标签**。
   **生产使用请改用单机 `train`**（含留出集校准），或把 MR 模型下载后另行校准。

答辩时建议把 MR 四步作为"我们设计的 Hadoop 并行化路径/可扩展架构"叙述，
即使现场演示用单机读 HDFS 也不空。

---

## 八、实测结果（完整数据集 N=281024）

### 模型规模

| 项 | 值 |
|---|---|
| 训练样本 N | 281,024（跳过 31 行尾部空行） |
| 词汇表 \|V\| | 9,764 |
| 标签种类 \|G\| | 20 |
| 模型文件大小 | 1.83 MB |

### 批量预测指标（140,512 条带 ground-truth 样本）

| 指标 | 数值 |
|---|---|
| MicroPrecision | 0.2318 |
| MicroRecall | 0.7740 |
| **MicroF1** | **0.3568** |
| **MacroF1** | **0.2511** |
| HammingLoss | 0.3588 |
| SubsetAccuracy | 0.0109 |

**per-genre F1**（20 个类别全部非零）：

| genre | F1 | P | R | support |
|---|---|---|---|---|
| drama | 0.6612 | 0.5002 | 0.9749 | 67648 |
| comedy | 0.5700 | 0.4305 | 0.8432 | 50208 |
| thriller | 0.4387 | 0.2849 | 0.9538 | 38112 |
| action | 0.4099 | 0.2829 | 0.7440 | 34496 |
| romance | 0.3866 | 0.2870 | 0.5922 | 26208 |
| adventure | 0.3190 | 0.1945 | 0.8866 | 24544 |
| crime | 0.2948 | 0.1832 | 0.7542 | 20704 |
| family | 0.2593 | 0.1627 | 0.6377 | 14752 |
| horror | 0.2403 | 0.1562 | 0.5207 | 15424 |
| sciencefiction | 0.2399 | 0.1462 | 0.6673 | 16352 |
| fantasy | 0.2016 | 0.1164 | 0.7506 | 13088 |
| music | 0.1691 | 0.1101 | 0.3642 | 5536 |
| mystery | 0.1687 | 0.0984 | 0.5898 | 10688 |
| history | 0.1375 | 0.0811 | 0.4503 | 6112 |
| documentary | 0.1374 | 0.2195 | 0.1000 | 2880 |
| animation | 0.1340 | 0.0762 | 0.5535 | 6880 |
| war | 0.1007 | 0.0580 | 0.3813 | 4448 |
| western | 0.0777 | 0.0434 | 0.3699 | 2336 |
| foreign | 0.0591 | 0.0310 | 0.6190 | 672 |
| tvmovie | 0.0156 | 0.0079 | 0.5000 | 128 |

> **SubsetAccuracy 仅 0.0109 是正常的**：该指标要求一部电影的**全部** genre
> 完全命中才算对，20 个类别下难度极高，论文与课程项目中普遍很低。
> 答辩被问到时按此解释即可。

### 阈值校准（留出集：90% 训练 / 10% 定阈值）

**TAGS 模式**（完整 tag 列表，批量预测用）：

| genre | tau | scale | 留出集 F1 |
|---|---|---|---|
| action | -585.4293 | 397.6297 | 0.4132 |
| drama | -1210.2281 | 392.1524 | 0.6609 |

**TEXT 模式**（自由文本剧情描述，前端用）：**按命中词数 k 分桶校准**，k=1..8 各一套阈值。

| 桶 | 平均 tau | 平均留出集 F1 | MicroF1 | MacroF1 | 平均预测标签数 |
|---|---|---|---|---|---|
| k=1 | -11.3183 | 0.3863 | 0.4835 | 0.3837 | 7.26 |
| k=2 | -11.4086 | 0.5270 | 0.5934 | 0.5229 | 5.85 |
| k=3 | -11.4273 | 0.6171 | 0.6711 | 0.6170 | 5.00 |
| k=8 | -11.3440 | 0.7068 | **0.7679** | **0.7068** | 3.99 |

（真实平均标签数 = 2.56。留出集样本数每桶 28102 条。）

F1 随 k 单调递增，符合"证据越多判别力越强"的预期；
平均预测标签数随 k 递减并趋近真实值，说明分桶确实抑制了短输入的过度预测。

**为何必须校准**：log 域分数天然是较大负数（实测 -50 ~ -1300），
任务书默认的 `tau=0.0` 会导致预测结果**恒为空**；
且各 genre 分数分布差异极大，单一全局阈值必然顾此失彼。

**为何 TEXT 模式必须分桶**（第二阶段修复的核心缺陷）：
TEXT 模式虽用权重归一化让分数不再随输入长度线性漂移，
但**噪声水平**仍与命中词数强相关——命中 1 个词时，若该词恰是某错误 genre 的
高权重 tag，就会得到很高的分数（噪声主导）；命中 8 个词时真实 genre 的信号
被平均放大、错误 genre 相互抵消（信号主导）。两者分布形态不同，共用一套阈值必然对短输入过松。

实测证据（未分桶时）：

| 输入 | 命中词数 | 激活标签数 |
|---|---|---|
| `murder` | 1 | **14** |
| 短句（侦探…rainy city） | 2~3 | **11** |
| 长句（…brutal murders in a dark rainy city） | 6~8 | 3 |

分桶后 `murder` 降到 12 个。剩余的过度激活是**朴素贝叶斯的固有特性**而非缺陷：
常见 tag（`murder` IDF=3.15、`violence`）在多个 genre 的正样本里都高频出现，
logCond 普遍偏高，会把多个 genre 一起抬过阈值；
而稀有 tag（`catburglar`、`egomania`、`narcism`，IDF≈8.37）只激活 3~4 个。

留出集数据表明，在"最大化 F1"目标下这个宽松阈值是**最优的**
（k=1 桶 MicroF1=0.4835，收紧会使 F1 下降），故算法层保留原行为；
但为满足前端展示需求，提供了可选的 **Top-N 截断**（见下）。

### Top-N 截断（可选，仅作用于 labels 格式）

```bash
# HTTP
curl -X POST "http://localhost:8080/predict?topN=3" \
     -H "Content-Type: application/json" -d '{"text":"murder"}'
# 命令行
java -cp "$CP" com.movie.sentiment.Main predict-single --model output/model.dat \
     --format api --text "murder" --top-n 3
```

实测（输入 `murder`，原本返回 12 个）：

| topN | 返回 |
|---|---|
| 0（默认，不截断） | 12 个：Thriller 0.844、Crime 0.83、Drama 0.8282、Mystery 0.8235、Horror 0.8062 … |
| 3 | Thriller 0.844、Crime 0.83、Drama 0.8282 |
| 5 | + Mystery 0.8235、Horror 0.8062 |

截断保留的是置信度最高的前 N 个，语义正确（murder → 惊悚/犯罪/剧情/悬疑/恐怖）。

> **设计边界**：Top-N **只作用于 labels 格式**。full schema 的 `predGenres`
> 必须与 `predictResult.txt` 严格一致（三处同源要求），故不受 `topN` 影响。
> 已用测试用例验证（test-http.ps1 第 15 组）。

### IDF 降权效果（真实 tag 数据，呼应 PPT 第03页）

| tag | IDF | DF | 出现率 |
|---|---|---|---|
| womandirector | 2.6096 | 20,672 | 7.36% |
| independentfilm | 2.6189 | 20,480 | 7.29% |
| duringcreditsstinger | 2.6572 | 19,712 | 7.01% |
| basedonnovel | 3.0990 | 12,672 | 4.51% |
| murder | 3.1455 | 12,096 | 4.30% |

IDF 分布：min=2.6096，max=8.3718，均值=7.7115；**最高/最低 = 3.2 倍**。

高频泛化 tag（如 `womandirector` 出现在 7.36% 的电影里）IDF 被压到 2.61，
稀有区分 tag（如 `catburglar`，DF=64）IDF 高达 8.37 —— 降权效果明确。

### IDF 消融实验（验证 IDF 降权的真实增益）

用 `--raw-tf true` 训练一个 IDF≡1 的纯词频模型（其余完全不变），对比：

| 指标 | 纯词频（IDF≡1） | TF-IDF（标准） | 变化 |
|---|---|---|---|
| MicroPrecision | 0.2164 | 0.2318 | **+7.1%** |
| MicroRecall | 0.8141 | 0.7740 | −4.9% |
| **MicroF1** | 0.3419 | **0.3568** | **+4.4%** |
| MacroF1 | 0.2405 | 0.2511 | **+4.4%** |
| HammingLoss | 0.4028 | 0.3588 | **−10.9%（改善）** |
| SubsetAccuracy | 0.0093 | 0.0109 | +17.2% |

**增益来源分析**：precision +7.1%、HammingLoss 改善 10.9%，recall −4.9%。
这个模式**完全符合 IDF 原理**——IDF 压住高频泛化 tag，模型不再因几个常见 tag
就到处乱贴类型，故**误报显著减少**；代价是少数靠泛化 tag 命中的情况丢失，召回略降。

**per-genre 变化**：20 个类型中 **18 个 F1 提升**，2 个略降。

| 提升最多 | 纯词频 → TF-IDF |
|---|---|
| romance | 0.3668 → 0.3866 (+0.0198) |
| documentary | 0.1183 → 0.1374 (+0.0191) |
| crime | 0.2759 → 0.2948 (+0.0189) |
| horror | 0.2235 → 0.2403 (+0.0168) |

| 下降 | 纯词频 → TF-IDF | support |
|---|---|---|
| foreign | 0.0617 → 0.0591 (−0.0026) | 672 |
| western | 0.0834 → 0.0777 (−0.0057) | 2336 |

两个下降项恰是**样本最少的类型**：样本少 → IDF 估计不稳定，
几个 tag 的权重变化即足以翻转判定结果。净收益为正（MacroF1 +4.4%）。

**诚实结论**：4.4% 的提升幅度不大，远不及 PPT 里 78%→85% 的跨度。
原因是数据特性不同——PPT 例子中"的""很"是**纯虚词**、零信息量，
去除收益巨大；而本项目的 tag 是**人工标注的主题词**，
像 `murder`、`violence` 即使高频本身也有区分度，故 IDF 的降权空间天然较小。

复现：
```bash
java -cp "$CP" com.movie.sentiment.Main train --input ../movies-from-kggle.txt \
     --model output/model-rawtf.dat --raw-tf true --calibrate-samples 20000
java -cp "$CP" com.movie.sentiment.Main predict --model output/model-rawtf.dat \
     --input ../movies-to-be-predicted.txt \
     --out output/predictResult-rawtf.txt --metrics output/metrics-rawtf.json
```

> `--raw-tf` 仅用于消融对比，正式训练/预测一律不加。

---

## 九、接口测试

```powershell
cd java代码
# 先启动服务：java -cp "$CP" com.movie.sentiment.Main serve --model output/model.dat --port 8090
powershell -ExecutionPolicy Bypass -File test-http.ps1 8090
```

覆盖 15 组场景、51 项断言，**全部通过**：
健康检查、标签列表、POST+tags、POST+text、full schema 八字段、
GET+query、CORS 预检、空输入 400、full schema 错误 200+error、
全 OOV、404、405、中文 UTF-8 字节校验、8 并发一致性、
Top-N 截断（labels 生效）、Top-N 不影响 full schema（三处同源）。

---

## 十、无 Maven 的手动 classpath 方案（重要，实测踩坑记录）

`hadoop-client` 的运行期传递依赖较多，**只放 hadoop-common + hdfs-client
会在运行期报 `NoClassDefFoundError`**。本项目 `lib/` 里已备齐实测可用的 27 个 jar：

**必需（编译期 API）：**
```
hadoop-common-3.3.6.jar
hadoop-hdfs-client-3.3.6.jar
hadoop-mapreduce-client-core-3.3.6.jar     ← Module-MR 需要
hadoop-auth-3.3.6.jar
```

**必需（运行期，缺任一个都会 NoClassDefFoundError）：**
```
woodstox-core-5.3.0.jar          ← Configuration 解析 XML 需要
stax2-api-4.2.1.jar              ← woodstox 依赖
commons-configuration2-2.1.1.jar ← Configuration 需要
commons-logging-1.1.3.jar
commons-collections-3.2.2.jar
slf4j-api-1.7.36.jar             ← Configuration 静态初始化需要
slf4j-nop-1.7.36.jar             ← 日志实现（nop = 静默，避免刷屏）
commons-compress-1.21.jar        ← fs.create 写文件需要
hadoop-shaded-guava-1.1.1.jar
hadoop-shaded-protobuf_3_7-1.1.1.jar  ← hdfs RPC 需要
hadoop-annotations-3.3.6.jar
htrace-core4-4.1.0-incubating.jar
commons-cli-1.2.jar
commons-lang3-3.12.0.jar
commons-io-2.8.0.jar
commons-math3-3.1.1.jar
protobuf-java-2.5.0.jar
guava-11.0.2.jar
jackson-core-asl-1.9.13.jar
jackson-mapper-asl-1.9.13.jar
re2j-1.1.jar
gson-2.2.4.jar
kerb-simplekdc-1.0.1.jar
```

> 注：`jackson-*-asl` 是 Hadoop 内部 RPC 序列化用的，
> **不属于**我们代码引入的第三方 JSON 库（我们的 JSON 全部手写 StringBuilder），
> 不违反"禁止第三方 JSON 库"的要求。

### 集群上运行时

若用 `hadoop jar` 提交作业，Hadoop 会自动把自身 jar 放进 classpath，
**不需要**手动指定上面这些，只要我们的 jar 里打了自己的 class 即可。

若要**脱离集群独立运行后端服务**（比如成员4 联调时只想跑 HTTP 服务），
需要把上述 jar 一起放进 classpath。

### JDK 23 上运行的额外参数

Hadoop 3.3.6 用了 `javax.security.auth.Subject.getSubject()`，
该方法在 JDK 18+ 已废弃、JDK 23 上默认抛
`UnsupportedOperationException: getSubject is supported only if a security manager is allowed`。
若开发机是 JDK 18 及以上，运行时加：

```bash
java -Djava.security.manager=allow -cp "$CP" com.movie.sentiment.Main ...
```

**课程集群若为 JDK 8 则不需要这个参数**。
本机实测：加此参数后 `file://` 读写全链路成功、无降级；
不加时会触发降级（仍能正确回退本地路径完成任务）。

### Windows 平台的 file:// 写限制

Windows 上 Hadoop 的 `file://` **写**操作需要 `winutils.exe`
（报错 `HADOOP_HOME and hadoop.home.dir are unset`），**读**不需要。
本项目实测：`file://` 读取成功，写入时触发降级到 java.io 本地写，
文件正确产出。**Linux 集群上无此限制**，读写均正常。

---

## 十一、给成员4 的联调要点

1. **模型已训练好**，`output/model.dat`（1.83 MB）含完整校准参数，直接用即可；
2. **启动服务**：`java -cp "$CP" com.movie.sentiment.Main serve --model output/model.dat --port 8080`
3. **默认返回 labels 格式**，与她规范完全一致，前端无需改动；
4. **想要详细字段**时加 `?schema=full`；
5. **她的后端若是 Java**，可直接调 `Main.predictForApi(model, text)` 免 HTTP 开销；
6. 详细字段清单与调用示例见同目录 **《给成员4的对接说明.md》**。

---

## 十二、鲁棒性 checklist（全部满足）

- ✅ train / 批量 predict / 后端 `/predict` 三处共用同一 `DataParser` 与同一 normalize
- ✅ OOV 静默跳过；全 OOV 返回空 predGenres + confidence=0，不崩溃、不外泄异常栈
- ✅ 脏行跳过 + WARN + 计数（WARN 限流前 5 条，超出汇总，避免刷屏），结束打印跳过行数
- ✅ CRLF 的 `\r` 处理；统一 UTF-8 读写
- ✅ 加1平滑不可省；全程 log 域防下溢；softmax 前减 max 平移防溢出；`Math.max(p, 1e-12)` 防 log(0)
- ✅ 后端模型单例、启动加载一次，不每请求读盘
- ✅ HDFS 连接失败自动 fallback 本地 file 并 WARN；`--no-fallback` 可关闭
- ✅ 模型缺失时报错信息清晰（指出期望路径 + `--model`/`--model-hdfs` 用法 + 4 条排查建议）
- ✅ `predictResult.txt` 每行字段 == 后端 full schema == predict-single 输出（复用同一序列化方法）
- ✅ 无硬编码 NameNode 地址 / 绝对路径 / 端口（全走占位参数 + 环境变量）

---

## 十三、开发过程中修复的真实缺陷（答辩可作为工程质量说明）

| # | 问题 | 后果 | 修法 |
|---|---|---|---|
| 1 | 固定 `tau=0.0` | log 域分数恒为负，**预测结果永远为空** | 训练期逐 genre 最优 F1 阈值校准 |
| 2 | 训练集内校准 | 记忆效应导致过度预测（平均 8.53 标签 vs 真实 2.58） | 改用留出集 90%/10% 划分 |
| 3 | TF 重复计数 | tag 出现 k 次时权重算成 k²·IDF | 先统计唯一词频再乘 IDF |
| 4 | genre 顺序错位 | 临时模型 LinkedHashMap 插入序变化，阈值张冠李戴（action 阈值算成 -7857） | 强制临时模型沿用全量 genre 顺序 |
| 5 | 统一绝对偏移 | 稀有类别阈值被推成正数，**7 个 genre 的 F1 全为 0** | 改按 scale 成比例的相对偏移，目标取 Micro/Macro F1 均值 |
| 6 | 短句激活 16/20 标签 | score 幅度与命中词数正相关，长列表阈值用在短文本上失配 | TEXT 模式改权重归一化打分（先验 + 加权平均对数似然） |
| 7 | 全 OOV 激活全部标签 | 零命中时回退纯先验（≈-2.83），远高于校准阈值（≈-10.5），sigmoid 饱和为 1.0 | 打分入口加"无有效证据"短路，返回空结果 |
| 8 | 校准留出集负容量 | 小样本时 `Math.max(1000, 10%)` 超过总数，`new ArrayList(-600)` 抛异常 | 对总数取上限并保证临时训练集非空 |
| 9 | 训练侧 S(g,t) 重复累加 | 文档内重复 tag 贡献 k²·TF·IDF，与预测侧口径不一致 | 文档内 (tag,genre) 组合去重（影响 56/50000 文档，指标不变） |
| 10 | MR 只读单个 part 文件 | Reducer 数 >1 时**静默丢失大部分统计量**，模型完全错误 | 改为遍历目录下全部 part 文件 |
| 11 | `[CONFIG]` 打到 stdout | **污染 `predict-single` 的纯 JSON 输出**，成员4 用管道 `ConvertFrom-Json` 直接解析失败 | 配置信息改走 stderr，stdout 只留纯 JSON（已实测单行、首字符 `{`） |
| 12 | 留出集划分负容量 | 小样本时 `Math.max(1000, 10%)` 超过总数，`new ArrayList(-600)` 抛异常 | 对总数取上限并保证临时训练集 ≥50%；样本 <20 时跳过校准并提示 |
| 13 | 短输入过度激活 | TEXT 模式共用一套阈值，输入 `murder`（命中 1 词）**激活 14/20 个标签**（真实均值 2.58） | 按命中词数 k=1..8 **分桶校准**，各桶平均标签数从 7.26 递减至 3.99；另加可选 `topN` 截断 |
| 14 | 训练侧 S(g,t) 重复累加 | 文档内重复 tag 贡献 k²·TF·IDF，**与预测侧口径不一致** | 文档内 (tag,genre) 组合去重（影响 56/50000 文档，指标不变，但两侧口径已统一） |
| 15 | MR 词表文件用 HdfsIO 写出 | HdfsIO 静态 conf（localhost:9000）与作业 conf（用户 `-D fs.defaultFS`）**不是同一个文件系统**，集群上作业 2 的 Mapper 读不到词表 | 改为直接用 driver 已持有的 `FileSystem fs` 写出，与两个作业同一 conf |
| 16 | MR 模型写出用默认 NameNode | 同类问题：模型会写到 localhost:9000 而非用户指定集群，**静默写错位置** | 在 `run()` 开头把 HdfsIO 的 NameNode 同步为作业的 `conf.get("fs.defaultFS")` |

其中 #7、#10、#11 都是"测试跑通但语义错误"的隐蔽缺陷，靠端到端测试与代码审查发现；
#13 是靠**对比不同长度输入的激活标签数**发现的（单一用例测试无法暴露）；
#15、#16 靠**审查 Configuration 传递路径**发现（本机无集群，只有真实集群才会暴露）。

**#13 的完整分析**（答辩可能被追问）：分桶后 `murder` 仍激活 12 个，这不是缺陷而是
朴素贝叶斯的固有特性——常见 tag（`murder` IDF=3.15）在多个 genre 的正样本里都高频出现，
logCond 普遍偏高，会把多个 genre 一起抬过阈值；稀有 tag（`catburglar` IDF=8.37）只激活 3~4 个。
留出集数据显示，在"最大化 F1"目标下这个宽松阈值**本身是最优的**
（k=1 桶 MicroF1=0.4835，收紧会使 F1 下降），故算法层保留原行为、
展示层提供可选 `topN` 截断（实测 `topN=3` → Thriller/Crime/Drama，语义正确）。
