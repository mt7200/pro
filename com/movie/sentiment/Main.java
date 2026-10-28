package com.movie.sentiment;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * 命令行入口：子命令分发与参数解析（手写，不依赖第三方 CLI 库）。
 *
 * <p>第二阶段在【不改动】第一阶段解析与算法逻辑的前提下，叠加：
 * HDFS 读写（Module-HDFS）、批量产物标准化（Module-BATCH）、HTTP 后端（Module-SERVER）。
 *
 * <p>子命令：
 * <pre>
 * train          --input ${HDFS_TRAIN} --model-hdfs ${HDFS_MODEL} --model-local ${LOCAL_MODEL}
 * predict        --model &lt;local|hdfs&gt; --input ${HDFS_PREDICT} --out ${PREDICT_RESULT}
 *                --metrics ${METRICS_OUT} [--format jsonl|tsv]
 * predict-single --model &lt;local|hdfs&gt; --text "..." [--format full|api|labels]
 * serve          --model &lt;local&gt; | --model-hdfs &lt;hdfs&gt; [--port ${SERVER_PORT}]
 * </pre>
 *
 * <p><b>路径规则</b>：所有路径既可写 HDFS URI（hdfs://nn:9000/...）、file:// URI，
 * 也可写纯本地路径（output/model.dat），由 {@link HdfsIO} 统一路由，无硬编码绝对路径。
 */
public class Main {

    // ─────────── 运行期环境占位符默认值（对应任务书 ${...}，均可命令行/环境变量覆盖） ───────────

    /** ${NAMENODE_URI} 默认值 */
    public static final String DEF_NAMENODE = HdfsIO.DEFAULT_NAMENODE_URI;
    /** ${HDFS_TRAIN} 默认值 */
    public static final String DEF_HDFS_TRAIN = "/input/movies-from-kggle.txt";
    /** ${HDFS_PREDICT} 默认值 */
    public static final String DEF_HDFS_PREDICT = "/input/movies-to-be-predicted.txt";
    /** ${HDFS_MODEL} 默认值 */
    public static final String DEF_HDFS_MODEL = "/output/model.dat";
    /** ${LOCAL_MODEL} 默认值 */
    public static final String DEF_LOCAL_MODEL = "output/model.dat";
    /** ${PREDICT_RESULT} 默认值（组长钦定文件名） */
    public static final String DEF_PREDICT_RESULT = "output/predictResult.txt";
    /** ${METRICS_OUT} 默认值 */
    public static final String DEF_METRICS_OUT = "output/metrics.json";
    /** ${SERVER_PORT} 默认值 */
    public static final int DEF_PORT = 8080;

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            printUsage();
            return;
        }

        String cmd = args[0].toLowerCase(Locale.ROOT);
        Map<String, String> opts = parseArgs(args, 1);
        // 所有子命令都先初始化 HDFS 配置（NameNode / 用户 / 降级开关）
        configureHdfs(opts);

        switch (cmd) {
            case "train":
                doTrain(opts);
                break;
            case "calibrate":
                doCalibrate(opts);
                break;
            case "predict":
                doPredict(opts);
                break;
            case "predict-single":
                doPredictSingle(opts);
                break;
            case "serve":
            case "server":
                doServe(opts);
                break;
            default:
                System.err.println("未知子命令: " + cmd);
                printUsage();
                System.exit(2);
        }
    }

    // ──────────────────── 参数解析 ────────────────────

    /**
     * 手写参数解析：支持 {@code --key value} 与 {@code --flag}（值为 "true"）。
     * 不引入 args4j 等第三方 CLI 库（规范禁止）。
     */
    private static Map<String, String> parseArgs(String[] args, int start) {
        Map<String, String> map = new LinkedHashMap<>();
        for (int i = start; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                String key = args[i].substring(2);
                // 下一个 token 若不是选项，则作为当前 key 的值；否则视为布尔开关
                String val = (i + 1 < args.length && !args[i + 1].startsWith("--"))
                        ? args[++i] : "true";
                map.put(key, val);
            }
        }
        return map;
    }

    /**
     * 取参数值，优先级：命令行 &gt; 环境变量 &gt; 默认值。
     * 这样 ${NAMENODE_URI} 等既能在命令行传，也能用环境变量注入，避免硬编码。
     */
    private static String opt(Map<String, String> opts, String key, String envKey, String def) {
        String v = opts.get(key);
        if (v != null && !v.trim().isEmpty()) {
            return v.trim();
        }
        String env = (envKey == null) ? null : System.getenv(envKey);
        if (env != null && !env.trim().isEmpty()) {
            return env.trim();
        }
        return def;
    }

    /** 初始化 HDFS 配置（NameNode 地址、Hadoop 用户、降级开关）。 */
    private static void configureHdfs(Map<String, String> opts) {
        String nameNode = opt(opts, "namenode", "NAMENODE_URI", DEF_NAMENODE);
        String user = opt(opts, "hadoop-user", "HADOOP_USER", HdfsIO.DEFAULT_HADOOP_USER);
        // --no-fallback 存在即禁用降级（用于严格验证集群连通性的场景）
        boolean fallback = !opts.containsKey("no-fallback");
        HdfsIO.configure(nameNode, user, fallback);
        // 【重要】配置信息必须走 stderr 而非 stdout：
        // predict-single 的 stdout 约定输出【纯 JSON】，供成员4 用管道直接取用
        // （如 `java ... predict-single ... > result.json` 或 PowerShell 的 ConvertFrom-Json）。
        // 若把 [CONFIG] 打到 stdout，会混入 JSON 文件导致解析失败（实测踩过此坑）。
        System.err.println("[CONFIG] NameNode=" + nameNode
                + (user.isEmpty() ? "" : "  user=" + user)
                + "  降级回退=" + (fallback ? "开启" : "已禁用(--no-fallback)"));
    }

    // ──────────────────── train 子命令 ────────────────────

    /**
     * 训练：HDFS/本地读训练集 → 建词汇表/IDF → 训 NB → 阈值校准 → 双写模型。
     *
     * <p>【双写策略】对齐 Module-HDFS 规范：模型同时写 ${HDFS_MODEL} 与 ${LOCAL_MODEL}。
     * 数据集读取走 HDFS 是硬性要求；模型落本地副本不违反该要求（模型是产物非输入数据集），
     * 且让后端默认读本地更稳，降低联调变量。
     */
    private static void doTrain(Map<String, String> opts) throws Exception {
        // 输入默认走 HDFS 路径（${HDFS_TRAIN}）；若集群不可用，HdfsIO 会自动降级到同名本地路径
        String inputPath = opt(opts, "input", "HDFS_TRAIN", DEF_HDFS_TRAIN);
        String modelHdfs = opt(opts, "model-hdfs", "HDFS_MODEL", DEF_HDFS_MODEL);
        String modelLocal = opt(opts, "model-local", "LOCAL_MODEL", DEF_LOCAL_MODEL);
        // --model 作为简写：只写单一路径时兼容第一阶段用法
        String modelSingle = opts.get("model");
        boolean userTau = opts.containsKey("tau");
        double tau = Double.parseDouble(opts.getOrDefault("tau", "0.0"));
        int calibSamples = Integer.parseInt(opts.getOrDefault("calibrate-samples", "20000"));
        boolean noCalibrate = Boolean.parseBoolean(opts.getOrDefault("no-calibrate", "false"));

        System.out.println("══════ 训练开始 ══════");
        System.out.println("训练集输入: " + inputPath);

        // 1) 解析训练数据（DataParser 内部已走 HdfsIO，支持 hdfs:// 与本地）
        DataParser parser = new DataParser();
        List<Sample> trainSamples = parser.parseFile(inputPath);
        int N = trainSamples.size();
        if (N == 0) {
            throw new IllegalStateException("训练集为空：请检查路径是否正确、文件是否存在。路径=" + inputPath
                    + "\n  提示：HDFS 不可用时会降级查找本地同名相对路径 "
                    + HdfsIO.toLocalFallback(inputPath));
        }
        System.out.println("训练样本数 N = " + N + "  (跳过脏行: " + parser.skippedLines + ")");

        // 2) 构建词汇表与 IDF（此统计量可在 Reduce 端聚合：DF 为可交换可结合的计数）
        //    --raw-tf 为【消融实验】开关：把 IDF 全置 1，退化为纯原始词频，
        //    用于量化 IDF 降权的实际增益（对应 PPT 第04页对比叙事）。正式训练不要加。
        boolean rawTfOnly = Boolean.parseBoolean(opts.getOrDefault("raw-tf", "false"));
        VocabIdfBuilder vocabBuilder = new VocabIdfBuilder();
        vocabBuilder.build(trainSamples, rawTfOnly);
        System.out.println("词汇表大小 |V| = " + vocabBuilder.vocabArr.length
                + (rawTfOnly ? "   【消融模式：IDF≡1，纯原始词频】" : "   【标准模式：TF-IDF】"));

        // 3) 训练 NB 模型（先用传入 tau 占位，随后校准覆盖）
        NbTrainer trainer = new NbTrainer();
        ModelParams model = trainer.train(trainSamples, vocabBuilder, tau, inputPath);
        System.out.println("标签种类数 |G| = " + model.G);

        // 4) 逐 genre 阈值校准
        //    必要性：log 域分数天然为较大负数（实测 -50 ~ -1200），固定 tau=0.0
        //    会导致预测结果恒为空；且各 genre 分数分布差异大，需分别定阈值。
        if (!noCalibrate) {
            // 4.1) 完整 tag 列表输入的阈值校准（批量预测用）
            //      留出集 90%/10% 划分，避免训练集内校准导致的过度预测
            NbTrainer.calibrate(model, trainSamples, calibSamples);
            // 4.2) 自由文本输入的阈值校准（后端剧情描述用）
            //      按【命中词数 k 分桶】各校准一套阈值：命中 1 个词与命中 8 个词的
            //      分数噪声水平差异极大，共用一套阈值会让短输入严重过度预测
            //      （实测未分桶时输入 "murder" 会激活 14/20 个标签，而真实平均仅 2.58 个）。
            NbTrainer.calibrateTextProfileByK(model, trainSamples, calibSamples);
        } else if (userTau) {
            System.out.println("[校准] 已按 --no-calibrate 跳过，使用用户指定全局 τ=" + tau);
        } else {
            System.out.println("[校准] 已跳过（--no-calibrate），全局 τ=" + tau
                    + "；注意 log 域分数为负，τ=0.0 将无法激活任何标签");
        }

        // 5) 保存模型（含校准阈值与尺度）
        if (modelSingle != null && !modelSingle.trim().isEmpty()) {
            // 简写模式：只写 --model 指定的单一路径（兼容第一阶段用法与调试）
            ModelSerializer.save(model, modelSingle.trim());
            System.out.println("[MODEL] 模型已保存至: " + modelSingle.trim());
        } else {
            // 双写模式（Module-HDFS 规范）：主路径通常为 HDFS，副本为本地
            int ok = ModelSerializer.dualWrite(model, modelHdfs, modelLocal);
            if (ok == 0) {
                throw new IOException("模型未写入任何路径，请检查 --model-hdfs / --model-local 参数");
            }
        }

        // ── 演示输出（供答辩截图）──

        // 各 genre 频率（一次遍历统计，避免 O(N×G) 重复扫描）
        System.out.println("\n── genre 频次统计（含逐 genre 校准阈值）──");
        Map<String, Integer> freqForDemo = new LinkedHashMap<>();
        for (Sample s : trainSamples) {
            for (String sg : s.genres) {
                // 此统计量可在 Reduce 端聚合：freq(g) 为可交换可结合的计数
                freqForDemo.merge(sg, 1, Integer::sum);
            }
        }
        for (int i = 0; i < model.G; i++) {
            String g = model.genreArr[i];
            int freq = freqForDemo.getOrDefault(g, 0);
            String tauStr = model.hasPerGenreCalibration()
                    ? String.format(Locale.ROOT, "%.4f", model.tauPerGenre[i]) : "未校准";
            System.out.printf("  %-20s freq=%-8d logPrior=%-10.4f tau=%s%n",
                    g, freq, model.logPrior[i], tauStr);
        }

        System.out.println("\n── TF-IDF 权重最高的 15 个 tag ──");
        printTopTfidf(trainSamples, vocabBuilder, 15);

        // 呼应 PPT 第03页 IDF 降权演示，但用真实 tag 数据而非"好吃/难吃"示意
        System.out.println("\n── 高频但 IDF 低的 tag（呼应 PPT IDF 降权）──");
        printHighDfLowIdf(vocabBuilder, 15);

        System.out.println("\n══════ 训练完成 ══════");
    }

    // ──────────────────── calibrate 子命令 ────────────────────

    /**
     * 对已有模型补做阈值校准（典型用途：MapReduce 产出的模型）。
     *
     * <p>MR 训练只在 driver 端汇聚计数量，无法在集群上做"留出集反复打分搜阈值"，
     * 所以 MrTrainDriver 写出的模型 tauPerGenre 为空，预测时退化为全局 tau=0.0。
     * 而 log 域分数是较大负数，tau=0.0 不会激活任何标签。
     *
     * <p>本命令不重算词表 / IDF / 概率，只复用与单机 train 完全相同的校准实现：
     * 90%/10% 留出切分 + 逐 genre 绝对阈值 + TEXT 按命中词数分桶，
     * 校准结果就地写回模型（本地 + 可选 HDFS）。
     */
    private static void doCalibrate(Map<String, String> opts) throws Exception {
        String srcModel = opts.containsKey("model")
                ? opts.get("model").trim() : opts.get("model-hdfs");
        if (srcModel == null || srcModel.trim().isEmpty()) {
            srcModel = DEF_LOCAL_MODEL;
        }
        String inputPath = opt(opts, "input", "HDFS_TRAIN", DEF_HDFS_TRAIN);
        String outLocal = opt(opts, "model-local", "LOCAL_MODEL", "output/model-calibrated.dat");
        String outHdfs = opts.get("out-model-hdfs");
        int calibSamples = Integer.parseInt(opts.getOrDefault("calibrate-samples", "20000"));

        System.out.println("══════ 阈值校准开始 ══════");
        System.out.println("待校准模型 : " + srcModel);
        System.out.println("校准数据集 : " + inputPath);
        System.out.println("本地输出   : " + outLocal
                + (outHdfs == null ? "" : "   HDFS输出: " + outHdfs));

        ModelParams model = loadModelOrFail(srcModel);
        System.out.printf(Locale.ROOT, "模型已加载: |V|=%d |G|=%d N=%d 逐genre校准=%s%n",
                model.V, model.G, model.N,
                model.hasPerGenreCalibration() ? "已启用" : "未启用");

        List<Sample> samples = new DataParser().parseFile(inputPath);
        System.out.println("校准样本数 N = " + samples.size());
        if (samples.size() < 20) {
            throw new IllegalStateException("校准样本不足 20 条，无法做留出校准：路径=" + inputPath);
        }

        NbTrainer.calibrate(model, samples, calibSamples);
        NbTrainer.calibrateTextProfileByK(model, samples, calibSamples);

        if (ModelSerializer.dualWrite(model, outHdfs, outLocal) == 0) {
            throw new IOException("模型未写入任何路径，请检查 --model-local / --out-model-hdfs");
        }

        System.out.println("\n── 校准后的逐 genre 阈值 ──");
        for (int i = 0; i < model.G; i++) {
            System.out.printf(Locale.ROOT, "  %-20s tau=%s%n", model.genreArr[i],
                    model.hasPerGenreCalibration()
                            ? String.format(Locale.ROOT, "%.4f", model.tauPerGenre[i]) : "未校准");
        }
        System.out.println("\n══════ 校准完成 ══════");
    }

    private static void printTopTfidf(List<Sample> samples, VocabIdfBuilder vb, int topK) {
        int V = vb.vocabArr.length;
        double[] sumTfidf = new double[V];   // 所有样本 TF-IDF 权重之和（此统计量可在 Reduce 端聚合）
        for (Sample s : samples) {
            for (String t : s.tags) {
                Integer idx = vb.vocabIndex.get(t);
                if (idx != null) {
                    // TFIDF(t,d) = TF(t,d) × IDF(t)
                    sumTfidf[idx] += VocabIdfBuilder.tf(t, s) * vb.idf[idx];
                }
            }
        }
        Integer[] indices = new Integer[V];
        for (int i = 0; i < V; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(sumTfidf[b], sumTfidf[a]));
        for (int i = 0; i < Math.min(topK, V); i++) {
            int idx = indices[i];
            System.out.printf(Locale.ROOT, "  %-22s TF-IDF总和=%-12.4f DF=%-7d IDF=%.4f%n",
                    vb.vocabArr[idx], sumTfidf[idx], vb.df[idx], vb.idf[idx]);
        }
    }

    private static void printHighDfLowIdf(VocabIdfBuilder vb, int topK) {
        int V = vb.vocabArr.length;
        Integer[] indices = new Integer[V];
        for (int i = 0; i < V; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Integer.compare(vb.df[b], vb.df[a]));
        System.out.println("  （DF 最高 = IDF 最低，即被降权最狠的泛化 tag）：");
        for (int i = 0; i < Math.min(topK, V); i++) {
            int idx = indices[i];
            System.out.printf(Locale.ROOT, "  %-22s DF=%-7d IDF=%.4f%n",
                    vb.vocabArr[idx], vb.df[idx], vb.idf[idx]);
        }
    }

    // ──────────────────── predict 子命令 ────────────────────

    /**
     * 批量预测：HDFS/本地读预测集 → 打分 → 写 ${PREDICT_RESULT}（JSON Lines）+ ${METRICS_OUT}。
     *
     * <p><b>产物格式钉死</b>：predictResult.txt 扩展名为 .txt 但内容是 JSON Lines，
     * 每行字段与后端 /predict 的 full schema 完全一致（复用 {@link JsonUtil#buildView}
     * + {@link JsonUtil#predictionLineJson}，三处同源杜绝漂移）。
     *
     * <p><b>为何默认 jsonl</b>：成员4 前端在实时接口不可用时可直接读该文件兜底 parse，
     * 字段与实时响应一致，前端无需写第二套解析逻辑；tsv 仅供老师/组长人工查看。
     */
    private static void doPredict(Map<String, String> opts) throws Exception {
        // 模型默认读本地副本（后端同源，避免每次批量预测都连 HDFS）；可用 --model-hdfs 指定集群路径
        String modelPath = opt(opts, "model", "LOCAL_MODEL", DEF_LOCAL_MODEL);
        String modelHdfs = opts.get("model-hdfs");
        if (modelHdfs != null && !modelHdfs.trim().isEmpty()) {
            modelPath = modelHdfs.trim();
        }
        String inputPath = opt(opts, "input", "HDFS_PREDICT", DEF_HDFS_PREDICT);
        String outPath = opt(opts, "out", "PREDICT_RESULT", DEF_PREDICT_RESULT);
        String metricsPath = opt(opts, "metrics", "METRICS_OUT", DEF_METRICS_OUT);
        // --format jsonl|tsv，默认 jsonl（理由见方法注释）
        String format = opts.getOrDefault("format", "jsonl").toLowerCase(Locale.ROOT);
        if (!"jsonl".equals(format) && !"tsv".equals(format)) {
            System.err.println("[WARN] 未知 --format=" + format + "，回退为 jsonl");
            format = "jsonl";
        }

        System.out.println("══════ 批量预测开始 ══════");
        System.out.println("模型路径  : " + modelPath);
        System.out.println("预测集路径: " + inputPath);
        System.out.println("结果输出  : " + outPath + "  (格式=" + format + ")");

        // 加载模型（严禁在预测阶段重建词汇表或重算 IDF/概率）
        ModelParams model = loadModelOrFail(modelPath);
        System.out.printf(Locale.ROOT,
                "模型已加载: |V|=%d |G|=%d N=%d τ=%.4f  逐genre校准=%s  文本校准=%s%n",
                model.V, model.G, model.N, model.tau,
                model.hasPerGenreCalibration() ? "已启用" : "未启用",
                model.hasTextCalibration() ? "已启用" : "未启用");

        NbPredictor predictor = buildPredictor(model, opts);


        // 解析预测数据（与训练共用同一 DataParser 与同一 normalize 规则）
        DataParser parser = new DataParser();
        List<Sample> predictSamples = parser.parseFile(inputPath);
        System.out.println("预测样本数: " + predictSamples.size()
                + "  (跳过脏行: " + parser.skippedLines + ")");

        // 评估收集容器（仅对带 ground-truth 的样本计算指标，否则标注 null）
        List<Set<String>> allTrue = new ArrayList<>();
        List<Set<String>> allPred = new ArrayList<>();
        List<JsonUtil.PredictionView> views = new ArrayList<>();   // 用于打印前 5 条对照表

        // 通过 HdfsIO 流式写出，避免 14 万条全部驻留内存再一次性写
        HdfsIO.mkdirsParent(outPath);
        try (OutputStream os = HdfsIO.createStream(outPath);
             BufferedWriter writer = new BufferedWriter(
                     new OutputStreamWriter(os, StandardCharsets.UTF_8), 1 << 16)) {

            int idx = 0;
            for (Sample s : predictSamples) {
                // 批量预测走 TAGS 模式（完整 tag 列表），与训练期校准口径一致
                NbPredictor.PredictionResult pr = predictor.predictByTags(Arrays.asList(s.tags));

                List<String> tags = Arrays.asList(s.tags);
                List<String> trueG = (s.genres != null && s.genres.length > 0)
                        ? Arrays.asList(s.genres) : null;

                // ★ 三处同源：统一走 buildView + predictionLineJson
                JsonUtil.PredictionView v = JsonUtil.buildView(model, idx, tags, trueG, pr);
                writer.write("tsv".equals(format)
                        ? JsonUtil.predictionTsvLine(v)
                        : JsonUtil.predictionLineJson(v));
                writer.newLine();

                if (views.size() < 5) {
                    views.add(v);
                }

                // 收集评估数据（trueGenres 为 null 表示该行无 ground-truth，不参与指标计算）
                allTrue.add(trueG == null ? null : new HashSet<>(trueG));
                allPred.add(new HashSet<>(v.predGenres));
                idx++;
            }
        }
        System.out.println("预测结果已写出: " + outPath);

        // ── 评估（仅对有 ground-truth 的样本；否则输出 null 并标注跳过）──
        List<Set<String>> evalTrue = new ArrayList<>();
        List<Set<String>> evalPred = new ArrayList<>();
        for (int i = 0; i < allTrue.size(); i++) {
            if (allTrue.get(i) != null) {
                evalTrue.add(allTrue.get(i));
                evalPred.add(allPred.get(i));
            }
        }
        if (!evalTrue.isEmpty()) {
            Metrics.EvalResult eval = Metrics.evaluate(evalTrue, evalPred, model.genreArr, model.genreIndex);
            // metrics 同样走 HdfsIO，可写 HDFS 或本地
            HdfsIO.writeText(metricsPath, JsonUtil.metricsJson(eval));
            System.out.println("评估指标已写出: " + metricsPath + "  (参与评估样本=" + evalTrue.size() + ")");
            System.out.printf(Locale.ROOT,
                    "  SubsetAccuracy=%.4f  MicroP=%.4f  MicroR=%.4f  MicroF1=%.4f  MacroF1=%.4f  HammingLoss=%.4f%n",
                    eval.subsetAccuracy, eval.microPrecision, eval.microRecall,
                    eval.microF1, eval.macroF1, eval.hammingLoss);
            // 平均标签数：用于判断是否过度/不足预测（答辩常被追问）
            double avgPred = 0, avgTrue = 0;
            for (int i = 0; i < evalPred.size(); i++) {
                avgPred += evalPred.get(i).size();
                avgTrue += evalTrue.get(i).size();
            }
            System.out.printf(Locale.ROOT, "  平均预测标签数=%.2f  平均真实标签数=%.2f%n",
                    avgPred / evalPred.size(), avgTrue / evalTrue.size());
        } else {
            System.out.println("预测集不含 ground-truth 标签，跳过评估（metrics 各项为 null）。");
            HdfsIO.writeText(metricsPath, "{\"n\":" + predictSamples.size()
                    + ",\"subsetAccuracy\":null,\"microF1\":null,\"macroF1\":null,"
                    + "\"hammingLoss\":null,\"perGenre\":{}}");
            System.out.println("评估指标（全 null）已写出: " + metricsPath);
        }

        // 打印前 5 条对照表（便于肉眼核验与截图进 PPT）
        System.out.println("\n── 前 5 条样本预测对照表 ──");
        System.out.printf("%-6s %-40s %-32s %-32s %s%n",
                "idx", "tags(截断)", "trueGenres", "predGenres", "conf");
        System.out.println(String.join("", Collections.nCopies(140, "-")));
        for (JsonUtil.PredictionView v : views) {
            System.out.printf(Locale.ROOT, "%-6d %-40s %-32s %-32s %.4f%n",
                    v.idx,
                    truncate(String.join(",", v.tags), 38),
                    v.trueGenres == null ? "N/A" : truncate(String.join(",", v.trueGenres), 30),
                    truncate(String.join(",", v.predGenres), 30),
                    v.confidence);
        }

        System.out.println("\n══════ 批量预测完成 ══════");
    }

    /**
     * 加载模型；失败时给出清晰的排错提示（指出期望路径与正确用法），不抛裸异常栈。
     */
    private static ModelParams loadModelOrFail(String modelPath) throws Exception {
        try {
            return ModelSerializer.load(modelPath);
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("模型文件反序列化失败（可能不是本程序生成的 model.dat）: "
                    + modelPath, e);
        } catch (Exception e) {
            System.err.println("══════ 模型加载失败 ══════");
            System.err.println("  期望路径: " + modelPath);
            System.err.println("  错误信息: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            System.err.println("  排查建议:");
            System.err.println("    1) 确认已先执行 train 生成模型；");
            System.err.println("    2) 本地模型用 --model <路径>，集群模型用 --model-hdfs hdfs://.../model.dat；");
            System.err.println("    3) HDFS 不可用时程序会自动降级查找本地同名相对路径: "
                    + HdfsIO.toLocalFallback(modelPath));
            System.err.println("    4) 当前 NameNode=" + HdfsIO.nameNodeUri()
                    + "（可用 --namenode 覆盖，需向成员2确认实际地址与端口）");
            throw e;
        }
    }

    // ──────────────────── predict-single 子命令 ────────────────────

    /**
     * 单条预测（联调用）。--format：
     * <ul>
     *   <li>{@code full}（默认）完整 schema，与 predictResult.txt 单行一致</li>
     *   <li>{@code api}/{@code labels} 成员4 接口格式 {@code {"success":..,"labels":[..]}}</li>
     * </ul>
     */
    private static void doPredictSingle(Map<String, String> opts) throws Exception {
        String modelPath = opt(opts, "model", "LOCAL_MODEL", DEF_LOCAL_MODEL);
        String modelHdfs = opts.get("model-hdfs");
        if (modelHdfs != null && !modelHdfs.trim().isEmpty()) {
            modelPath = modelHdfs.trim();
        }
        String text = opts.get("text");
        String format = opts.getOrDefault("format", "full").toLowerCase(Locale.ROOT);

        if (text == null || text.trim().isEmpty()) {
            System.err.println("predict-single 需要 --text 参数，例如：");
            System.err.println("  --text \"A space marine battles aliens on a distant planet.\"");
            System.err.println("  --text \"spacewar,alien,marine\"");
            System.exit(2);
            return;
        }

        ModelParams model = loadModelOrFail(modelPath);
        NbPredictor predictor = buildPredictor(model, opts);


        // 统一走 resolveTags：自动识别"原始数据行 / tag 串 / 自然语言句子"三种形态，
        // 与后端 /predict 使用完全相同的入口，保证实时与批量同源。
        List<String> tags = TextTokenizer.resolveTags(text, model);
        if (tags.isEmpty()) {
            System.err.println("[WARN] 未能从输入中识别出任何词汇表内的特征词（全 OOV，不报错）");
        } else {
            System.err.println("[INFO] 解析出 " + tags.size() + " 个特征词: "
                    + truncate(String.join(",", tags), 80));
        }

        NbPredictor.PredictionResult pr = predictor.predictAuto(tags);
        JsonUtil.PredictionView v = JsonUtil.buildView(model, 0, tags, null, pr);

        if ("api".equals(format) || "labels".equals(format)) {
            // --top-n 仅作用于 labels 格式（可选截断，默认 0 = 不截断）；
            // full schema 必须与 predictResult.txt 严格一致，不受影响。
            int topN = Integer.parseInt(opts.getOrDefault("top-n", "0"));
            System.out.println(JsonUtil.predictionLabelsJson(v, topN));
        } else {
            System.out.println(JsonUtil.predictionLineJson(v));
        }
    }

    // ──────────────────── serve 子命令（Module-SERVER） ────────────────────

    /**
     * 启动 HTTP 后端。模型在启动时【一次性】反序列化到内存，之后所有请求复用，
     * 严禁每请求读盘/读 HDFS（对齐 Module-SERVER 规范）。
     */
    private static void doServe(Map<String, String> opts) throws Exception {
        String modelPath = opt(opts, "model", "LOCAL_MODEL", DEF_LOCAL_MODEL);
        // --model-hdfs 与 --model 二选一：前者从 HDFS 读模型，后者读本地副本
        String modelHdfs = opts.get("model-hdfs");
        if (modelHdfs != null && !modelHdfs.trim().isEmpty()) {
            modelPath = modelHdfs.trim();
            System.out.println("[SERVE] 模型来源: HDFS " + modelPath);
        } else {
            System.out.println("[SERVE] 模型来源: 本地 " + modelPath
                    + "  (如需从集群读请加 --model-hdfs " + DEF_HDFS_MODEL + ")");
        }
        int port = Integer.parseInt(opt(opts, "port", "SERVER_PORT", String.valueOf(DEF_PORT)));

        ModelParams model = loadModelOrFail(modelPath);
        // 启动即构造 predictor 单例（无状态、线程安全），每请求复用
        new PredictServer(model, port,
                Double.parseDouble(opt(opts, "z-threshold", "Z_THRESHOLD", "0")),
                Double.parseDouble(opt(opts, "min-prior", "MIN_PRIOR", "0.005")),
                loadGenreZThresholds(opts.get("z-thresholds"), model)).start();
    }

    /**
     * 统一构造 predictor：解析 --z-threshold / --min-prior / --z-thresholds。
     *
     * <p>--z-thresholds 指向逐 genre 阈值表（JSON，形如 {"drama":0.95,...}），
     * 用于替代全局统一的 z 阈值；文件不存在或解析失败时自动回退到统一阈值，
     * 保证老命令（只给 --z-threshold）行为完全不变。
     */
    private static NbPredictor buildPredictor(ModelParams model, Map<String, String> opts) {
        double z = Double.parseDouble(opt(opts, "z-threshold", "Z_THRESHOLD", "0"));
        double minPrior = Double.parseDouble(opt(opts, "min-prior", "MIN_PRIOR", "0.005"));
        return new NbPredictor(model, z, minPrior,
                loadGenreZThresholds(opts.get("z-thresholds"), model));
    }

    /**
     * 读取逐 genre 的 z 阈值表；返回 null 表示不启用（走统一阈值）。
     * 只做最朴素的 key:number 解析，缺项 genre 保持全局阈值对应的默认值 1.25。
     */
    static double[] loadGenreZThresholds(String path, ModelParams model) {
        if (path == null || path.isEmpty()) {
            return null;
        }
        File f = new File(path);
        if (!f.isFile()) {
            System.err.println("[WARN] 阈值表不存在，回退到统一 z 阈值: " + path);
            return null;
        }
        try {
            String text = new String(java.nio.file.Files.readAllBytes(f.toPath()),
                    StandardCharsets.UTF_8);
            double[] out = new double[model.G];
            Arrays.fill(out, 1.25);
            java.util.regex.Matcher m = java.util.regex.Pattern
                    .compile("\"([A-Za-z]+)\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)")
                    .matcher(text);
            int hit = 0;
            while (m.find()) {
                int gi = indexOfGenre(model.genreArr, m.group(1));
                if (gi >= 0) {
                    out[gi] = Double.parseDouble(m.group(2));
                    hit++;
                }
            }
            if (hit == 0) {
                System.err.println("[WARN] 阈值表未匹配到任何 genre，回退统一阈值: " + path);
                return null;
            }
            System.out.println("[DECISION] 已加载逐 genre z 阈值: " + hit + " 项 <- " + path);
            return out;
        } catch (Exception e) {
            System.err.println("[WARN] 阈值表解析失败，回退统一阈值: " + e.getMessage());
            return null;
        }
    }

    private static int indexOfGenre(String[] genreArr, String name) {
        for (int i = 0; i < genreArr.length; i++) {
            if (genreArr[i].equalsIgnoreCase(name)) {
                return i;
            }
        }
        return -1;
    }

    // ══════════════ 供成员4 后端直接调用的公共 API（免 HTTP 开销） ══════════════
    //
    // 这三个方法内部一律走 JsonUtil.buildView，与批量 predict、HTTP handler
    // 完全同源，保证"实时 / 兜底 / 批量"三处打分与输出 schema 一致。

    /**
     * 解析原始文本（数据行 / tag 串 / 自然语言均可），返回单条【完整 schema】预测 JSON。
     * 字段与 predictResult.txt 单行一致。
     */
    public static String predictSingleJson(ModelParams model, String rawLine) {
        NbPredictor predictor = new NbPredictor(model);
        List<String> tags = TextTokenizer.resolveTags(rawLine, model);
        NbPredictor.PredictionResult pr = predictor.predictAuto(tags);
        return JsonUtil.predictionLineJson(JsonUtil.buildView(model, 0, tags, null, pr));
    }

    /**
     * 给定 tag 列表，返回单条【完整 schema】预测 JSON。
     */
    public static String predictSingleJsonByTags(ModelParams model, List<String> tags) {
        NbPredictor predictor = new NbPredictor(model);
        List<String> norm = normalizeTags(tags);
        NbPredictor.PredictionResult pr = predictor.predictAuto(norm);
        return JsonUtil.predictionLineJson(JsonUtil.buildView(model, 0, norm, null, pr));
    }

    /**
     * 给定 tag 列表，返回【成员4 labels 格式】JSON：
     * <pre>{"success":true,"message":"预测成功","labels":[{"name":"Action","confidence":0.66}]}</pre>
     */
    public static String predictApiJsonByTags(ModelParams model, List<String> tags) {
        NbPredictor predictor = new NbPredictor(model);
        List<String> norm = normalizeTags(tags);
        NbPredictor.PredictionResult pr = predictor.predictAuto(norm);
        JsonUtil.PredictionView v = JsonUtil.buildView(model, 0, norm, null, pr);
        if (v.predGenres.isEmpty()) {
            v.message = "未识别到已知特征词，请补充更具体的剧情描述";
        }
        return JsonUtil.predictionLabelsJson(v);
    }

    /**
     * 把自然语言剧情描述转为【成员4 labels 格式】响应 JSON。
     * 这是成员4 后端最省事的一个入口。
     */
    public static String predictForApi(ModelParams model, String text) {
        if (text == null || text.trim().isEmpty()) {
            return JsonUtil.apiErrorJson("请输入电影描述");
        }
        NbPredictor predictor = new NbPredictor(model);
        List<String> tags = TextTokenizer.resolveTags(text.trim(), model);
        NbPredictor.PredictionResult pr = predictor.predictAuto(tags);
        JsonUtil.PredictionView v = JsonUtil.buildView(model, 0, tags, null, pr);
        if (tags.isEmpty()) {
            v.message = "未识别到已知特征词，请补充更具体的剧情描述";
        }
        return JsonUtil.predictionLabelsJson(v);
    }

    /**
     * 归一化 tag：与训练期【同一】规则（trim + 小写 + 剥离包裹引号），
     * 保证后端传入的 tag 与词汇表口径一致。复用 DataParser 的元素抽取逻辑，
     * 避免"两处各写一遍 normalize"导致不一致。
     */
    static List<String> normalizeTags(List<String> raw) {
        List<String> out = new ArrayList<>();
        if (raw == null) {
            return out;
        }
        for (String t : raw) {
            if (t == null) {
                continue;
            }
            String s = t.trim();
            if (s.isEmpty()) {
                continue;
            }
            // 若前端误传带引号的字面量（如 'action' 或 "loveofone'slife"），
            // 走 DataParser 的同一套引号剥离规则；否则直接小写。
            if ((s.startsWith("'") || s.startsWith("\"")) && s.length() > 1) {
                String[] extracted = DataParser.extractTokens("[" + s + "]");
                if (extracted.length > 0) {
                    for (String e : extracted) {
                        out.add(e);
                    }
                    continue;
                }
            }
            out.add(s.toLowerCase(Locale.ROOT));
        }
        return out;
    }

    // ──────────────────── 工具方法 ────────────────────

    private static String truncate(String s, int maxLen) {
        if (s == null) {
            return "";
        }
        if (s.length() <= maxLen) {
            return s;
        }
        return s.substring(0, maxLen - 3) + "...";
    }

    private static void printUsage() {
        System.out.println("《大数据技术》多标签电影分类 — 成员3 算法模块");
        System.out.println();
        System.out.println("用法（路径支持 hdfs:// / file:// / 本地相对路径，由 HdfsIO 统一路由）：");
        System.out.println();
        System.out.println("  1) 训练（读 HDFS 数据集，模型双写 HDFS + 本地）");
        System.out.println("     java com.movie.sentiment.Main train \\");
        System.out.println("        --input " + DEF_HDFS_TRAIN + " \\");
        System.out.println("        --model-hdfs " + DEF_HDFS_MODEL + " --model-local " + DEF_LOCAL_MODEL);
        System.out.println("        [--namenode hdfs://nn:8020] [--tau 0.0] [--no-calibrate] [--no-fallback]");
        System.out.println();
        System.out.println("  2) 批量预测（产出 predictResult.txt 兜底文件 + metrics.json）");
        System.out.println("     java com.movie.sentiment.Main predict \\");
        System.out.println("        --model " + DEF_LOCAL_MODEL + " --input " + DEF_HDFS_PREDICT + " \\");
        System.out.println("        --out " + DEF_PREDICT_RESULT + " --metrics " + DEF_METRICS_OUT + " [--format jsonl|tsv]");
        System.out.println();
        System.out.println("  3) 启动 HTTP 后端（供成员4 前端实时调用）");
        System.out.println("     java com.movie.sentiment.Main serve --model " + DEF_LOCAL_MODEL + " [--port " + DEF_PORT + "]");
        System.out.println("     （或 --model-hdfs " + DEF_HDFS_MODEL + " 从集群读模型）");
        System.out.println();
        System.out.println("  4) 单条预测（联调自测）");
        System.out.println("     java com.movie.sentiment.Main predict-single --model " + DEF_LOCAL_MODEL);
        System.out.println("        --text \"A space marine battles aliens...\" [--format full|api]");
        System.out.println();
        System.out.println("  5) MapReduce 训练（需集群 YARN；产物不含阈值校准，需再跑 calibrate）");
        System.out.println("     hadoop jar sentiment-mr.jar com.movie.sentiment.mr.MrTrainDriver \\");
        System.out.println("        -input " + DEF_HDFS_TRAIN + " -model " + DEF_HDFS_MODEL);
        System.out.println();
        System.out.println("  6) 对已有模型补做阈值校准（MR 模型必跑，否则 tau=0.0 无法激活标签）");
        System.out.println("     java com.movie.sentiment.Main calibrate \\");
        System.out.println("        --model-hdfs /output/model-mr.dat --input " + DEF_HDFS_TRAIN + " \\");
        System.out.println("        --model-local output/model-mr-calibrated.dat [--out-model-hdfs /output/model-mr-calibrated.dat]");
        System.out.println();
        System.out.println("默认值（均可用环境变量覆盖：NAMENODE_URI / HDFS_TRAIN / HDFS_PREDICT /");
        System.out.println("        HDFS_MODEL / LOCAL_MODEL / PREDICT_RESULT / METRICS_OUT / SERVER_PORT）：");
        System.out.println("  --namenode    = " + DEF_NAMENODE + "   ← 本项目实测 NameNode");
        System.out.println("  --input(train)= " + DEF_HDFS_TRAIN);
        System.out.println("  --input(pred) = " + DEF_HDFS_PREDICT);
        System.out.println("  --model-hdfs  = " + DEF_HDFS_MODEL);
        System.out.println("  --model-local = " + DEF_LOCAL_MODEL);
        System.out.println("  --out         = " + DEF_PREDICT_RESULT + "   ← 组长钦定文件名（.txt 扩展名，内容为 JSON Lines）");
        System.out.println("  --metrics     = " + DEF_METRICS_OUT);
        System.out.println("  --port        = " + DEF_PORT);
        System.out.println("  --format      = jsonl（成员4 前端可直接 parse；tsv 供人工查看）");
        System.out.println();
        System.out.println("后端接口：");
        System.out.println("  POST /predict          {\"tags\":[\"action\",\"adventure\"]}  或  {\"text\":\"剧情描述\"}");
        System.out.println("  GET  /predict?text=action,adventure");
        System.out.println("       默认返回成员4 格式 {\"success\":..,\"labels\":[{\"name\":..,\"confidence\":..}]}");
        System.out.println("       加 ?schema=full 返回完整字段 {\"idx\":..,\"tags\":..,\"scores\":..,\"probs\":..}");
        System.out.println("  GET  /genres           全部标签显示名（前端做下拉/图例）");
        System.out.println("  GET  /health           模型规模与校准状态");
        System.out.println();
        System.out.println("提示：本机无 Hadoop 集群时，HDFS 路径会自动降级到同名本地相对路径，");
        System.out.println("      例如 /input/movies-from-kggle.txt → input/movies-from-kggle.txt（相对当前目录）。");
    }
}
