package com.movie.sentiment.mr;

import com.movie.sentiment.HdfsIO;
import com.movie.sentiment.ModelSerializer;
import com.movie.sentiment.ModelParams;
import com.movie.sentiment.NbTrainer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Module-MR 主驱动：用两个 MapReduce 作业在集群上并行完成训练，产出与单机【完全一致】的模型。
 *
 * <p><b>对应 PPT 第05页四步的并行化</b>：
 * <ol>
 *   <li>分词/解析 —— 下放各 Mapper（复用 {@code DataParser.parseLine} 无状态纯函数）；</li>
 *   <li>统计 DF(t) / freq(g) / N —— {@link DfCountJob}，Combiner+Reducer 求和；</li>
 *   <li>累加 S(g,t) = Σ TFIDF(t,d) —— {@link TfIdfSumJob}，Mapper 算局部贡献、Reducer 求和；</li>
 *   <li>全局归一 + 平滑 —— driver 端调用 {@link NbTrainer#assembleModel}
 *       （需 N、|V|、totalLabelOcc 等全局量，无法在单个 Reducer 内完成）。</li>
 * </ol>
 *
 * <p><b>为什么第 4 步放在 driver</b>：P(t|g)=(S(g,t)+1)/(Z(g)+|V|) 需要 Z(g)=Σ_t S(g,t)
 * 覆盖【全部】tag，而单个 Reducer 只看到某个 key 分组；先验也需要 totalLabelOcc 这个全局量。
 * 这与任务书"局部贡献下放 Mapper、全局归一下放 Reducer/driver"的要求一致。
 *
 * <p><b>与单机的一致性保证</b>（本模块最重要的正确性要求）：
 * <ul>
 *   <li>解析：Mapper 与单机同用 {@code DataParser.parseLine}；</li>
 *   <li>DF 文档内去重：{@link DfCountJob.DfMapper} 与 {@code VocabIdfBuilder.build} 同口径；</li>
 *   <li>IDF 公式：driver 与 {@code VocabIdfBuilder} 同为 log(N/(DF+1))；</li>
 *   <li>先验/条件概率/平滑/log 域：一律调用 {@link NbTrainer#assembleModel}，公式只有一份实现；</li>
 *   <li>genre 顺序：driver 按"genre 名字典序"固定，与作业 1 输出顺序无关，
 *       避免 LinkedHashMap 插入序在不同并行度下变化（第一阶段踩过这个坑）。</li>
 * </ul>
 *
 * <p><b>不做阈值校准</b>：校准需在留出集上反复打分搜最优 F1，属于单机内存操作，
 * 分布式化收益低、复杂度高。MR 产出的模型 tauPerGenre 为空，
 * {@code NbPredictor} 会自动回退到全局 tau（已实现逐级回退）。
 * 若需要校准，可把 MR 模型下载到本地后执行 {@code Main train --no-calibrate} 之外的校准流程，
 * 或直接用单机 train 的模型供后端使用。
 *
 * <p><b>运行方式</b>（详见 README）：
 * <pre>
 *   hadoop jar sentiment-mr.jar com.movie.sentiment.mr.MrTrainDriver \
 *       -input /input/movies-from-kggle.txt \
 *       -model /output/model-mr.dat \
 *       -tmp   /tmp/sentiment-mr
 * </pre>
 *
 * <p><b>诚实说明</b>：本模块为加分/架构演示用途。开发机无 Hadoop 集群，
 * 因此只做到【编译通过 + 逻辑自检】，未在真实 YARN 上实跑；
 * 联调 deadline 前若跑不通，以 Module-HDFS 的单机读 HDFS 为准（见 README）。
 */
public class MrTrainDriver extends Configured implements Tool {

    @Override
    public int run(String[] args) throws Exception {
        // ── 参数解析（用 Hadoop 自带的 GenericOptionsParser 风格，不引第三方 CLI 库）──
        String input = "/input/movies-from-kggle.txt";   // ${HDFS_TRAIN}
        String modelOut = "/output/model-mr.dat";        // ${HDFS_MODEL}
        String modelLocal = null;                        // 可选：同时落一份本地副本
        String tmpRoot = "/tmp/sentiment-mr";            // 中间产物目录（作业输出）

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-input":      input = args[++i]; break;
                case "-model":      modelOut = args[++i]; break;
                case "-model-local": modelLocal = args[++i]; break;
                case "-tmp":        tmpRoot = args[++i]; break;
                default:
                    System.err.println("未知参数: " + args[i]);
                    printUsage();
                    return 2;
            }
        }

        Configuration conf = getConf();

        // 把 HdfsIO 的 NameNode 同步为【本次作业的 conf】（含用户 -D fs.defaultFS=...）。
        //
        // 【为什么必须同步】HdfsIO 持有自己的静态 Configuration，默认 fs.defaultFS
        // 是 localhost:9000。而 main() 里的 HdfsIO.configure(null,...) 只设了默认值，
        // 拿不到 ToolRunner 解析出的 -D 参数。若不同步，最后
        // ModelSerializer.save("/output/model-mr.dat") 会把模型写到 localhost:9000，
        // 而不是用户指定的集群——在真实集群上这是静默写错位置。
        // （词表中间文件已在 writeVocabFile 里直接用作业的 fs，不受此影响。）
        String defaultFS = conf.get("fs.defaultFS");
        String hadoopUser = conf.get("hadoop.job.ugi", "");
        if (hadoopUser.contains(",")) {
            hadoopUser = hadoopUser.substring(0, hadoopUser.indexOf(','));   // 去掉 ",password" 部分
        }
        HdfsIO.configure(defaultFS, hadoopUser, true);

        FileSystem fs = FileSystem.get(conf);
        Path inputPath = new Path(input);

        System.out.println("══════ MapReduce 训练开始 ══════");
        System.out.println("训练集输入 : " + input);
        System.out.println("模型输出   : " + modelOut);
        System.out.println("中间目录   : " + tmpRoot);

        if (!fs.exists(inputPath)) {
            System.err.println("[ERROR] 训练集不存在: " + input);
            return 3;
        }

        Path dfOutDir = new Path(tmpRoot, "df-count");
        Path vocabFile = new Path(tmpRoot, "vocab-idf.tsv");
        Path sOutDir = new Path(tmpRoot, "tfidf-sum");

        // 中间目录必须清空，否则 MapReduce 作业因输出目录已存在而失败
        cleanTmp(fs, dfOutDir);
        cleanTmp(fs, sOutDir);

        // ══════════ 作业 1：DF(t) / freq(g) / N ══════════
        System.out.println("\n── 作业 1/2：统计 DF(t) 与 freq(g) ──");
        Job dfJob = DfCountJob.run(conf, inputPath, dfOutDir);

        // N 从 Counter 读取（避免为拿一个数字再跑一轮作业）
        Counter sampleCounter = dfJob.getCounters()
                .findCounter(DfCountJob.Counter.SAMPLES);
        Counter dirtyCounter = dfJob.getCounters()
                .findCounter(DfCountJob.Counter.DIRTY_LINES);
        int N = (int) sampleCounter.getValue();
        long dirty = dirtyCounter.getValue();
        System.out.println("  样本总数 N = " + N + "  (脏行跳过 = " + dirty + ")");
        if (N <= 0) {
            System.err.println("[ERROR] N=0，训练集可能格式不符或路径为空");
            return 4;
        }

        // 读取作业 1 输出：每行 "<t:tag|g:genre>\t<count>"
        // 必须遍历目录下【全部】 part 文件：Reducer 数 >1 时输出会分片到多个文件，
        // 只读 part-r-00000 会静默丢失大部分统计量，导致模型完全错误。
        Map<String, Integer> dfMap = new LinkedHashMap<>();     // tag → DF
        Map<String, Integer> genreFreq = new LinkedHashMap<>(); // genre → freq(g)
        readCounts(fs, dfOutDir, dfMap, genreFreq);
        System.out.println("  词汇表大小 |V| = " + dfMap.size()
                + "   genre 种类数 |G| = " + genreFreq.size());
        if (dfMap.isEmpty() || genreFreq.isEmpty()) {
            System.err.println("[ERROR] 未读到 DF/genre 统计，检查作业 1 输出与 part 文件名");
            return 5;
        }

        // ══════════ driver 端：算 IDF、固定 genre 顺序、写词表文件 ══════════
        String[] vocabArr = dfMap.keySet().toArray(new String[0]);
        Map<String, Integer> vocabIndex = new HashMap<>();
        double[] idf = new double[vocabArr.length];
        int[] df = new int[vocabArr.length];
        for (int i = 0; i < vocabArr.length; i++) {
            vocabIndex.put(vocabArr[i], i);
            df[i] = dfMap.get(vocabArr[i]);
            // IDF(t) = log(N / (DF(t) + 1))   ← 与 VocabIdfBuilder 同一公式，+1 防除零
            idf[i] = Math.log((double) N / (df[i] + 1));
        }

        // genre 顺序固定为【字典序】：与并行度、Reducer 数量无关，保证可复现。
        // （第一阶段实测：若依赖 LinkedHashMap 插入序，不同数据子集会得到不同顺序，
        //   导致按索引搬运的阈值张冠李戴。）
        String[] genreArr = genreFreq.keySet().toArray(new String[0]);
        java.util.Arrays.sort(genreArr);

        // 写词表+IDF 文件，供作业 2 的 Mapper 通过 DistributedCache 读取
        writeVocabFile(fs, vocabFile, vocabArr, idf, genreArr);
        System.out.println("  词表文件已写出: " + vocabFile
                + "  (" + vocabArr.length + " 词 + " + genreArr.length + " genre)");

        // ══════════ 作业 2：S(g,t) ══════════
        System.out.println("\n── 作业 2/2：累加 S(g,t) = Σ TFIDF(t,d) ──");
        Job sJob = TfIdfSumJob.run(conf, inputPath, vocabFile, sOutDir);
        Counter s2Counter = sJob.getCounters().findCounter(TfIdfSumJob.Counter.SAMPLES);
        System.out.println("  作业 2 处理样本数 = " + s2Counter.getValue()
                + "  (作业 1 为 " + N + "，两者应相等)");
        if (s2Counter.getValue() != N) {
            // 不相等说明两个作业看到的输入不一致（如分片边界问题），模型将不可信
            System.err.println("[WARN] 两个作业样本数不一致，请检查输入路径是否被并发修改");
        }

        // 读取作业 2 输出：每行 "<gIdx>\t<tIdx>\t<S值>"（同样需遍历全部 part 文件）
        double[][] S = new double[genreArr.length][vocabArr.length];
        long sEntries = readSMatrix(fs, sOutDir, S);
        System.out.println("  非零 S(g,t) 条目数 = " + sEntries);

        // ══════════ 第 4 步：全局归一 + 平滑（复用单机同一份实现）══════════
        // ★ 关键：不在 MR 模块里重写公式，直接调用 NbTrainer.assembleModel，
        //   保证分布式路径与单机路径的先验/条件概率/平滑/log 域完全一致。
        System.out.println("\n── 装配模型（归一 + 加1平滑 + log 域）──");
        ModelParams model = NbTrainer.assembleModel(
                vocabArr, vocabIndex, idf, N, genreFreq, S, genreArr,
                0.0, "mapreduce:" + input);

        // MR 不做阈值校准（理由见类注释）：置空数组，NbPredictor 会自动回退全局 tau
        model.tauPerGenre = null;
        model.scalePerGenre = null;
        model.tauPerGenreText = null;
        model.scalePerGenreText = null;

        System.out.printf("  N=%d |V|=%d |G|=%d tau=%.4f（未校准，预测时回退全局 tau）%n",
                model.N, model.V, model.G, model.tau);

        // ── 保存模型（走 HdfsIO，支持 HDFS 与本地双写）──
        if (modelLocal != null) {
            ModelSerializer.dualWrite(model, modelOut, modelLocal);
        } else {
            ModelSerializer.save(model, modelOut);
            System.out.println("[MODEL] 已写入: " + modelOut);
        }

        System.out.println("\n══════ MapReduce 训练完成 ══════");
        System.out.println("提示：本模型未做阈值校准，直接用于预测会因 tau=0.0 而不激活任何标签。");
        System.out.println("      生产使用请改用单机 train（含留出集校准），或参考类注释的校准说明。");
        return 0;
    }

    /** 清理中间目录（MapReduce 要求输出目录不存在）。 */
    private static void cleanTmp(FileSystem fs, Path dir) throws Exception {
        if (fs.exists(dir)) {
            fs.delete(dir, true);
            System.out.println("  已清理中间目录: " + dir);
        }
    }

    /**
     * 读取作业 1 输出（DF / genre 频次）。
     *
     * <p>遍历输出目录下【全部】 part 文件：Reducer 数 >1 时结果会分散在多个
     * part-r-00000 / part-r-00001 ... 中，只读单个文件会静默丢失统计量。
     * 同时兼容不同 Hadoop 版本的命名差异（part-r-* / part-00000 等）。
     */
    private static void readCounts(FileSystem fs, Path outDir,
                                   Map<String, Integer> dfMap,
                                   Map<String, Integer> genreFreq) throws Exception {
        for (Path part : listPartFiles(fs, outDir)) {
            try (BufferedReader br = new BufferedReader(new java.io.InputStreamReader(
                    fs.open(part), java.nio.charset.StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (line.trim().isEmpty()) continue;
                    String[] parts = line.split("\t");
                    if (parts.length < 2) continue;
                    String key = parts[0];
                    int count;
                    try {
                        count = Integer.parseInt(parts[1].trim());
                    } catch (NumberFormatException e) {
                        continue;   // 容错：跳过异常行，不中断整个读取
                    }
                    if (key.startsWith(DfCountJob.PREFIX_TAG)) {
                        dfMap.put(key.substring(DfCountJob.PREFIX_TAG.length()), count);
                    } else if (key.startsWith(DfCountJob.PREFIX_GENRE)) {
                        genreFreq.put(key.substring(DfCountJob.PREFIX_GENRE.length()), count);
                    }
                }
            }
        }
    }

    /**
     * 读取作业 2 输出，填入 S 矩阵。
     * 行格式：{@code <gIdx>\t<tIdx>\t<S值>}（TextOutputFormat 会在 key/value 间插入 \t）
     *
     * <p>同样遍历目录下全部 part 文件，理由见 {@link #readCounts}。
     *
     * @return 实际读到的非零条目数
     */
    private static long readSMatrix(FileSystem fs, Path outDir, double[][] S) throws Exception {
        long entries = 0L;
        for (Path part : listPartFiles(fs, outDir)) {
            try (BufferedReader br = new BufferedReader(new java.io.InputStreamReader(
                    fs.open(part), java.nio.charset.StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (line.trim().isEmpty()) continue;
                    String[] parts = line.split("\t");
                    if (parts.length < 3) continue;
                    try {
                        int gIdx = Integer.parseInt(parts[0].trim());
                        int tIdx = Integer.parseInt(parts[1].trim());
                        double val = Double.parseDouble(parts[2].trim());
                        if (gIdx >= 0 && gIdx < S.length && tIdx >= 0 && tIdx < S[0].length) {
                            S[gIdx][tIdx] = val;
                            entries++;
                        }
                    } catch (NumberFormatException e) {
                        continue;   // 容错
                    }
                }
            }
        }
        return entries;
    }

    /**
     * 列出作业输出目录下的全部 part 文件（排除 _SUCCESS、.crc 校验文件、日志目录）。
     *
     * <p>兼容不同 Hadoop 版本命名：part-r-00000（有 Reducer）、part-00000（map-only）、
     * part-r-00000.gz 等，只要以 "part-" 开头且是普通文件即视为有效输出。
     */
    private static java.util.List<Path> listPartFiles(FileSystem fs, Path outDir) throws Exception {
        java.util.List<Path> result = new java.util.ArrayList<>();
        if (outDir == null || !fs.exists(outDir)) {
            return result;
        }
        org.apache.hadoop.fs.FileStatus[] statuses = fs.listStatus(outDir);
        for (org.apache.hadoop.fs.FileStatus st : statuses) {
            if (!st.isFile()) continue;                       // 跳过子目录（如 _logs）
            String name = st.getPath().getName();
            if (name.startsWith("part-") && !name.endsWith(".crc")) {
                result.add(st.getPath());
            }
        }
        // 按文件名排序，保证多次读取顺序一致（结果与顺序无关，但便于复现调试）
        result.sort(java.util.Comparator.comparing(p -> p.getName()));
        return result;
    }

    /**
     * 写出词表+IDF 文件（供作业 2 的 Mapper 通过 DistributedCache 读取）。
     * 格式见 {@link TfIdfSumJob.TfIdfMapper#setup}。
     *
     * <p><b>为什么直接用传入的 FileSystem 而不走 HdfsIO</b>：
     * driver 里的两个作业使用 {@code ToolRunner} 传入的 Configuration
     * （含用户通过 {@code -D fs.defaultFS=...} 指定的集群地址），
     * 而 {@code HdfsIO} 持有的是它自己的静态 Configuration（默认 localhost:9000）。
     * 若用 HdfsIO 写这个文件，在集群上会把词表写到【另一个文件系统】，
     * 导致作业 2 的 Mapper 读不到而失败。
     * 因此这里必须复用与作业相同的 fs 实例，保证读写落在同一处。
     */
    private static void writeVocabFile(FileSystem fs, Path vocabFile,
                                       String[] vocabArr, double[] idf, String[] genreArr)
            throws Exception {
        StringBuilder sb = new StringBuilder(vocabArr.length * 40);
        sb.append("#V ").append(vocabArr.length).append('\n');
        sb.append("#G ").append(genreArr.length).append('\n');
        for (int i = 0; i < vocabArr.length; i++) {
            // %.17g 保证 double 往返无损：Mapper 端解析回来的 IDF 必须与此处完全相等，
            // 否则 S(g,t) 会与单机结果产生细微偏差，破坏"分布式 == 单机"的一致性承诺。
            sb.append('t').append(i).append('\t').append(vocabArr[i])
                    .append('\t').append(String.format(java.util.Locale.ROOT, "%.17g", idf[i]))
                    .append('\n');
        }
        for (int i = 0; i < genreArr.length; i++) {
            sb.append('g').append(i).append('\t').append(genreArr[i]).append('\n');
        }
        // 用与作业同一个 fs 写出（覆盖已存在文件）
        try (java.io.OutputStream os = fs.create(vocabFile, true);
             java.io.BufferedWriter w = new java.io.BufferedWriter(
                     new java.io.OutputStreamWriter(os, java.nio.charset.StandardCharsets.UTF_8),
                     1 << 16)) {
            w.write(sb.toString());
        }
    }

    private static void printUsage() {
        System.out.println("用法: hadoop jar sentiment-mr.jar com.movie.sentiment.mr.MrTrainDriver \\");
        System.out.println("        -input <训练集HDFS路径>  [-model <模型输出路径>] \\");
        System.out.println("        [-model-local <本地副本路径>] [-tmp <中间目录>]");
        System.out.println();
        System.out.println("默认值:");
        System.out.println("  -input       = /input/movies-from-kggle.txt   （${HDFS_TRAIN}）");
        System.out.println("  -model       = /output/model-mr.dat           （${HDFS_MODEL}）");
        System.out.println("  -model-local = 不写本地副本");
        System.out.println("  -tmp         = /tmp/sentiment-mr              （两个作业的中间输出目录）");
        System.out.println();
        System.out.println("说明: 本作业不含阈值校准，产出的模型 tau=0.0 会导致预测不激活任何标签；");
        System.out.println("      生产使用请改用单机 train（含留出集校准）。详见类注释与 README。");
    }

    /** 入口：交给 ToolRunner 以便自动处理 Hadoop 通用参数（-D、-fs、-jt 等）。 */
    public static void main(String[] args) throws Exception {
        // 初始化 HdfsIO：供最后 ModelSerializer.save/dualWrite 写模型时使用
        // （词表中间文件不走 HdfsIO，而是用与作业同一个 FileSystem，见 writeVocabFile 注释）。
        HdfsIO.configure(null, null, true);
        int code = ToolRunner.run(new Configuration(), new MrTrainDriver(), args);
        System.exit(code);
    }
}
