package com.movie.sentiment.mr;

import com.movie.sentiment.DataParser;
import com.movie.sentiment.Sample;
import com.movie.sentiment.VocabIdfBuilder;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Module-MR 第 2 个作业：累加条件概率分子 S(g,t)。
 *
 * <p>对应 PPT 第05页四步并行化的第 3 步（按标签累加特征权重）。
 *
 * <pre>
 *   S(g,t) = Σ_{d : g ∈ genres(d)} TFIDF(t,d)
 *   TFIDF(t,d) = TF(t,d) × IDF(t)
 * </pre>
 *
 * <p><b>可交换可结合性</b>：S(g,t) 是"单文档局部贡献之和"，加法满足交换律与结合律，
 * 因此「Mapper 算局部贡献 → Combiner 局部求和 → Reducer 全局求和」与单机一遍扫描
 * 结果<b>完全相同</b>。此统计量可在 Reduce 端聚合。
 *
 * <p><b>为什么需要两个作业</b>：TFIDF 依赖 IDF，而 IDF = log(N/(DF+1)) 是【全局量】，
 * 必须先扫一遍全量数据拿到 DF 才能算。所以：
 * <ol>
 *   <li>{@link DfCountJob}：扫全量 → DF(t) / freq(g) / N（Counter 回传）；</li>
 *   <li>driver 端算出 IDF 并落盘为词表文件（见 {@link MrTrainDriver}）；</li>
 *   <li>本作业：Mapper 通过 DistributedCache 拿到词表+IDF，算每篇文档的局部
 *       TFIDF 贡献并按 (g,t) 输出，Reducer 求和得到 S(g,t)。</li>
 * </ol>
 * 这正是"单文档局部贡献（可下放 Mapper）+ 全局归一平滑（需全局量，放 driver）"的拆分，
 * 与第一阶段代码注释里为 MR 预留的接口一致。
 *
 * <p><b>键设计</b>：输出键为 {@code g<genreIdx>\t t<tabIdx>}（用索引而非原文，
 * 避免 genre/tag 里的特殊字符干扰 TSV 解析，也省 shuffle 字节）。
 * Reducer 输出 {@code <gIdx>\t<tIdx>\t<S值>}，driver 端按索引填回矩阵。
 *
 * <p><b>规模提示</b>：中间记录数 ≈ N × 平均tag数 × 平均genre数
 * （28 万 × 20 × 3 ≈ 1700 万条）。已用 Combiner 在 map 端先聚合以压缩 shuffle；
 * 若集群 shuffle 压力大，可考虑下面 Mapper 注释里的"文档内预聚合"优化。
 */
public class TfIdfSumJob {

    /** DistributedCache 中词表文件的缓存名（driver 端写入，Mapper 端读取） */
    public static final String VOCAB_CACHE_NAME = "vocab_idf.tsv";

    /** Hadoop Counter：本作业成功解析的样本数，用于与 DfCountJob 的 N 交叉校验 */
    public enum Counter {
        SAMPLES, DIRTY_LINES
    }

    /**
     * 配置并提交作业。
     *
     * @param conf       Hadoop 配置
     * @param inputPath  训练集路径（${HDFS_TRAIN}）
     * @param vocabFile  词表+IDF 文件路径（driver 由 DfCountJob 结果生成，须在 HDFS 或本地可见）
     * @param outputDir  作业输出目录（必须不存在）
     */
    public static Job run(Configuration conf, Path inputPath, Path vocabFile, Path outputDir)
            throws IOException, ClassNotFoundException, InterruptedException {
        Job job = Job.getInstance(conf, "MultiLabel-S(g,t)-TfIdf-Sum");
        job.setJarByClass(TfIdfSumJob.class);

        job.setMapperClass(TfIdfMapper.class);
        job.setCombinerClass(SumReducer.class);   // Combiner 与 Reducer 同一实现（求和可结合）
        job.setReducerClass(SumReducer.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        // 把词表+IDF 分发到每个 Mapper 节点（9764 词约几百 KB，走 DistributedCache 合适）
        //
        // 【关键】URI 必须带 "#VOCAB_CACHE_NAME" fragment：Hadoop 用它作为任务工作目录里
        // 符号链接的名字。若不带 fragment，链接名会取 URI 最后一段（如 vocab-idf.tsv），
        // 与 Mapper 中约定的 VOCAB_CACHE_NAME 不一致，导致 Mapper 读不到文件。
        java.net.URI vocabUri = java.net.URI.create(
                vocabFile.toUri().toString() + "#" + VOCAB_CACHE_NAME);
        job.addCacheFile(vocabUri);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputDir);

        job.waitForCompletion(true);
        return job;
    }

    /**
     * Mapper：读词表 → 对每篇文档算局部 TFIDF → 按 (g,t) 输出贡献。
     *
     * <p><b>setup 只执行一次</b>：词表与 IDF 在 setup 中加载到内存数组，
     * 避免每条记录重复解析（Mapper 会被调用 N 次，setup 只 1 次）。
     */
    public static class TfIdfMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {

        /** tag 原文 → 词汇表下标 */
        private Map<String, Integer> vocabIndex;
        /** 词汇表下标 → IDF */
        private double[] idf;
        /** genre 原文 → genre 下标（driver 端确定顺序后写入词表文件） */
        private Map<String, Integer> genreIndex;

        private final Text outKey = new Text();
        private final DoubleWritable outVal = new DoubleWritable();

        /** 文档内 tag 去重集合（保证一个文档对每个 (g,t) 只贡献一次） */
        private final Set<String> seenTags = new HashSet<>();
        /** 文档内 tag → TF 计数（一次遍历得到，避免 VocabIdfBuilder.tf 的重复扫描） */
        private final Map<String, Integer> tfMap = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            vocabIndex = new HashMap<>();
            genreIndex = new HashMap<>();

            // 从 DistributedCache 打开词表文件。
            //
            // 【为何不用 context.getLocalCacheFiles()】该 API 在 Hadoop 2.x 起已废弃，
            // 且返回的路径在某些版本带 "#缓存名" 符号链接后缀，处理起来易出错。
            //
            // 【采用的方式】DistributedCache 会把缓存文件放到任务的当前工作目录，
            // 并建立以【URI 最后一段】命名的符号链接。故按约定名直接读本地文件最可靠：
            //   1) 先试当前工作目录下的 VOCAB_CACHE_NAME（绝大多数 Hadoop 版本命中）；
            //   2) 失败则从 getCacheFiles() 拿原始 URI，用 FileSystem 打开（兼容旧版本）。
            BufferedReader br = null;
            java.io.File localFile = new java.io.File(VOCAB_CACHE_NAME);
            if (localFile.exists() && localFile.canRead()) {
                br = new BufferedReader(new InputStreamReader(
                        new java.io.FileInputStream(localFile), StandardCharsets.UTF_8));
            } else {
                java.net.URI[] uris = context.getCacheFiles();
                if (uris != null && uris.length > 0) {
                    // 注意：driver 端 addCacheFile 时给 URI 加了 "#缓存名" fragment，
                    // 直接 new Path(uri) 会把 fragment 当成路径的一部分而解析失败，
                    // 故先按 '#' 截断字符串再构造 Path（等价于剥离 fragment，
                    // 且避开 java.net.URI 多参构造抛 URISyntaxException 的处理成本）。
                    String uriStr = uris[0].toString();
                    int hash = uriStr.indexOf('#');
                    if (hash >= 0) {
                        uriStr = uriStr.substring(0, hash);
                    }
                    Path p = new Path(uriStr);
                    br = new BufferedReader(new InputStreamReader(
                            FileSystem.get(p.toUri(), context.getConfiguration()).open(p),
                            StandardCharsets.UTF_8));
                }
            }
            if (br == null) {
                throw new IOException("未能打开词表文件 " + VOCAB_CACHE_NAME
                        + "（当前工作目录与 DistributedCache 均未找到）；"
                        + "请确认 driver 已生成该文件并调用 job.addCacheFile。");
            }

            int vocabSize = 0;
            try {
                String line;
                // 文件格式（三种前缀行，driver 写出）：
                //   #V <词汇表大小>            ← 头，用于分配 idf 数组
                //   #G <genre数>
                //   t<idx>\t<tag原文>\t<idf>   ← 词汇表项
                //   g<idx>\t<genre原文>        ← genre 顺序（决定 genreIndex）
                while ((line = br.readLine()) != null) {
                    if (line.startsWith("#V")) {
                        vocabSize = Integer.parseInt(line.substring(2).trim());
                        idf = new double[vocabSize];
                    } else if (line.startsWith("#G")) {
                        // genre 数仅用于校验，实际数量由 g 行数决定
                        continue;
                    } else if (line.startsWith("t")) {
                        String[] parts = line.split("\t");
                        if (parts.length >= 3) {
                            int idx = Integer.parseInt(parts[0].substring(1));
                            vocabIndex.put(parts[1], idx);
                            if (idf != null && idx < idf.length) {
                                idf[idx] = Double.parseDouble(parts[2]);
                            }
                        }
                    } else if (line.startsWith("g")) {
                        String[] parts = line.split("\t");
                        if (parts.length >= 2) {
                            genreIndex.put(parts[1], Integer.parseInt(parts[0].substring(1)));
                        }
                    }
                }
            } finally {
                br.close();
            }
            if (idf == null) {
                throw new IOException("词表文件缺少 #V 头行，无法确定词汇表大小");
            }
            context.getConfiguration().setInt("mr.vocab.size", vocabIndex.size());
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = DataParser.stripTrailingCR(value.toString());
            if (line.trim().isEmpty()) {
                return;
            }

            // ★ 复用第一阶段无状态纯函数，保证与单机解析口径 100% 一致
            Sample s = DataParser.parseLine(line, 0);
            if (s == null) {
                context.getCounter(Counter.DIRTY_LINES).increment(1L);
                return;
            }
            context.getCounter(Counter.SAMPLES).increment(1L);

            // ── 1. 一次遍历统计文档内 TF（同 VocabIdfBuilder.tf 语义，但只扫一遍）──
            tfMap.clear();
            for (String t : s.tags) {
                // 此统计量为【单文档局部贡献】，可下放 Mapper
                tfMap.merge(t, 1, Integer::sum);
            }

            // ── 2. 文档的 genre 下标集合（多标签：一篇文档可属多个 g）──
            // 文档内去重：S(g,t) 的定义是"对 g 的正样本求和"，同一文档对同一 (g,t) 只贡献一次
            int[] gIdxArr = new int[s.genres.length];
            int gCnt = 0;
            seenTags.clear();
            for (String g : s.genres) {
                Integer gi = genreIndex.get(g);
                if (gi == null) {
                    continue;   // genre 不在全集（防御），静默跳过
                }
                if (seenTags.add(g)) {
                    gIdxArr[gCnt++] = gi;
                }
            }
            if (gCnt == 0) {
                return;   // 该文档无有效 genre，对任何 S(g,t) 都无贡献
            }

            // ── 3. 对每个 (g,t) 输出局部 TFIDF 贡献 ──
            // TFIDF(t,d) = TF(t,d) × IDF(t)   ← 对应规范 TF-IDF 公式
            for (Map.Entry<String, Integer> e : tfMap.entrySet()) {
                Integer tIdx = vocabIndex.get(e.getKey());
                if (tIdx == null) {
                    continue;   // OOV：静默跳过，不抛异常（与单机一致）
                }
                double tfidf = e.getValue() * idf[tIdx];
                if (tfidf == 0.0) {
                    continue;   // IDF 为 0（DF+1==N）时无信息量，跳过以减少 shuffle
                }
                // 键用索引，避免 genre/tag 原文里的特殊字符干扰 TSV
                for (int i = 0; i < gCnt; i++) {
                    outKey.set(gIdxArr[i] + "\t" + tIdx);
                    outVal.set(tfidf);
                    context.write(outKey, outVal);
                }
            }
            // 【可选优化】若集群 shuffle 压力大，可在此先按 gIdx 把本文档的贡献
            // 聚合进一个 Map<Integer, double[]>（g → 稀疏 t→tfidf），map 结束时一次性输出，
            // 可把中间记录数从 N×|tags|×|genres| 降到 N×|genres|×|distinctTags|。
            // 因 Combiner 已在 map 端做了同类聚合，此处默认不再叠加，保持代码简洁。
        }
    }

    /**
     * Combiner / Reducer 共用：对同一 (g,t) 的局部贡献求和 → S(g,t)。
     *
     * <p>求和满足结合律，故两端可用同一实现；结果与单机一遍扫描完全相同。
     * 此统计量可在 Reduce 端聚合。
     */
    public static class SumReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {

        private final DoubleWritable outVal = new DoubleWritable();

        @Override
        protected void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double sum = 0.0;
            for (DoubleWritable v : values) {
                sum += v.get();
            }
            outVal.set(sum);
            context.write(key, outVal);
        }
    }
}
