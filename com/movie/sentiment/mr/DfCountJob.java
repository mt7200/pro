package com.movie.sentiment.mr;

import com.movie.sentiment.DataParser;
import com.movie.sentiment.Sample;
import com.movie.sentiment.TokenFilter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * Module-MR 第 1 个作业：统计【文档频率 DF(t)】与【标签频次 freq(g)】。
 *
 * <p>对应 PPT 第05页四步并行化的第 1~2 步（分词/统计 → 求 DF 与词频）。
 *
 * <p><b>可交换可结合性</b>（任务书硬性要求，也是能否用 MapReduce 的前提）：
 * DF(t) 与 freq(g) 都是"计数"，计数满足交换律与结合律，因此
 * 「各 Mapper 输出局部计数 → Combiner 局部求和 → Reducer 全局求和」
 * 的结果与「单机一遍扫描累加」<b>完全相同</b>。此统计量可在 Reduce 端聚合。
 *
 * <p><b>一致性关键点</b>：Mapper 直接复用第一阶段的
 * {@link DataParser#parseLine(String, int)}（无状态纯静态函数），
 * 保证分布式路径与单机路径的解析口径 100% 一致，绝不出现两套解析规则。
 *
 * <p><b>键前缀</b>：tag 与 genre 可能同名（如都有 "action"），
 * 故输出键加前缀 {@code t:} / {@code g:} 区分，避免两类计数被错误合并。
 *
 * <p><b>输出</b>：TSV，每行 {@code <前缀:名称>\t<计数>}，由 {@link MrTrainDriver} 读取。
 * 样本总数 N 通过 Hadoop Counter 回传 driver，避免额外一轮作业。
 */
public class DfCountJob {

    /** 输出键前缀：tag（用于 DF 统计） */
    public static final String PREFIX_TAG = "t:";
    /** 输出键前缀：genre（用于 freq 统计） */
    public static final String PREFIX_GENRE = "g:";

    /** Hadoop Counter 组名与计数项：driver 端读取 N 与脏行数 */
    public enum Counter {
        /** 成功解析的样本数，即训练集规模 N */
        SAMPLES,
        /** 跳过的脏行数（[...] 块数 < 2），对齐单机路径的 skippedLines */
        DIRTY_LINES
    }

    /**
     * 配置并提交作业。
     *
     * @param conf      Hadoop 配置（含 fs.defaultFS）
     * @param inputPath 训练集在 HDFS 的路径（${HDFS_TRAIN}）
     * @param outputDir 作业输出目录（必须不存在，由 driver 负责清理）
     * @return 提交并等待完成的 Job（driver 从中读 Counter 取 N）
     */
    public static Job run(Configuration conf, Path inputPath, Path outputDir)
            throws IOException, ClassNotFoundException, InterruptedException {
        Job job = Job.getInstance(conf, "MultiLabel-DF-and-GenreFreq-Count");
        job.setJarByClass(DfCountJob.class);

        job.setMapperClass(DfMapper.class);
        job.setCombinerClass(CountSumReducer.class);   // Combiner：局部求和，大幅减少 shuffle
        job.setReducerClass(CountSumReducer.class);   // Reducer：全局求和（与 Combiner 同一实现）

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(LongWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputDir);

        // 提高并行度：按输入分片自然切分即可，这里不强制设置 map 数，
        // 由 InputFormat 依 HDFS block 大小决定（课程集群默认 128MB/块）
        job.waitForCompletion(true);
        return job;
    }

    /**
     * Mapper：一行（一部电影）→ 若干 {@code (t:tag, 1)} 与 {@code (g:genre, 1)}。
     *
     * <p><b>文档内去重</b>：DF(t) 的定义是"含 t 的样本条数"，故同一文档内
     * tag 重复出现只能计 1 次；freq(g) = Σ_d [g∈genres(d)] 是指示函数求和，
     * 同一文档内 genre 重复也只能计 1 次。此处与单机 {@code NbTrainer} 口径严格一致。
     */
    public static class DfMapper extends Mapper<LongWritable, Text, Text, LongWritable> {

        /** 复用对象，避免每条记录都 new（Mapper 高频调用，减少 GC 压力） */
        private final Text outKey = new Text();
        private final LongWritable ONE = new LongWritable(1L);
        private final Set<String> dedup = new HashSet<>();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            // CRLF 处理：与单机路径同一套 stripTrailingCR
            String line = DataParser.stripTrailingCR(value.toString());
            if (line.trim().isEmpty()) {
                return;   // 空行跳过（不计入脏行，与单机一致）
            }

            // ★ 复用第一阶段无状态纯函数，保证解析口径完全一致
            Sample s = DataParser.parseLine(line, 0);
            if (s == null) {
                context.getCounter(Counter.DIRTY_LINES).increment(1L);
                return;   // 脏行跳过 + 计数，不抛异常中断作业
            }
            context.getCounter(Counter.SAMPLES).increment(1L);

            // DF(t)：文档内每个唯一 tag 计 1
            dedup.clear();
            for (String t : s.tags) {
                // 与单机训练同一口径：过滤非英文字母/数字构成的词条
                if (!TokenFilter.isValid(t)) {
                    continue;
                }
                if (dedup.add(t)) {
                    outKey.set(PREFIX_TAG + t);
                    context.write(outKey, ONE);
                }
            }

            // freq(g)：文档内每个唯一 genre 计 1（多标签：一条样本可贡献多个 g）
            dedup.clear();
            for (String g : s.genres) {
                if (dedup.add(g)) {
                    outKey.set(PREFIX_GENRE + g);
                    context.write(outKey, ONE);
                }
            }
        }
    }

    /**
     * Combiner / Reducer 共用实现：对同一键的计数求和。
     *
     * <p>求和满足结合律，因此 Combiner（map 端局部聚合）与 Reducer（全局聚合）
     * 可以用完全相同的代码，且最终结果与单机一遍扫描相同。
     * 此统计量可在 Reduce 端聚合。
     */
    public static class CountSumReducer
            extends Reducer<Text, LongWritable, Text, LongWritable> {

        private final LongWritable outVal = new LongWritable();

        @Override
        protected void reduce(Text key, Iterable<LongWritable> values, Context context)
                throws IOException, InterruptedException {
            long sum = 0L;
            for (LongWritable v : values) {
                sum += v.get();
            }
            outVal.set(sum);
            context.write(key, outVal);
        }
    }
}
