package com.movie.sentiment;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 数据解析器：一行文本 → Sample。
 * 完全无状态纯函数，第二阶段 Mapper 直接复用。
 *
 * 解析规则（硬性规范）：
 * 1) 用正则抓取所有 [...] 块，第1块=标签列(genre)，第2块=特征列(tag)。
 *    常量 LABEL_BLOCK_INDEX=1, FEATURE_BLOCK_INDEX=2，方向相反时对调即可。
 * 2) 从块内提取元素：单引号包裹或双引号包裹，保留内部撇号如 loveofone'slife。
 * 3) 统一转小写（Locale.ROOT）。
 * 4) 处理 CRLF，跳过空行。
 */
public class DataParser {

    /** 标签(genre)列在第几个 [...] 块（1-based） */
    public static final int LABEL_BLOCK_INDEX = 1;

    /** 特征(tag)列在第几个 [...] 块（1-based） */
    public static final int FEATURE_BLOCK_INDEX = 2;

    /** 最少需要的 [...] 块数 */
    public static final int MIN_BLOCKS = 2;

    // 匹配 [...] 块（不跨行，内容非贪婪）
    private static final Pattern BLOCK_PATTERN = Pattern.compile("\\[[^\\]]*\\]");

    // 提取单引号或双引号包裹的元素；group(1)匹配单引号内容，group(2)匹配双引号内容
    private static final Pattern ELEM_PATTERN = Pattern.compile("'([^']*)'|\"([^\"]*)\"");

    /** 解析计数：跳过的脏行数，由调用方在遍历结束后读取 */
    public int skippedLines = 0;

    /** 最多打印多少条脏行 WARN，超出只汇总计数，避免刷屏淹没正常输出 */
    private static final int MAX_WARN_PRINT = 5;

    private int warnPrinted = 0;

    /**
     * 从文件读取全部样本；跳过脏行（块数不足、空行）并 WARN。
     *
     * <p>第二阶段改造：底层读取委托 {@link HdfsIO#openReader}，因此 filePath 既可是
     * HDFS URI（{@code hdfs://nn:9000/input/movies-from-kggle.txt}），
     * 也可是 {@code file://} URI 或纯本地路径，调用方无需区分。
     * 解析规则本身【完全未改动】，保证 train / predict / 后端三处 100% 一致。
     *
     * @param filePath 训练或预测文件路径（UTF-8，支持 hdfs:// 与本地）
     * @return 样本列表
     */
    public List<Sample> parseFile(String filePath) throws IOException {
        List<Sample> samples = new ArrayList<>();
        try (BufferedReader br = HdfsIO.openReader(filePath)) {
            String line;
            int lineNum = 0;
            while ((line = br.readLine()) != null) {
                lineNum++;
                line = stripTrailingCR(line);      // 处理 CRLF
                if (line.trim().isEmpty()) {
                    skippedLines++;
                    warn("行" + lineNum + "：空行，跳过");
                    continue;
                }
                Sample s = parseLine(line, lineNum);
                if (s == null) {
                    skippedLines++;
                    warn("行" + lineNum + "：脏数据（[...] 块数<" + MIN_BLOCKS + "），跳过");
                    continue;
                }
                samples.add(s);
            }
        }
        // 超出打印上限的脏行统一汇总一条
        int suppressed = skippedLines - warnPrinted;
        if (suppressed > 0) {
            System.err.println("[WARN] 另有 " + suppressed + " 行脏数据被跳过（不再逐条打印），合计跳过 "
                    + skippedLines + " 行");
        }
        return samples;
    }

    /** 打印脏行警告（受 MAX_WARN_PRINT 限流） */
    private void warn(String msg) {
        if (warnPrinted < MAX_WARN_PRINT) {
            System.err.println("[WARN] " + msg);
            warnPrinted++;
        }
    }

    /**
     * 解析单行 → Sample（无状态纯函数，Mapper 直接复用）。
     * 返回 null 表示脏行。
     *
     * @param rawLine  原始行文本（已 strip \r）
     * @param lineNum  行号（用于输出定位）
     * @return Sample 或 null
     */
    public static Sample parseLine(String rawLine, int lineNum) {
        // 步骤1：抓取所有 [...] 块
        Matcher blockMatcher = BLOCK_PATTERN.matcher(rawLine);
        List<String> blocks = new ArrayList<>(4);
        while (blockMatcher.find()) {
            blocks.add(blockMatcher.group());
        }
        if (blocks.size() < MIN_BLOCKS) {
            return null;   // 脏行
        }

        // 步骤2：取对应列（常量索引，便于对调）
        String labelBlock = blocks.get(LABEL_BLOCK_INDEX - 1);   // 转 0-based
        String featureBlock = blocks.get(FEATURE_BLOCK_INDEX - 1);

        // 步骤3：提取元素并转小写
        String[] genres = extractTokens(labelBlock);
        String[] tags = extractTokens(featureBlock);

        return new Sample(genres, tags, lineNum, rawLine);
    }

    /**
     * 从 [...] 块字符串中提取所有 token。
     * 对单引号/双引号包裹的元素做正确剥离，保留内部撇号。
     * 所有 token 统一转小写（Locale.ROOT）。
     * 对每个 token 做 trim。
     */
    public static String[] extractTokens(String block) {
        Matcher m = ELEM_PATTERN.matcher(block);
        List<String> tokens = new ArrayList<>();
        while (m.find()) {
            // group(1) 非空 = 单引号包裹；group(2) 非空 = 双引号包裹
            String token = m.group(1) != null ? m.group(1) : m.group(2);
            if (token != null) {
                token = token.trim();
                if (!token.isEmpty()) {
                    // 统一转小写（与训练预测全程一致）
                    tokens.add(token.toLowerCase(Locale.ROOT));
                }
            }
        }
        return tokens.toArray(new String[0]);
    }

    /** 去掉行尾的 \r（CRLF → LF） */
    public static String stripTrailingCR(String line) {
        if (line != null && line.endsWith("\r")) {
            return line.substring(0, line.length() - 1);
        }
        return line;
    }
}