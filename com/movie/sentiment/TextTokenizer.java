package com.movie.sentiment;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 自由文本分词器：把前端传来的自然语言剧情描述转成模型可识别的 tag 候选。
 *
 * <p>为什么需要这个类：训练数据的特征是【拼接式 tag】（如 cultureclash、spacewar、
 * loveofone'slife），而前端传的是【带空格的英文句子】（如 "A hero travels through
 * space to save his family."）。两者形态不同，必须做一层映射，否则全部 OOV、模型失效。
 *
 * <p>映射策略（两步，召回优先）：
 * <ol>
 *   <li>逐词精确匹配：把句子切成单词（保留词内撇号），小写后查词汇表；</li>
 *   <li>相邻词拼接匹配：把连续 n 个词（n=2..NGRAM_MAX）去掉空格拼接后再查词汇表，
 *       用于命中 space travel → spacetravel、love of one's life → loveofone'slife 这类形态。</li>
 * </ol>
 *
 * <p>本类为无状态纯静态函数，第二阶段 Mapper 可直接复用。
 */
public final class TextTokenizer {

    /** 拼接匹配的最大词组长度（n-gram 上限） */
    public static final int NGRAM_MAX = 4;

    /**
     * 单词匹配正则：字母数字串，允许词内撇号（如 one's）。
     * 不含前后撇号，避免把引号误当词字符。
     */
    private static final Pattern WORD_PATTERN =
            Pattern.compile("[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*");

    private TextTokenizer() {
        // 工具类，禁止实例化
    }

    /**
     * 把任意文本切成小写单词列表（保留词内撇号）。
     *
     * @param text 原始文本，可为 null
     * @return 单词列表，永不为 null
     */
    public static List<String> tokenize(String text) {
        List<String> words = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return words;
        }
        Matcher m = WORD_PATTERN.matcher(text);
        while (m.find()) {
            String w = m.group().toLowerCase(Locale.ROOT);
            if (!w.isEmpty()) {
                words.add(w);
            }
        }
        return words;
    }

    /**
     * 把自由文本映射为词汇表内命中的 tag 候选（已去重、保持出现顺序）。
     *
     * <p>返回结果只包含 vocab 中真实存在的词，因此下游可直接当作 Sample.tags 使用，
     * TF/IDF 计算与批量预测完全一致。
     *
     * @param text  前端传入的剧情描述
     * @param vocab tag 词汇表集合（来自 ModelParams.vocabIndex.keySet()）
     * @return 命中的 tag 列表；若一个都没命中则返回空列表（不返回 null）
     */
    public static List<String> toCandidateTags(String text, Set<String> vocab) {
        List<String> words = tokenize(text);
        if (words.isEmpty() || vocab == null || vocab.isEmpty()) {
            return new ArrayList<>();
        }

        // LinkedHashSet：去重 + 保持首次出现顺序（TF 统计依赖重复次数，故下面另行处理重复）
        LinkedHashSet<String> hits = new LinkedHashSet<>();

        int n = words.size();
        for (int i = 0; i < n; i++) {
            StringBuilder concat = new StringBuilder(words.get(i));
            // 策略1：单词本身命中
            String single = concat.toString();
            if (vocab.contains(single)) {
                hits.add(single);
            }
            // 策略2：向后拼接 1..NGRAM_MAX-1 个词，去空格后命中
            for (int j = i + 1; j < n && j - i < NGRAM_MAX; j++) {
                concat.append(words.get(j));
                String gram = concat.toString();
                if (vocab.contains(gram)) {
                    hits.add(gram);
                }
            }
        }
        return new ArrayList<>(hits);
    }

    /**
     * 把自由文本映射为 tag 列表，并【保留重复次数】以便 TF 统计生效。
     *
     * <p>与 {@link #toCandidateTags} 的区别：同一个 tag 在文中多次出现时会重复入列，
     * 从而 TF(t,d) &gt; 1，与训练期 tag 列表的语义保持一致。
     *
     * @param text  前端传入的剧情描述
     * @param vocab tag 词汇表集合
     * @return 含重复的 tag 列表
     */
    public static List<String> toCandidateTagsWithTf(String text, Set<String> vocab) {
        List<String> words = tokenize(text);
        List<String> result = new ArrayList<>();
        if (words.isEmpty() || vocab == null || vocab.isEmpty()) {
            return result;
        }
        int n = words.size();
        for (int i = 0; i < n; i++) {
            StringBuilder concat = new StringBuilder(words.get(i));
            if (vocab.contains(concat.toString())) {
                result.add(concat.toString());
            }
            for (int j = i + 1; j < n && j - i < NGRAM_MAX; j++) {
                concat.append(words.get(j));
                String gram = concat.toString();
                if (vocab.contains(gram)) {
                    result.add(gram);
                }
            }
        }
        return result;
    }

    /**
     * 智能解析用户输入，兼容三种形态（供 predict-single 与 HTTP 接口共用）：
     * <ol>
     *   <li>训练集原始行（含两个 [...] 块）→ 直接走 DataParser；</li>
     *   <li>逗号/空格分隔的 tag 串 → 拆分后按词汇表过滤；</li>
     *   <li>自然语言句子 → 走 {@link #toCandidateTagsWithTf} 做词汇映射。</li>
     * </ol>
     *
     * @param text  用户输入原文
     * @param model 已加载模型（提供词汇表）
     * @return 可直接喂给 NbPredictor 的 tag 列表
     */
    public static List<String> resolveTags(String text, ModelParams model) {
        if (text == null) {
            return new ArrayList<>();
        }
        String cleaned = DataParser.stripTrailingCR(text).trim();
        if (cleaned.isEmpty()) {
            return new ArrayList<>();
        }

        // 形态1：完整数据行（含 [genre...] 与 [tag...] 两个块）
        if (cleaned.indexOf('[') >= 0) {
            Sample s = DataParser.parseLine(cleaned, 0);
            if (s != null && s.tags.length > 0) {
                List<String> tags = new ArrayList<>(s.tags.length);
                for (String t : s.tags) {
                    tags.add(t);
                }
                return tags;
            }
        }

        // 形态2/3：先按分隔符粗切，命中词汇表的直接保留（tag 串场景）
        Set<String> vocab = model.vocabIndex.keySet();
        List<String> direct = new ArrayList<>();
        for (String part : cleaned.split("[,;\\s]+")) {
            String t = part.trim().toLowerCase(Locale.ROOT);
            if (!t.isEmpty() && vocab.contains(t)) {
                direct.add(t);
            }
        }
        // 粗切已能覆盖大部分词汇（说明输入本身就是 tag 串），直接返回
        if (!direct.isEmpty()) {
            return direct;
        }

        // 形态3：自然语言句子 → 词级 + n-gram 拼接映射
        return toCandidateTagsWithTf(cleaned, vocab);
    }
}
