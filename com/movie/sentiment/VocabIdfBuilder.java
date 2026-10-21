package com.movie.sentiment;

import java.util.*;

/**
 * 词汇表与 IDF 构建器。
 * 遍历训练集，构建 tag 词汇表（vocabIndex）、文档频率 DF、IDF。
 *
 * 所有统计量设计为可增量合并（merge 友好），满足"各 Mapper 输出局部计数、
 * Reducer 端求和"即与单机一遍扫描完全一致的要求。
 */
public class VocabIdfBuilder {

    /** tag → (词汇表索引, 从0开始) */
    public Map<String, Integer> vocabIndex;
    /** 词汇表（索引→tag） */
    public String[] vocabArr;

    /** 文档频率：DF[tIndex] = tags 中包含词 t 的样本条数  — 可在 Reduce 端聚合 */
    public int[] df;

    /** IDF[tIndex] = log(N / (DF[t]+1))，+1 防除零 */
    public double[] idf;

    /** 训练样本总数 */
    public int N;

    /**
     * 从训练样本列表构建词汇表与 IDF（标准模式：TF-IDF）。
     */
    public void build(List<Sample> trainSamples) {
        build(trainSamples, false);
    }

    /**
     * 从训练样本列表构建词汇表与 IDF。
     *
     * @param rawTfOnly true = 【消融实验】模式：把所有 IDF 置为 1.0，
     *                  此时 TFIDF(t,d) = TF(t,d) × 1 = TF(t,d)，即退化为"纯原始词频"。
     *                  用于验证 IDF 降权到底带来多少增益（对应 PPT 第04页的对比叙事）。
     *                  默认 false = 标准 TF-IDF，公式为 IDF(t)=log(N/(DF(t)+1))。
     *
     *                  <p>【注意】此开关仅用于消融对比，正式训练/预测一律 false。
     *                  DF 仍照常统计并保留在 df[] 中，便于对比时输出真实 DF。
     */
    public void build(List<Sample> trainSamples, boolean rawTfOnly) {
        N = trainSamples.size();
        Map<String, Integer> dfMap = new LinkedHashMap<>();

        // 第一遍：统计 DF（每样本文档内去重计数）
        // 文档内去重是 DF 定义的要求：DF(t)="含 t 的样本条数"，一篇文档只计一次
        for (Sample s : trainSamples) {
            Set<String> seen = new HashSet<>();
            for (String t : s.tags) {
                if (seen.add(t)) {
                    // 此统计量可在 Reduce 端聚合：各 Mapper 对 (t, 1) 求和
                    dfMap.merge(t, 1, Integer::sum);
                }
            }
        }

        // 构建 vocabIndex / vocabArr
        int V = dfMap.size();
        vocabIndex = new LinkedHashMap<>();
        vocabArr = new String[V];
        df = new int[V];
        idf = new double[V];

        int idx = 0;
        for (Map.Entry<String, Integer> e : dfMap.entrySet()) {
            vocabIndex.put(e.getKey(), idx);
            vocabArr[idx] = e.getKey();
            df[idx] = e.getValue();
            idx++;
        }

        // 计算 IDF
        for (int i = 0; i < V; i++) {
            if (rawTfOnly) {
                // 消融：IDF 恒为 1 → TFIDF 退化为纯 TF（原始词频）
                idf[i] = 1.0;
            } else {
                // IDF(t) = log(N / (DF(t) + 1))   （+1 防除零，对应规范 IDF 公式）
                idf[i] = Math.log((double) N / (df[i] + 1));
            }
        }
    }

    /** TF(t, d) = tag t 在样本 d 的 tags 数组中出现的次数 */
    public static int tf(String t, Sample d) {
        int count = 0;
        for (String tag : d.tags) {
            if (tag.equals(t)) count++;
        }
        return count;
    }
}