package com.movie.sentiment;

import java.util.*;

/**
 * 多标签评估指标计算。
 * 当预测行带 ground-truth genres 时计算；否则为 null。
 */
public class Metrics {

    /**
     * 评估结果容器。
     */
    public static class EvalResult {
        public int n;                                    // 参与评估的样本数
        public Double subsetAccuracy;                    // 严格匹配准确率
        public Double microPrecision, microRecall, microF1;
        public Double macroPrecision, macroRecall, macroF1;
        public Double hammingLoss;
        public Map<String, PerGenre> perGenre;

        public static class PerGenre {
            public double precision, recall, f1;
            public int support;
        }
    }

    /**
     * 计算评估指标。
     *
     * @param allTrue  每个样本的真实 genre 集合（多标签）
     * @param allPred  每个样本的预测 genre 集合
     * @param genreArr 所有 genre 名称
     * @param genreIndex genre→索引
     * @return EvalResult
     */
    public static EvalResult evaluate(List<Set<String>> allTrue, List<Set<String>> allPred,
                                       String[] genreArr, Map<String, Integer> genreIndex) {
        int n = allTrue.size();
        int G = genreArr.length;

        EvalResult r = new EvalResult();
        r.n = n;

        // ── Per-genre 计数 ──
        int[] tpPerG = new int[G];
        int[] fpPerG = new int[G];
        int[] fnPerG = new int[G];
        int[] supportPerG = new int[G];

        // ── global micro 计数 ──
        int globalTP = 0, globalFP = 0, globalFN = 0;
        int exactMatch = 0;
        int hammingTotal = 0;  // 分母 = n * |G|

        for (int i = 0; i < n; i++) {
            Set<String> tSet = allTrue.get(i);
            Set<String> pSet = allPred.get(i);

            // support 统计
            for (String g : tSet) {
                Integer gi = genreIndex.get(g);
                if (gi != null) supportPerG[gi]++;
            }

            // subset (exact match) accuracy
            if (tSet.equals(pSet)) exactMatch++;

            // Hamming loss 逐 label
            for (int gIdx = 0; gIdx < G; gIdx++) {
                String g = genreArr[gIdx];
                boolean inTrue = tSet.contains(g);
                boolean inPred = pSet.contains(g);
                if (inTrue && inPred) {
                    tpPerG[gIdx]++;
                    globalTP++;
                } else if (!inTrue && inPred) {
                    fpPerG[gIdx]++;
                    globalFP++;
                } else if (inTrue && !inPred) {
                    fnPerG[gIdx]++;
                    globalFN++;
                }
                // !inTrue && !inPred → TN（不计）
                if (inTrue != inPred) hammingTotal++;
            }
        }

        // ── Micro ──
        if (globalTP + globalFP > 0) {
            r.microPrecision = (double) globalTP / (globalTP + globalFP);
        } else {
            r.microPrecision = 0.0;
        }
        if (globalTP + globalFN > 0) {
            r.microRecall = (double) globalTP / (globalTP + globalFN);
        } else {
            r.microRecall = 0.0;
        }
        if (r.microPrecision + r.microRecall > 0) {
            r.microF1 = 2.0 * r.microPrecision * r.microRecall / (r.microPrecision + r.microRecall);
        } else {
            r.microF1 = 0.0;
        }

        // ── Macro ──
        double sumP = 0, sumR = 0, sumF = 0;
        int countG = 0;
        r.perGenre = new LinkedHashMap<>();
        for (int gIdx = 0; gIdx < G; gIdx++) {
            EvalResult.PerGenre pg = new EvalResult.PerGenre();
            int tp = tpPerG[gIdx];
            int fp = fpPerG[gIdx];
            int fn = fnPerG[gIdx];
            pg.support = supportPerG[gIdx];

            if (tp + fp > 0) {
                pg.precision = (double) tp / (tp + fp);
            } else {
                pg.precision = 0.0;
            }
            if (tp + fn > 0) {
                pg.recall = (double) tp / (tp + fn);
            } else {
                pg.recall = 0.0;
            }
            if (pg.precision + pg.recall > 0) {
                pg.f1 = 2.0 * pg.precision * pg.recall / (pg.precision + pg.recall);
            } else {
                pg.f1 = 0.0;
            }

            r.perGenre.put(genreArr[gIdx], pg);

            // 只有 support > 0 才计入 macro 平均（常见做法；也可全部计入）
            // 这里按全部 genre 计入，与 sklearn 默认一致
            sumP += pg.precision;
            sumR += pg.recall;
            sumF += pg.f1;
            countG++;
        }
        r.macroPrecision = countG > 0 ? sumP / countG : 0.0;
        r.macroRecall    = countG > 0 ? sumR / countG : 0.0;
        r.macroF1        = countG > 0 ? sumF / countG : 0.0;

        // ── Subset accuracy ──
        r.subsetAccuracy = (double) exactMatch / n;

        // ── Hamming loss ──
        r.hammingLoss = (double) hammingTotal / (n * G);

        return r;
    }
}