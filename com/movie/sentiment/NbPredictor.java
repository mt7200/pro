package com.movie.sentiment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 朴素贝叶斯预测器（多标签：one-vs-rest + 阈值激活）。
 *
 * <p>核心公式（对应任务书规范）：
 * <pre>
 *   w_t      = TF(t,x) * IDF(t)                      // 特征权重，OOV 跳过
 *   score(g) = logPrior[g] + Σ_{t∈tags∩V} w_t * logCond[g][t]   // log 域累加，防下溢
 *   激活     : score(g) > tauPerGenre[g]              // 多标签，不是 argmax
 *   softmax  : prob(g) = exp(score(g)-max) / Σ exp(score(g')-max) // 平移防溢出
 *   sigmoid  : conf(g) = 1/(1+exp(-(score(g)-tau[g])/scale[g]))   // 逐标签独立置信度
 * </pre>
 *
 * <p>所有参数来自 ModelParams，预测阶段【严禁】重算词汇表 / IDF / 概率。
 */
public class NbPredictor {

    /**
     * 输入模式：决定使用哪一套校准阈值。
     *
     * <p>TAGS  — 输入是完整 tag 列表（批量预测文件 / predict-single 传数据行），
     *           与训练分布一致，用 tauPerGenre；
     * <p>TEXT — 输入是前端自由文本剧情描述，通常只命中少量特征词，
     *           分数分布整体偏高，用 tauPerGenreText（按短列表分布单独校准）。
     */
    public enum InputMode { TAGS, TEXT }

    private final ModelParams model;

    public NbPredictor(ModelParams model) {
        this.model = model;
    }

    public ModelParams getModel() {
        return model;
    }

    /** 对 Sample 做预测（完整 tag 列表，TAGS 模式）。 */
    public PredictionResult predict(Sample sample) {
        return predictByTags(Arrays.asList(sample.tags), InputMode.TAGS);
    }

    /** 给定 tag 列表，按 TAGS 模式预测（保持旧调用签名兼容）。 */
    public PredictionResult predictByTags(List<String> tags) {
        return predictByTags(tags, InputMode.TAGS);
    }

    /**
     * 自由文本预测入口：先做词汇映射，再按 TEXT 模式打分。
     * 供成员4后端 / HTTP 接口调用。
     */
    public PredictionResult predictText(String freeText) {
        List<String> tags = TextTokenizer.resolveTags(freeText, model);
        return predictByTags(tags, InputMode.TEXT);
    }

    /**
     * 命中词数达到此值时，视为"完整 tag 列表"而非"短文本"，切换到 TAGS 模式校准。
     * 与 NbTrainer.TEXT_K_MAX（自由文本校准的抽样上限）保持一致：
     * 校准分布覆盖 1..8 个词，故 &gt;8 个词时按长列表处理更准确。
     */
    public static final int TEXT_MODE_MAX_TAGS = 8;

    /**
     * 自适应预测：根据实际命中的特征词数量自动选择校准模式。
     *
     * <p>为什么需要：同一份模型要同时服务两种输入——批量预测文件里的完整 tag 列表
     * （约 20 个词，分数低）与前端自由文本描述（约 2~6 个词，分数高）。
     * 两者的 score 分布不同，必须用各自校准的阈值，否则短文本会激活几乎所有标签。
     *
     * @param tags 已解析的特征词列表
     * @return 预测结果
     */
    public PredictionResult predictAuto(List<String> tags) {
        InputMode mode = tags.size() > TEXT_MODE_MAX_TAGS ? InputMode.TAGS : InputMode.TEXT;
        return predictByTags(tags, mode);
    }

    /**
     * 计算 tag 列表的 TF-IDF 权重映射（vocabIndex → w_t）。
     *
     * <p>实现要点：先统计每个唯一 tag 的 TF，再乘 IDF，避免"边遍历边累加"
     * 导致重复 tag 被计成 k² 倍。TF/IDF 均为【单文档局部贡献】，可下放 Mapper。
     */
    public Map<Integer, Double> computeWeights(List<String> tags) {
        // 1) 统计唯一 tag 的词频（仅统计在词汇表内的，OOV 静默跳过）
        Map<Integer, Integer> tfMap = new HashMap<>();
        for (String tag : tags) {
            Integer tIdx = model.vocabIndex.get(tag);
            if (tIdx == null) continue;   // OOV：不在词汇表中，静默跳过，不抛异常
            tfMap.merge(tIdx, 1, Integer::sum);
        }
        // 2) w_t = TF(t,d) * IDF(t)   （对应规范 TF-IDF 公式）
        Map<Integer, Double> weights = new HashMap<>(tfMap.size() * 2);
        for (Map.Entry<Integer, Integer> e : tfMap.entrySet()) {
            weights.put(e.getKey(), e.getValue() * model.idf[e.getKey()]);
        }
        return weights;
    }

    /**
     * 给定权重映射，计算各 genre 的原始 log 分数（不做激活/softmax）。
     * score(g) = logPrior[g] + Σ w_t * logCond[g][t]
     *
     * <p>严格对应任务书规范的打分公式，【批量预测路径（TAGS 模式）使用此式】。
     * 此步为【全局归一/平滑】结果的使用端，校准阶段同样复用。
     */
    public double[] computeRawScores(Map<Integer, Double> weights) {
        int G = model.G;
        double[] scores = new double[G];
        for (int gIdx = 0; gIdx < G; gIdx++) {
            double score = model.logPrior[gIdx];
            double[] condRow = model.logCond[gIdx];
            for (Map.Entry<Integer, Double> e : weights.entrySet()) {
                score += e.getValue() * condRow[e.getKey()];
            }
            scores[gIdx] = score;
        }
        return scores;
    }

    /**
     * 自由文本专用的【权重归一化】打分（TEXT 模式使用）。
     *
     * <p>为什么需要与原始公式不同：score(g) = logPrior[g] + Σ w_t·logCond[g][t]
     * 中求和项的负值幅度与命中 tag 的总权重 W 成正比。批量预测样本平均含约 20 个 tag
     * （W 大 → 分数低至 -1200），而前端一句短描述常只命中 2~6 个词（W 小 → 分数偏高），
     * 两者分数不在同一量纲上，任何固定阈值都会失配 —— 这才是"短文本激活 16/20 个标签"
     * 的根本原因（仅靠重新校准阈值只是打补丁，换一个长度的输入又会失效）。
     *
     * <p>归一化做法：把求和换成【加权平均对数似然】，得到与输入长度无关的量：
     * <pre>
     *   scoreNorm(g) = logPrior[g] + ( Σ w_t·logCond[g][t] ) / W,   W = Σ w_t
     * </pre>
     * 判别力不变（真实 genre 的平均对数似然更接近 0，错误 genre 更负），
     * 但对 W 不敏感，因此 2 个词和 20 个词的输入可直接共用同一套阈值。
     *
     * <p>注意：此式【仅用于前端自由文本】，不影响任务书规定的批量预测公式，
     * 模型主体参数（logPrior / logCond / idf）完全未改动。
     *
     * @param weights 特征权重映射
     * @return 归一化后的各 genre 分数
     */
    public double[] computeNormalizedScores(Map<Integer, Double> weights) {
        int G = model.G;
        double[] scores = new double[G];

        // 总权重 W（防御：全 OOV 时 W=0，退化为纯先验）
        double W = 0.0;
        for (double w : weights.values()) {
            W += w;
        }
        if (W <= 0.0) {
            for (int gIdx = 0; gIdx < G; gIdx++) {
                scores[gIdx] = model.logPrior[gIdx];
            }
            return scores;
        }

        for (int gIdx = 0; gIdx < G; gIdx++) {
            double sum = 0.0;
            double[] condRow = model.logCond[gIdx];
            for (Map.Entry<Integer, Double> e : weights.entrySet()) {
                sum += e.getValue() * condRow[e.getKey()];
            }
            scores[gIdx] = model.logPrior[gIdx] + sum / W;
        }
        return scores;
    }

    /**
     * 按输入模式选择打分方式：TAGS → 原始求和公式；TEXT → 权重归一化公式。
     * 训练期校准与预测期必须走同一分支，保证阈值与分数量纲一致。
     */
    public double[] computeScores(Map<Integer, Double> weights, InputMode mode) {
        return mode == InputMode.TEXT
                ? computeNormalizedScores(weights)
                : computeRawScores(weights);
    }

    /**
     * 取 genre g 在指定输入模式下的激活阈值。
     *
     * <p>回退链（保证旧模型文件也能用）：
     * TEXT 分桶阈值 → TEXT 不分桶阈值 → TAGS 逐 genre 阈值 → 全局 tau。
     *
     * @param k 命中的词汇表内特征词数（TEXT 模式下用于选桶；TAGS 模式忽略）
     */
    public double tauOf(int gIdx, InputMode mode, int k) {
        if (mode == InputMode.TEXT) {
            // 优先用分桶阈值：不同命中词数的分数分布形态不同，必须各用各的阈值
            int bucket = bucketIndexOf(k);
            if (bucket >= 0 && model.hasTextCalibrationByK()) {
                return model.tauPerGenreTextByK[bucket][gIdx];
            }
            if (model.hasTextCalibration()) {
                return model.tauPerGenreText[gIdx];
            }
        }
        if (model.hasPerGenreCalibration()) {
            return model.tauPerGenre[gIdx];
        }
        return model.tau;   // 未校准的旧模型：回退全局 tau
    }

    /** 取 genre g 的 sigmoid 尺度（与 {@link #tauOf} 同样的模式/分桶/回退逻辑）。 */
    public double scaleOf(int gIdx, InputMode mode, int k) {
        double scale;
        if (mode == InputMode.TEXT) {
            int bucket = bucketIndexOf(k);
            if (bucket >= 0 && model.hasTextCalibrationByK()) {
                scale = model.scalePerGenreTextByK[bucket][gIdx];
            } else if (model.hasTextCalibration()) {
                scale = model.scalePerGenreText[gIdx];
            } else if (model.hasPerGenreCalibration()) {
                scale = model.scalePerGenre[gIdx];
            } else {
                scale = 1.0;
            }
        } else if (model.hasPerGenreCalibration()) {
            scale = model.scalePerGenre[gIdx];
        } else {
            scale = 1.0;
        }
        // 防御：尺度非正会让 sigmoid 失效
        return (scale > 0) ? scale : 1e-6;
    }

    /**
     * 把命中词数 k 映射到分桶下标。
     *
     * <p>k 取值 1..textCalibKMax 时下标为 k-1；
     * k 超过上限时用【最后一桶】——因为 k 越大分数分布越稳定，
     * 用最大桶足够准确，也避免为极大 k 预留海量桶。
     * k<=0 或无分桶校准时返回 -1，由调用方走回退链。
     */
    private int bucketIndexOf(int k) {
        if (k <= 0 || !model.hasTextCalibrationByK()) {
            return -1;
        }
        return Math.min(k, model.textCalibKMax) - 1;
    }

    /**
     * 逐标签独立置信度：sigmoid((score - tau) / scale)，取值 0~1。
     *
     * <p>各标签相互独立、【不相加为 1】，正是接口规范第 4 节要求的形式。
     * 未校准的旧模型退化为以全局 tau 为中心的陡峭 sigmoid（近似激活指示）。
     */
    public double labelConfidence(int gIdx, double score, InputMode mode, int k) {
        double z = (score - tauOf(gIdx, mode, k)) / scaleOf(gIdx, mode, k);
        // 防溢出：|z| 很大时直接取端点，避免 exp 溢出为 Infinity
        if (z > 30) return 1.0;
        if (z < -30) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * 给定 tag 列表（已小写化）与输入模式，返回完整预测结果。
     * 供成员4后端 / HTTP 接口 / predict-single / 批量预测直接调用。
     */
    public PredictionResult predictByTags(List<String> tags, InputMode mode) {
        int G = model.G;

        // ── 1. 特征权重 + 分数（TAGS 用原始求和公式，TEXT 用权重归一化公式）──
        // 必须与 NbTrainer 校准时使用的分支一致，保证阈值与分数同量纲
        Map<Integer, Double> weights = computeWeights(tags);

        // ── 1.5 无有效证据短路 ──
        // 规范要求：全 OOV 时返回空 predGenres + confidence=0 + 结构化提示，不崩溃。
        //
        // 【为什么必须短路】零命中时 computeNormalizedScores 会走 W<=0 分支回退到
        // 纯先验 logPrior（实测 ≈ -2.83），而 TEXT 阈值是在"至少命中一个特征词"的
        // 分数分布（实测 ≈ -10.5）上校准的。两者不同量纲，-2.83 远高于阈值，
        // 会导致【全部标签被激活且 sigmoid 饱和为 1.0】——曾实测出 17/17 个标签、
        // confidence 全为 1.0 的错误结果。
        //
        // W<=0 也一并短路：若输入只命中 IDF<=0 的超高频 tag
        // （IDF=log(N/(DF+1))，当 DF+1>N 时为负），同样属于"无区分性证据"。
        double W = 0.0;
        for (double w : weights.values()) {
            W += w;
        }
        if (weights.isEmpty() || W <= 0.0) {
            return emptyResult(G);
        }

        double[] scores = computeScores(weights, mode);

        // 命中词汇表的【唯一特征词数】k：TEXT 模式用它选择对应的校准桶。
        // 用 weights.size()（已按 vocabIndex 去重）而非 tags.size()，
        // 因为决定分数噪声水平的是"命中了几个有效特征"，OOV 词不产生影响。
        int k = weights.size();

        // ── 2. 激活判定（多标签核心）：score(g) > tau(g, mode, k) ──
        List<String> predGenres = new ArrayList<>();
        List<Integer> activatedIdx = new ArrayList<>();
        for (int gIdx = 0; gIdx < G; gIdx++) {
            // 注意用"正向比较"而非 !(score <= tau)：后者在 score 为 NaN 时会误判为激活
            if (scores[gIdx] > tauOf(gIdx, mode, k)) {
                predGenres.add(model.genreArr[gIdx]);
                activatedIdx.add(gIdx);
            }
        }

        // ── 3. Softmax 相对置信（先减 max 平移，防溢出）──
        double maxScore = Double.NEGATIVE_INFINITY;
        for (int gIdx = 0; gIdx < G; gIdx++) {
            if (scores[gIdx] > maxScore) maxScore = scores[gIdx];
        }
        double[] probs = new double[G];
        double sumExp = 0.0;
        for (int gIdx = 0; gIdx < G; gIdx++) {
            probs[gIdx] = Math.exp(scores[gIdx] - maxScore);
            sumExp += probs[gIdx];
        }
        double maxProb = 0.0;
        for (int gIdx = 0; gIdx < G; gIdx++) {
            probs[gIdx] /= sumExp;
            if (probs[gIdx] > maxProb) maxProb = probs[gIdx];
        }

        // ── 4. 逐标签独立置信度（sigmoid），与 softmax 互不干扰 ──
        // 必须与激活判定用【同一个 k】选桶，否则会出现"标签已激活但 confidence 很低"
        // 这类自相矛盾的输出（阈值与 sigmoid 中心不一致所致）。
        double[] labelConf = new double[G];
        for (int gIdx = 0; gIdx < G; gIdx++) {
            labelConf[gIdx] = labelConfidence(gIdx, scores[gIdx], mode, k);
        }

        // confidence = 被激活类 softmax prob 之和（保留原字段语义）
        double confidence = 0.0;
        for (int gIdx : activatedIdx) {
            confidence += probs[gIdx];
        }

        // ── 5. 组装返回（LinkedHashMap 保持 genre 稳定顺序）──
        Map<String, Double> scoreMap = new LinkedHashMap<>();
        Map<String, Double> probMap = new LinkedHashMap<>();
        Map<String, Double> confMap = new LinkedHashMap<>();
        for (int gIdx = 0; gIdx < G; gIdx++) {
            String g = model.genreArr[gIdx];
            scoreMap.put(g, scores[gIdx]);
            probMap.put(g, probs[gIdx]);
            confMap.put(g, labelConf[gIdx]);
        }

        return new PredictionResult(predGenres, scoreMap, probMap, confMap,
                confidence, maxProb, scores, probs, labelConf, activatedIdx);
    }

    /**
     * 无有效证据时的空结果：不激活任何标签、confidence/maxProb 为 0。
     *
     * <p>scores/probs/labelConf 仍按 genre 顺序填入 0.0，保证 full schema 的
     * scores/probs 对象结构完整（前端 parse 不会因缺字段报错），
     * 同时让"零证据"在数值上可辨识（而非 -2.83 这类会被误读为先验偏好的值）。
     */
    private PredictionResult emptyResult(int G) {
        Map<String, Double> scoreMap = new LinkedHashMap<>();
        Map<String, Double> probMap = new LinkedHashMap<>();
        Map<String, Double> confMap = new LinkedHashMap<>();
        for (int gIdx = 0; gIdx < G; gIdx++) {
            String g = model.genreArr[gIdx];
            scoreMap.put(g, 0.0);
            probMap.put(g, 0.0);
            confMap.put(g, 0.0);
        }
        return new PredictionResult(
                new ArrayList<String>(),          // predGenres：空
                scoreMap, probMap, confMap,
                0.0,                              // confidence = 0（规范要求）
                0.0,                              // maxProb = 0
                new double[G], new double[G], new double[G],
                new ArrayList<Integer>());        // activatedIdx：空
    }

    /** 预测结果结构体 */
    public static class PredictionResult {
        /** 激活的 genre 名称列表（多标签） */
        public final List<String> predGenres;
        /** genre → log 分数 */
        public final Map<String, Double> scores;
        /** genre → softmax 相对概率（相加为 1） */
        public final Map<String, Double> probs;
        /** genre → 逐标签独立置信度（0~1，不相加为 1） */
        public final Map<String, Double> labelConfidences;
        /** 被激活类的 softmax prob 之和 */
        public final double confidence;
        /** 最大 softmax prob */
        public final double maxProb;

        /** 原始数组视图（按 genreIndex），供 HTTP 层高效取值 */
        public final double[] scoreArr;
        public final double[] probArr;
        public final double[] labelConfArr;
        public final List<Integer> activatedIdx;

        public PredictionResult(List<String> predGenres,
                                Map<String, Double> scores,
                                Map<String, Double> probs,
                                Map<String, Double> labelConfidences,
                                double confidence, double maxProb,
                                double[] scoreArr, double[] probArr, double[] labelConfArr,
                                List<Integer> activatedIdx) {
            this.predGenres = predGenres;
            this.scores = scores;
            this.probs = probs;
            this.labelConfidences = labelConfidences;
            this.confidence = confidence;
            this.maxProb = maxProb;
            this.scoreArr = scoreArr;
            this.probArr = probArr;
            this.labelConfArr = labelConfArr;
            this.activatedIdx = activatedIdx;
        }
    }
}
