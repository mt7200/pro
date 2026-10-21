package com.movie.sentiment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * 朴素贝叶斯训练器。
 * 计算 logPrior 与 logCond，组装 ModelParams；并做逐 genre 阈值校准。
 *
 * <p>统计量设计为可增量合并（各 Mapper 输出局部贡献 → Reducer 端求和 = 单机结果），
 * 为第二阶段 MapReduce 改造留好接口。
 */
public class NbTrainer {

    /**
     * 从训练样本、词汇表、IDF 训练完整模型（genre 顺序按数据首次出现顺序）。
     */
    public ModelParams train(List<Sample> trainSamples, VocabIdfBuilder vocabBuilder, double tau, String trainSource) {
        return train(trainSamples, vocabBuilder, tau, trainSource, null);
    }

    /**
     * 从训练样本、词汇表、IDF 训练完整模型。
     *
     * @param fixedGenreOrder 非 null 时，强制 genre 按此顺序排列（用于校准期训练临时模型）。
     *                        必要性：genreArr 的顺序由 LinkedHashMap 插入序决定，即"哪个 genre
     *                        最先出现"。临时模型只用 90% 数据，罕见标签（如 tvmovie，freq=256）
     *                        的首次出现位置会变，导致 genreArr 顺序与全量模型【不一致】；
     *                        若不固定顺序，按索引搬运阈值就会张冠李戴。
     */
    public ModelParams train(List<Sample> trainSamples, VocabIdfBuilder vocabBuilder, double tau,
                             String trainSource, String[] fixedGenreOrder) {
        int V = vocabBuilder.vocabArr.length;

        // ── 1. 统计 freq(g) ──
        // 此统计量可在 Reduce 端聚合：freq(g) 为可交换可结合的计数，
        // 各 Mapper 输出局部 freq、Reducer 求和 == 单机一遍扫描的结果。
        Map<String, Integer> genreFreq = new LinkedHashMap<>();
        for (Sample s : trainSamples) {
            for (String g : s.genres) {
                genreFreq.merge(g, 1, Integer::sum);
            }
        }

        // ── 2. 统计 S(g,t) ──
        // S(g,t) = Σ_{d: g∈genres(d)} TFIDF(t,d)
        // 此统计量可在 Reduce 端聚合：单文档局部贡献 tfidf 可直接相加（满足结合律）。
        // 【性能说明】外层遍历 tag、内层遍历 genre，并对 (g,t) 组合去重，
        // 复杂度约 O(N × |tags| × |genres_of_doc))，实测 28 万样本约 1 分钟。
        double[][] S = new double[fixedGenreOrder != null ? fixedGenreOrder.length : genreFreq.size()][V];
        // 先确定 genre 顺序，才能把 genreFreq/S 的下标固定下来
        String[] genreArr;
        Map<String, Integer> genreIndex = new HashMap<>();
        if (fixedGenreOrder != null) {
            genreArr = fixedGenreOrder.clone();
        } else {
            genreArr = new String[genreFreq.size()];
            int gi = 0;
            for (String g : genreFreq.keySet()) {
                genreArr[gi++] = g;
            }
        }
        for (int i = 0; i < genreArr.length; i++) {
            genreIndex.put(genreArr[i], i);
        }
        S = new double[genreArr.length][V];

        double[] idf = vocabBuilder.idf;
        for (Sample s : trainSamples) {
            // 对每个 (tag, genre) 组合只累加一次，避免重复 tag 被计多次
            Set<String> seenTags = new HashSet<>();
            for (String tag : s.tags) {
                if (!seenTags.add(tag)) continue;   // 同一文档内重复 tag 只算一次 TF-IDF 贡献
                Integer tIdx = vocabBuilder.vocabIndex.get(tag);
                if (tIdx == null) continue;         // 不在词汇表中（防御）
                // TFIDF(t,d) = TF(t,d) × IDF(t)  ← 单文档局部贡献
                double tfidf = VocabIdfBuilder.tf(tag, s) * idf[tIdx];
                // 多标签：一条样本可贡献多个 g
                for (String g : s.genres) {
                    Integer gIdxObj = genreIndex.get(g);
                    if (gIdxObj == null) continue;   // 固定顺序下该 genre 不在索引表中，跳过
                    S[gIdxObj][tIdx] += tfidf;
                }
            }
        }

        // ── 3. 归一 + 平滑 + 组装（与 MapReduce 共用同一份实现）──
        return assembleModel(vocabBuilder.vocabArr, vocabBuilder.vocabIndex, idf,
                vocabBuilder.N, genreFreq, S, genreArr, tau, trainSource);
    }

    /**
     * 【全局归一 + 平滑 + 组装】—— 由【已聚合的计数】装配完整模型。
     *
     * <p><b>为什么单独抽出这个方法</b>：单机训练（{@link #train}）与 MapReduce 路径
     * （{@code com.movie.sentiment.mr.MrTrainDriver}）都必须走这同一段"归一+平滑"公式。
     * 若 MR 里另写一份，两处公式极易漂移（改了一处忘另一处），违反一致性要求。
     * 抽出后：Mapper/Reducer 只负责产出【局部计数并求和】，
     * driver 端拿到聚合结果调用本方法即可，公式只有一份实现。
     *
     * <p>本方法内的两步都属于"需要全局量"的收尾计算，必须在聚合完成后执行：
     * <ul>
     *   <li>先验：prior(g) = (freq(g)+1) / (totalLabelOcc + |G|)，需 totalLabelOcc = Σ_g freq(g)；</li>
     *   <li>条件概率：P(t|g) = (S(g,t)+1) / (Z(g) + |V|)，需 Z(g) = Σ_t S(g,t)。</li>
     * </ul>
     * 全程 log 域存储，取概率前 Math.max(p, 1e-12) 防 log(0)（对应规范"鲁棒性"条目）。
     *
     * @param vocabArr    词汇表（tag 原文数组）
     * @param vocabIndex  tag → 下标
     * @param idf         各 tag 的 IDF（按 vocabIndex）
     * @param N           训练样本总数（IDF 与模型元信息用）
     * @param genreFreq   已聚合的 freq(g)（此统计量可在 Reduce 端聚合）
     * @param S           已聚合的 S(g,t) 矩阵，下标须与 genreArr / vocabArr 对齐
     * @param genreArr    genre 顺序（决定 genreIndex，单机与 MR 必须一致）
     * @param tau         全局阈值占位（后续校准会覆盖）
     * @param trainSource 训练数据来源标识（写入模型便于溯源）
     * @return 完整模型参数
     */
    public static ModelParams assembleModel(String[] vocabArr, Map<String, Integer> vocabIndex,
                                            double[] idf, int N,
                                            Map<String, Integer> genreFreq, double[][] S,
                                            String[] genreArr, double tau, String trainSource) {
        int V = vocabArr.length;
        int G = genreArr.length;
        Map<String, Integer> genreIndex = new HashMap<>();
        for (int i = 0; i < G; i++) {
            genreIndex.put(genreArr[i], i);
        }

        // ── 先验：prior(g) = (freq(g) + 1) / (totalLabelOcc + |G|) ──
        // totalLabelOcc = Σ_g freq(g)：所有标签出现次数之和（多标签口径，一条样本可贡献多个 g）
        // 此统计量可在 Reduce 端聚合
        long totalLabelOcc = 0L;
        for (int f : genreFreq.values()) {
            totalLabelOcc += f;
        }
        double[] logPrior = new double[G];
        for (int i = 0; i < G; i++) {
            String g = genreArr[i];
            // 固定 genre 顺序时，罕见标签（如 tvmovie，freq=256）可能未落在本次子集里，
            // 故用 getOrDefault 兜底，避免 Integer 拆箱 NPE
            int freq = genreFreq.getOrDefault(g, 0);
            // 加1平滑（拉普拉斯），对应规范 prior 公式
            double prior = (freq + 1.0) / (totalLabelOcc + G);
            logPrior[i] = Math.log(Math.max(prior, 1e-12));
        }

        // ── 条件概率：P(t|g) = (S(g,t) + 1) / (Z(g) + |V|) ──
        // Z(g) = Σ_t S(g,t)，此步为"全局归一/平滑"，依赖 Reduce 端聚合后的 S(g,t)
        double[][] logCond = new double[G][V];
        for (int gIdx = 0; gIdx < G; gIdx++) {
            double Z = 0.0;
            for (int tIdx = 0; tIdx < V; tIdx++) {
                Z += S[gIdx][tIdx];
            }
            // Z(g) 也可在 Reduce 端对各 Mapper 的局部 Z 求和得到（加法可结合）
            for (int tIdx = 0; tIdx < V; tIdx++) {
                // 分子+1、分母+|V| 即加1平滑，防 0 概率（对应规范条件概率公式）
                double prob = (S[gIdx][tIdx] + 1.0) / (Z + V);
                logCond[gIdx][tIdx] = Math.log(Math.max(prob, 1e-12));
            }
        }

        // ── 组装 ModelParams ──
        ModelParams model = new ModelParams();
        model.genreArr = genreArr;
        model.genreIndex = genreIndex;
        model.vocabArr = vocabArr;
        model.vocabIndex = vocabIndex;
        model.idf = idf;
        model.logPrior = logPrior;
        model.logCond = logCond;
        model.N = N;
        model.V = V;
        model.G = G;
        model.tau = tau;
        model.trainedAtMillis = System.currentTimeMillis();
        model.trainSource = trainSource;
        return model;
    }

    // ══════════════════════ 阈值校准（留出集，消除内样本偏差） ══════════════════════

    /** 留出集比例：用 90% 训练临时模型，在剩余 10% 上校准阈值 */
    public static final double HOLDOUT_RATIO = 0.1;

    /** 自由文本模拟的 tag 命中数上限（k 取 1..TEXT_K_MAX 均匀随机） */
    private static final int TEXT_K_MAX = 8;

    /**
     * 逐 genre 阈值校准（one-vs-rest 最优 F1 阈值）—— 用于【完整 tag 列表】输入。
     *
     * <p>为什么必须校准：朴素贝叶斯的 log 域分数
     * score(g) = logPrior[g] + Σ w_t·logCond[g][t] 天然是较大的负数（实测 -50 ~ -1300），
     * 任何固定阈值（如任务书默认的 tau=0.0）都无法激活标签，预测结果恒为空；
     * 且不同 genre 的分数分布差异极大，用单一全局阈值必然顾此失彼。
     *
     * <p>为什么用留出集：若直接在训练集上搜阈值，模型对这些样本存在记忆效应，
     * 正样本分数被系统性抬高，得到的阈值过于宽松，应用到新数据时会严重过度预测
     * （实测平均每样本预测 8.6 个标签，而真实仅 2.6 个）。故用 90% 样本训练
     * 一个临时模型，在剩余 10% 留出样本上搜索阈值，再把阈值套用到全量模型上。
     *
     * @param model        已用全量数据训练好的模型（阈值字段为空），本方法就地写入校准结果
     * @param trainSamples 全量训练样本
     * @param maxSamples   留出集中最多用于校准的样本数（控制耗时）
     */
    public static void calibrate(ModelParams model, List<Sample> trainSamples, int maxSamples) {
        calibrateInternal(model, trainSamples, maxSamples, false);
    }

    /**
     * 自由文本（前端剧情描述）专用阈值校准。
     *
     * <p>为什么需要单独校准：score(g) 的负值幅度与命中 tag 的总权重 W 正相关。
     * 训练样本平均含约 20 个 tag（W 大 → 分数低至 -1200），而前端一句短描述
     * 通常只命中 2~6 个词汇（W 小 → 分数偏高）。若沿用按长列表校准的阈值，
     * 短文本会激活几乎所有标签（实测一句 10 词的句子激活 16/20 个标签，毫无区分度）。
     *
     * <p>做法：TEXT 模式改用【权重归一化】打分（见 NbPredictor.computeNormalizedScores），
     * 使分数与输入长度无关；再在这个分布上做留出集最优 F1 校准。
     * 模型主体参数（logPrior / logCond / idf）完全不变，仍严格遵守任务书公式。
     *
     * @param model        模型（就地写入 tauPerGenreText / scalePerGenreText）
     * @param trainSamples 全量训练样本
     * @param maxSamples   留出集中最多用于校准的样本数
     */
    public static void calibrateTextProfile(ModelParams model, List<Sample> trainSamples, int maxSamples) {
        calibrateInternal(model, trainSamples, maxSamples, true);
    }

    /**
     * 校准主流程：切分留出集 → 训练临时模型 → 在留出集上搜最优阈值 → 写入目标模型。
     *
     * @param textProfile true = 自由文本模式（短列表降采样 + 归一化打分）
     */
    private static void calibrateInternal(ModelParams model, List<Sample> trainSamples,
                                          int maxSamples, boolean textProfile) {
        // 1) 确定性切分留出集（与分桶校准共用同一实现，保证两条路径口径一致）
        Holdout h = splitHoldout(trainSamples, textProfile ? "TEXT" : "TAGS");
        if (h == null) {
            return;   // 样本不足，splitHoldout 内已打印说明
        }

        // 2) 用 90% 数据训练临时模型（与分桶校准共用同一实现）
        ModelParams tempModel = buildTempModel(model, h.tempTrain);

        // 3) 在留出集上采集分数并搜索最优阈值
        NbPredictor predictor = new NbPredictor(tempModel);
        List<double[]> scores = new ArrayList<>();
        List<Set<String>> truth = new ArrayList<>();
        collectScores(predictor, h.holdout, maxSamples, textProfile, TEXT_K_MAX,
                new Random(4242L), scores, truth);

        // 阈值是在临时模型上搜的，但要套用到全量模型；两者参数仅差 10% 数据，
        // 分数分布几乎一致，远比"在训练集内校准"可靠。
        scanOptimalTau(model, tempModel, scores, truth, textProfile);
    }

    /** 留出集切分结果 */
    private static class Holdout {
        List<Sample> tempTrain;   // 90%：用于训练临时模型
        List<Sample> holdout;     // 10%：用于搜索阈值（消除训练集内校准的记忆偏差）
    }

    /**
     * 确定性切分留出集：90% 训练临时模型，10% 用于搜阈值。
     *
     * <p>抽成独立方法的原因：{@link #calibrateInternal}（TAGS / 不分桶 TEXT）
     * 与 {@link #calibrateTextProfileByK}（分桶 TEXT）必须用【完全相同】的切分，
     * 否则两套阈值来自不同数据划分，无法比较也难复现。
     *
     * @param tag 日志标识（TAGS / TEXT / TEXT-k 分桶）
     * @return 切分结果；样本不足 20 条时返回 null（调用方应跳过校准）
     */
    private static Holdout splitHoldout(List<Sample> trainSamples, String tag) {
        int total = trainSamples.size();
        if (total < 20) {
            // 样本太少时无法做有意义的留出集校准（临时训练集几乎为空，阈值不可信）
            System.out.println("[校准-" + tag + "] 训练样本仅 " + total
                    + " 条（<20），跳过留出集校准；如需校准请提供更充分的训练集。");
            return null;
        }
        // 留出集规模：目标 10%，但至少 1000 条以保证阈值估计稳定；
        // 【边界修复】小样本下 Math.max(1000, ...) 可能超过 total，导致
        // total - holdoutSize 为负、new ArrayList(负数) 抛 IllegalArgumentException。
        // 故必须对 total 取上限，并保证临时训练集至少留下 50% 数据可训。
        int holdoutSize = Math.max(1000, (int) (total * HOLDOUT_RATIO));
        holdoutSize = Math.min(holdoutSize, Math.max(1, total / 2));

        // 固定种子 → 多次训练结果完全一致，可复现
        List<Integer> idx = new ArrayList<>(total);
        for (int i = 0; i < total; i++) idx.add(i);
        java.util.Collections.shuffle(idx, new Random(20261020L));

        Holdout h = new Holdout();
        h.tempTrain = new ArrayList<>(total - holdoutSize);
        h.holdout = new ArrayList<>(holdoutSize);
        for (int i = 0; i < total; i++) {
            Sample s = trainSamples.get(idx.get(i));
            if (i < holdoutSize) h.holdout.add(s); else h.tempTrain.add(s);
        }
        System.out.println("[校准-" + tag + "] 留出集=" + h.holdout.size()
                + " 条，临时训练集=" + h.tempTrain.size() + " 条");
        return h;
    }

    /**
     * 用 90% 数据训练临时模型，复用全量模型的词汇表/IDF/genre 顺序。
     *
     * <p><b>关键</b>：必须把全量模型的 genreArr 顺序传给临时模型，否则两者的 genre
     * 索引不一致，按索引搬运阈值会张冠李戴（第一阶段实测让 action 阈值变成 -7857）。
     * 复用全量 idf/vocabIndex 也避免了二次扫描 28 万样本。
     */
    private static ModelParams buildTempModel(ModelParams model, List<Sample> tempTrain) {
        VocabIdfBuilder vb = new VocabIdfBuilder();
        vb.vocabArr = model.vocabArr;
        vb.vocabIndex = model.vocabIndex;
        vb.df = new int[model.V];
        vb.idf = model.idf;
        vb.N = model.N;
        return new NbTrainer().train(tempTrain, vb, model.tau, model.trainSource, model.genreArr);
    }

    /**
     * 按【固定命中词数 k】采集留出集分数，用于分桶校准。
     *
     * <p>与 {@link #collectScores} 的区别：后者随机取 k∈[1,TEXT_K_MAX]，得到的是
     * 混合分布；本方法固定 k，得到单一分布，才能让每个桶有自己的阈值。
     *
     * @param fixedK 该桶对应的命中词数
     */
    private static void collectScoresFixedK(NbPredictor predictor, List<Sample> samples,
                                            int maxSamples, int fixedK, Random rnd,
                                            List<double[]> outScores, List<Set<String>> outTruth) {
        int total = samples.size();
        int step = Math.max(1, total / Math.max(1, maxSamples));
        for (int i = 0; i < total; i += step) {
            Sample s = samples.get(i);
            if (s.tags.length == 0) {
                continue;   // 无 tag 的样本对任何 k 都无贡献
            }
            // 随机保留恰好 fixedK 个 tag（不足则全取），模拟前端命中 k 个特征词的情形
            int k = Math.min(s.tags.length, fixedK);
            List<String> pool = new ArrayList<>(Arrays.asList(s.tags));
            java.util.Collections.shuffle(pool, rnd);
            List<String> tags = pool.subList(0, k);

            Map<Integer, Double> weights = predictor.computeWeights(tags);
            if (weights.isEmpty()) {
                continue;   // 全 OOV：预测期会短路返回空结果，校准也应排除，否则拉偏阈值
            }
            // 必须与 NbPredictor.predictByTags 的 TEXT 分支一致（权重归一化打分），
            // 否则阈值与分数不同量纲
            outScores.add(predictor.computeScores(weights, NbPredictor.InputMode.TEXT));
            outTruth.add(new HashSet<>(Arrays.asList(s.genres)));
        }
    }

    /**
     * 【自由文本分桶校准】按命中词数 k=1..TEXT_K_MAX 分别校准一套阈值。
     *
     * <p><b>为什么必须分桶</b>（第二阶段修复的核心缺陷）：
     * TEXT 模式虽然用权重归一化让分数不再随输入长度线性漂移，但【噪声水平】仍与
     * 命中词数强相关——命中 1 个词时，若该词恰是某错误 genre 的高权重 tag，
     * 就会得到很高的分数（噪声主导）；命中 8 个词时真实 genre 的信号被平均放大、
     * 错误 genre 相互抵消（信号主导）。两者分布形态不同，共用一套阈值必然对短输入过松。
     *
     * <p><b>实测证据</b>（未分桶时）：
     * <ul>
     *   <li>输入 {@code "murder"}（命中 1 词）→ 激活 14/20 个标签</li>
     *   <li>输入完整侦探长句（命中 6~8 词）→ 只激活 3 个</li>
     *   <li>而真实平均标签数仅 2.58</li>
     * </ul>
     * 分桶后每个 k 用各自分布定阈值，短输入的过度预测被显著抑制。
     *
     * <p>同时仍写入不分桶的 {@code tauPerGenreText}（取 k 的中位桶），
     * 作为旧模型兼容与 k 超出范围时的回退。
     *
     * @param model        目标模型（就地写入 tauPerGenreTextByK / scalePerGenreTextByK）
     * @param trainSamples 全量训练样本
     * @param maxSamples   每个桶最多使用的留出样本数
     */
    public static void calibrateTextProfileByK(ModelParams model, List<Sample> trainSamples,
                                               int maxSamples) {
        Holdout h = splitHoldout(trainSamples, "TEXT-分桶");
        if (h == null) {
            return;
        }
        ModelParams tempModel = buildTempModel(model, h.tempTrain);
        NbPredictor predictor = new NbPredictor(tempModel);

        int kMax = TEXT_K_MAX;
        double[][] tauByK = new double[kMax][];
        double[][] scaleByK = new double[kMax][];

        for (int k = 1; k <= kMax; k++) {
            List<double[]> scores = new ArrayList<>();
            List<Set<String>> truth = new ArrayList<>();
            // 每个桶用不同但固定的种子，保证可复现且各桶抽样相互独立
            collectScoresFixedK(predictor, h.holdout, maxSamples, k,
                    new Random(4242L + k), scores, truth);

            if (scores.size() < 20) {
                // 该桶样本过少（如留出集里 tag 数普遍不足 k），阈值不可信。
                // 复制相邻较小桶的结果，保证数组完整、预测期不会取到未初始化的桶。
                System.out.println("[校准-TEXT k=" + k + "] 有效样本仅 " + scores.size()
                        + " 条（<20），复用 k=" + (k - 1) + " 的阈值");
                tauByK[k - 1] = (k > 1) ? tauByK[k - 2].clone() : new double[model.G];
                scaleByK[k - 1] = (k > 1) ? scaleByK[k - 2].clone() : newScaleOnes(model.G);
                continue;
            }

            // ★ 复用同一份扫描算法（scanTauCore），8 个桶不会各写一遍公式
            TauScanResult res = scanTauCore(model, scores, truth);
            double[] tauOut = res.tau;
            double[] scaleOut = res.scale;

            // 每桶独立搜索全局偏移：不同 k 的过度预测程度不同，不能共用一个 k 系数
            double kOffset = searchGlobalOffset(model.genreArr, tauOut, scaleOut,
                    scores, truth, "-TEXT k=" + k);
            for (int gIdx = 0; gIdx < model.G; gIdx++) {
                tauOut[gIdx] += kOffset * scaleOut[gIdx];
            }

            tauByK[k - 1] = tauOut;
            scaleByK[k - 1] = scaleOut;

            // 每桶只打印一行汇总，避免 8 桶 × 20 genre = 160 行刷屏
            double avgTau = 0.0, avgF1 = 0.0;
            for (int gIdx = 0; gIdx < model.G; gIdx++) { avgTau += tauOut[gIdx]; avgF1 += res.f1[gIdx]; }
            System.out.printf(java.util.Locale.ROOT,
                    "  [TEXT k=%d] 样本=%-6d 平均tau=%-10.4f 平均holdoutF1=%.4f  偏移=%.3f%n",
                    k, scores.size(), avgTau / model.G, avgF1 / model.G, kOffset);
        }

        model.tauPerGenreTextByK = tauByK;
        model.scalePerGenreTextByK = scaleByK;
        model.textCalibKMax = kMax;

        // 不分桶字段取【中位桶】作为回退值：k 超出 kMax 或旧代码路径时使用
        int mid = kMax / 2;
        model.tauPerGenreText = tauByK[Math.min(mid, kMax - 1)].clone();
        model.scalePerGenreText = scaleByK[Math.min(mid, kMax - 1)].clone();

        System.out.println("[校准-TEXT 分桶] 完成，共 " + kMax + " 个桶（k=1.." + kMax
                + "），回退桶取 k=" + (Math.min(mid, kMax - 1) + 1));
    }

    /** 生成全 1 的 scale 数组（用于样本不足的桶，使 sigmoid 不至于退化） */
    private static double[] newScaleOnes(int G) {
        double[] s = new double[G];
        Arrays.fill(s, 1.0);
        return s;
    }

    /**
     * 采集校准用的 (scores, truth) 对。
     *
     * @param shorten  是否做"短列表"降采样（自由文本模式用）
     * @param kMax     降采样时每条样本保留的最大 tag 数
     * @param rnd      随机源（shorten=true 时必填）
     * @param outScores 输出：每条样本的分数数组
     * @param outTruth  输出：每条样本的真实 genre 集合
     */
    private static void collectScores(NbPredictor predictor, List<Sample> samples, int maxSamples,
                                      boolean shorten, int kMax, Random rnd,
                                      List<double[]> outScores, List<Set<String>> outTruth) {
        int total = samples.size();
        int step = Math.max(1, total / Math.max(1, maxSamples));
        for (int i = 0; i < total; i += step) {
            Sample s = samples.get(i);
            List<String> tags;
            if (shorten && s.tags.length > 0) {
                // 随机保留 k 个 tag，模拟前端短句只命中少量特征的情形
                int k = Math.min(s.tags.length, 1 + rnd.nextInt(kMax));
                List<String> pool = new ArrayList<>(Arrays.asList(s.tags));
                java.util.Collections.shuffle(pool, rnd);
                tags = pool.subList(0, k);
            } else {
                tags = Arrays.asList(s.tags);
            }
            Map<Integer, Double> weights = predictor.computeWeights(tags);
            // 按模式选用打分方式：TAGS→原始求和公式，TEXT→权重归一化公式
            // 必须与 NbPredictor.predictByTags 的分支一致，否则阈值与分数不同量纲
            NbPredictor.InputMode mode = shorten
                    ? NbPredictor.InputMode.TEXT : NbPredictor.InputMode.TAGS;
            outScores.add(predictor.computeScores(weights, mode));
            outTruth.add(new HashSet<>(Arrays.asList(s.genres)));
        }
    }

    /**
     * 阈值扫描结果：逐 genre 的最优阈值、sigmoid 尺度与留出集 F1。
     * 独立成类是为了让"扫描算法"与"打印/写入 model"解耦，
     * 从而被 TAGS 模式与自由文本的多个分桶【共用同一份实现】。
     */
    private static class TauScanResult {
        double[] tau;
        double[] scale;
        double[] f1;
    }

    /**
     * 逐 genre 最优 F1 阈值扫描的【核心算法】：纯计算，不打印、不写 model。
     *
     * <p>算法：把该 genre 的留出集分数升序排序，从"全部激活"起逐步上移阈值，
     * 增量维护 TP/FP，在 O(M log M) 内扫出使 F1 最大的阈值。
     * scale 取正负样本分数标准差的均值，使 sigmoid 过渡平滑而不过陡。
     *
     * <p>抽成独立方法的原因：自由文本需要按命中词数分成多个桶，每桶都要跑一遍
     * 完全相同的扫描。若各处复制粘贴公式，改一处漏一处必然导致口径漂移。
     */
    private static TauScanResult scanTauCore(ModelParams target,
                                             List<double[]> allScores,
                                             List<Set<String>> allTruth) {
        int G = target.G;
        int M = allScores.size();
        TauScanResult r = new TauScanResult();
        r.tau = new double[G];
        r.scale = new double[G];
        r.f1 = new double[G];

        for (int gIdx = 0; gIdx < G; gIdx++) {
            String g = target.genreArr[gIdx];

            double[] scores = new double[M];
            boolean[] positive = new boolean[M];
            int posCount = 0;
            double sumPos = 0.0, sumNeg = 0.0;
            for (int i = 0; i < M; i++) {
                scores[i] = allScores.get(i)[gIdx];
                positive[i] = allTruth.get(i).contains(g);
                if (positive[i]) { posCount++; sumPos += scores[i]; }
                else { sumNeg += scores[i]; }
            }

            // 该 genre 在留出集中从未出现（极罕见标签可能发生）：阈值取该列最大值，永不激活
            if (posCount == 0) {
                double maxS = Double.NEGATIVE_INFINITY;
                for (double sc : scores) if (sc > maxS) maxS = sc;
                r.tau[gIdx] = maxS;
                r.scale[gIdx] = 1.0;
                continue;
            }

            // 按 score 升序排序索引
            Integer[] order = new Integer[M];
            for (int i = 0; i < M; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));

            // 从"全部激活"（阈值低于最小分数）开始，逐步上移阈值，增量维护 TP/FP
            int tp = posCount, fp = M - posCount;
            double bestF1 = f1(tp, fp, posCount);
            double bestTau = scores[order[0]] - 1e-9;

            for (int k = 0; k < M; k++) {
                int idx = order[k];
                // 阈值上移越过该样本，它不再被激活
                if (positive[idx]) tp--; else fp--;
                double candF1 = f1(tp, fp, posCount);
                if (candF1 > bestF1) {
                    bestF1 = candF1;
                    bestTau = scores[idx];
                }
            }
            r.tau[gIdx] = bestTau;
            r.f1[gIdx] = bestF1;

            // 尺度：正负样本分数标准差的均值，保证 sigmoid 过渡平滑且不过陡
            double meanPos = sumPos / posCount;
            double meanNeg = sumNeg / Math.max(1, M - posCount);
            double varPos = 0.0, varNeg = 0.0;
            for (int i = 0; i < M; i++) {
                double d = scores[i] - (positive[i] ? meanPos : meanNeg);
                if (positive[i]) varPos += d * d; else varNeg += d * d;
            }
            varPos /= Math.max(1, posCount);
            varNeg /= Math.max(1, M - posCount);
            double scale = (Math.sqrt(varPos) + Math.sqrt(varNeg)) / 2.0;
            // 防御：尺度过小会让 sigmoid 退化成阶跃函数
            r.scale[gIdx] = Math.max(scale, 1e-6);
        }
        return r;
    }

    /**
     * 对采集到的分数做逐 genre 最优 F1 阈值扫描，写入目标模型。
     *
     * @param target      阈值写入的目标模型（全量模型）
     * @param scoreModel  产生这些分数的模型（临时模型），用于取 genreArr
     * @param textProfile true → 写 tauPerGenreText/scalePerGenreText；false → 写 tauPerGenre/scalePerGenre
     */
    private static void scanOptimalTau(ModelParams target, ModelParams scoreModel,
                                       List<double[]> allScores, List<Set<String>> allTruth,
                                       boolean textProfile) {
        int G = target.G;
        int M = allScores.size();

        // ★ 扫描算法与分桶校准共用同一实现（scanTauCore），不重复公式
        TauScanResult res = scanTauCore(target, allScores, allTruth);
        double[] tauOut = res.tau;
        double[] scaleOut = res.scale;

        System.out.printf("[校准%s] 留出集 %d 样本（全量的 %.0f%%），逐 genre 最优 F1 阈值搜索%n",
                textProfile ? "·自由文本" : "", M, HOLDOUT_RATIO * 100);
        System.out.printf("  %-20s %-14s %-10s %s%n", "genre", "tau", "scale", "holdoutF1");
        for (int gIdx = 0; gIdx < G; gIdx++) {
            System.out.printf("  %-20s %-14.4f %-10.4f %.4f%n",
                    target.genreArr[gIdx], tauOut[gIdx], scaleOut[gIdx], res.f1[gIdx]);
        }

        // ── 全局偏移校准：抑制 one-vs-rest 的系统性过度预测 ──
        double kOffset = searchGlobalOffset(target.genreArr, tauOut, scaleOut,
                allScores, allTruth, textProfile ? "·自由文本" : "");

        // 按 scale 成比例融入阈值：tau_g += k * scale_g
        // 用"多少个标准差"而非绝对值，保证每个 genre 受到同等统计强度的调整
        for (int gIdx = 0; gIdx < G; gIdx++) {
            tauOut[gIdx] += kOffset * scaleOut[gIdx];
        }

        if (textProfile) {
            target.tauPerGenreText = tauOut;
            target.scalePerGenreText = scaleOut;
        } else {
            target.tauPerGenre = tauOut;
            target.scalePerGenre = scaleOut;
            // 全局 tau 取逐 genre 阈值中位数，便于日志与旧接口展示
            double[] sorted = tauOut.clone();
            Arrays.sort(sorted);
            target.tau = sorted[sorted.length / 2];
        }
    }

    /**
     * 搜索全局阈值偏移系数 k，使留出集上的【MicroF1 与 MacroF1 的均值】最大。
     * 最终每个 genre 的阈值调整为 tau_g += k · scale_g。
     *
     * <p>为什么需要这一步：逐 genre 独立求最优 F1 只保证每个标签自身表现最好，
     * 完全不约束"一条样本预测几个标签"。高频标签会被系统性放宽——例如 drama
     * 出现在约 48% 的样本里，只要"几乎总是预测 drama"就能拿到约 0.66 的 F1，
     * 于是它的阈值极低、几乎恒激活。20 个标签中有多个如此，叠加后就严重过度预测
     * （实测平均每样本预测 8.58 个标签，而真实仅 2.56 个），Hamming loss 也被拉高。
     *
     * <p>为什么用【相对偏移】而非绝对偏移（关键）：各 genre 的分数量级差异极大
     * （drama tau≈-1210、documentary tau≈-17），若统一加一个绝对值（如 +257），
     * 高频类别只是略微收紧，而罕见类别阈值会被推成正数 → 永不激活 →
     * per-genre 报表出现 7 个类别全零、MacroF1 从 0.25 掉到 0.18。
     * 改用 k·scale_g（scale 即该 genre 分数标准差）后，调整量对各 genre 是
     * "同等统计强度"（都是 k 个标准差），罕见类别不会被一刀切清零。
     *
     * <p>目标取 Micro 与 Macro 的均值：单看 MicroF1（按标签实例加权，罕见类别占比极小）
     * 会偏好"牺牲稀有类换整体"的解；单看 MacroF1 则可能放任过度预测。两者均值兼顾。
     *
     * @return 最优偏移系数 k（已打印诊断信息）
     */
    private static double searchGlobalOffset(String[] genreArr, double[] tauOut, double[] scaleOut,
                                             List<double[]> allScores, List<Set<String>> allTruth,
                                             String label) {
        int G = genreArr.length;
        int M = allScores.size();

        // 各 genre 在留出集中的正例数（support），support=0 的不计入 macro 平均
        int[] support = new int[G];
        double avgTrue = 0.0;
        for (Set<String> t : allTruth) {
            avgTrue += t.size();
            for (String g : t) {
                Integer gi = indexOf(genreArr, g);
                if (gi != null) support[gi]++;
            }
        }
        avgTrue /= Math.max(1, M);
        int macroCount = 0;
        for (int g = 0; g < G; g++) if (support[g] > 0) macroCount++;
        if (macroCount == 0) macroCount = G;

        double bestK = 0.0;
        double bestObj = -1.0;
        double bestMicro = 0.0, bestMacro = 0.0, bestAvgPred = 0.0;

        // k ∈ [-0.5, +3.0] 个标准差，共 141 个候选（正区间更宽：主要用于抑制过度预测）
        final int STEPS = 140;
        final double LO = -0.5, HI = 3.0;
        for (int s = 0; s <= STEPS; s++) {
            double k = LO + (HI - LO) * s / STEPS;

            int[] tp = new int[G], fp = new int[G], fn = new int[G];
            for (int i = 0; i < M; i++) {
                double[] sc = allScores.get(i);
                Set<String> truth = allTruth.get(i);
                for (int gIdx = 0; gIdx < G; gIdx++) {
                    boolean pred = sc[gIdx] > tauOut[gIdx] + k * scaleOut[gIdx];
                    boolean real = truth.contains(genreArr[gIdx]);
                    if (pred && real) tp[gIdx]++;
                    else if (pred) fp[gIdx]++;
                    else if (real) fn[gIdx]++;
                }
            }

            // Micro：所有标签实例汇总
            int sumTp = 0, sumFp = 0, sumFn = 0;
            for (int g = 0; g < G; g++) { sumTp += tp[g]; sumFp += fp[g]; sumFn += fn[g]; }
            double microP = (sumTp + sumFp) > 0 ? (double) sumTp / (sumTp + sumFp) : 0.0;
            double microR = (sumTp + sumFn) > 0 ? (double) sumTp / (sumTp + sumFn) : 0.0;
            double microF1 = (microP + microR) > 0 ? 2 * microP * microR / (microP + microR) : 0.0;

            // Macro：各 genre F1 的算术平均（仅统计 support>0 的类别）
            double sumF1 = 0.0;
            for (int g = 0; g < G; g++) {
                if (support[g] == 0) continue;
                double p = (tp[g] + fp[g]) > 0 ? (double) tp[g] / (tp[g] + fp[g]) : 0.0;
                double r = (tp[g] + fn[g]) > 0 ? (double) tp[g] / (tp[g] + fn[g]) : 0.0;
                sumF1 += (p + r) > 0 ? 2 * p * r / (p + r) : 0.0;
            }
            double macroF1 = sumF1 / macroCount;

            double obj = (microF1 + macroF1) / 2.0;
            if (obj > bestObj) {
                bestObj = obj;
                bestK = k;
                bestMicro = microF1;
                bestMacro = macroF1;
                bestAvgPred = (double) (sumTp + sumFp) / Math.max(1, M);
            }
        }

        System.out.printf("[校准%s] 相对偏移搜索: k=%.3f 个标准差  MicroF1=%.4f  MacroF1=%.4f  "
                        + "目标均值=%.4f  平均预测标签=%.2f (真实=%.2f)%n",
                label, bestK, bestMicro, bestMacro, bestObj,
                bestAvgPred, avgTrue);
        return bestK;
    }

    /** 在数组中查找元素下标，未找到返回 null */
    private static Integer indexOf(String[] arr, String target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(target)) return i;
        }
        return null;
    }

    /** 二分类 F1：tp=真正例, fp=假正例, posTotal=真实正例总数 */
    private static double f1(int tp, int fp, int posTotal) {
        if (tp == 0) return 0.0;
        double precision = (double) tp / (tp + fp);
        double recall = (double) tp / posTotal;
        return 2.0 * precision * recall / (precision + recall);
    }
}
