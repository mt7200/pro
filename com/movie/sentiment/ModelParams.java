package com.movie.sentiment;

import java.io.Serializable;
import java.util.Map;

/**
 * 可序列化模型容器（Java 原生序列化，固定 serialVersionUID）。
 *
 * 字段说明：
 *   genreArr / genreIndex  — 标签(genre)全集
 *   vocabArr  / vocabIndex — tag 词汇表
 *   idf[]                  — 按 vocabIndex 的 IDF 值
 *   logPrior[]             — 按 genreIndex 的先验概率对数值
 *   logCond[][]            — [genreIndex][vocabIndex] 条件概率对数值
 *   N, V, G                — 样本数、词汇表大小、标签种类数
 *   tau                    — 全局激活阈值（可调，兼容保留）
 *   tauPerGenre[]          — 逐 genre 校准阈值（按 genreIndex），one-vs-rest 最优 F1 阈值
 *   scalePerGenre[]        — 逐 genre 的 sigmoid 尺度（按 genreIndex），用于把 score 映射为 0~1 独立置信度
 *   trainedAtMillis        — 训练完成时间戳
 *   trainSource            — 训练文件来源路径（记录用）
 */
public class ModelParams implements Serializable {
    private static final long serialVersionUID = 2026102001L;

    public String[] genreArr;
    // 【关于 javac -Xlint:serial 警告】下面两个字段声明为 Map 接口，
    // 而 Map 接口本身不 extends Serializable，故编译器提示"非序列化字段"。
    // 实际存入的实现类是 HashMap / LinkedHashMap，二者【都实现了 Serializable】，
    // 因此运行期序列化完全正常（本项目已多次实测 save/load 往返成功）。
    // 保留 Map 接口类型是任务书对模型容器字段的明确要求，属规范优先。
    public Map<String, Integer> genreIndex;
    public String[] vocabArr;
    public Map<String, Integer> vocabIndex;
    public double[] idf;
    public double[] logPrior;
    public double[][] logCond;

    public int N;   // 训练样本总数
    public int V;   // 词汇表大小
    public int G;   // genre 种类数
    public double tau;
    public long trainedAtMillis;
    public String trainSource;

    /**
     * 逐 genre 校准阈值（按 genreIndex）。
     * 由 NbTrainer.calibrate 在训练集上以"最大化该 genre 的 F1"为目标搜索得到，
     * 解决 log 域分数天然为负、全局 tau=0.0 无法激活任何标签的问题。
     * 旧模型文件反序列化后可能为 null，此时回退到全局 tau。
     */
    public double[] tauPerGenre;

    /**
     * 逐 genre 的 sigmoid 尺度（按 genreIndex），用于把 log 分数映射为 0~1 独立置信度：
     *   confidence(g) = 1 / (1 + exp(-(score(g) - tauPerGenre[g]) / scalePerGenre[g]))
     * 满足接口规范"逐标签独立、不要求相加等于 1"的要求。
     */
    public double[] scalePerGenre;

    /**
     * 【自由文本输入】专用的逐 genre 校准阈值。
     *
     * <p>存在原因：score(g) 的负值幅度与命中 tag 数量正相关。批量预测的样本
     * 平均含约 20 个 tag（分数很低），而前端一句剧情描述通常只命中 2~6 个词汇
     * （分数明显偏高）。若沿用按长列表校准的阈值，短文本会激活几乎所有标签。
     * 故对"随机抽取少量 tag 的短列表分布"单独校准一套阈值。
     */
    public double[] tauPerGenreText;

    /** 【自由文本输入】专用的逐 genre sigmoid 尺度 */
    public double[] scalePerGenreText;

    /**
     * 【自由文本输入】按【命中词数 k】分桶的逐 genre 阈值。
     * 下标为 [k-1][genreIdx]，k 取值 1..textCalibKMax。
     *
     * <p><b>为什么必须分桶</b>：TEXT 模式用权重归一化打分后，分数虽然不再随输入
     * 长度线性漂移，但【噪声水平】仍与命中词数强相关——
     * 命中 1 个词时，该词恰好是某错误 genre 的高权重 tag 就会得到很高的分数
     * （噪声主导）；命中 8 个词时，真实 genre 的信号被平均放大、错误 genre 被抵消
     * （信号主导）。两者分布形态不同，共用一套阈值必然对短输入过松。
     *
     * <p><b>实测证据</b>（未分桶时）：输入 "murder"（命中 1 词）激活 14/20 个标签，
     * 输入完整侦探长句（命中 6~8 词）只激活 3 个，而真实平均标签数仅 2.58。
     * 分桶后每个 k 用各自的分布定阈值，短输入的过度预测被显著抑制。
     *
     * <p>旧模型文件反序列化后此字段为 null，
     * {@link NbPredictor} 会逐级回退到 {@link #tauPerGenreText} 再到 {@link #tau}。
     */
    public double[][] tauPerGenreTextByK;

    /** 【自由文本输入】按命中词数分桶的逐 genre sigmoid 尺度，下标 [k-1][genreIdx] */
    public double[][] scalePerGenreTextByK;

    /**
     * 分桶上限（即 {@link #tauPerGenreTextByK} 的第一维长度）。
     * 预测时命中词数 k 超过此值则使用最后一桶（k 越大分布越稳定，用最大桶足够）。
     */
    public int textCalibKMax;

    /** 逐 genre 阈值是否可用（旧模型文件兼容判断） */
    public boolean hasPerGenreCalibration() {
        return tauPerGenre != null && tauPerGenre.length == G
                && scalePerGenre != null && scalePerGenre.length == G;
    }

    /** 自由文本专用阈值是否可用（不分桶的版本，作为分桶缺失时的回退） */
    public boolean hasTextCalibration() {
        return tauPerGenreText != null && tauPerGenreText.length == G
                && scalePerGenreText != null && scalePerGenreText.length == G;
    }

    /**
     * 自由文本【分桶】阈值是否可用。
     * 校验维度完整性，避免旧模型或写入不全时数组越界。
     */
    public boolean hasTextCalibrationByK() {
        if (tauPerGenreTextByK == null || scalePerGenreTextByK == null || textCalibKMax <= 0) {
            return false;
        }
        if (tauPerGenreTextByK.length < textCalibKMax
                || scalePerGenreTextByK.length < textCalibKMax) {
            return false;
        }
        for (int k = 0; k < textCalibKMax; k++) {
            if (tauPerGenreTextByK[k] == null || tauPerGenreTextByK[k].length != G) {
                return false;
            }
            if (scalePerGenreTextByK[k] == null || scalePerGenreTextByK[k].length != G) {
                return false;
            }
        }
        return true;
    }
}