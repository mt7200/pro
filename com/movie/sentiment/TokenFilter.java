package com.movie.sentiment;

import java.util.regex.Pattern;

/**
 * 训练阶段的词条过滤。
 *
 * <p>任务书要求"在训练过程中过滤包含非英文字符或数字构成的词语"，
 * 本类给出统一判定：只有完全由英文字母或数字组成的词条才参与训练统计。
 * 含连字符、撇号、括号、不间断空格等非字母数字字符的 tag（如
 * {@code loveofone'slife}、{@code middle-earth(tolkien)}）在训练阶段被丢弃，
 * 避免标点噪声进入词汇表与共现计数。
 *
 * <p>作用范围：词汇表构建（{@link VocabIdfBuilder}）与 MapReduce 的 DF 统计
 * （{@code DfCountJob}）。预测阶段沿用既有词汇表查找，不做二次过滤，
 * 保证训练与预测共用同一份 {@code vocabIndex}。
 */
public final class TokenFilter {

    /** 合法词条：仅小写英文字母与数字（词条在解析阶段已统一转小写） */
    private static final Pattern VALID = Pattern.compile("[a-z0-9]+");

    /** 训练阶段累计被过滤的词条次数，供日志打印 */
    public static long filteredCount = 0L;

    private TokenFilter() {
    }

    /**
     * 判断词条是否合法（纯字母/数字）。
     *
     * @param token 已转小写的词条
     * @return true 表示保留，false 表示过滤
     */
    public static boolean isValid(String token) {
        return token != null && VALID.matcher(token).matches();
    }
}
