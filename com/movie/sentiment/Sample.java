package com.movie.sentiment;

/**
 * 样本 POJO：一部电影的标签（特征）与类型（标签）集合。
 * 字段为 public final，只读不可变。
 */
public class Sample implements java.io.Serializable {
    private static final long serialVersionUID = 20261020L;

    /** 多标签 ground-truth：该电影的 genre 集合，如 ["Action","Adventure","Fantasy"] */
    public final String[] genres;

    /** 特征：该电影的 tag 列表，如 ["spacewar","alien","3d"] */
    public final String[] tags;

    /** 样本在原始文件中的行号（1-based），用于输出定位 */
    public final int lineNumber;

    /** 原始行原文（用于调试时回看，可选 null） */
    public final String rawLine;

    public Sample(String[] genres, String[] tags, int lineNumber, String rawLine) {
        this.genres = genres;
        this.tags = tags;
        this.lineNumber = lineNumber;
        this.rawLine = rawLine;
    }
}