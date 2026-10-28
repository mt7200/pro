package com.movie.sentiment;

/**
 * 课程任务书要求的统一执行入口。
 *
 * <p>任务书规定"工程中应该包含一个名 BD.java 的程序，该程序是整合工程的执行入口
 * （包括 main()，程序可直接运行）"。本类只做入口转发，全部子命令（train / predict /
 * predict-single / serve）仍由 {@link Main} 实现，避免出现第二套参数解析。
 *
 * <p>用法示例：
 * <pre>
 *   java -cp "target/classes:lib/*" com.movie.sentiment.BD train \
 *        --input hdfs://master:9000/input_2024083048/movies-from-kggle.txt \
 *        --model-local output/model.dat
 *   java -cp "target/classes:lib/*" com.movie.sentiment.BD serve --model output/model.dat
 * </pre>
 */
public class BD {

    public static void main(String[] args) throws Exception {
        Main.main(args);
    }
}
