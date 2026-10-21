package com.movie.sentiment;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

/**
 * 模型序列化器：save / load ModelParams（Java 原生 Object 序列化）。
 *
 * <p>第二阶段改造：底层 IO 全部委托 {@link HdfsIO}，因此 {@code filePath} 既可是
 * 本地路径（{@code output/model.dat}），也可是 HDFS/URI（{@code hdfs://nn:9000/output/model.dat}
 * 或 {@code file:///...}），调用方无需区分。
 *
 * <p><b>硬性约束</b>：预测/后端阶段严禁重建词汇表或重算 IDF/概率，
 * 所有参数必须来自本文件加载的模型（对齐第一阶段与第二阶段规范）。
 */
public class ModelSerializer {

    /**
     * 保存模型到指定路径（本地或 HDFS 均可）。
     * ObjectOutputStream 套 HdfsIO.createStream，对应规范"ObjectOutputStream 套 fs.create"。
     */
    public static void save(ModelParams model, String filePath) throws IOException {
        try (OutputStream os = HdfsIO.createStream(filePath);
             ObjectOutputStream oos = new ObjectOutputStream(os)) {
            oos.writeObject(model);
            oos.flush();
        }
    }

    /**
     * 从指定路径加载模型（本地或 HDFS 均可）。
     * ObjectInputStream 套 HdfsIO.openStream，对应规范"ObjectInputStream 套 fs.open"。
     */
    public static ModelParams load(String filePath) throws IOException, ClassNotFoundException {
        try (InputStream is = HdfsIO.openStream(filePath);
             ObjectInputStream ois = new ObjectInputStream(is)) {
            return (ModelParams) ois.readObject();
        }
    }

    /**
     * 序列化为字节数组（供需要一次拿到完整字节的场景，如 HDFS 双写复用同一份字节）。
     */
    public static byte[] toBytes(ModelParams model) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream(1 << 20);
        try (ObjectOutputStream oos = new ObjectOutputStream(bos)) {
            oos.writeObject(model);
        }
        return bos.toByteArray();
    }

    /** 从字节数组反序列化模型。 */
    public static ModelParams fromBytes(byte[] bytes) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
            return (ModelParams) ois.readObject();
        }
    }

    /**
     * 双写：把同一份模型字节同时写入主路径（通常为 HDFS）与副本路径（通常为本地）。
     *
     * <p><b>规范依据</b>：Module-HDFS 的"双写策略"——模型写一份到 ${HDFS_MODEL}
     * 满足"产物在集群"，同时落一份到 ${LOCAL_MODEL} 供后端与组长打包，降低联调变量。
     * 数据集读取走 HDFS 是硬性要求；模型落本地副本不违反该要求（模型是产物而非输入数据集）。
     *
     * @param model     待写模型
     * @param primary   主路径，null/空则跳过（如仅要本地副本）
     * @param secondary 副本路径，null/空则跳过
     * @return 实际写成功的路径数量
     */
    public static int dualWrite(ModelParams model, String primary, String secondary) throws IOException {
        byte[] bytes = toBytes(model);   // 只序列化一次，两处写同一份字节，保证内容一致
        int ok = 0;
        if (primary != null && !primary.trim().isEmpty()) {
            HdfsIO.writeBytes(primary, bytes);
            ok++;
            System.out.println("[MODEL] 已写入主路径: " + primary + " (" + bytes.length + " 字节)");
        }
        if (secondary != null && !secondary.trim().isEmpty()
                && !secondary.trim().equals(primary == null ? null : primary.trim())) {
            HdfsIO.writeBytes(secondary, bytes);
            ok++;
            System.out.println("[MODEL] 已写入本地副本: " + secondary + " (" + bytes.length + " 字节)");
        }
        return ok;
    }
}
