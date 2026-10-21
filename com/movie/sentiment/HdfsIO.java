package com.movie.sentiment;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.security.UserGroupInformation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.nio.charset.StandardCharsets;

/**
 * Module-HDFS：HDFS 统一读写工具类。
 *
 * <p>设计目标（对齐第二阶段 Module-HDFS 规范）：让【同一份】train/predict/serve 代码
 * 在 hdfs:// 与 file:// 之间透明切换——调用方只传路径字符串，不关心底层是集群还是本地。
 *
 * <p><b>路由策略</b>（{@link #openStream} 等处统一判断）：
 * <ul>
 *   <li>路径<b>不含 scheme</b>（如 {@code output/model.dat}）→ 走 java.io 直连本地。
 *       理由：后端服务只需读本地模型副本，不应被迫依赖 Hadoop 运行时；
 *       同时避免集群抖动影响纯本地操作。</li>
 *   <li>路径<b>含 scheme</b>（{@code hdfs://} / {@code file://}）→ 走
 *       {@code FileSystem.get(URI, Configuration)}，由 Hadoop 天然支持两者。</li>
 * </ul>
 *
 * <p><b>降级策略</b>（对齐组长"降级方案"哲学的代码级体现）：HDFS 连接失败时，
 * 打印明确 WARN 并自动回退到【同名本地路径】重试，保证联调 deadline 不因集群问题卡死。
 * 回退可用 {@code --no-fallback} 关闭。URI → 本地路径的映射规则见 {@link #toLocalFallback}。
 *
 * <p><b>线程安全</b>：配置项为进程级静态量，须在 {@code Main} 启动阶段一次性
 * {@link #configure}，之后只读，可被多线程并发调用。
 */
public class HdfsIO {

    // ─────────────── 运行期环境占位符（对应任务书的 ${...}，全部可命令行/环境变量覆盖） ───────────────

    /** ${NAMENODE_URI} 默认值：向成员2/组长确认实际 NameNode 地址与端口（常见 8020 或 9000） */
    public static final String DEFAULT_NAMENODE_URI = "hdfs://localhost:9000";

    /** ${HADOOP_USER} 默认值：空 = 简单认证，不指定用户（课程集群一般不开 Kerberos） */
    public static final String DEFAULT_HADOOP_USER = "";

    private static String nameNodeUri = DEFAULT_NAMENODE_URI;
    private static String hadoopUser = DEFAULT_HADOOP_USER;
    private static boolean fallbackEnabled = true;
    private static boolean configured = false;

    /** Hadoop 客户端配置（进程级复用，避免每次读写重建） */
    private static Configuration conf;

    private HdfsIO() {
    }

    // ─────────────────────────── 配置 ───────────────────────────

    /**
     * 在程序启动阶段调用一次，设置 NameNode 地址、Hadoop 用户与降级开关。
     *
     * @param nameNode ${NAMENODE_URI}，null/空则用默认值
     * @param user     ${HADOOP_USER}，null/空则不指定（简单认证）
     * @param fallback 是否允许 HDFS 失败时回退本地，对应 {@code --no-fallback} 关闭
     */
    public static synchronized void configure(String nameNode, String user, boolean fallback) {
        if (nameNode != null && !nameNode.trim().isEmpty()) {
            nameNodeUri = nameNode.trim();
        }
        hadoopUser = (user == null) ? "" : user.trim();
        fallbackEnabled = fallback;

        conf = new Configuration();
        // fs.defaultFS 决定不带 scheme 的相对 HDFS 路径（如 /input/x.txt）解析到哪个集群
        conf.set("fs.defaultFS", nameNodeUri);
        // 课程集群通常为简单认证；显式声明以免继承到机器上的 kerberos 配置
        conf.set("hadoop.security.authentication", "simple");
        // 关闭递归删除保护之外的冗余校验，缩短连接失败时的等待，便于快速降级
        conf.setInt("ipc.client.connect.max.retries", 2);
        conf.setInt("ipc.client.connect.retry.interval", 1000);

        applySimpleAuth();
        configured = true;
    }

    /**
     * 简单认证：若指定了 ${HADOOP_USER}，以该远程用户身份访问 HDFS。
     *
     * <p><b>Kerberos 切换点</b>：课程集群一般不开 Kerberos，故用 createRemoteUser。
     * 若成员2 的集群启用了 Kerberos，把下面替换为：
     * <pre>
     *   UserGroupInformation.setConfiguration(conf);
     *   UserGroupInformation.loginUserFromKeytab(principal, keytabPath);
     * </pre>
     * 其余读写代码无需改动。
     */
    private static void applySimpleAuth() {
        if (hadoopUser.isEmpty()) {
            return;
        }
        try {
            UserGroupInformation ugi = UserGroupInformation.createRemoteUser(hadoopUser);
            UserGroupInformation.setLoginUser(ugi);
            System.out.println("[HDFS] 以简单认证用户身份访问: " + hadoopUser);
        } catch (Throwable t) {
            // 捕获 Throwable 而非 IOException：setLoginUser 不声明受检异常，
            // 且当 hadoop-auth/security 依赖缺失时会抛 NoClassDefFoundError，
            // 此时应跳过用户设置、以默认身份继续，而非让整个程序启动失败。
            System.err.println("[WARN] 设置 Hadoop 用户失败（将以默认身份继续）: " + t);
        }
    }

    private static Configuration conf() {
        if (!configured || conf == null) {
            // 允许未显式 configure 就直接使用（取全部默认值），降低调用方负担
            configure(null, null, true);
        }
        return conf;
    }

    public static boolean isFallbackEnabled() {
        return fallbackEnabled;
    }

    public static String nameNodeUri() {
        return nameNodeUri;
    }

    // ─────────────────────────── 路径判定 ───────────────────────────

    /** 取路径的 scheme；无 scheme 返回 null（视为纯本地路径）。 */
    public static String scheme(String uriOrPath) {
        if (uriOrPath == null) {
            return null;
        }
        int i = uriOrPath.indexOf("://");
        return i > 0 ? uriOrPath.substring(0, i) : null;
    }

    /** 是否为远程路径（hdfs:// 等，不含 file://）。仅用于日志与判断，不参与路由。 */
    private static boolean isRemote(String uriOrPath) {
        String s = scheme(uriOrPath);
        return s != null && !"file".equalsIgnoreCase(s);
    }

    /**
     * 路由判断：是否交给 Hadoop FileSystem 处理。
     *
     * <p>规则对齐 Hadoop 自身的路径约定（这点很关键，否则 Windows 下会解析错）：
     * <ul>
     *   <li>含 scheme（{@code hdfs://} / {@code file://}）→ 走 Hadoop；</li>
     *   <li>以 {@code /} 开头的绝对路径（如 {@code /input/movies.txt}）→ 走 Hadoop，
     *       由 {@code fs.defaultFS}（即 ${NAMENODE_URI}）决定落到哪个集群。
     *       <b>不能</b>当本地路径处理：Windows 下 {@code /input/x} 会被 java.io
     *       解析成当前盘符根目录 {@code C:\input\x}，与项目目录完全不符；</li>
     *   <li>其余相对路径（如 {@code output/model.dat}）→ java.io 直连本地，
     *       不依赖 Hadoop 运行时（后端只读本地模型副本时无需集群）。</li>
     * </ul>
     */
    private static boolean useHadoopFs(String uriOrPath) {
        if (uriOrPath == null) {
            return false;
        }
        if (scheme(uriOrPath) != null) {
            return true;
        }
        return uriOrPath.startsWith("/");
    }

    /**
     * 把 HDFS 路径映射为【同名本地路径】，用于降级回退。
     *
     * <p>映射规则（去掉 scheme+authority 与前导斜杠，相对当前工作目录解析）：
     * <pre>
     *   hdfs://localhost:9000/output/model.dat  →  output/model.dat
     *   hdfs://nn:8020/input/movies.txt         →  input/movies.txt
     *   /input/movies.txt（无 scheme 绝对路径） →  input/movies.txt
     * </pre>
     * 之所以相对 CWD 而非根目录：Windows 下 {@code /output/x} 会落到当前盘符根目录，
     * 与项目工作目录不符；相对 CWD 才能命中 train 双写产生的本地副本，
     * 或用户在项目下建立的 {@code input/} 数据镜像目录。
     */
    public static String toLocalFallback(String uriOrPath) {
        if (uriOrPath == null) {
            return null;
        }
        String rest;
        int i = uriOrPath.indexOf("://");
        if (i >= 0) {
            // 去掉 scheme 与 authority（host:port）
            rest = uriOrPath.substring(i + 3);
            int slash = rest.indexOf('/');
            rest = (slash < 0) ? "" : rest.substring(slash + 1);
        } else {
            rest = uriOrPath;
        }
        // 去掉可能残留的多个前导斜杠，转为相对 CWD 的路径
        while (rest.startsWith("/")) {
            rest = rest.substring(1);
        }
        return rest;
    }

    // ─────────────────────────── 读 ───────────────────────────

    /**
     * 打开文本读取器（UTF-8）。
     *
     * <p><b>注意</b>：返回的 BufferedReader 由【调用方】负责关闭（try-with-resources），
     * 本方法不持有资源。对应规范 Module-HDFS 的 openReader 约定。
     *
     * @param uriOrPath hdfs:// 或 file:// URI，或本地路径
     */
    public static BufferedReader openReader(String uriOrPath) throws IOException {
        return new BufferedReader(new InputStreamReader(openStream(uriOrPath), StandardCharsets.UTF_8),
                1 << 16);
    }

    /**
     * 打开原始字节输入流（模型反序列化用）。调用方负责关闭。
     *
     * <p>降级逻辑：Hadoop 打开失败 → WARN → 回退本地同名路径（若允许）。
     */
    public static InputStream openStream(String uriOrPath) throws IOException {
        if (uriOrPath == null || uriOrPath.trim().isEmpty()) {
            throw new IOException("路径为空，无法读取");
        }
        String p = uriOrPath.trim();

        if (!useHadoopFs(p)) {
            // 纯本地路径：java.io 直连，不依赖 Hadoop 运行时
            return openLocal(p);
        }

        try {
            FileSystem fs = FileSystem.get(URI.create(p), conf());
            Path path = new Path(p);
            if (!fs.exists(path)) {
                throw new IOException("路径不存在: " + p);
            }
            // 说明：FileSystem.get 返回的是【缓存共享】实例，此处只关流、不关 fs，
            // 否则会影响同一进程内其它使用该 FS 的代码。
            return fs.open(path);
        } catch (Throwable t) {
            // 捕获 Throwable 而非仅 IOException：Hadoop jar 缺失时抛 NoClassDefFoundError，
            // 此时也应降级到本地，保证成员4 在精简 classpath 下仍能起后端。
            if (!fallbackEnabled) {
                throw new IOException("读取失败且已禁用降级(--no-fallback): " + p + " → " + t, t);
            }
            String local = toLocalFallback(p);
            System.err.println("[WARN] HDFS 读取失败: " + p);
            System.err.println("[WARN]   原因: " + t.getClass().getSimpleName() + ": " + t.getMessage());
            System.err.println("[WARN]   自动回退本地同名路径: " + local
                    + "  (如需禁用降级请加 --no-fallback)");
            return openLocal(local);
        }
    }

    private static InputStream openLocal(String localPath) throws IOException {
        File f = new File(localPath);
        if (!f.exists()) {
            throw new IOException("本地文件不存在: " + f.getAbsolutePath()
                    + "（请确认已训练并生成模型，或用 --model/--model-hdfs 指定正确路径）");
        }
        return new FileInputStream(f);
    }

    /** 一次性读取整个文本文件为字符串（UTF-8）。适合小文件如配置、指标。 */
    public static String readAllText(String uriOrPath) throws IOException {
        try (InputStream in = openStream(uriOrPath);
             ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) != -1) {
                bos.write(buf, 0, n);
            }
            return new String(bos.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    /** 一次性读取整个文件为字节数组（模型反序列化用）。 */
    public static byte[] readAllBytes(String uriOrPath) throws IOException {
        try (InputStream in = openStream(uriOrPath);
             ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) != -1) {
                bos.write(buf, 0, n);
            }
            return bos.toByteArray();
        }
    }

    // ─────────────────────────── 写 ───────────────────────────

    /** 打开原始字节输出流（会自动创建父目录、覆盖同名文件）。调用方负责关闭。 */
    public static OutputStream createStream(String uriOrPath) throws IOException {
        if (uriOrPath == null || uriOrPath.trim().isEmpty()) {
            throw new IOException("路径为空，无法写入");
        }
        String p = uriOrPath.trim();

        if (!useHadoopFs(p)) {
            return createLocal(p);
        }
        try {
            FileSystem fs = FileSystem.get(URI.create(p), conf());
            Path path = new Path(p);
            mkdirsParentRemote(fs, path);
            return fs.create(path, true);   // true = 覆盖已存在文件
        } catch (Throwable t) {
            if (!fallbackEnabled) {
                throw new IOException("写入失败且已禁用降级(--no-fallback): " + p + " → " + t, t);
            }
            String local = toLocalFallback(p);
            System.err.println("[WARN] HDFS 写入失败: " + p);
            System.err.println("[WARN]   原因: " + t.getClass().getSimpleName() + ": " + t.getMessage());
            System.err.println("[WARN]   自动回退本地同名路径: " + local);
            return createLocal(local);
        }
    }

    private static OutputStream createLocal(String localPath) throws IOException {
        File f = new File(localPath);
        File parent = f.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("无法创建本地目录: " + parent.getAbsolutePath());
        }
        return new FileOutputStream(f);
    }

    private static void mkdirsParentRemote(FileSystem fs, Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null && !fs.exists(parent)) {
            fs.mkdirs(parent);
        }
    }

    /** 写文本（UTF-8）。自动建父目录、覆盖同名文件。 */
    public static void writeText(String uriOrPath, String content) throws IOException {
        try (OutputStream os = createStream(uriOrPath);
             BufferedWriter w = new BufferedWriter(
                     new OutputStreamWriter(os, StandardCharsets.UTF_8), 1 << 16)) {
            w.write(content);
        }
    }

    /** 写字节（模型序列化用）。自动建父目录、覆盖同名文件。 */
    public static void writeBytes(String uriOrPath, byte[] data) throws IOException {
        try (OutputStream os = createStream(uriOrPath)) {
            os.write(data);
        }
    }

    // ─────────────────────────── 其它 ───────────────────────────

    /** 判断路径是否存在（自动区分 HDFS / 本地）。 */
    public static boolean exists(String uriOrPath) {
        if (uriOrPath == null || uriOrPath.trim().isEmpty()) {
            return false;
        }
        String p = uriOrPath.trim();
        if (!useHadoopFs(p)) {
            return new File(p).exists();
        }
        try {
            FileSystem fs = FileSystem.get(URI.create(p), conf());
            return fs.exists(new Path(p));
        } catch (Throwable t) {
            if (fallbackEnabled) {
                return new File(toLocalFallback(p)).exists();
            }
            return false;
        }
    }

    /** 创建路径的父目录（写文件前调用；本地与 HDFS 均支持）。 */
    public static void mkdirsParent(String uriOrPath) throws IOException {
        if (uriOrPath == null || uriOrPath.trim().isEmpty()) {
            return;
        }
        String p = uriOrPath.trim();
        if (!useHadoopFs(p)) {
            File f = new File(p).getAbsoluteFile().getParentFile();
            if (f != null && !f.exists()) {
                f.mkdirs();
            }
            return;
        }
        FileSystem fs = FileSystem.get(URI.create(p), conf());
        mkdirsParentRemote(fs, new Path(p));
    }
}
