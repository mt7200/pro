package com.movie.hdfs;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.security.UserGroupInformation;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 成员2：HDFS 文件上传与校验工具（Java API 方式）。
 *
 * <p>用途：不依赖 {@code hdfs dfs -put} 命令行，直接用 Hadoop FileSystem API
 * 把本地数据集上传到集群，并在上传后做端到端校验，满足课程要求中
 * “通过 api 进行上传文件到 hdfs” 与 “检验上传是否成功”。
 *
 * <p><b>与算法模块的约定</b>：上传目标路径使用成员3 代码中的默认值
 * （{@code /input/movies-from-kggle.txt}、{@code /input/movies-to-be-predicted.txt}），
 * 这样成员3 无需改代码即可直接读集群数据。
 *
 * <p><b>为什么不用 {@code FileSystem.copyFromLocalFile}</b>：
 * 该方法不带进度、不能在上传过程中同步计算本地 MD5，也无法在同一个命令里
 * 完成“上传 + 回读校验”。本工具用流式复制显式控制这三件事。
 *
 * <p><b>校验口径</b>：{@code FileSystem.getFileChecksum()} 返回的是 HDFS 内部的
 * MD5-of-MD5（按块与 CRC 计算），与本地文件的 MD5 不是同一口径，不能直接比较。
 * 因此本工具的 {@code --verify} 会<b>把远端文件完整读回</b>再算 MD5，
 * 与上传时同步计算的本地 MD5 比较，属于真正的端到端校验。
 *
 * <p><b>失败策略</b>：本工具是集群连通性/权限的验收工具，<b>不做本地降级</b>，
 * 连不上或写不进去直接以非 0 退出码失败，避免“看起来成功、其实写到了本地”。
 *
 * <p>用法见 {@code help} 子命令。
 */
public final class HdfsUploader {

    /** 上传缓冲区 1 MiB：与 HDFS 客户端默认写包大小量级一致，避免过小包拖慢吞吐 */
    private static final int BUFFER_SIZE = 1 << 20;

    /** 本项目实测集群的 NameNode 默认值，仍可用 --namenode 或环境变量覆盖 */
    private static final String DEFAULT_NAMENODE = "hdfs://master:9000";

    private HdfsUploader() {
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            printUsage();
            System.exit(2);
        }
        String cmd = args[0];
        try {
            if ("help".equals(cmd) || "-h".equals(cmd) || "--help".equals(cmd)) {
                printUsage();
                return;
            }
            Map<String, String> opts = parseOptions(args, 1);
            if ("put".equals(cmd)) {
                put(opts);
            } else if ("ls".equals(cmd)) {
                ls(opts);
            } else if ("stat".equals(cmd)) {
                stat(opts);
            } else {
                System.err.println("[ERROR] 未知子命令: " + cmd);
                printUsage();
                System.exit(2);
            }
        } catch (IllegalArgumentException e) {
            System.err.println("[ERROR] 参数错误: " + e.getMessage());
            System.err.println("        运行 help 查看用法。");
            System.exit(2);
        } catch (Throwable t) {
            System.err.println("[ERROR] 操作失败: " + t.getClass().getSimpleName() + ": " + t.getMessage());
            printTroubleshooting();
            System.exit(3);
        }
    }

    // ─────────────────────────── put ───────────────────────────

    private static void put(Map<String, String> opts) throws Exception {
        String localPath = require(opts, "local");
        String remotePath = require(opts, "remote");
        boolean overwrite = flag(opts, "overwrite");
        boolean verify = flag(opts, "verify");
        int replication = intOpt(opts, "replication", 0);

        File local = new File(localPath);
        if (!local.isFile()) {
            throw new IllegalArgumentException("本地文件不存在或不是普通文件: " + local.getAbsolutePath());
        }

        FileSystem fs = connect(opts);
        try {
            Path dst = new Path(remotePath);
            Path parent = dst.getParent();
            if (parent != null && !fs.exists(parent)) {
                System.out.println("[INFO] 创建目录: " + parent);
                fs.mkdirs(parent);
            }
            if (fs.exists(dst) && !overwrite) {
                throw new IllegalArgumentException("目标已存在: " + fs.makeQualified(dst)
                        + "（如需覆盖请加 --overwrite）");
            }

            long total = local.length();
            MessageDigest localMd5 = newMd5();
            long copied = 0L;
            int nextMark = 25;
            long t0 = System.nanoTime();

            System.out.println("[INFO] 开始上传: " + local.getAbsolutePath());
            System.out.println("[INFO] 目标路径: " + fs.makeQualified(dst));

            try (InputStream in = new FileInputStream(local);
                 FSDataOutputStream out = fs.create(dst, true)) {
                byte[] buf = new byte[BUFFER_SIZE];
                int n;
                while ((n = in.read(buf)) > 0) {
                    out.write(buf, 0, n);
                    localMd5.update(buf, 0, n);
                    copied += n;
                    int pct = total == 0 ? 100 : (int) (copied * 100 / total);
                    if (pct >= nextMark) {
                        System.out.printf("[INFO] 进度 %3d%%  %,d / %,d 字节%n", pct, copied, total);
                        nextMark += 25;
                    }
                }
            }

            long costMs = (System.nanoTime() - t0) / 1_000_000L;

            // 统一权限为 644：属主可读写、同组与其他用户只读。
            // 成员3 的 train/predict 只需读，644 足够；目录由 mkdirs 建成 755。
            fs.setPermission(dst, new FsPermission((short) 0644));
            if (replication > 0) {
                fs.setReplication(dst, (short) replication);
            }

            FileStatus st = fs.getFileStatus(dst);
            String localDigest = hex(localMd5.digest());
            String remoteDigest = null;
            boolean md5Ok = true;

            if (verify) {
                System.out.println("[INFO] 校验中：回读远端文件并计算 MD5 ...");
                DigestResult rr = md5Remote(fs, dst);
                remoteDigest = rr.md5;
                md5Ok = rr.bytes == local.length() && localDigest.equals(remoteDigest);
            }

            System.out.println();
            System.out.println("================ 上传结果 ================");
            System.out.println("本地文件   : " + local.getAbsolutePath());
            System.out.println("本地大小   : " + String.format("%,d", local.length()) + " 字节");
            System.out.println("远端路径   : " + fs.makeQualified(dst));
            System.out.println("远端大小   : " + String.format("%,d", st.getLen()) + " 字节");
            System.out.println("耗时       : " + costMs + " ms"
                    + (costMs > 0 ? String.format("（%.2f MiB/s）", total / 1048576.0 / (costMs / 1000.0)) : ""));
            System.out.println("MD5(本地)  : " + localDigest);
            if (verify) {
                System.out.println("MD5(远端)  : " + remoteDigest);
                System.out.println("MD5 一致   : " + (md5Ok ? "是" : "否"));
            }
            System.out.println("权限/属主  : " + st.getPermission() + "  " + st.getOwner() + ":" + st.getGroup());
            System.out.println("副本/块大小: " + st.getReplication() + " 副本 / "
                    + String.format("%,d", st.getBlockSize()) + " 字节");
            System.out.println("修改时间   : " + formatTime(st.getModificationTime()));
            System.out.println("==========================================");
            System.out.println(md5Ok ? "[OK] 上传成功" : "[FAIL] 上传后校验不一致");

            if (!md5Ok) {
                System.exit(4);
            }
        } finally {
            fs.close();
        }
    }

    // ─────────────────────────── ls / stat ───────────────────────────

    private static void ls(Map<String, String> opts) throws Exception {
        String remotePath = opt(opts, "remote", "/input");
        FileSystem fs = connect(opts);
        try {
            Path p = new Path(remotePath);
            if (!fs.exists(p)) {
                throw new IllegalArgumentException("路径不存在: " + fs.makeQualified(p));
            }
            System.out.println("[INFO] NameNode: " + fs.getUri());
            System.out.println(String.format("%-12s %-10s %-10s %14s %6s %14s  %s",
                    "权限", "属主", "属组", "大小(字节)", "副本", "块大小", "路径"));
            for (FileStatus st : fs.listStatus(p)) {
                System.out.println(String.format("%-12s %-10s %-10s %,14d %6d %,14d  %s",
                        st.getPermission(), st.getOwner(), st.getGroup(), st.getLen(),
                        st.getReplication(), st.getBlockSize(), st.getPath()));
            }
        } finally {
            fs.close();
        }
    }

    private static void stat(Map<String, String> opts) throws Exception {
        String remotePath = require(opts, "remote");
        FileSystem fs = connect(opts);
        try {
            Path p = new Path(remotePath);
            if (!fs.exists(p)) {
                throw new IllegalArgumentException("路径不存在: " + fs.makeQualified(p));
            }
            FileStatus st = fs.getFileStatus(p);
            System.out.println("路径       : " + st.getPath());
            System.out.println("类型       : " + (st.isDirectory() ? "目录" : "文件"));
            System.out.println("大小       : " + String.format("%,d", st.getLen()) + " 字节");
            System.out.println("权限/属主  : " + st.getPermission() + "  " + st.getOwner() + ":" + st.getGroup());
            System.out.println("副本数     : " + st.getReplication());
            System.out.println("块大小     : " + String.format("%,d", st.getBlockSize()) + " 字节");
            System.out.println("修改时间   : " + formatTime(st.getModificationTime()));
        } finally {
            fs.close();
        }
    }

    // ─────────────────────────── 连接与工具方法 ───────────────────────────

    /**
     * 建立 FileSystem 连接。
     *
     * <p>参数优先级：命令行 --namenode / --user &gt; 环境变量 NAMENODE_URI / HADOOP_USER &gt; 默认值。
     * 第三参数 user 直接交给 {@code FileSystem.get(URI, conf, user)}，
     * 由 Hadoop 以简单认证的远程用户身份访问，避免直接改全局 UGI 带来的副作用。
     */
    private static FileSystem connect(Map<String, String> opts) throws IOException, InterruptedException {
        String namenode = firstNonEmpty(opt(opts, "namenode", null),
                System.getenv("NAMENODE_URI"), DEFAULT_NAMENODE);
        String user = firstNonEmpty(opt(opts, "user", null), System.getenv("HADOOP_USER"), null);

        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", namenode);
        // 课程集群通常为简单认证；显式声明，避免继承机器上的 Kerberos 配置
        conf.set("hadoop.security.authentication", "simple");
        // 连接失败时尽快报错，不长时间挂起（验收工具要的是“立刻暴露问题”）
        conf.setInt("ipc.client.connect.max.retries", 3);
        conf.setInt("ipc.client.connect.retry.interval", 2000);

        try {
            FileSystem fs = FileSystem.get(URI.create(namenode), conf, user);
            System.out.println("[INFO] NameNode = " + fs.getUri());
            System.out.println("[INFO] 当前 HDFS 用户 = " + currentUser());
            return fs;
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw ie;
        }
    }

    private static String currentUser() {
        try {
            return UserGroupInformation.getCurrentUser().getShortUserName();
        } catch (Throwable t) {
            return System.getProperty("user.name", "unknown");
        }
    }

    /** 回读远端文件并计算 MD5（同时统计字节数，防止截断）。 */
    private static DigestResult md5Remote(FileSystem fs, Path p) throws IOException {
        MessageDigest md = newMd5();
        long bytes = 0L;
        try (FSDataInputStream in = fs.open(p)) {
            byte[] buf = new byte[BUFFER_SIZE];
            int n;
            while ((n = in.read(buf)) > 0) {
                md.update(buf, 0, n);
                bytes += n;
            }
        }
        return new DigestResult(hex(md.digest()), bytes);
    }

    private static MessageDigest newMd5() {
        try {
            return MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("JVM 不支持 MD5", e);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            String h = Integer.toHexString(b & 0xFF);
            if (h.length() == 1) {
                sb.append('0');
            }
            sb.append(h);
        }
        return sb.toString();
    }

    private static String formatTime(long millis) {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date(millis));
    }

    private static String firstNonEmpty(String a, String b, String c) {
        if (a != null && !a.trim().isEmpty()) {
            return a.trim();
        }
        if (b != null && !b.trim().isEmpty()) {
            return b.trim();
        }
        return c;
    }

    // ─────────────────────────── 参数解析 ───────────────────────────

    private static Map<String, String> parseOptions(String[] args, int from) {
        Map<String, String> opts = new LinkedHashMap<String, String>();
        for (int i = from; i < args.length; i++) {
            String a = args[i];
            if (!a.startsWith("--")) {
                throw new IllegalArgumentException("无法识别的参数: " + a);
            }
            String key = a.substring(2);
            if (key.isEmpty()) {
                throw new IllegalArgumentException("空参数名");
            }
            String val = "true";
            if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                val = args[++i];
            }
            opts.put(key, val);
        }
        return opts;
    }

    private static String opt(Map<String, String> opts, String key, String def) {
        String v = opts.get(key);
        return (v == null) ? def : v;
    }

    private static String require(Map<String, String> opts, String key) {
        String v = opts.get(key);
        if (v == null || v.trim().isEmpty() || "true".equals(v)) {
            throw new IllegalArgumentException("缺少必填参数 --" + key);
        }
        return v.trim();
    }

    private static boolean flag(Map<String, String> opts, String key) {
        String v = opts.get(key);
        return v != null && ("true".equalsIgnoreCase(v) || "1".equals(v) || "yes".equalsIgnoreCase(v));
    }

    private static int intOpt(Map<String, String> opts, String key, int def) {
        String v = opts.get(key);
        if (v == null) {
            return def;
        }
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("--" + key + " 需要整数，实际为: " + v);
        }
    }

    // ─────────────────────────── 帮助与排错 ───────────────────────────

    private static void printUsage() {
        System.out.println("HdfsUploader —— 成员2 HDFS 上传/校验工具（Java API）");
        System.out.println();
        System.out.println("用法:");
        System.out.println("  java -cp \"out:<hadoop classpath>\" com.movie.hdfs.HdfsUploader put \\");
        System.out.println("       --local <本地文件> --remote <HDFS路径> [--verify] [--overwrite] \\");
        System.out.println("       [--namenode hdfs://host:port] [--user <hdfs用户>] [--replication N]");
        System.out.println();
        System.out.println("  java ... HdfsUploader ls   [--remote /input] [--namenode ...]");
        System.out.println("  java ... HdfsUploader stat --remote /input/movies-to-be-predicted.txt");
        System.out.println();
        System.out.println("参数:");
        System.out.println("  --local       本地文件路径（必填，put）");
        System.out.println("  --remote      HDFS 目标路径（必填，put/stat；ls 默认 /input）");
        System.out.println("  --verify      上传后回读全文计算 MD5，与本地 MD5 比对");
        System.out.println("  --overwrite   目标已存在时覆盖（默认不覆盖，直接失败）");
        System.out.println("  --namenode    NameNode 地址，默认取环境变量 NAMENODE_URI，再退回 "
                + DEFAULT_NAMENODE);
        System.out.println("  --user        简单认证下的远程用户名，默认取环境变量 HADOOP_USER");
        System.out.println("  --replication 副本数，默认沿用集群 dfs.replication");
        System.out.println();
        System.out.println("示例（成员2 的实验命令）:");
        System.out.println("  java -cp \"out:$HADOOP_CP\" com.movie.hdfs.HdfsUploader put \\");
        System.out.println("       --local ../movies-to-be-predicted.txt \\");
        System.out.println("       --remote /input/movies-to-be-predicted.txt --verify");
    }

    private static void printTroubleshooting() {
        System.err.println();
        System.err.println("排错建议:");
        System.err.println("  1. NameNode 地址/端口是否正确：--namenode hdfs://<host>:<8020|9000>");
        System.err.println("  2. 集群是否已启动：jps 应能看到 NameNode / DataNode；hdfs dfsadmin -report 能否执行");
        System.err.println("  3. 主机名解析与防火墙：/etc/hosts 是否有该主机名，端口是否放通");
        System.err.println("  4. 权限：目标目录是否允许当前用户写；必要时 hdfs dfs -chmod -R 755 /input");
        System.err.println("  5. classpath 是否包含 Hadoop 客户端 jar：hadoop classpath");
        System.err.println("  6. Windows 本机用 file:// 调试时需 winutils.exe；连真实 HDFS 集群无此问题，建议在集群客户端执行");
    }

    private static final class DigestResult {
        final String md5;
        final long bytes;

        DigestResult(String md5, long bytes) {
            this.md5 = md5;
            this.bytes = bytes;
        }
    }
}
