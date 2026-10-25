#!/usr/bin/env bash
# ============================================================================
# 成员2：HDFS 数据上传与校验
#   步骤 1  命令行方式上传训练集  movies-from-kggle.txt        （hdfs dfs -put）
#   步骤 2  Java API 方式上传测试集 movies-to-be-predicted.txt（HdfsUploader）
#   步骤 3  校验：文件大小 / 行数 / 权限 / 属主
#
# 在【集群客户端节点】执行（Linux）。用法：
#   export NAMENODE="hdfs://master:9000"       # 本项目实测值
#   export HDFS_USER="maototo"                 # 可选；不填则用当前系统用户
#   bash scripts/hdfs-upload.sh
#
# 脚本不会掩盖失败：任一步失败都会打印 [FAIL] 并继续记录后续检查，
# 最后以退出码 1 结束，便于在报告里如实记录问题。
# ============================================================================
set -u

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

NAMENODE="${NAMENODE:-hdfs://master:9000}"        # 本项目实测 NameNode
HDFS_INPUT="${HDFS_INPUT:-/input}"
HDFS_USER="${HDFS_USER:-maototo}"
TRAIN_LOCAL="${TRAIN_LOCAL:-$BASE_DIR/../大数据课程考核26/大数据课程考核26/movies-from-kggle.txt}"
TEST_LOCAL="${TEST_LOCAL:-$BASE_DIR/../大数据课程考核26/大数据课程考核26/movies-to-be-predicted.txt}"

TRAIN_HDFS="$HDFS_INPUT/movies-from-kggle.txt"
TEST_HDFS="$HDFS_INPUT/movies-to-be-predicted.txt"

LOG_DIR="$BASE_DIR/output"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/hdfs-upload-$(date +%Y%m%d-%H%M%S).log"

# 所有输出同时写屏幕和日志文件（报告的证据来源）
exec > >(tee -a "$LOG") 2>&1

FAILED=0
section() { printf '\n==================== %s ====================\n' "$*"; }
fail()    { echo "[FAIL] $*"; FAILED=1; }

section "0. 环境与集群状态"
echo "执行时间   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "NameNode   : $NAMENODE"
echo "HDFS 目录  : $HDFS_INPUT"
echo "训练集本地 : $TRAIN_LOCAL"
echo "测试集本地 : $TEST_LOCAL"
echo
echo "--- hadoop version ---"
hadoop version || fail "hadoop 命令不可用"
echo
echo "--- fs.defaultFS / dfs.replication ---"
hdfs getconf -confKey fs.defaultFS || fail "读取 fs.defaultFS 失败"
hdfs getconf -confKey dfs.replication || fail "读取 dfs.replication 失败"
echo
echo "--- jps ---"
jps || true
echo
echo "--- hdfs dfsadmin -report（前 25 行）---"
hdfs dfsadmin -report 2>&1 | head -n 25 || fail "dfsadmin -report 失败"

for f in "$TRAIN_LOCAL" "$TEST_LOCAL"; do
    if [ ! -f "$f" ]; then
        fail "本地数据文件不存在: $f"
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "[ABORT] 本地数据文件缺失，终止。"
    exit 1
fi

section "1. 命令行上传训练集（hdfs dfs -put）"
hdfs dfs -mkdir -p "$HDFS_INPUT" || fail "创建 $HDFS_INPUT 失败"
hdfs dfs -ls -d "$HDFS_INPUT" || true
echo
echo "--- 上传前目录内容 ---"
hdfs dfs -ls -h "$HDFS_INPUT" || true
echo
echo "--- 执行上传：hdfs dfs -put -f $TRAIN_LOCAL $HDFS_INPUT/ ---"
if hdfs dfs -put -f "$TRAIN_LOCAL" "$HDFS_INPUT/"; then
    echo "[OK] 训练集上传命令返回成功"
else
    fail "训练集上传失败"
fi
echo
echo "--- 上传后目录内容 ---"
hdfs dfs -ls -h "$HDFS_INPUT" || true

section "2. 设置并检查权限（成员3 需要只读权限）"
hdfs dfs -chmod 755 "$HDFS_INPUT" || fail "chmod 755 $HDFS_INPUT 失败"
hdfs dfs -chmod 644 "$TRAIN_HDFS" || fail "chmod 644 $TRAIN_HDFS 失败"
hdfs dfs -stat "路径=%n 属主=%u 属组=%g 副本=%r 权限=%a 文件字节=%b 块大小=%o 修改时间=%Y" "$HDFS_INPUT" || true
hdfs dfs -stat "路径=%n 属主=%u 属组=%g 副本=%r 权限=%a 文件字节=%b 块大小=%o 修改时间=%Y" "$TRAIN_HDFS" || true

section "3. Java API 上传测试集（HdfsUploader）"
HADOOP_CP="$(hadoop classpath)" || fail "hadoop classpath 失败"
mkdir -p "$BASE_DIR/out"
echo "--- 编译 HdfsUploader ---"
if javac -encoding UTF-8 -cp "$HADOOP_CP" -d "$BASE_DIR/out" \
        "$BASE_DIR/com/movie/hdfs/HdfsUploader.java"; then
    echo "[OK] 编译通过"
else
    fail "编译失败"
fi

echo
echo "--- 执行 Java API 上传（含回读 MD5 校验）---"
JAVA_CMD=(java -Dfile.encoding=UTF-8 -cp "$BASE_DIR/out:$HADOOP_CP"
          com.movie.hdfs.HdfsUploader put
          --local "$TEST_LOCAL" --remote "$TEST_HDFS" --namenode "$NAMENODE"
          --overwrite --verify)
if [ -n "$HDFS_USER" ]; then
    JAVA_CMD=(java -Dfile.encoding=UTF-8 -cp "$BASE_DIR/out:$HADOOP_CP"
              com.movie.hdfs.HdfsUploader put
              --local "$TEST_LOCAL" --remote "$TEST_HDFS" --namenode "$NAMENODE"
              --user "$HDFS_USER" --overwrite --verify)
fi
if "${JAVA_CMD[@]}"; then
    echo "[OK] Java API 上传并通过 MD5 校验"
else
    fail "Java API 上传或校验失败"
fi

section "4. 上传结果校验（大小 / 行数 / 内容）"
echo "--- 远端文件清单 ---"
hdfs dfs -ls -h "$HDFS_INPUT" || true
echo
echo "--- 本地大小 vs HDFS 大小 ---"
printf 'train 本地 : %s 字节\n' "$(wc -c < "$TRAIN_LOCAL")"
printf 'train HDFS : %s 字节\n' "$(hdfs dfs -cat "$TRAIN_HDFS" | wc -c)"
printf 'test  本地 : %s 字节\n' "$(wc -c < "$TEST_LOCAL")"
printf 'test  HDFS : %s 字节\n' "$(hdfs dfs -cat "$TEST_HDFS" | wc -c)"
echo
echo "--- 行数对比 ---"
printf 'train 本地 : %s 行\n' "$(wc -l < "$TRAIN_LOCAL")"
printf 'train HDFS : %s 行\n' "$(hdfs dfs -cat "$TRAIN_HDFS" | wc -l)"
printf 'test  本地 : %s 行\n' "$(wc -l < "$TEST_LOCAL")"
printf 'test  HDFS : %s 行\n' "$(hdfs dfs -cat "$TEST_HDFS" | wc -l)"
printf 'test  记录数（末尾无换行，wc -l 会少 1）: %s\n' "$(awk 'END{print NR}' "$TEST_LOCAL")"
echo
echo "--- 首行内容（确认数据格式）---"
hdfs dfs -head "$TRAIN_HDFS" | head -n 1 || true
echo
echo "--- HDFS 自身校验和（MD5-of-MD5，仅作集群侧一致性参考）---"
hdfs dfs -checksum "$TRAIN_HDFS" || true
hdfs dfs -checksum "$TEST_HDFS" || true

section "5. 结束"
echo "日志文件: $LOG"
if [ "$FAILED" -eq 0 ]; then
    echo "[DONE] 全部步骤成功，可直接把本日志对应截图贴入报告。"
    exit 0
else
    echo "[DONE] 存在失败步骤，请按日志中的 [FAIL] 排查。"
    exit 1
fi
