package com.movie.sentiment;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;

/**
 * Module-SERVER：HTTP 后端服务（仅用 JDK 内置 com.sun.net.httpserver，无 Web 框架）。
 *
 * <p><b>路由</b>：
 * <ul>
 *   <li>{@code POST /predict} — JSON body，兼容 {@code {"tags":["a","b"]}} 与 {@code {"text":"a b,c"}}</li>
 *   <li>{@code GET  /predict?text=a,b,c} — query 传参，便于 curl/浏览器直接自测</li>
 *   <li>{@code GET  /genres} — 全部标签显示名（前端做下拉/图例）</li>
 *   <li>{@code GET  /health} — 模型规模与校准状态</li>
 * </ul>
 *
 * <p><b>响应 schema</b>（两套并存，由 {@code ?schema=} 选择，默认 labels）：
 * <ul>
 *   <li>{@code labels}（默认，成员4《前后端预测接口对接规范》）：
 *       {@code {"success":true,"message":"预测成功","labels":[{"name":"Action","confidence":0.66}]}}
 *       <br>业务错误返 400 + {@code {"success":false,"message":"..","labels":[]}}</li>
 *   <li>{@code full}（第二阶段 Module-BATCH 钉死的完整 schema，与 predictResult.txt 单行一致）：
 *       {@code {"idx":0,"tags":[..],"trueGenres":null,"predGenres":[..],"scores":{..},"probs":{..},
 *       "confidence":..,"maxProb":..}}
 *       <br>业务错误返 200 + {@code {"error":"..","predGenres":[]}}，便于前端统一 parse</li>
 * </ul>
 *
 * <p><b>三处同源</b>：本 handler 与批量 predict、predict-single 一律经
 * {@link JsonUtil#buildView} 构建视图再序列化，杜绝字段漂移。
 *
 * <p><b>CORS 必做</b>（联调最大坑）：所有响应带 Allow-Origin/Methods/Headers，
 * OPTIONS 预检直接 204。成员4 前端可能跑在 file:// 或异端口，跨域必须放开。
 *
 * <p><b>模型单例</b>：启动时一次性反序列化到内存，所有请求复用同一个无状态
 * {@link NbPredictor}（线程安全），严禁每请求读盘/读 HDFS。
 */
public class PredictServer {

    /** 默认端口，对应 ${SERVER_PORT} */
    public static final int DEFAULT_PORT = 8080;

    /** 请求体大小上限（1 MB），防御异常大请求打爆内存 */
    private static final int MAX_BODY_BYTES = 1024 * 1024;

    /** 模型单例（启动时加载一次） */
    private final ModelParams model;
    /** 预测器单例：无状态、可多线程并发复用 */
    private final NbPredictor predictor;
    private final int port;
    private HttpServer server;

    public PredictServer(ModelParams model, int port) {
        this(model, port, 0.0, 0.0);
    }

    /**
     * @param zThreshold 行内 z 分数阈值，&gt;0 启用相对阈值决策（提高精确率）
     * @param minPrior   长尾 genre 先验下限，低于此值的 genre 不参与激活
     */
    public PredictServer(ModelParams model, int port, double zThreshold, double minPrior) {
        this(model, port, zThreshold, minPrior, null);
    }

    /**
     * @param zTauPerGenre 逐 genre 的 z 阈值表（null → 统一使用 zThreshold）
     */
    public PredictServer(ModelParams model, int port, double zThreshold, double minPrior,
                         double[] zTauPerGenre) {
        this.model = model;
        this.predictor = new NbPredictor(model, zThreshold, minPrior, zTauPerGenre);
        this.port = port;
    }

    /** 启动服务并阻塞（Ctrl+C 退出）。绑定 0.0.0.0 以便成员4 从其它机器访问。 */
    public void start() throws IOException {
        // 0.0.0.0：局域网内成员4 可用 http://<本机IP>:port 访问（不只是 localhost）
        server = HttpServer.create(new InetSocketAddress("0.0.0.0", port), 0);
        server.createContext("/predict", this::handlePredict);
        server.createContext("/genres", this::handleGenres);
        server.createContext("/health", this::handleHealth);
        // 固定线程池支持前端并发请求（JDK 标准库，非第三方）
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();

        System.out.println("══════ HTTP 预测服务已启动 ══════");
        System.out.println("监听地址: 0.0.0.0:" + port);
        System.out.println("预测接口: POST http://localhost:" + port + "/predict");
        System.out.println("          GET  http://localhost:" + port + "/predict?text=action,adventure");
        System.out.println("标签列表: GET  http://localhost:" + port + "/genres");
        System.out.println("健康检查: GET  http://localhost:" + port + "/health");
        System.out.printf("模型摘要: N=%d |V|=%d |G|=%d τ=%.4f  逐genre校准=%s  文本校准=%s%n",
                model.N, model.V, model.G, model.tau,
                model.hasPerGenreCalibration() ? "已启用" : "未启用",
                model.hasTextCalibration() ? "已启用" : "未启用");
        System.out.println();
        System.out.println("curl 自测命令（供成员4/组长一键验证）：");
        System.out.println("  curl -X POST http://localhost:" + port + "/predict "
                + "-H \"Content-Type: application/json\" -d \"{\\\"tags\\\":[\\\"action\\\",\\\"adventure\\\"]}\"");
        System.out.println("  curl \"http://localhost:" + port + "/predict?text=spacewar,alien,marine\"");
        System.out.println();
        System.out.println("按 Ctrl+C 停止服务");
    }

    /** 停止服务（供测试/嵌入调用）。 */
    public void stop() {
        if (server != null) {
            server.stop(0);
        }
    }

    // ──────────────────── /predict ────────────────────

    private void handlePredict(HttpExchange ex) throws IOException {
        // 响应 schema：labels（默认，成员4 前端）| full（第二阶段完整字段）
        boolean fullSchema = "full".equalsIgnoreCase(
                JsonUtil.extractQueryParam(ex.getRequestURI().getQuery(), "schema"));
        // 可选 Top-N 截断（仅对 labels 格式生效，full schema 必须与 predictResult.txt 一致）
        int topN = parseIntOr(JsonUtil.extractQueryParam(ex.getRequestURI().getQuery(), "topN"), 0);
        try {
            // CORS 预检：直接 204 + CORS 头，不读 body
            if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                send(ex, 204, "");
                return;
            }

            String method = ex.getRequestMethod().toUpperCase(Locale.ROOT);
            if (!"POST".equals(method) && !"GET".equals(method)) {
                respondError(ex, fullSchema, 405, "该接口仅支持 POST 与 GET 请求");
                return;
            }

            // ── 取输入：POST 读 body（tags 优先，其次 text）；GET 读 query ──
            List<String> rawTags = null;
            String text = null;

            if ("POST".equals(method)) {
                String body = readBody(ex);
                // 入参形式(1)：{"tags":["a","b",...]} → 直接作为 tag 数组
                rawTags = JsonUtil.extractJsonStringArray(body, "tags");
                // 入参形式(2)：{"text":"a b,c|d"} → 按 [,|\s]+ 切分
                if (rawTags == null || rawTags.isEmpty()) {
                    text = JsonUtil.extractJsonStringField(body, "text");
                }
                // TODO 上线前与成员4敲定其一：目前两种入参都兼容（tags 优先），
                //      确认前端固定用哪种后可收窄，减少歧义。
            } else {
                String q = ex.getRequestURI().getQuery();
                text = JsonUtil.extractQueryParam(q, "text");
                if (text == null || text.trim().isEmpty()) {
                    String tagsParam = JsonUtil.extractQueryParam(q, "tags");
                    if (tagsParam != null && !tagsParam.trim().isEmpty()) {
                        text = tagsParam;   // ?tags=a,b 同样按分隔符切
                    }
                }
            }

            // ── 输入为空：参数不合法 ──
            boolean noInput = (rawTags == null || rawTags.isEmpty())
                    && (text == null || text.trim().isEmpty());
            if (noInput) {
                respondError(ex, fullSchema, 400, "请输入电影描述或 tag 列表");
                return;
            }

            // ── 解析为 tag 数组：与训练期【同一】normalize（trim + 小写 + 去包裹引号）──
            List<String> tags;
            if (rawTags != null && !rawTags.isEmpty()) {
                tags = Main.normalizeTags(rawTags);
            } else {
                // text 形式：优先按分隔符切（tag 串），再经 TextTokenizer 映射自然语言。
                // TextTokenizer.resolveTags 已同时处理这两种形态，与后端其它入口同源。
                tags = TextTokenizer.resolveTags(text.trim(), model);
            }

            // ── 预测：全 OOV 不报错，返回空 predGenres + confidence=0 + 结构化提示 ──
            NbPredictor.PredictionResult pr = predictor.predictAuto(tags);
            JsonUtil.PredictionView v = JsonUtil.buildView(model, 0, tags, null, pr);

            if (v.predGenres.isEmpty()) {
                v.message = tags.isEmpty()
                        ? "未识别到已知特征词，请补充更具体的剧情描述"
                        : "未能激活任何类型标签（输入特征区分度不足）";
            }
            send(ex, 200, fullSchema
                    ? JsonUtil.predictionLineJson(v)
                    : JsonUtil.predictionLabelsJson(v, topN));

        } catch (RuntimeException e) {
            // 业务/未预期异常：不外泄异常栈给前端，统一转结构化 JSON
            System.err.println("[ERROR] /predict 处理失败: " + e);
            e.printStackTrace();
            respondError(ex, fullSchema, 500, "服务器内部错误，请稍后重试");
        } finally {
            ex.close();
        }
    }

    /**
     * 业务错误响应（两种 schema 各自的口径，见类注释）：
     * <ul>
     *   <li>labels：成员4 规范要求的 400/500 + {@code {"success":false,"message":..,"labels":[]}}</li>
     *   <li>full：第二阶段规范要求的 200 + {@code {"error":"..","predGenres":[]}}，
     *       让前端能统一 parse（不因状态码分支）</li>
     * </ul>
     */
    private static void respondError(HttpExchange ex, boolean fullSchema,
                                     int statusForLabels, String message) throws IOException {
        if (fullSchema) {
            // full schema：一律 200，错误信息放 error 字段，其余字段给空值保证结构完整
            String json = "{\"idx\":0,\"tags\":[],\"trueGenres\":null,\"predGenres\":[]"
                    + ",\"scores\":{},\"probs\":{},\"confidence\":0.0,\"maxProb\":0.0"
                    + ",\"error\":\"" + JsonUtil.escapeJsonString(message) + "\"}";
            send(ex, 200, json);
        } else {
            send(ex, statusForLabels, JsonUtil.apiErrorJson(message));
        }
    }

    // ──────────────────── /genres ────────────────────

    /**
     * GET /genres → {@code {"success":true,"message":"ok","genres":["Action","Adventure",...]}}
     * 供前端做标签下拉/图例；非成员4 规范约定的核心接口，字段独立不影响 /predict。
     */
    private void handleGenres(HttpExchange ex) throws IOException {
        try {
            if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                send(ex, 204, "");
                return;
            }
            StringBuilder sb = new StringBuilder(256);
            sb.append("{\"success\":true,\"message\":\"ok\",\"genres\":[");
            for (int i = 0; i < model.G; i++) {
                if (i > 0) sb.append(',');
                sb.append('"').append(JsonUtil.escapeJsonString(
                        JsonUtil.genreDisplayName(model.genreArr[i]))).append('"');
            }
            sb.append("]}");
            send(ex, 200, sb.toString());
        } finally {
            ex.close();
        }
    }

    // ──────────────────── /health ────────────────────

    /** GET /health → 模型规模与校准状态，供组长/成员4 快速确认后端已正确加载模型。 */
    private void handleHealth(HttpExchange ex) throws IOException {
        try {
            if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                send(ex, 204, "");
                return;
            }
            String json = "{\"success\":true,\"message\":\"ok\""
                    + ",\"N\":" + model.N
                    + ",\"V\":" + model.V
                    + ",\"G\":" + model.G
                    + ",\"tau\":" + String.format(Locale.ROOT, "%.4f", model.tau)
                    + ",\"calibrated\":" + model.hasPerGenreCalibration()
                    + ",\"textCalibrated\":" + model.hasTextCalibration()
                    + ",\"trainSource\":\"" + JsonUtil.escapeJsonString(
                            model.trainSource == null ? "" : model.trainSource) + "\""
                    + "}";
            send(ex, 200, json);
        } finally {
            ex.close();
        }
    }

    // ──────────────────── IO 工具 ────────────────────

    /**
     * 安全解析整数：query 参数缺失或非法时返回默认值，不抛异常。
     * 用于 {@code ?topN=3} 这类可选数值参数（前端传错值也不应让请求 500）。
     */
    private static int parseIntOr(String s, int def) {
        if (s == null || s.trim().isEmpty()) {
            return def;
        }
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    /** 读取请求体（UTF-8，带大小上限保护）。 */
    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream in = ex.getRequestBody()) {
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            int read;
            int total = 0;
            while ((read = in.read(chunk)) != -1) {
                total += read;
                if (total > MAX_BODY_BYTES) {
                    throw new IOException("请求体超过 " + MAX_BODY_BYTES + " 字节上限");
                }
                buf.write(chunk, 0, read);
            }
            return new String(buf.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    /**
     * 统一响应：UTF-8 JSON + CORS 头。
     *
     * <p>CORS 对齐 Module-SERVER 规范（联调最大坑）：成员4 前端可能跑在 file://
     * 或异端口，必须放开跨域，否则浏览器 fetch 会被 CORS 策略拦下。
     */
    private static void send(HttpExchange ex, int status, String json) throws IOException {
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=UTF-8");
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        byte[] bytes = (json == null) ? new byte[0] : json.getBytes(StandardCharsets.UTF_8);
        // 204 无响应体：长度传 -1，否则浏览器会一直等 body 而超时
        ex.sendResponseHeaders(status, bytes.length == 0 ? -1 : bytes.length);
        if (bytes.length > 0) {
            try (OutputStream os = ex.getResponseBody()) {
                os.write(bytes);
            }
        }
    }
}
