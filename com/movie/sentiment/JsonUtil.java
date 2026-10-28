package com.movie.sentiment;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * JSON 工具类：纯手写 StringBuilder 拼接 JSON。
 * 不引入任何第三方 JSON 库。
 * 所有对外输出的字符串值均经过 escapeJsonString 处理。
 */
public class JsonUtil {

    /**
     * 对 JSON 字符串值做最小转义：\ → \\, " → \", 换行 → \n, 回车 → \r, tab → \t, 控制字符 → \\uXXXX。
     */
    public static String escapeJsonString(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.toString();
    }

    /** 把 String 数组拼成 JSON 数组字符串，如 ["Action","Adventure"] */
    public static String jsonArray(String[] arr) {
        if (arr == null) return "null";
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(',');
            sb.append('"').append(escapeJsonString(arr[i])).append('"');
        }
        sb.append(']');
        return sb.toString();
    }

    /**
     * 把 List&lt;String&gt; 拼成 JSON 数组字符串
     */
    public static String jsonArray(List<String> list) {
        if (list == null) return "null";
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append('"').append(escapeJsonString(list.get(i))).append('"');
        }
        sb.append(']');
        return sb.toString();
    }

    /**
     * 提取 JSON 字符串数组字段的值（最小手写解析，不引第三方库）。
     * 用于解析后端请求体 {@code {"tags":["action","adventure"]}}。
     *
     * @param json  请求体原文
     * @param field 字段名（如 "tags"）
     * @return 元素列表（已 trim）；字段不存在或非数组返回 null
     */
    public static List<String> extractJsonStringArray(String json, String field) {
        if (json == null) return null;
        String key = "\"" + field + "\"";
        int ki = json.indexOf(key);
        if (ki < 0) return null;

        int i = ki + key.length();
        while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
        if (i >= json.length() || json.charAt(i) != ':') return null;
        i++;
        while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
        if (i >= json.length() || json.charAt(i) == 'n') return null;   // null 字面量
        if (json.charAt(i) != '[') return null;
        i++;

        List<String> out = new ArrayList<String>();
        while (i < json.length()) {
            char c = json.charAt(i);
            if (Character.isWhitespace(c) || c == ',') { i++; continue; }
            if (c == ']') return out;                                   // 数组结束
            if (c != '"') { i++; continue; }                            // 跳过非字符串元素
            i++;
            StringBuilder sb = new StringBuilder();
            while (i < json.length()) {
                char ch = json.charAt(i++);
                if (ch == '"') break;
                if (ch == '\\' && i < json.length()) {
                    char e = json.charAt(i++);
                    switch (e) {
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case 'b': sb.append('\b'); break;
                        case 'f': sb.append('\f'); break;
                        default:  sb.append(e);
                    }
                } else {
                    sb.append(ch);
                }
            }
            out.add(sb.toString());
        }
        return out;   // 未闭合，返回已解析部分（容错，不抛异常）
    }

    /**
     * 提取 query string 中某个参数的值（已做 URL 解码）。
     * 用于支持 {@code GET /predict?text=action,adventure}。
     *
     * @param query HttpExchange.getRequestURI().getQuery() 的原始串，可为 null
     * @param name  参数名
     * @return 解码后的值；不存在返回 null
     */
    public static String extractQueryParam(String query, String name) {
        if (query == null || query.isEmpty()) return null;
        for (String pair : query.split("&")) {
            int eq = pair.indexOf('=');
            String k = (eq < 0) ? pair : pair.substring(0, eq);
            if (!k.equals(name)) continue;
            String v = (eq < 0) ? "" : pair.substring(eq + 1);
            try {
                // UTF-8 解码：%20 → 空格，%E4%B8%AD → 中文
                return java.net.URLDecoder.decode(v, "UTF-8");
            } catch (Exception e) {
                return v.replace('+', ' ');   // 解码失败退化为最小处理
            }
        }
        return null;
    }

    // ══════════════ 预测结果统一序列化（三处同源） ══════════════
    //
    // 规范要求：predictResult.txt 每行 JSON 字段 == 后端 /predict 响应字段
    //          == 成员4 兜底 parse 字段。为避免"两处各写一遍导致漂移"，
    //          批量预测、后端 handler、predict-single 三处一律调用本方法。

    /**
     * 预测结果视图：承载一次预测的全部产出，供序列化为 JSON / TSV。
     *
     * <p>与 {@link NbPredictor.PredictionResult} 的区别：本类字段已按【对外展示口径】
     * 处理完毕（genre 恢复显示大小写、数组顺序确定、tags 已 normalize），
     * 因此序列化方法无需再依赖模型对象，三处调用都能拿到完全一致的输出。
     */
    public static class PredictionView {
        public int idx;
        public List<String> tags;
        public List<String> trueGenres;      // 无 ground-truth 时为 null
        public List<String> predGenres;
        public Map<String, Double> scores;
        public Map<String, Double> probs;
        public Map<String, Double> labelConfs;  // 逐标签独立置信度（labels 格式用）
        public double confidence;
        public double maxProb;
        public String message = "预测成功";      // labels 格式的 message 字段
        public boolean success = true;

        public PredictionView() {
            this.tags = new ArrayList<String>();
            this.predGenres = new ArrayList<String>();
            this.scores = new LinkedHashMap<String, Double>();
            this.probs = new LinkedHashMap<String, Double>();
            this.labelConfs = new LinkedHashMap<String, Double>();
        }
    }

    /**
     * 由预测结果构建统一的对外视图（三处同源的【唯一】入口）。
     *
     * <p>批量 predictResult.txt、后端 /predict（两种 schema）、predict-single 一律
     * 先调用本方法得到视图，再交给序列化方法输出，从而杜绝"两处各拼一遍 JSON 导致字段漂移"。
     *
     * <p><b>大小写口径</b>（重要，已在 README 说明）：
     * <ul>
     *   <li>{@code predGenres} 与 {@code trueGenres} 保持模型内部的<b>小写</b>形式，
     *       两者可直接比对，也让 predictResult.txt 自洽（评估指标正是基于小写集合计算的）；</li>
     *   <li>成员4 的 labels 格式在序列化时再转成可读大小写（Action / ScienceFiction），
     *       与其规范示例一致。</li>
     * </ul>
     *
     * @param model      已加载模型，用于把 genre 索引映射回名称
     * @param idx        样本序号（批量预测为行号，单条为 0）
     * @param tags       已 normalize 的特征词
     * @param trueGenres ground-truth；无则为 null（序列化输出 null 而非空数组）
     * @param pr         预测器产出
     */
    public static PredictionView buildView(ModelParams model, int idx, List<String> tags,
                                           List<String> trueGenres,
                                           NbPredictor.PredictionResult pr) {
        PredictionView v = new PredictionView();
        v.idx = idx;
        if (tags != null) {
            v.tags = tags;
        }
        v.trueGenres = trueGenres;

        // predGenres 按【逐标签置信度降序】排列，与 labels 格式顺序保持一致，
        // 这样同一份视图序列化成两种格式时，标签顺序不会互相矛盾。
        List<Integer> order = new ArrayList<>(pr.activatedIdx);
        order.sort(new java.util.Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                return Double.compare(pr.labelConfArr[b], pr.labelConfArr[a]);
            }
        });
        List<String> ordered = new ArrayList<>(order.size());
        for (int gIdx : order) {
            // 模型内部 genre 统一小写（如 action），此处保持小写以便与 trueGenres 直接比对
            ordered.add(model.genreArr[gIdx]);
        }
        v.predGenres = ordered;

        v.scores = pr.scores;
        v.probs = pr.probs;
        v.labelConfs = pr.labelConfidences;
        v.confidence = pr.confidence;
        v.maxProb = pr.maxProb;
        return v;
    }

    /**
     * 生成【完整 schema】的预测结果 JSON 单行（predictResult.txt 默认格式、后端 ?schema=full）。
     *
     * <p>schema 与第一阶段/第二阶段规范钉死的字段一致：
     * <pre>
     * {"idx":&lt;int&gt;,"tags":[..],"trueGenres":[..]|null,"predGenres":[..],
     *  "scores":{g:double},"probs":{g:double},"confidence":double,"maxProb":double}
     * </pre>
     */
    public static String predictionLineJson(PredictionView v) {
        return predictionJson(v.idx, v.tags.toArray(new String[0]),
                v.trueGenres == null ? null : v.trueGenres.toArray(new String[0]),
                v.predGenres, v.scores, v.probs, v.confidence, v.maxProb);
    }

    /**
     * 生成【成员4 labels 格式】的预测结果 JSON（后端 /predict 默认响应、前端兜底 parse）。
     *
     * <pre>{"success":true,"message":"预测成功","labels":[{"name":"Action","confidence":0.66}]}</pre>
     *
     * <p>视图内 predGenres 为小写，此处统一经 {@link #genreDisplayName} 转为可读大小写，
     * 与成员4 规范示例（"Action"）一致。labels 顺序沿用视图顺序（已按置信度降序）。
     */
    public static String predictionLabelsJson(PredictionView v) {
        return predictionLabelsJson(v, 0);
    }

    /**
     * 同上，但支持【Top-N 截断】（可选，默认 0 = 不截断）。
     *
     * <p><b>为什么提供这个开关</b>：朴素贝叶斯在输入特征词很少时存在固有的过度激活——
     * 常见 tag（如 {@code murder}，IDF=3.15）在多个 genre 的正样本里都高频出现，
     * 其 logCond 普遍偏高，会把多个 genre 的分数一起抬过阈值。
     * 实测单个词 {@code "murder"} 会激活 12 个标签，而稀有 tag
     * （如 {@code catburglar}，IDF=8.37）只激活 3~4 个。
     *
     * <p>留出集数据表明，在"最大化 F1"目标下这个宽松阈值是【最优的】
     * （k=1 桶 MicroF1=0.4835，收紧会使 F1 下降），所以算法层保留原行为；
     * 但前端展示 12 个置信度都在 0.83~0.85 的标签缺乏区分度，故提供截断选项。
     *
     * <p><b>只作用于 labels 格式</b>：full schema 的 {@code predGenres} 必须与
     * {@code predictResult.txt} 严格一致（三处同源要求），因此不受本参数影响。
     *
     * @param topN &gt;0 时只保留置信度最高的前 N 个标签；&lt;=0 表示不截断
     */
    public static String predictionLabelsJson(PredictionView v, int topN) {
        int n = v.predGenres.size();
        if (topN > 0 && topN < n) {
            n = topN;   // 视图已按置信度降序，直接截前 n 个即可
        }
        List<String> display = new ArrayList<>(n);
        double[] confs = new double[n];
        for (int i = 0; i < n; i++) {
            String g = v.predGenres.get(i);
            display.add(genreDisplayName(g));
            Double c = v.labelConfs.get(g);
            confs[i] = (c == null) ? 0.0 : c;
        }
        return apiSuccessJson(display, confs, v.message);
    }

    /**
     * 生成【TSV 人类可读】单行，供老师/组长用文本编辑器直接查看。
     * 列：idx \t predGenres \t trueGenres \t confidence \t maxProb \t tags(截断)
     */
    public static String predictionTsvLine(PredictionView v) {
        StringBuilder sb = new StringBuilder(256);
        sb.append(v.idx).append('\t');
        sb.append(join(v.predGenres, ",")).append('\t');
        sb.append(v.trueGenres == null ? "-" : join(v.trueGenres, ",")).append('\t');
        sb.append(formatConfidence(v.confidence)).append('\t');
        sb.append(formatConfidence(v.maxProb)).append('\t');
        String t = join(v.tags, ",");
        if (t.length() > 80) t = t.substring(0, 80) + "...";   // 截断，保持行宽可读
        sb.append(t);
        return sb.toString();
    }

    private static String join(List<String> list, String sep) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(sep);
            sb.append(list.get(i));
        }
        return sb.toString();
    }

    /** 把 Map<String, Double> 拼成 JSON 对象字符串，如 {"Action":0.8,"Drama":0.2} */
    public static String jsonDoubleMap(Map<String, Double> map) {
        if (map == null) return "null";
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Double> e : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(escapeJsonString(e.getKey())).append("\":");
            if (e.getValue() == null || Double.isNaN(e.getValue()) || Double.isInfinite(e.getValue())) {
                sb.append("null");
            } else {
                // 保留 6 位小数，兼顾精度与可读性
                sb.append(String.format("%.6f", e.getValue()));
            }
        }
        sb.append('}');
        return sb.toString();
    }

    /**
     * 生成单条预测的 JSON 对象字符串。
     *
     * @param idx        样本索引（0-based）
     * @param tags       tag 数组（原始）
     * @param trueGenres 真实 genre 数组（可为 null）
     * @param predGenres 预测 genre 列表
     * @param scores     各 genre 的 log 分数
     * @param probs      各 genre 的 softmax 概率
     * @param confidence 被激活类 prob 之和
     * @param maxProb    最大 prob
     * @return JSON 字符串（一行）
     */
    public static String predictionJson(int idx, String[] tags, String[] trueGenres,
                                        List<String> predGenres,
                                        Map<String, Double> scores,
                                        Map<String, Double> probs,
                                        double confidence, double maxProb) {
        StringBuilder sb = new StringBuilder(512);
        sb.append("{\"idx\":").append(idx);
        sb.append(",\"tags\":").append(jsonArray(tags));
        sb.append(",\"trueGenres\":").append(trueGenres != null ? jsonArray(trueGenres) : "null");
        sb.append(",\"predGenres\":").append(jsonArray(predGenres));
        sb.append(",\"scores\":").append(jsonDoubleMap(scores));
        sb.append(",\"probs\":").append(jsonDoubleMap(probs));
        sb.append(",\"confidence\":").append(String.format("%.6f", confidence));
        sb.append(",\"maxProb\":").append(String.format("%.6f", maxProb));
        sb.append('}');
        return sb.toString();
    }

    /**
     * 生成 metrics.json 内容字符串。
     */
    public static String metricsJson(Metrics.EvalResult r) {
        StringBuilder sb = new StringBuilder(1024);
        sb.append("{\"n\":").append(r.n);
        sb.append(",\"subsetAccuracy\":").append(r.subsetAccuracy != null ? String.format("%.6f", r.subsetAccuracy) : "null");
        // 2026-10-28 补：评审要求指标文件自带 P/R，不能只给 F1（否则无法核对）
        sb.append(",\"microPrecision\":").append(r.microPrecision != null ? String.format("%.6f", r.microPrecision) : "null");
        sb.append(",\"microRecall\":").append(r.microRecall != null ? String.format("%.6f", r.microRecall) : "null");
        sb.append(",\"microF1\":").append(r.microF1 != null ? String.format("%.6f", r.microF1) : "null");
        sb.append(",\"macroPrecision\":").append(r.macroPrecision != null ? String.format("%.6f", r.macroPrecision) : "null");
        sb.append(",\"macroRecall\":").append(r.macroRecall != null ? String.format("%.6f", r.macroRecall) : "null");
        sb.append(",\"macroF1\":").append(r.macroF1 != null ? String.format("%.6f", r.macroF1) : "null");
        sb.append(",\"hammingLoss\":").append(r.hammingLoss != null ? String.format("%.6f", r.hammingLoss) : "null");
        sb.append(",\"perGenre\":{");
        boolean firstG = true;
        for (Map.Entry<String, Metrics.EvalResult.PerGenre> e : r.perGenre.entrySet()) {
            if (!firstG) sb.append(',');
            firstG = false;
            Metrics.EvalResult.PerGenre pg = e.getValue();
            sb.append('"').append(escapeJsonString(e.getKey())).append("\":{");
            sb.append("\"precision\":").append(String.format("%.6f", pg.precision));
            sb.append(",\"recall\":").append(String.format("%.6f", pg.recall));
            sb.append(",\"f1\":").append(String.format("%.6f", pg.f1));
            sb.append(",\"support\":").append(pg.support);
            sb.append('}');
        }
        sb.append("}}");
        return sb.toString();
    }

    // ══════════════ 以下为前后端接口对接（成员4）所需 ══════════════

    /**
     * genre 显示名：模型内部统一小写（action），对外展示恢复可读大小写（Action）。
     * 对少数拼接词做专门映射，避免 Sciencefiction / Tvmovie 这类不美观结果。
     */
    public static String genreDisplayName(String lowerGenre) {
        if (lowerGenre == null || lowerGenre.isEmpty()) return lowerGenre;
        switch (lowerGenre) {
            case "sciencefiction": return "ScienceFiction";
            case "tvmovie":        return "TVMovie";
            default:
                return Character.toUpperCase(lowerGenre.charAt(0)) + lowerGenre.substring(1);
        }
    }

    /**
     * 接口成功响应（对应《前后端预测接口对接规范》第 2 节）：
     * <pre>
     * {"success":true,"message":"预测成功",
     *  "labels":[{"name":"Action","confidence":0.92}, ...]}
     * </pre>
     *
     * <p>confidence 为【逐标签独立】的 sigmoid 置信度，取值 0~1，
     * 多个标签之间【不要求相加等于 1】（符合规范第 4 节说明）。
     *
     * @param names       标签显示名（已按置信度降序）
     * @param confidences 对应置信度，长度须与 names 一致
     */
    public static String apiSuccessJson(List<String> names, double[] confidences) {
        return apiSuccessJson(names, confidences, "预测成功");
    }

    /**
     * 接口成功响应（可自定义 message）。
     *
     * @param message 提示信息，如"预测成功"；未命中特征词时可传更具体的引导语
     */
    public static String apiSuccessJson(List<String> names, double[] confidences, String message) {
        StringBuilder sb = new StringBuilder(256);
        sb.append("{\"success\":true,\"message\":\"")
                .append(escapeJsonString(message)).append("\",\"labels\":[");
        int n = Math.min(names.size(), confidences.length);
        for (int i = 0; i < n; i++) {
            if (i > 0) sb.append(',');
            sb.append("{\"name\":\"").append(escapeJsonString(names.get(i))).append('"');
            sb.append(",\"confidence\":").append(formatConfidence(confidences[i])).append('}');
        }
        sb.append("]}");
        return sb.toString();
    }

    /**
     * 接口错误响应（对应规范第 3 节第 4 条），HTTP 400/500 均用此统一格式：
     * <pre>{"success":false,"message":"请输入电影描述","labels":[]}</pre>
     */
    public static String apiErrorJson(String message) {
        return "{\"success\":false,\"message\":\"" + escapeJsonString(message) + "\",\"labels\":[]}";
    }

    /**
     * 置信度格式化：保留 4 位小数并去掉无意义的尾随零，输出为合法 JSON 数字。
     * 规范明确要求 confidence 是数字（如 0.92），【不能】是字符串 "92%"。
     * 同时防御 NaN/Infinity（JSON 不允许），退化为 0。
     */
    public static String formatConfidence(double c) {
        if (Double.isNaN(c) || Double.isInfinite(c)) return "0.0";
        // 夹到 [0,1]，防止极端数值破坏前端百分比展示
        if (c < 0.0) c = 0.0;
        if (c > 1.0) c = 1.0;
        String s = String.format(Locale.ROOT, "%.4f", c);
        // 去尾随零：0.9200 → 0.92；但保证至少一位小数（1.0000 → 1.0）
        if (s.indexOf('.') >= 0) {
            s = s.replaceAll("0+$", "");
            if (s.endsWith(".")) s += "0";
        }
        return s;
    }

    /**
     * 最小 JSON 字符串字段提取器（不引第三方库）。
     * 用于解析前端请求体 {"text":"..."}，支持 \" \\ \n \r \t 及 unicode 转义还原。
     *
     * @param json  请求体原文
     * @param field 字段名（如 "text"）
     * @return 字段字符串值；字段不存在返回 null
     */
    public static String extractJsonStringField(String json, String field) {
        if (json == null) return null;
        String key = "\"" + field + "\"";
        int ki = json.indexOf(key);
        if (ki < 0) return null;

        int i = ki + key.length();
        // 跳过空白找冒号
        while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
        if (i >= json.length() || json.charAt(i) != ':') return null;
        i++;
        while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
        if (i >= json.length()) return null;
        // null 字面量
        if (json.charAt(i) == 'n') {
            if (json.startsWith("null", i)) return null;
            return null;
        }
        if (json.charAt(i) != '"') return null;   // 非字符串值，本接口不需要
        i++;

        StringBuilder sb = new StringBuilder();
        while (i < json.length()) {
            char c = json.charAt(i++);
            if (c == '"') {
                return sb.toString();             // 字符串结束
            }
            if (c == '\\') {
                if (i >= json.length()) break;
                char e = json.charAt(i++);
                switch (e) {
                    case '"':  sb.append('"');  break;
                    case '\\': sb.append('\\'); break;
                    case '/':  sb.append('/');  break;
                    case 'b':  sb.append('\b'); break;
                    case 'f':  sb.append('\f'); break;
                    case 'n':  sb.append('\n'); break;
                    case 'r':  sb.append('\r'); break;
                    case 't':  sb.append('\t'); break;
                    case 'u':
                        if (i + 4 <= json.length()) {
                            try {
                                sb.append((char) Integer.parseInt(json.substring(i, i + 4), 16));
                            } catch (NumberFormatException ex) {
                                sb.append('?');
                            }
                            i += 4;
                        }
                        break;
                    default: sb.append(e);
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();   // 未闭合引号，返回已读部分（容错）
    }
}