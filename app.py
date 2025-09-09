# app.py  —— 连续 Trigger Day + 6 模型 / 2 编码器 + E2 百分位说明 + 简易登录
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ========== 0) 登录 ==========
st.set_page_config(page_title="Gn Starting Protocol Prediction", layout="wide")
st.title("🧬 Personalized Gn Starting Protocol Prediction System")

# 简易账号；用户名：0001~0006；密码：123456
VALID_USERS = {f"{i:04d}": "123456" for i in range(1, 7)}

with st.sidebar:
    st.header("🔐 Login")
    username = st.text_input("User ID", value="", key="uid")
    password = st.text_input("Password", value="", type="password", key="pwd")
    login_ok = (username in VALID_USERS) and (password == VALID_USERS[username])

if not login_ok:
    # 不显示任何文字提示
    st.stop()

st.markdown("🔍 Please fill in the baseline information and up to three sets of hormone monitoring data (some values can be missing).")

from pathlib import Path

# 找到 app.py 所在目录
BASE_DIR = Path(__file__).resolve().parent

# 模型目录：优先环境变量 MODEL_DIR，否则使用仓库里的 final_models/
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "final_models")))

# 统一为“bundle”结构：{"pipeline": pipe, "features": [...]}
def load_bundle(name):
    path = MODEL_DIR / name
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["features"]

# 6 个模型（2 回归-核心，2 分类-核心，2 回归-核心+动态）
reg_start_model,     F_START   = load_bundle("reg_start_model.pkl")
reg_total_model,     F_TOTAL   = load_bundle("reg_total_model.pkl")
clf_drug_model,      F_DRUG    = load_bundle("clf_drug_model.pkl")
clf_protocol_model,  F_PROTO   = load_bundle("clf_protocol_model.pkl")
reg_trigger_model,   F_TRIG    = load_bundle("reg_trigger_model.pkl")   # ✅ 连续 Trigger Day
reg_days_model,      F_DAYS    = load_bundle("reg_days_model.pkl")

# 2 个编码器
drug_encoder     = joblib.load(os.path.join(MODEL_DIR, "drug_encoder.pkl"))
protocol_encoder = joblib.load(os.path.join(MODEL_DIR, "protocol_encoder.pkl"))

# E2 分位信息（只存了统计值，而不是原始数组）
e2_percentiles   = joblib.load(os.path.join(MODEL_DIR, "e2_percentiles.pkl"))

# ========== 2) 特征定义（与训练一致）==========
MODEL_CORE_FEATURES = F_START  # 与存档保持一致
# 动态/全部特征从 reg_trigger_model / reg_days_model 的 bundle 取
MODEL_ALL_FEATURES  = F_TRIG

# 为了界面输入更友好（显示顺序不影响模型）
UI_CORE_ORDER = [
    "年龄", "体重指数",
    "(基础内分泌)FSH", "(基础内分泌)LH", "(基础内分泌)PRL",
    "(基础内分泌)E2", "(基础内分泌)T", "(基础内分泌)AMH",
    "左窦卵泡数", "右窦卵泡数"
]

# ========== 3) 采集用户输入 ==========
user_input = {}
st.sidebar.header("📌 Baseline Information")
for feat in UI_CORE_ORDER:
    user_input[feat] = st.sidebar.number_input(feat, value=0.0, format="%.2f")

st.sidebar.header("📊 Dynamic Monitoring Data (can be partially missing)")
for i in range(1, 3 + 1):
    st.sidebar.markdown(f"###### Monitoring {i}")
    for prefix in ["血E2", "血LH", "血FSH", "血P"]:
        key = f"{prefix}_{i}"
        user_input[key] = st.sidebar.number_input(key, value=np.nan, format="%.2f")
    user_input[f"Day_{i}"] = st.sidebar.number_input(f"Day_{i}", value=np.nan, format="%.0f")

# 额外三个输入（允许缺失）
user_input["最大卵泡测定日3"] = st.sidebar.number_input("Max follicle measurement day 3", value=np.nan, format="%.0f")
user_input["左侧最大卵泡直径3"] = st.sidebar.number_input("Left max follicle diameter 3", value=np.nan, format="%.2f")
user_input["右侧最大卵巢直径3"] = st.sidebar.number_input("Right max follicle diameter 3", value=np.nan, format="%.2f")

# 构建输入 DF，并用 0.0 简单填补（训练管道里仍有 Imputer 兜底）
input_df = pd.DataFrame([user_input])
input_df_filled = input_df.fillna(0.0)

# 严格按训练时顺序取列
X_core = input_df_filled[MODEL_CORE_FEATURES]
X_all  = input_df_filled[MODEL_ALL_FEATURES]

# ========== 4) 推理 ==========
dose_pred        = float(reg_start_model.predict(X_core)[0])
total_dose_pred  = float(reg_total_model.predict(X_core)[0])
drug_label       = int(clf_drug_model.predict(X_core)[0])
protocol_label   = int(clf_protocol_model.predict(X_core)[0])
drug_pred        = drug_encoder.inverse_transform([drug_label])[0]
protocol_pred    = protocol_encoder.inverse_transform([protocol_label])[0]
trigger_day_cont = float(reg_trigger_model.predict(X_all)[0])            # ✅ 连续
trigger_day_int  = int(np.rint(trigger_day_cont))
total_days_pred  = float(reg_days_model.predict(X_all)[0])

# ========== 5) 结果展示（尽量保持你原格式）==========
st.subheader("🎯 Prediction Results")
st.markdown(f"""
- 💉 **Recommended Gn starting dose**: {dose_pred:.0f} IU  
- 💊 **Recommended drug type**: {drug_pred}  
- 🧩 **Recommended protocol**: {protocol_pred}  
- 📦 **Predicted total Gn dose**: {total_dose_pred:.0f} IU  
- ⏳ **Predicted total Gn days**: {total_days_pred:.1f} days  
- 🚦 **Recommended Trigger day**: **Day {trigger_day_int}** (continuous prediction: {trigger_day_cont:.2f})
""")

# ========== 6) E2 百分位工具 ==========
# e2_percentiles 结构示例：{"基础E2": {"n":..., "p5":..., "p25":..., "p50":..., "p75":..., "p95":..., "min":..., "max":...}, ...}

def get_stats(key):
    obj = e2_percentiles.get(key, None)
    if not isinstance(obj, dict):
        return None
    # 只要到的值，用于近似百分位插值
    return {
        "n": obj.get("n"),
        "min": obj.get("min"),
        "p5": obj.get("p5"),
        "p25": obj.get("p25"),
        "p50": obj.get("p50"),
        "p75": obj.get("p75"),
        "p95": obj.get("p95"),
        "max": obj.get("max"),
    }

def approx_percentile(x, s):
    """
    用 (min, P5, P25, P50, P75, P95, max) 的分段线性插值，估算 x 的总体百分位。
    返回整数百分位（0–100），异常返回 None。
    """
    if s is None or x is None or np.isnan(x):
        return None
    # 关键节点
    knots = []
    for val, p in [(s.get("min"), 0), (s.get("p5"), 5), (s.get("p25"), 25),
                   (s.get("p50"), 50), (s.get("p75"), 75), (s.get("p95"), 95), (s.get("max"), 100)]:
        if val is not None:
            knots.append((float(val), p))
    if not knots:
        return None
    # 边界裁剪
    xs = [k[0] for k in knots]
    ps = [k[1] for k in knots]
    if x <= xs[0]:
        return int(ps[0])
    if x >= xs[-1]:
        return int(ps[-1])
    # 查所在区间线性插值
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, p0 = xs[i-1], ps[i-1]
            x1, p1 = xs[i], ps[i]
            # 避免除零
            if x1 == x0:
                return int(round((p0 + p1) / 2))
            t = (x - x0) / (x1 - x0)
            p = p0 + t * (p1 - p0)
            return int(round(p))
    return None

# ========== 7) Serum E2 percentile plot（保持原图形样式）==========
st.subheader("📈 Serum E2 percentile plot")
fig, ax = plt.subplots()
percentile_text = []

for i, key in enumerate(["血E2_1", "血E2_2", "血E2_3"], start=1):
    stats = get_stats(key)
    val = float(input_df_filled[key].values[0])
    if stats is not None:
        # 显示 P25–P75 柱
        if (stats["p25"] is not None) and (stats["p75"] is not None):
            ax.plot([i, i], [stats["p25"], stats["p75"]], linewidth=6)
        # 当前值与 P 值标注
        if not np.isnan(val):
            pr = approx_percentile(val, stats)  # ✅ 新增：近似百分位
            ax.scatter(i, val, s=40)
            label_txt = f"{val:.0f}" + (f" (P{pr})" if pr is not None else "")
            ax.text(i + 0.1, val, label_txt, fontsize=9)
            if pr is not None:
                percentile_text.append(f"- **{key}**: {val:.0f} pg/mL, at **P{pr}**")
            else:
                percentile_text.append(f"- **{key}**: {val:.0f} pg/mL (reference P25–P75)")

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["E2_1", "E2_2", "E2_3"])
ax.set_ylabel("E2 (pg/mL)")
ax.set_title("Serum E2 percentile plot")
st.pyplot(fig)

if percentile_text:
    st.markdown("🔢 **Percentile explanation:**")
    for text in percentile_text:
        st.markdown(text)

# ========== 8) Baseline E2 percentile plot（保持原样式）==========
st.subheader("📊 Baseline E2 percentile plot")
base_key = "(基础内分泌)E2"
base_val = float(input_df_filled[base_key].values[0])
base_stats = get_stats("基础E2")

if base_stats is not None and not np.isnan(base_val):
    fig2, ax2 = plt.subplots()
    if base_stats["p25"] is not None and base_stats["p75"] is not None:
        ax2.plot([1, 1], [base_stats["p25"], base_stats["p75"]], linewidth=6, label="P25–P75")
    if base_stats["p50"] is not None:
        ax2.axhline(base_stats["p50"], linestyle='--', label="P50")
    ax2.scatter(1, base_val, s=80, label=f"Your value: {base_val:.0f}")
    ax2.set_xlim(0.5, 1.5)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["Baseline E2"])
    ax2.set_ylabel("E2 (pg/mL)")
    ax2.set_title("Baseline E2 percentile plot")
    ax2.legend()
    st.pyplot(fig2)

    pr_base = approx_percentile(base_val, base_stats)  # ✅ 使用近似百分位
    if pr_base is not None:
        st.markdown(f"🔢 Your **Baseline E2** value is **{base_val:.0f} pg/mL**, at about **P{pr_base}**.")
    else:
        st.markdown(f"🔢 Your **Baseline E2** value is **{base_val:.0f} pg/mL** (reference P25–P75).")
else:
    st.warning("⚠️ Baseline E2 missing or no reference data available, cannot display percentile plot.")


