import math
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Polygon
import platform

# OS별 폰트 설정
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"  # Streamlit Cloud용

plt.rcParams["axes.unicode_minus"] = False

# 한국어 피처 매핑
feature_kor_map = {
    "marital_status": "결혼 여부",
    "no_verification": "검증 없음",
    "claim_year": "청구 연도",
    "address_change_ind": "주소 변경 여부",
    "witness_present_ind": "목격자 존재 여부",
    "high_education_ind": "고학력 여부",
    "accident_parking": "주차 사고 여부",
    "vehicle_price_per_driver_age": "차량 가격 대비 운전자 나이",
    "liab_prct_sq": "책임 비율 제곱",
    "age_over_safety": "안전 대비 나이",
    "age_of_driver": "운전자 나이",
    "age_of_driver_sq": "운전자 나이 제곱",
    "female_x_zip_fraud_rate_oof": "여성 X 지역 사기율",
    "zip_rate_x_accident_highway": "지역 사고율 X 고속도로 사고",
    "safety_risk_score": "안전 위험 점수",
    "base_points": "기본 점수",
    "total_score": "총점"
}


# -------------------------------------------------
# 페이지 설정
# -------------------------------------------------
st.set_page_config(
    page_title="Fraud Scorecard Dashboard",
    layout="wide"
)

st.title("🚗 Insurance Fraud Scorecard Dashboard")
st.caption("Claim ID를 입력하거나 선택하면 score gauge와 feature contribution waterfall을 확인할 수 있습니다.")

# -------------------------------------------------
# 데이터 로드
# -------------------------------------------------
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "Data"

    score_df = pd.read_csv(DATA_DIR / "train_scorecard_result.csv")
    partial_df = pd.read_csv(DATA_DIR / "train_partial_score_breakdown_long.csv")

    # claim_id 타입 통일
    score_df["claim_id"] = score_df["claim_id"].astype(str)
    partial_df["claim_id"] = partial_df["claim_id"].astype(str)

    return score_df, partial_df

score_df, partial_df = load_data()

# -------------------------------------------------
# 기본 유효성 검사
# -------------------------------------------------
required_score_cols = {"claim_id", "score"}
required_partial_cols = {"claim_id", "feature", "contribution"}

missing_score = required_score_cols - set(score_df.columns)
missing_partial = required_partial_cols - set(partial_df.columns)

if missing_score:
    st.error(f"train_scorecard_result.csv에 필요한 컬럼이 없습니다: {missing_score}")
    st.stop()

if missing_partial:
    st.error(f"train_partial_score_breakdown_long.csv에 필요한 컬럼이 없습니다: {missing_partial}")
    st.stop()

# -------------------------------------------------
# 사이드바 - claim 선택
# -------------------------------------------------
st.sidebar.header("🔎 Claim 선택")

claim_ids = sorted(score_df["claim_id"].unique().tolist())

search_claim = st.sidebar.text_input("Claim ID 입력", value=claim_ids[0] if claim_ids else "")

if search_claim and search_claim in claim_ids:
    selected_claim = search_claim
else:
    selected_claim = st.sidebar.selectbox("또는 목록에서 선택", claim_ids)

# -------------------------------------------------
# 선택 claim 데이터 추출
# -------------------------------------------------
row = score_df.loc[score_df["claim_id"] == selected_claim].iloc[0]
claim_partial = partial_df.loc[partial_df["claim_id"] == selected_claim].copy()

# base_points / total_score 제거
claim_partial = claim_partial[
    ~claim_partial["feature"].isin(["base_points", "total_score"])
].copy()

claim_partial["contribution"] = pd.to_numeric(claim_partial["contribution"], errors="coerce").fillna(0.0)

claim_partial["feature_kor"] = claim_partial["feature"].map(feature_kor_map).fillna(claim_partial["feature"])

# score 정보
score = float(row["score"])
pred_proba_bad = float(row["pred_proba_bad"]) if "pred_proba_bad" in row.index else np.nan
score_band = row["score_band"] if "score_band" in row.index else "N/A"
fraud_label = row["fraud"] if "fraud" in row.index else "N/A"

# -------------------------------------------------
# 위험 수준 계산
# -------------------------------------------------
def get_risk_level(score_value: float) -> str:
    if score_value < 450:
        return "High Risk"
    elif score_value < 500:
        return "Medium-High Risk"
    elif score_value < 550:
        return "Medium Risk"
    elif score_value < 600:
        return "Low Risk"
    else:
        return "Very Low Risk"

risk_level = get_risk_level(score)

# -------------------------------------------------
# Gauge Chart
# -------------------------------------------------
def draw_scorecard_gauge(
    score,
    min_score=400,
    max_score=650,
    bands=None,
    title="FRAUD CUSTOMER SCORE"
):
    if bands is None:
        bands = [
            (400, 450, "#b14632", "<450"),
            (450, 500, "#d0863a", "450-500"),
            (500, 550, "#dec349", "500-550"),
            (550, 600, "#94be55", "550-600"),
            (600, 650, "#5a9460", "600+"),
        ]

    score = float(np.clip(score, min_score, max_score))

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")

    center = (0, 0)
    radius = 1.0
    width = 0.42

    def score_to_angle(x):
        return 180 - (x - min_score) / (max_score - min_score) * 180

    # 반원 구간
    for start, end, color, label in bands:
        theta1 = score_to_angle(end)
        theta2 = score_to_angle(start)

        wedge = Wedge(
            center,
            radius,
            theta1,
            theta2,
            width=width,
            facecolor=color,
            edgecolor="white",
            linewidth=2.5
        )
        ax.add_patch(wedge)

        mid_score = (start + end) / 2
        ang = np.deg2rad(score_to_angle(mid_score))
        r_text = radius + 0.18
        x_text = r_text * np.cos(ang)
        y_text = r_text * np.sin(ang)

        ax.text(
            x_text, y_text, label,
            ha="center", va="center",
            fontsize=13, fontweight="bold"
        )

    # 바늘
    angle = np.deg2rad(score_to_angle(score))
    needle_len = 0.78
    needle_half_width = 0.03

    tip = np.array([needle_len * np.cos(angle), needle_len * np.sin(angle)])
    left = np.array([
        needle_half_width * np.cos(angle + np.pi / 2),
        needle_half_width * np.sin(angle + np.pi / 2)
    ])
    right = np.array([
        needle_half_width * np.cos(angle - np.pi / 2),
        needle_half_width * np.sin(angle - np.pi / 2)
    ])

    needle = Polygon([left, right, tip], closed=True, color="#222222")
    ax.add_patch(needle)
    ax.add_patch(Circle((0, 0), 0.09, color="#222222"))
    ax.add_patch(Circle((0, 0), 0.04, color="white"))

    # 텍스트
    ax.text(0, 1.28, title, ha="center", va="center", fontsize=28, fontweight="bold")
    ax.text(-0.86, -0.02, "HIGH RISK", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#b14632")
    ax.text(0.86, -0.02, "LOW RISK", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#5a9460")
    ax.text(0, -0.24, f"Score: {score:.1f}", ha="center", va="center",
            fontsize=22, fontweight="bold")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.35, 1.45)

    return fig

# -------------------------------------------------
# Waterfall Chart
# -------------------------------------------------
def draw_waterfall(df: pd.DataFrame, title: str = "Feature Contribution Waterfall"):
    # 0 아닌 값 중심으로 보기 좋게
    plot_df = df.copy()
    plot_df = plot_df.sort_values("contribution", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    cumulative = 0
    y_positions = np.arange(len(plot_df))

    for i, row_ in plot_df.iterrows():
        value = row_["contribution"]
        color = "#2e8b57" if value > 0 else "#c0392b" if value < 0 else "#b0b0b0"

        ax.barh(
            y=i,
            width=value,
            left=cumulative,
            color=color,
            edgecolor="white"
        )

        text_x = cumulative + value + (0.4 if value >= 0 else -0.4)
        ha = "left" if value >= 0 else "right"

        ax.text(
            text_x,
            i,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=10
        )

        cumulative += value

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["feature_kor"])
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Contribution")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig

# -------------------------------------------------
# KPI 영역
# -------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Claim ID", selected_claim)
k2.metric("Score", f"{score:.1f}")
k3.metric("Risk Level", risk_level)
k4.metric("Fraud Probability", f"{pred_proba_bad:.3f}" if not np.isnan(pred_proba_bad) else "N/A")

# -------------------------------------------------
# 메인 레이아웃
# -------------------------------------------------
left_col, right_col = st.columns([1.1, 1.2])

with left_col:
    st.subheader("Score Gauge")
    fig_gauge = draw_scorecard_gauge(score=score)
    st.pyplot(fig_gauge, use_container_width=True)

with right_col:
    st.subheader("Feature Contribution")
    if claim_partial.empty:
        st.warning("선택한 claim에 대한 contribution 데이터가 없습니다.")
    else:
        fig_waterfall = draw_waterfall(claim_partial, title=f"Waterfall for Claim {selected_claim}")
        st.pyplot(fig_waterfall, use_container_width=True)

# -------------------------------------------------
# 하단 상세 정보
# -------------------------------------------------
st.markdown("---")
detail_col1, detail_col2 = st.columns([1, 1])

with detail_col1:
    st.subheader("Claim Summary")
    summary_items = {
        "claim_id": selected_claim,
        "score": round(score, 3),
        "score_band": score_band,
        "risk_level": risk_level,
        "pred_proba_bad": round(pred_proba_bad, 6) if not np.isnan(pred_proba_bad) else "N/A",
        "fraud": fraud_label
    }
    st.dataframe(pd.DataFrame(summary_items.items(), columns=["Field", "Value"]), use_container_width=True)

with detail_col2:
    st.subheader("Top Contributions")
    if not claim_partial.empty:
        contrib_view = claim_partial.copy()
        contrib_view["abs_contribution"] = contrib_view["contribution"].abs()
        contrib_view = contrib_view.sort_values("abs_contribution", ascending=False).drop(columns="abs_contribution")
        st.dataframe(contrib_view, use_container_width=True)
    else:
        st.info("표시할 contribution 데이터가 없습니다.")