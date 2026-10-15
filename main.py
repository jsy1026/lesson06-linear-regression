import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ 기온 예측기")
st.caption(
    "서울의 과거 연평균기온을 분석하고 "
    "선형 회귀로 연도별 예상 기온을 계산합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
CUTOFF_YEAR = 2025
MIN_OBSERVATIONS = 300


# --------------------------------------------------
# 데이터 읽기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {', '.join(sorted(missing))}"
        )

    # 날짜와 기온을 올바른 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행은 분석에서 제외
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 기준 기간 이후 데이터 제외
    df = df[df["날짜"].dt.year <= CUTOFF_YEAR].copy()

    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일 수와 평균기온 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도 제외
    yearly = yearly[
        yearly["관측일수"] >= MIN_OBSERVATIONS
    ].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 앱 실행
# --------------------------------------------------
try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.info("인터넷 연결과 CSV 주소, 파일의 열 이름을 확인해 주세요.")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 하려면 조건을 만족하는 연도가 2개 이상 필요합니다.")
    st.stop()

# 선형 회귀: 기온 = 기울기 * 연도 + 절편
years = yearly["연도"].to_numpy(dtype=float)
temperatures = yearly["평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(years, temperatures, 1)

# 피어슨 상관계수
correlation = np.corrcoef(years, temperatures)[0, 1]

start_year = int(years.min())
end_year = int(years.max())
year_count = len(yearly)


def predict_temperature(year):
    """학습한 선형 회귀식으로 연평균기온을 추정합니다."""
    return slope * year + intercept


# --------------------------------------------------
# 핵심 통계
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("회귀 분석 연도 수", f"{year_count:,}년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수", f"{correlation:.4f}")

st.caption(
    f"분석 조건: 1900년 이후 표시 여부와 관계없이 "
    f"기준 연도 {CUTOFF_YEAR}년까지의 자료 중 "
    f"관측일 수가 {MIN_OBSERVATIONS}일 이상인 연도만 사용합니다."
)

# --------------------------------------------------
# 연도 선택 및 예상 기온
# --------------------------------------------------
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

predicted_temperature = predict_temperature(selected_year)

st.subheader(f"📅 {selected_year}년 예상 연평균기온")

st.markdown(
    f"""
    <div style="
        padding: 24px;
        border-radius: 16px;
        background-color: rgba(100, 149, 237, 0.12);
        text-align: center;
        margin-bottom: 16px;
    ">
        <div style="font-size: 16px; opacity: 0.8;">
            선형 회귀 예측값
        </div>
        <div style="font-size: 48px; font-weight: 700;">
            {predicted_temperature:.2f} °C
        </div>
        <div style="font-size: 14px; opacity: 0.75;">
            과거 연평균기온의 선형 추세를 연장한 추정치
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Plotly 산점도 + 회귀 직선
# --------------------------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(
            size=7,
            color="#3182CE",
            opacity=0.8,
        ),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

# 학습 구간에 맞춘 회귀 직선
line_years = np.array([start_year, end_year], dtype=float)
line_temperatures = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(color="#E53E3E", width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 슬라이더에서 선택한 연도의 예측 위치
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(
            size=13,
            color="#DD6B20",
            symbol="diamond",
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            "예측 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    template="plotly_white",
    height=560,
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=20, r=20, t=50, b=20),
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(128,128,128,0.15)",
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(128,128,128,0.15)",
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 회귀식 및 데이터 표
# --------------------------------------------------
st.subheader("🧮 회귀 분석 결과")

st.latex(
    rf"\hat{{y}} = {slope:.6f}x {intercept:+.3f}"
)

st.write(
    f"- 회귀 분석에 사용한 연도 수: **{year_count:,}개**"
)
st.write(f"- 시작 연도: **{start_year}년**")
st.write(f"- 끝 연도: **{end_year}년**")
st.write(f"- 피어슨 상관계수: **{correlation:.4f}**")
st.write(f"- 연도당 회귀 기울기: **{slope:.4f} °C/년**")

if correlation > 0:
    st.write("연도와 연평균기온 사이에 양의 선형 상관관계가 있습니다.")
elif correlation < 0:
    st.write("연도와 연평균기온 사이에 음의 선형 상관관계가 있습니다.")
else:
    st.write("연도와 연평균기온 사이의 선형 상관관계가 없습니다.")

with st.expander("분석에 사용한 연도별 데이터 보기"):
    display_data = yearly.copy()
    display_data["평균기온"] = display_data["평균기온"].round(2)
    st.dataframe(display_data, use_container_width=True, hide_index=True)

st.caption(
    "주의: 상관관계는 인과관계를 의미하지 않습니다. "
    "이 모델은 연도와 연평균기온 사이의 직선 추세만 학습하므로, "
    "1900년 이전이나 2025년 이후의 예측값 및 장기 전망에는 "
    "큰 불확실성이 있을 수 있습니다."
)
