# -*- coding: utf-8 -*-
"""
영상 상태평가 웹 데모 (Streamlit)
----------------------------------
condition_assessment.py의 분석 로직을 그대로 재사용해서,
브라우저에서 영상 파일을 업로드하면 상태점수와 그래프를 바로 보여준다.

실행 방법
  1) 같은 폴더에 condition_assessment.py 파일이 있어야 함
  2) 터미널에서: pip install streamlit
  3) 터미널에서: streamlit run app.py
  4) 자동으로 브라우저 창이 뜸 (안 뜨면 터미널에 나오는 http://localhost:8501 주소를 직접 열기)
"""
import os
import tempfile

import streamlit as st

import condition_assessment as ca
import db

st.set_page_config(page_title="영상 상태평가 데모", layout="wide")

st.title("시청각 자료 상태평가 데모")
st.caption("영상 파일을 올리면 선명도·노이즈·대비·세로줄(스크래치)을 분석해 훼손 의심도를 점수로 보여줍니다.")

# Supabase가 설정돼 있으면 로그인한 사람만 사용 가능 (설정 없으면 로그인 없이 동작)
if db.is_configured() and not db.require_login():
    st.stop()

with st.sidebar:
    st.header("분석 설정")
    sample_every = st.slider(
        "분석 간격 (프레임)", min_value=1, max_value=30, value=5,
        help="숫자가 클수록 듬성듬성 분석해서 빠르지만 정밀도는 떨어짐"
    )
    zthresh = st.slider(
        "이상치 민감도 (z-score)", min_value=1.5, max_value=4.0, value=2.5, step=0.1,
        help="낮출수록 더 많은 프레임을 이상치로 잡음"
    )
    damage_run = st.slider(
        "손상 vs 장면전환 기준(연속 프레임 수)", min_value=1, max_value=10, value=3,
        help="이 값 이하로 짧게 튀면 손상 후보, 길게 이어지면 장면전환 후보"
    )
    st.divider()
    st.caption("점수 기준값(고정 절대 기준)은 condition_assessment.py 상단의 "
               "SHARPNESS_RANGE / NOISE_RANGE / CONTRAST_RANGE에서 조정할 수 있습니다.")

uploaded = st.file_uploader(
    "영상 파일을 올려주세요 (mp4, wmv, avi, mov, mkv 등)",
    type=["mp4", "wmv", "avi", "mov", "mkv", "mts", "m4v", "mpg", "mpeg"],
)

if uploaded is not None:
    # cv2.VideoCapture는 실제 파일 경로가 필요해서, 업로드된 내용을 임시 파일로 저장
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.video(uploaded)

    with st.spinner("분석 중입니다... 영상 길이에 따라 시간이 걸릴 수 있어요."):
        audio_summary = None
        try:
            df = ca.analyze_video(
                tmp_path,
                sample_every=sample_every,
                outlier_z=zthresh,
                max_damage_run=damage_run,
            )
            # 오디오(음성 누락/무음/음량)도 같이 분석 — tmp 파일을 지우기 전에 해야 함
            audio_result = ca.analyze_audio(tmp_path)
            audio_summary = ca.summarize_audio(audio_result)
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            df = None
        finally:
            os.unlink(tmp_path)

    if df is not None:
        summary = ca.summarize_condition(df, sample_every=sample_every, audio_summary=audio_summary)
        v = summary["values"]

        # 분석 결과 요약을 Supabase에 누적 저장 (설정돼 있을 때만, 같은 파일·설정은 1번만)
        if db.is_configured():
            save_key = (uploaded.name, uploaded.size, sample_every, zthresh, damage_run)
            saved_keys = st.session_state.setdefault("saved_keys", set())
            if save_key not in saved_keys:
                ok, msg = db.save_result(uploaded.name, summary, sample_every)
                if ok:
                    saved_keys.add(save_key)
                    st.toast("누적 기록에 저장했습니다")
                else:
                    st.warning(f"누적 기록 저장 실패: {msg}")

        st.subheader("종합 상태점수")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("종합 상태점수 (0~100)", f"{v['mean_score']:.1f}")
        col2.metric("평균 선명도", f"{df['sharpness'].mean():.1f}")
        col3.metric("평균 노이즈", f"{df['noise'].mean():.2f}")
        col4.metric("평균 대비", f"{df['contrast'].mean():.1f}")

        st.subheader("다기준 판정")
        n_criteria = len(summary["grades"])
        st.caption(
            "평균 점수 하나만 보면 짧고 심한 손상이 전체 평균에 묻히는 '평균 희석' 오류가 생길 수 있어, "
            "QCTools(구간별 임계값 flag)·TLVQM(저품질 구간 가중)·MediaSCORE(다기준 결합 판정) 방식을 참고해 "
            f"아래 {n_criteria}개 기준 중 가장 나쁜 등급을 최종 판정으로 삼습니다. (세로줄/가로줄/음성은 "
            "2026-09-30 실제 소장자료 재검증에서 실제 훼손과 일치함이 확인되어 판정 기준에 포함됨)"
        )
        crit_info = [
            ("전체 평균 점수", v["mean_score"], summary["grades"]["mean_score"]),
            ("최저 10% 구간 평균", v["p10_score"], summary["grades"]["p10_score"]),
            ("손상 후보 비율(%)", v["damage_ratio"], summary["grades"]["damage_ratio"]),
            ("최장 손상 구간(프레임)", v["max_run_span"], summary["grades"]["max_run_span"]),
            ("세로줄/스크래치 비율(%)", v["scratch_ratio"], summary["grades"]["scratch_ratio"]),
            ("가로줄/수평선 비율(%)", v["hline_ratio"], summary["grades"]["hline_ratio"]),
        ]
        if "audio_missing" in v:
            crit_info.append(("음성 트랙", "누락", summary["grades"]["audio_missing"]))
        else:
            if "audio_silence_ratio" in v:
                crit_info.append(("무음 구간 비율(%)", v["audio_silence_ratio"], summary["grades"]["audio_silence_ratio"]))
            if "audio_max_silence_run" in v:
                crit_info.append(("최장 무음 지속(초)", v["audio_max_silence_run"], summary["grades"]["audio_max_silence_run"]))
            if "audio_mean_dbfs" in v:
                crit_info.append(("평균 음량(dBFS)", v["audio_mean_dbfs"], summary["grades"]["audio_mean_dbfs"]))

        crit_cols = st.columns(3)
        for i, (label, value, grade) in enumerate(crit_info):
            display_val = value if isinstance(value, str) else f"{value:.1f}"
            crit_cols[i % 3].metric(label, display_val, grade)
        if "audio_missing" not in v and "audio_silence_ratio" not in v:
            st.caption("※ 음성 분석 안 됨 — ffmpeg를 못 찾았습니다. `pip install imageio-ffmpeg` 후 다시 실행하면 포함됩니다.")

        verdict = summary["verdict"]
        reason_txt = f" (사유: {', '.join(summary['reasons'])})" if summary["reasons"] else ""
        if verdict == "위험":
            st.error(f"상태 나쁨 — 훼손 가능성이 높습니다{reason_txt}. 우선 복원 검토를 권장합니다.")
        elif verdict == "주의":
            st.warning(f"상태 주의 — 일부 기준에서 문제가 감지되었습니다{reason_txt}. 상세 확인을 권장합니다.")
        else:
            st.success(f"상태 양호 — {n_criteria}개 기준 모두에서 뚜렷한 훼손 신호가 적습니다.")

        n_damage = int((df["outlier_type"] == "손상 후보").sum())
        n_scene = int((df["outlier_type"] == "장면전환/동작 후보").sum())
        n_scratch = int(df["has_scratch"].sum())
        n_hline = int(df["has_hline"].sum())
        st.write(
            f"이상 프레임: 손상 후보 **{n_damage}개** · 장면전환/동작 후보 **{n_scene}개** · "
            f"세로줄/스크래치 의심 **{n_scratch}개** · 가로줄/수평선 의심 **{n_hline}개**"
        )

        st.subheader("훼손 의심 구간 한눈에 보기")
        st.caption(
            "배경색(빨강/노랑/초록)은 점수 구간, 빨간 띠는 손상 후보로 잡힌 구간, "
            "아래쪽 세로 눈금(｜)은 스크래치(보라)·가로줄(파랑) 감지 프레임입니다."
        )
        timeline_fig = ca.build_timeline_figure(df, sample_every=sample_every)
        st.pyplot(timeline_fig)

        st.subheader("프레임별 추이")
        chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs(
            ["종합 상태점수", "선명도", "노이즈", "스크래치(세로줄)", "가로줄(수평선)"]
        )
        with chart_tab1:
            st.line_chart(df.set_index("frame_idx")["condition_score"])
        with chart_tab2:
            st.line_chart(df.set_index("frame_idx")["sharpness"])
        with chart_tab3:
            st.line_chart(df.set_index("frame_idx")["noise"])
        with chart_tab4:
            st.line_chart(df.set_index("frame_idx")["scratch_cols"])
            st.caption(
                "프레임당 의심 세로줄 컬럼 개수 (최근 여러 프레임에 걸쳐 같은 자리에서 "
                f"{ca.SCRATCH_PERSIST_MIN_HITS}번 이상 반복돼야 '스크래치 있음'으로 확정)"
            )
        with chart_tab5:
            st.line_chart(df.set_index("frame_idx")["hline_rows"])
            st.caption(
                "프레임당 의심 가로줄 행 개수 (세로줄과 같은 방식으로 최근 여러 프레임에 걸쳐 "
                f"같은 자리에서 {ca.SCRATCH_PERSIST_MIN_HITS}번 이상 반복돼야 '가로줄 있음'으로 확정)"
            )

        if n_damage > 0:
            st.subheader("손상 후보 프레임 목록")
            st.write(df.loc[df["outlier_type"] == "손상 후보", "frame_idx"].tolist())

        with st.expander("프레임별 상세 데이터 보기 (CSV 원본)"):
            st.dataframe(df)
            st.download_button(
                "CSV로 다운로드",
                df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{os.path.splitext(uploaded.name)[0]}_frame_metrics.csv",
            )
else:
    st.info("위에서 영상 파일을 올리면 분석이 시작됩니다.")

# ---- 누적 분석 기록 (Supabase 설정 시에만 표시) ----
if db.is_configured():
    st.divider()
    st.subheader("누적 분석 기록")
    hist, hist_msg = db.load_results()
    if hist is None:
        st.warning(f"기록을 불러오지 못했습니다: {hist_msg}")
    elif hist.empty:
        st.caption("아직 저장된 기록이 없습니다. 영상을 분석하면 여기에 쌓입니다.")
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric("누적 분석 건수", len(hist))
        h2.metric("위험 판정", int((hist["verdict"] == "위험").sum()))
        h3.metric("평균 상태점수", f"{hist['mean_score'].mean():.1f}")
        show_cols = ["created_at", "user_email", "file_name", "verdict", "mean_score", "p10_score",
                     "damage_ratio", "scratch_ratio", "hline_ratio",
                     "audio_silence_ratio", "audio_mean_dbfs", "reasons"]
        show_cols = [c for c in show_cols if c in hist.columns]
        st.dataframe(hist[show_cols], use_container_width=True)
        st.download_button(
            "누적 기록 CSV 다운로드",
            hist.to_csv(index=False).encode("utf-8-sig"),
            file_name="assessment_history.csv",
        )
