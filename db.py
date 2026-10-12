# -*- coding: utf-8 -*-
"""
로그인 + 분석 결과 누적 저장 (Supabase)
----------------------------------------
- 로그인: Supabase Auth(이메일+비밀번호). 계정은 Supabase 대시보드에서 관리자가 직접 만든다.
- 저장: 영상 자체는 저장하지 않고 분석 '결과 요약(점수·판정·지표)'만 표에 한 줄씩 쌓는다.
- Supabase 주소/키가 설정되어 있지 않으면 로그인·저장 기능이 모두 꺼지고 앱은 평소처럼 동작한다.

설정 방법 (둘 중 하나)
  - 로컬: 폴더에 .streamlit/secrets.toml 파일을 만들고 아래 두 줄을 적는다
        SUPABASE_URL = "https://xxxx.supabase.co"
        SUPABASE_KEY = "anon public 키"
  - Streamlit Cloud: 앱 Settings > Secrets 칸에 같은 두 줄을 붙여넣는다
"""
import pandas as pd
import requests
import streamlit as st

TABLE = "assessments"
TIMEOUT = 15


# ---------------- 설정 ----------------
def _config():
    try:
        url = st.secrets["SUPABASE_URL"].rstrip("/")
        key = st.secrets["SUPABASE_KEY"]
        return url, key
    except Exception:
        return None


def is_configured():
    return _config() is not None


# ---------------- 로그인 ----------------
def current_user():
    auth = st.session_state.get("auth")
    return auth["email"] if auth else None


def _token(key):
    auth = st.session_state.get("auth")
    return auth["token"] if auth else key


def _headers(key):
    return {
        "apikey": key,
        "Authorization": f"Bearer {_token(key)}",
        "Content-Type": "application/json",
    }


def login(email, password):
    """이메일/비밀번호로 로그인. (성공 여부, 메시지) 반환."""
    cfg = _config()
    if cfg is None:
        return False, "Supabase 설정 없음"
    url, key = cfg
    try:
        r = requests.post(
            f"{url}/auth/v1/token", params={"grant_type": "password"},
            json={"email": email, "password": password},
            headers={"apikey": key, "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            st.session_state["auth"] = {
                "token": data["access_token"],
                "email": data.get("user", {}).get("email", email),
            }
            return True, "ok"
        return False, "이메일 또는 비밀번호가 올바르지 않습니다."
    except Exception as e:
        return False, f"로그인 서버 연결 실패: {e}"


def logout():
    st.session_state.pop("auth", None)
    st.session_state.pop("saved_keys", None)


def require_login():
    """로그인돼 있으면 True. 아니면 로그인 폼을 그리고 False 반환 (호출한 쪽에서 st.stop())."""
    if current_user():
        with st.sidebar:
            st.caption(f"로그인: {current_user()}")
            if st.button("로그아웃"):
                logout()
                st.rerun()
        return True

    st.subheader("로그인")
    with st.form("login_form"):
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인")
    if submitted:
        ok, msg = login(email.strip(), password)
        if ok:
            st.rerun()
        else:
            st.error(msg)
    st.caption("계정은 관리자가 발급합니다. 계정이 필요하면 관리자에게 문의하세요.")
    return False


def _expired_check(r):
    """401이면 로그인 세션 만료로 보고 로그인 정보를 지운다."""
    if r.status_code == 401:
        logout()
        return True
    return False


# ---------------- 저장/조회 ----------------
def save_result(file_name, summary, sample_every):
    """분석 요약 1건을 표에 추가. (성공 여부, 메시지) 반환."""
    cfg = _config()
    if cfg is None:
        return False, "Supabase 설정 없음"
    url, key = cfg
    v = summary["values"]
    row = {
        "file_name": file_name,
        "verdict": summary["verdict"],
        "mean_score": float(v["mean_score"]),
        "p10_score": float(v["p10_score"]),
        "damage_ratio": float(v["damage_ratio"]),
        "max_run_span": float(v["max_run_span"]),
        "scratch_ratio": float(v["scratch_ratio"]),
        "hline_ratio": float(v["hline_ratio"]),
        "audio_missing": "audio_missing" in v,
        "audio_silence_ratio": float(v["audio_silence_ratio"]) if "audio_silence_ratio" in v else None,
        "audio_max_silence_run": float(v["audio_max_silence_run"]) if "audio_max_silence_run" in v else None,
        "audio_mean_dbfs": float(v["audio_mean_dbfs"]) if "audio_mean_dbfs" in v else None,
        "reasons": ", ".join(summary.get("reasons", [])),
        "sample_every": int(sample_every),
    }
    try:
        r = requests.post(f"{url}/rest/v1/{TABLE}", json=row,
                          headers={**_headers(key), "Prefer": "return=minimal"},
                          timeout=TIMEOUT)
        if r.status_code in (200, 201, 204):
            return True, "ok"
        if _expired_check(r):
            return False, "로그인이 만료되었습니다. 새로고침 후 다시 로그인해 주세요."
        return False, f"{r.status_code} {r.text[:200]}"
    except Exception as e:
        return False, str(e)


def load_results(limit=500):
    """누적된 기록을 최신순 DataFrame으로 반환. 실패 시 (None, 메시지)."""
    cfg = _config()
    if cfg is None:
        return None, "Supabase 설정 없음"
    url, key = cfg
    try:
        r = requests.get(f"{url}/rest/v1/{TABLE}",
                         params={"select": "*", "order": "created_at.desc", "limit": limit},
                         headers=_headers(key), timeout=TIMEOUT)
        if r.status_code != 200:
            if _expired_check(r):
                return None, "로그인이 만료되었습니다. 새로고침 후 다시 로그인해 주세요."
            return None, f"{r.status_code} {r.text[:200]}"
        df = pd.DataFrame(r.json())
        if not df.empty:
            df["created_at"] = (pd.to_datetime(df["created_at"], utc=True)
                                .dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M"))
        return df, "ok"
    except Exception as e:
        return None, str(e)
