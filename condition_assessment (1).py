"""
영상 상태평가 툴 (Condition Assessment Tool)
----------------------------------------------
훼손 정도를 육안이 아닌 객관적 수치로 평가하기 위한 스크립트.

측정 지표
  1) 선명도 (Sharpness)      - Laplacian variance. 낮을수록 흐릿함/초점 손실
  2) 노이즈 (Noise level)    - 원본과 미디언 필터 결과의 차이(잔차) 표준편차. 높을수록 노이즈 많음
  3) 밝기 (Brightness)       - 평균 밝기. 극단값은 노출/열화 문제 가능성
  4) 대비 (Contrast)         - 밝기의 표준편차. 낮을수록 색바램/저대비 의심
  5) 채도 (Saturation)       - 색바램(color fading) 여부 참고 지표
  6) 프레임 간 이상치         - 인접 프레임과의 차이가 급격히 튀는 지점
                               (순간적 결손·드롭아웃 후보)
  7) 세로 스크래치/줄         - 한 프레임 안에서 좌우 이웃 컬럼과 밝기가 유독 다른
                               좁은 세로줄 개수. frame_diff로는 안 잡히는, 여러
                               프레임에 걸쳐 같은 자리에 계속 있는 손상을 잡기 위함
  8) 가로줄(수평선)           - 7)과 같은 방식을 행(row) 방향에 적용. 세로줄 지표만으로는
                               못 잡는 가로 방향 줄무늬 손상을 잡기 위함(2026-09-30 추가)
  9) 음성(오디오) 상태        - 무음 구간 비율·최장 무음 지속시간·평균 음량(dBFS)·음성 트랙
                               누락 여부. 화면(영상) 지표만으로는 전혀 안 잡히는 오디오 손상을
                               같은 다기준 판정에 포함하기 위함(2026-09-30 추가). ffmpeg 실행
                               파일이 필요하며, imageio-ffmpeg 패키지를 설치하면 별도로 ffmpeg를
                               설치하지 않아도 자동으로 내장된 실행 파일을 사용한다
                               (pip install imageio-ffmpeg).

이상치 세분화 (손상 후보 vs 장면전환/동작 후보)
  프레임 간 차이가 튀는 원인은 실제 손상(스크래치·드롭아웃)뿐 아니라
  컷 전환·카메라 패닝·빠른 동작에서도 똑같이 발생한다. 이를 구분하기 위해
  "연속으로 이상치가 몇 프레임 이어지는가"를 기준으로 나눈다:
    - 연속 길이가 짧음(기본 3프레임 이하) -> "손상 후보"
      (순간적으로 튀었다가 곧 원래대로 돌아옴 = 드롭아웃/스크래치 패턴에 가까움)
    - 연속 길이가 김(기본 4프레임 이상)   -> "장면전환/동작 후보"
      (여러 프레임에 걸쳐 계속 달라짐 = 컷 전환이나 패닝/동작일 가능성이 높음)
  이 기준(--damage-run)은 경험적 휴리스틱이며, 실제 소장 자료를 여러 건
  돌려보면서 팀 내에서 임계값을 보정하는 걸 권장한다. 최종 판단은
  diff_check.py 같은 시각화 도구로 직접 눈으로 확인하는 걸 권장한다.

출력
  - frame_metrics.csv : 프레임별 원시 수치 (outlier_type 포함)
  - condition_report.png : 지표별 추이 그래프 (손상 후보/장면전환 후보 구분 표시)
  - 콘솔 요약 : 전체 평균/이상 프레임 목록(유형별) + 0~100 종합 점수(상대 지표)

주의
  종합 상태점수(condition_score)는 고정된 절대 기준값(SHARPNESS_RANGE 등)을 기준으로
  0~100점을 매기므로, 같은 영상 안에서뿐 아니라 서로 다른 영상끼리도 점수를 비교할 수
  있습니다. (이전 버전은 영상 자기 자신의 최소/최대값으로 정규화해서, 영상 전체가 고르게
  나쁘면 오히려 100점이 나오고 여러 영상 간 비교가 불가능한 문제가 있었습니다 — 수정됨.)
  다만 기준값 자체는 일반적인 SD급 방송 아카이브 영상을 가정한 경험적 값이므로, 실제
  소장 자료를 여러 건 돌려보면서 팀 내 기준으로 보정하는 걸 권장합니다
  (--sharpness-range, --noise-range, --contrast-range 옵션으로 조정 가능).
"""

import argparse
import os
import subprocess
from collections import deque
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장 (서버/터미널 환경 대응)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- 한글 폰트 설정 (Windows 기준 맑은 고딕, 없으면 기본 폰트로 대체) ----
for candidate in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if candidate in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams["axes.unicode_minus"] = False


def sharpness_score(gray):
    """Laplacian variance: 값이 낮을수록 흐릿함(초점/디테일 손실)"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def noise_score(gray):
    """미디언 필터 전후 잔차의 표준편차: 값이 높을수록 노이즈 많음"""
    denoised = cv2.medianBlur(gray, 5)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)
    return float(np.std(residual))


def brightness_contrast(gray):
    return float(np.mean(gray)), float(np.std(gray))


def saturation_score(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def frame_diff_score(prev_gray, gray):
    """인접 프레임과의 절대 차이 평균: 급격히 튀면 스크래치/드롭아웃 후보"""
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))


# ---- 세로 스크래치 탐지 설정 ----
# 1차 검증에서, 순수 랜덤 노이즈만 있는 영상에서도 통계적 임계값(평균+3표준편차)만으로는
# 프레임마다 몇 개 컬럼이 우연히 '이상치'로 잡혀 오탐(false positive)이 심하게 나는 문제가
# 발견되었다. 이를 막기 위해 두 가지 조건을 함께 건다:
#   1) 밝기 편차 절댓값이 SCRATCH_ABS_MIN 이상이어야 함(우연한 미세한 노이즈 배제)
#   2) 같은 x좌표(±허용범위) 부근이 최근 여러 분석 프레임에 걸쳐 반복 확인돼야 함
#      (실제 스크래치는 여러 프레임에 계속 같은 자리에 있고, 노이즈로 인한 우연한 튐은
#      프레임마다 다른 위치에서 무작위로 발생하므로 반복되지 않는다)
SCRATCH_ABS_MIN = 12.0     # 편차 절댓값 최소 기준(0~255 스케일)
SCRATCH_Z = 4.0            # 통계적 이상치 기준(표준편차 배수)
SCRATCH_PERSIST_WINDOW = 5     # 최근 몇 개 분석 프레임을 살펴볼지
SCRATCH_PERSIST_MIN_HITS = 3   # 그중 몇 번 이상 비슷한 위치에서 잡혀야 '진짜 스크래치'로 볼지
SCRATCH_COL_TOLERANCE = 2      # 같은 스크래치로 볼 x좌표 오차 허용범위(픽셀)
SCRATCH_STATIC_FRAME_DIFF_MIN = 1.0
# 화면이 거의 안 움직이는 프레임(경고문 타이틀 카드 등 완전 정지 화면)은 프레임 전체가
# 그대로이기 때문에 글자·그래픽 테두리 같은 콘텐츠 경계선도 "여러 프레임에 걸쳐 같은
# 자리에 계속 있는" 것처럼 보여 스크래치로 오탐된다. frame_diff(전체 화면 움직임)가 이
# 값보다 작으면 그 프레임은 스크래치 후보 계산에서 아예 제외한다.


def scratch_columns(gray, abs_min=SCRATCH_ABS_MIN, z_thresh=SCRATCH_Z):
    """한 프레임 안에서 좌우 이웃 컬럼들과 비교했을 때 유독 밝기가 다른(=세로줄 의심)
    컬럼들의 x좌표 배열을 반환한다. 통계적 이상치(z_thresh) & 절대 편차(abs_min)를
    모두 만족해야 후보로 잡는다."""
    col_mean = gray.mean(axis=0).astype(np.float64)
    if col_mean.shape[0] < 15:
        return np.array([], dtype=int)
    kernel = np.ones(11) / 11.0
    # 가장자리 컬럼에서 numpy convolve가 암묵적으로 0을 채워 넣어 생기는
    # 가짜 편차(false positive)를 막기 위해 reflect 패딩 후 'valid'로 계산
    padded = np.pad(col_mean, (5, 5), mode="reflect")
    local_avg = np.convolve(padded, kernel, mode="valid")
    deviation = np.abs(col_mean - local_avg)
    std = deviation.std()
    if std < 1e-6:
        return np.array([], dtype=int)
    threshold = max(deviation.mean() + z_thresh * std, abs_min)
    return np.where(deviation > threshold)[0]


def scratch_score(gray):
    """scratch_columns()로 잡힌 후보 컬럼의 개수(프레임 단위 참고용 지표, 그래프 표시용).
    실제 '스크래치 있음' 판정(has_scratch)은 analyze_video 안에서 여러 프레임에 걸친
    지속성까지 확인한 뒤 결정한다 — 이 함수 값만으로는 오탐이 섞여 있을 수 있다."""
    return int(len(scratch_columns(gray)))


# ---- 가로줄(수평선) 탐지 ----
# 실제 소장자료 검증(2026-09-30) 중, 세로줄 탐지기는 정상 범위인데 가로 방향으로 심한
# 줄무늬가 있는 영상이 오히려 "상태 양호"로 나오는 사례가 확인됐다. 원인은 scratch_columns가
# 컬럼(세로) 방향으로만 편차를 보기 때문에 가로 방향 손상은 애초에 측정 대상이 아니었던 것.
# 같은 방식(로컬 평균 대비 편차 + 절대 하한 + 여러 프레임에 걸친 지속성 확인)을 행(row)
# 방향에도 그대로 적용해서 가로줄을 잡는다.
def line_rows(gray, abs_min=SCRATCH_ABS_MIN, z_thresh=SCRATCH_Z):
    """한 프레임 안에서 위아래 이웃 행들과 비교했을 때 유독 밝기가 다른(=가로줄 의심)
    행들의 y좌표 배열을 반환한다. scratch_columns()와 완전히 같은 방식을 행 방향에 적용."""
    row_mean = gray.mean(axis=1).astype(np.float64)
    if row_mean.shape[0] < 15:
        return np.array([], dtype=int)
    kernel = np.ones(11) / 11.0
    padded = np.pad(row_mean, (5, 5), mode="reflect")
    local_avg = np.convolve(padded, kernel, mode="valid")
    deviation = np.abs(row_mean - local_avg)
    std = deviation.std()
    if std < 1e-6:
        return np.array([], dtype=int)
    threshold = max(deviation.mean() + z_thresh * std, abs_min)
    return np.where(deviation > threshold)[0]


def line_score(gray):
    """line_rows()로 잡힌 후보 행의 개수(프레임 단위 참고용 지표). has_hline과 마찬가지로
    실제 판정은 analyze_video 안에서 지속성까지 확인한 뒤 결정한다."""
    return int(len(line_rows(gray)))


# ---- 음성(오디오) 상태 분석 ----
# 실제 소장자료 검증 중, 화면(영상) 지표만으로는 음성 누락·무음·음량 저하 같은 오디오
# 손상을 전혀 잡지 못한다는 지적(2026-09-30)에 따라 추가. cv2.VideoCapture는 오디오를
# 읽지 못하므로, ffmpeg로 오디오 트랙만 PCM으로 뽑아서 별도로 분석한다. ffmpeg를 시스템에
# 따로 설치할 필요 없이 `pip install imageio-ffmpeg`만 하면 그 패키지가 내려받아 둔 ffmpeg
# 실행 파일을 자동으로 찾아 쓴다 — 패키지가 없으면 오디오 분석은 건너뛰고 기존 영상 지표만으로
# 판정한다(하위 호환).
AUDIO_SILENCE_DBFS = -45.0        # 이 dBFS보다 작으면(더 조용하면) 그 구간은 '무음'으로 본다
AUDIO_SILENCE_MIN_RUN_SEC = 2.0   # 무음이 이 시간 이상 이어져야 의미 있는 '무음 구간'으로 취급
                                   # (지금은 참고용 상수 — 실제 최장 구간 계산은 summarize_audio에서 함)


def _get_ffmpeg_exe():
    """imageio-ffmpeg가 자동으로 받아둔 ffmpeg 실행 파일 경로를 반환. 패키지가 없으면 None."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def analyze_audio(video_path, window_sec=0.5, sample_rate=16000):
    """영상 파일에서 오디오 트랙을 추출해 구간별(window_sec) 음량(dBFS)을 계산한다.
    반환값 3가지 경우:
      - DataFrame          : 정상적으로 오디오를 추출·분석함
      - "no_audio" (문자열) : ffmpeg는 실행됐지만 오디오 트랙이 없거나 추출된 음성이 없음
                              (음성 누락으로 간주)
      - None                : ffmpeg 실행 파일 자체를 못 찾아 분석 불가(imageio-ffmpeg 미설치 등)
    """
    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        return None

    cmd = [
        ffmpeg_exe, "-v", "error", "-i", video_path,
        "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None

    raw = proc.stdout
    if not raw or len(raw) < 2:
        return "no_audio"

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = max(1, int(window_sec * sample_rate))
    n_windows = max(1, len(audio) // win)

    rows = []
    for i in range(n_windows):
        seg = audio[i * win:(i + 1) * win]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        dbfs = 20.0 * np.log10(rms) if rms > 1e-9 else -100.0
        rows.append({"window_idx": i, "time_sec": i * window_sec, "rms": rms, "dbfs": dbfs})

    if not rows:
        return "no_audio"
    return pd.DataFrame(rows)


def summarize_audio(audio_result, window_sec=0.5):
    """analyze_audio()의 반환값을 받아 무음 비율/최장 무음 지속시간/평균 음량을 계산한다.
    audio_result가 DataFrame이 아니면("no_audio" 또는 None) 그대로 되돌려 보낸다."""
    if not isinstance(audio_result, pd.DataFrame):
        return audio_result

    is_silent = audio_result["dbfs"] < AUDIO_SILENCE_DBFS
    silence_ratio = float(is_silent.mean() * 100.0)

    max_run = 0
    run = 0
    for s in is_silent:
        if s:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    max_silence_run_sec = max_run * window_sec

    # 완전 무음 구간(-100dBFS, log(0) 대체값)은 평균 음량 계산에서 제외해서
    # '전체가 무음이라 평균도 매우 낮다'는 왜곡을 줄인다(무음 여부는 별도 지표로 이미 잡힘).
    audible = audio_result.loc[~is_silent, "dbfs"]
    mean_dbfs = float(audible.mean()) if len(audible) else -100.0

    return {
        "silence_ratio": silence_ratio,
        "max_silence_run_sec": max_silence_run_sec,
        "mean_dbfs": mean_dbfs,
    }


# ---- 고정 절대 기준값 (여러 영상을 서로 비교할 수 있도록 영상마다 다시 계산하지 않음) ----
# 일반적인 SD급 방송 아카이브 영상을 기준으로 잡은 경험적 값입니다.
# 실제 소장 자료를 더 돌려보면서 팀 내 기준으로 보정하는 걸 권장합니다(--sharpness-range 등으로 조정 가능).
SHARPNESS_RANGE = (20.0, 400.0)   # Laplacian 분산: 이 아래면 심하게 흐림, 이 위면 충분히 선명
NOISE_RANGE = (2.0, 25.0)        # 잔차 표준편차: 이 아래면 깨끗함, 이 위면 노이즈 심함
CONTRAST_RANGE = (15.0, 70.0)    # 밝기 표준편차: 이 아래면 저대비(색바램 의심), 이 위면 정상 대비


def normalize(series, lo, hi, invert=False):
    """
    0~100 스케일로 정규화. 이전 버전은 영상 자기 자신의 min/max로 정규화했는데,
    그러면 영상 전체가 고르게 나쁘면(변화가 없으면) 오히려 100점이 나오는 문제가 있었고,
    여러 영상 간 점수를 비교할 수도 없었다. 그래서 lo/hi를 고정된 절대 기준값으로 받아
    영상 간에도 비교 가능한 절대 스케일로 바꿨다.
    invert=True면 값이 낮을수록 좋은 지표(예: 노이즈)를 뒤집는다.
    """
    s = series.copy()
    norm = (s - lo) / (hi - lo) * 100.0
    norm = norm.clip(lower=0.0, upper=100.0)
    return (100 - norm) if invert else norm


def _damage_run_spans(df, sample_every):
    """'손상 후보'로 표시된 프레임들을 연속 구간(run)으로 묶어 각 구간의
    프레임 스팬(마지막-첫 frame_idx)을 리스트로 반환한다. 평균 하나로는
    보이지 않는, 짧지만 국소적인 손상 구간의 '길이'를 별도로 보기 위함."""
    idxs = df.index[df["outlier_type"] == "손상 후보"].tolist()
    if not idxs:
        return []
    spans = []
    run = [idxs[0]]
    for cur in idxs[1:]:
        prev_frame_idx = df.loc[run[-1], "frame_idx"]
        cur_frame_idx = df.loc[cur, "frame_idx"]
        if cur_frame_idx - prev_frame_idx <= sample_every:
            run.append(cur)
        else:
            spans.append(df.loc[run[-1], "frame_idx"] - df.loc[run[0], "frame_idx"])
            run = [cur]
    spans.append(df.loc[run[-1], "frame_idx"] - df.loc[run[0], "frame_idx"])
    return spans


# ---- 종합 판정 기준 (선행연구 참고) ----
# 평균 하나만 보면 "평균 희석형" 오류(짧고 심한 손상이 전체 평균에 묻힘)가 생기므로,
# 여러 기준을 함께 보고 그중 가장 나쁜 등급을 최종 판정으로 삼는다(QCTools의
# "구간 하나라도 임계값을 벗어나면 flag" 방식 + TLVQM의 저품질 구간 가중(worst-case
# pooling) 개념 + MediaSCORE의 다기준 결합 판정 방식을 참고함).
#   - mean_score   : 영상 전체 평균 상태점수
#   - p10_score    : 하위 10% 구간 평균(전체는 괜찮아도 일부 구간이 심하게 나쁘면 낮아짐)
#   - damage_ratio  : 손상 후보(frame_diff 급변)로 잡힌 프레임의 비율(%)
#   - max_run_span  : 가장 길게 이어진 손상 구간의 프레임 스팬(국소 손상의 지속 정도)
#   - scratch_ratio : 세로 스크래치/줄 의심 프레임의 비율(%) — frame_diff로는 안 잡히는
#                     "여러 프레임에 걸쳐 같은 자리에 계속 있는" 손상(스크래치·먼지줄)을 잡기 위한 지표
#
# scratch_ratio는 실제 소장자료 검증(2026-09-29)에서, 화면이 거의 안 움직이는 구간
# (경고문 타이틀 카드처럼 완전히 정지된 화면)에서 콘텐츠 자체의 경계선(글자·그래픽 테두리)을
# 스크래치로 오인하는 오류가 다수 확인되었다. 프레임 자체의 움직임(frame_diff)이 거의 없는
# 구간은 스크래치 판정에서 제외하도록 보강했다(SCRATCH_STATIC_FRAME_DIFF_MIN). 이후 실제
# 소장자료 재검증(2026-09-30)에서 세로줄(95%)·가로줄(75%) 판정이 모두 실제 훼손과 일치하는
# 것으로 육안 확인되어, scratch_ratio·hline_ratio 둘 다 최종 판정(verdict) 기준에 포함시켰다.
# 다만 여전히 경험적 임계값이므로, 자료를 더 돌려보면서 오탐 사례가 나오면
# CRITERIA_THRESHOLDS의 good/warn 값을 조정하거나 필요하면 다시 참고 지표로 되돌리는 걸 권장한다.
# audio_* 3개는 오디오 분석이 가능했을 때만(analyze_audio가 DataFrame을 반환했을 때만)
# values/grades에 추가되고, audio_missing은 오디오 트랙 자체가 없을 때만 추가된다. 즉
# VERDICT_CRITERIA에 이름이 있어도 실제로 그 영상에서 계산되지 않았으면 판정에서 자동으로
# 빠진다(summarize_condition의 verdict_grades는 grades에 실제로 있는 키만 본다).
VERDICT_CRITERIA = ["mean_score", "p10_score", "damage_ratio", "max_run_span",
                     "scratch_ratio", "hline_ratio",
                     "audio_silence_ratio", "audio_max_silence_run",
                     "audio_mean_dbfs", "audio_missing"]
CRITERIA_THRESHOLDS = {
    "mean_score":    {"good": 65, "warn": 40, "higher_is_better": True},
    "p10_score":     {"good": 50, "warn": 30, "higher_is_better": True},
    "damage_ratio":  {"good": 3,  "warn": 10, "higher_is_better": False},
    "max_run_span":  {"good": 5,  "warn": 15, "higher_is_better": False},
    "scratch_ratio": {"good": 2,  "warn": 10, "higher_is_better": False},
    "hline_ratio":   {"good": 2,  "warn": 10, "higher_is_better": False},
    "audio_silence_ratio":   {"good": 5,   "warn": 20,  "higher_is_better": False},
    "audio_max_silence_run": {"good": 3,   "warn": 10,  "higher_is_better": False},  # 초
    "audio_mean_dbfs":       {"good": -20, "warn": -35, "higher_is_better": True},   # dBFS(0에 가까울수록 큰 소리)
}
CRITERIA_LABELS = {
    "mean_score": "전체 평균 상태점수",
    "p10_score": "최저 10% 구간 평균 점수",
    "damage_ratio": "손상 후보 프레임 비율(%)",
    "max_run_span": "최장 손상 구간 길이(프레임)",
    "scratch_ratio": "세로줄/스크래치 의심 비율(%)",
    "hline_ratio": "가로줄/수평선 의심 비율(%)",
    "audio_silence_ratio": "무음 구간 비율(%)",
    "audio_max_silence_run": "최장 무음 지속시간(초)",
    "audio_mean_dbfs": "평균 음량(dBFS)",
    "audio_missing": "음성 트랙 누락 여부",
}


def _grade(value, cfg):
    good, warn, higher = cfg["good"], cfg["warn"], cfg["higher_is_better"]
    if higher:
        if value >= good:
            return "양호"
        return "주의" if value >= warn else "위험"
    else:
        if value <= good:
            return "양호"
        return "주의" if value <= warn else "위험"


def summarize_condition(df, sample_every=1, audio_summary=None):
    """여러 기준을 함께 계산해서 딕셔너리로 반환한다.
    verdict는 VERDICT_CRITERIA에 담긴 기준 중, 이 영상에서 실제로 계산된 것들만 놓고
    가장 나쁜 등급(보수적 판정)을 고른다. reasons에는 그 등급을 유발한 기준명이 담긴다.

    audio_summary: summarize_audio()의 반환값(선택).
      - dict            : 오디오 정상 분석됨 -> audio_silence_ratio/audio_max_silence_run/
                          audio_mean_dbfs를 values/grades에 추가하고 판정에 반영
      - "no_audio"      : 음성 트랙 없음/추출 실패 -> audio_missing을 '위험'으로 추가
      - None(기본값)     : 오디오 분석을 안 했거나 못 함(ffmpeg 없음 등) -> 오디오 관련 지표
                          없이 기존 영상 지표만으로 판정(하위 호환)
    """
    total = len(df)
    n_damage = int((df["outlier_type"] == "손상 후보").sum())
    spans = _damage_run_spans(df, sample_every)
    n_scratch = int(df["has_scratch"].sum()) if "has_scratch" in df.columns else 0
    n_hline = int(df["has_hline"].sum()) if "has_hline" in df.columns else 0

    values = {
        "mean_score": float(df["condition_score"].mean()),
        "p10_score": float(df["condition_score"].quantile(0.10)),
        "damage_ratio": (n_damage / total * 100.0) if total else 0.0,
        "max_run_span": float(max(spans)) if spans else 0.0,
        "scratch_ratio": (n_scratch / total * 100.0) if total else 0.0,
        "hline_ratio": (n_hline / total * 100.0) if total else 0.0,
    }
    grades = {k: _grade(v, CRITERIA_THRESHOLDS[k]) for k, v in values.items()}

    if audio_summary == "no_audio":
        values["audio_missing"] = 100.0
        grades["audio_missing"] = "위험"
    elif isinstance(audio_summary, dict):
        values["audio_silence_ratio"] = audio_summary["silence_ratio"]
        values["audio_max_silence_run"] = audio_summary["max_silence_run_sec"]
        values["audio_mean_dbfs"] = audio_summary["mean_dbfs"]
        for k in ("audio_silence_ratio", "audio_max_silence_run", "audio_mean_dbfs"):
            grades[k] = _grade(values[k], CRITERIA_THRESHOLDS[k])
    # audio_summary가 None이면(분석 안 함/불가) 오디오 관련 키를 아예 추가하지 않는다.

    order = {"양호": 0, "주의": 1, "위험": 2}
    # 최종 판정(verdict)은 VERDICT_CRITERIA 중 이 영상에서 실제로 계산된 기준만 놓고 정한다.
    verdict_grades = {k: g for k, g in grades.items() if k in VERDICT_CRITERIA}
    verdict = max(verdict_grades.values(), key=lambda g: order[g])
    reasons = [CRITERIA_LABELS[k] for k, g in verdict_grades.items() if g == verdict and verdict != "양호"]

    return {
        "values": values,
        "grades": grades,
        "verdict": verdict,
        "reasons": reasons,
    }


def classify_outlier_runs(df, sample_every, max_damage_run=3):
    """
    연속된 이상치 프레임을 하나의 '런(run)'으로 묶어서,
    런 길이가 짧으면 '손상 후보', 길면 '장면전환/동작 후보'로 분류.
    sample_every로 샘플링했을 때 프레임 번호 간격이 sample_every인 것끼리
    같은 런으로 취급한다.
    """
    outlier_type = pd.Series([""] * len(df), index=df.index, dtype=object)
    df = df.reset_index(drop=True)

    idxs = df.index[df["is_outlier"]].tolist()
    if not idxs:
        return outlier_type.values

    runs = []
    run = [idxs[0]]
    for cur in idxs[1:]:
        prev_frame_idx = df.loc[run[-1], "frame_idx"]
        cur_frame_idx = df.loc[cur, "frame_idx"]
        if cur_frame_idx - prev_frame_idx <= sample_every:
            run.append(cur)
        else:
            runs.append(run)
            run = [cur]
    runs.append(run)

    result = [""] * len(df)
    for run in runs:
        label = "손상 후보" if len(run) <= max_damage_run else "장면전환/동작 후보"
        for i in run:
            result[i] = label

    return result


def analyze_video(video_path, sample_every=1, outlier_z=2.5, max_damage_run=3,
                   sharpness_range=SHARPNESS_RANGE, noise_range=NOISE_RANGE,
                   contrast_range=CONTRAST_RANGE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[정보] 총 {total_frames} 프레임, FPS {fps:.2f}, 분석 간격: {sample_every}프레임마다 1장")

    rows = []
    prev_gray = None
    idx = 0
    scratch_history = deque(maxlen=SCRATCH_PERSIST_WINDOW)
    line_history = deque(maxlen=SCRATCH_PERSIST_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = sharpness_score(gray)
            noise = noise_score(gray)
            bright, contrast = brightness_contrast(gray)
            sat = saturation_score(frame)
            fdiff = frame_diff_score(prev_gray, gray)

            # 화면이 거의 안 움직이는 프레임(정지 타이틀 카드 등)은 콘텐츠 경계선이 스크래치로
            # 오탐되므로 후보 계산 자체에서 제외한다(prev_gray가 없는 첫 프레임도 마찬가지).
            is_static_frame = (prev_gray is None) or (fdiff < SCRATCH_STATIC_FRAME_DIFF_MIN)
            cols = np.array([], dtype=int) if is_static_frame else scratch_columns(gray)
            # 지속성 확인: 이번 프레임의 후보 컬럼이 최근 SCRATCH_PERSIST_WINDOW개 분석
            # 프레임(이번 프레임 포함) 중 SCRATCH_PERSIST_MIN_HITS번 이상, 비슷한 x좌표에서
            # 반복됐을 때만 '진짜 스크래치'로 확정한다(무작위 노이즈로 인한 1회성 오탐 배제).
            window = list(scratch_history) + [cols]
            has_scratch_frame = False
            for c in cols:
                hits = sum(
                    1 for arr in window
                    if arr.size and np.any(np.abs(arr - c) <= SCRATCH_COL_TOLERANCE)
                )
                if hits >= SCRATCH_PERSIST_MIN_HITS:
                    has_scratch_frame = True
                    break
            scratch_history.append(cols)

            # 가로줄(수평선)도 같은 정지 화면 제외 + 지속성 확인 로직을 행 방향에 적용
            line_rows_ = np.array([], dtype=int) if is_static_frame else line_rows(gray)
            line_window = list(line_history) + [line_rows_]
            has_hline_frame = False
            for r in line_rows_:
                hits = sum(
                    1 for arr in line_window
                    if arr.size and np.any(np.abs(arr - r) <= SCRATCH_COL_TOLERANCE)
                )
                if hits >= SCRATCH_PERSIST_MIN_HITS:
                    has_hline_frame = True
                    break
            line_history.append(line_rows_)

            rows.append({
                "frame_idx": idx,
                "time_sec": (idx / fps) if fps else None,
                "sharpness": sharp,
                "noise": noise,
                "brightness": bright,
                "contrast": contrast,
                "saturation": sat,
                "frame_diff": fdiff,
                "scratch_cols": len(cols),
                "has_scratch": has_scratch_frame,
                "hline_rows": len(line_rows_),
                "has_hline": has_hline_frame,
            })
            prev_gray = gray
        idx += 1

    cap.release()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("분석할 프레임이 없습니다. 영상 경로/코덱을 확인하세요.")

    # ---- 이상 프레임 탐지 (프레임 간 차이 기준 z-score) ----
    mean_fd, std_fd = df["frame_diff"].mean(), df["frame_diff"].std()
    if std_fd > 1e-9:
        df["frame_diff_z"] = (df["frame_diff"] - mean_fd) / std_fd
    else:
        df["frame_diff_z"] = 0.0
    df["is_outlier"] = df["frame_diff_z"].abs() > outlier_z

    # ---- 이상치 세분화: 손상 후보 vs 장면전환/동작 후보 (연속 길이 기준) ----
    df["outlier_type"] = classify_outlier_runs(df, sample_every, max_damage_run)

    # (has_scratch는 위 프레임 반복문에서 지속성 검증까지 마치고 이미 계산됨)

    # ---- 종합 점수 (0~100, 고정 절대 기준값 기준 -> 영상 간 비교 가능) ----
    sh_lo, sh_hi = sharpness_range
    no_lo, no_hi = noise_range
    co_lo, co_hi = contrast_range
    df["score_sharpness"] = normalize(df["sharpness"], sh_lo, sh_hi)          # 높을수록 좋음
    df["score_noise"] = normalize(df["noise"], no_lo, no_hi, invert=True)    # 낮을수록 좋음 -> invert
    df["score_contrast"] = normalize(df["contrast"], co_lo, co_hi)           # 높을수록 좋음(저대비 방지)
    df["condition_score"] = (
        df["score_sharpness"] * 0.4 +
        df["score_noise"] * 0.4 +
        df["score_contrast"] * 0.2
    )

    return df


# ---- 손상 위치 한눈에 보기 (타임라인 차트) ----
# 지표별로 나뉜 그래프(save_report의 4분할)만으로는 "영상의 어느 구간에 손상이 있는지"를
# 한눈에 파악하기 어렵다는 피드백(2026-09-30)에 따라, 종합 상태점수 곡선 하나 위에
# (1) 배경색으로 양호/주의/위험 점수 구간을 표시하고, (2) 손상 후보로 잡힌 프레임 구간을
# 빨간 띠로 겹쳐 강조하고, (3) 스크래치/가로줄이 감지된 프레임을 그래프 아래쪽 눈금으로
# 표시하는 단일 차트 하나로 정리한다. 색상은 배치 툴(batch_assessment.py)의 판정 색상과
# 통일해서 같은 색이 항상 같은 의미(위험=빨강/주의=노랑/양호=초록)를 갖도록 한다.
STATUS_COLOR = {"위험": "#C0504D", "주의": "#E2B93B", "양호": "#548235"}


def build_timeline_figure(df, sample_every=1):
    """훼손 의심 구간을 한눈에 보여주는 타임라인 차트를 matplotlib Figure로 만들어 반환한다.
    파일로 저장하려면 save_timeline_chart()를, Streamlit 등에 바로 그리려면 이 함수를 쓴다."""
    summary = summarize_condition(df, sample_every=sample_every)
    good_th = CRITERIA_THRESHOLDS["mean_score"]["good"]
    warn_th = CRITERIA_THRESHOLDS["mean_score"]["warn"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # 배경: mean_score 등급 경계와 같은 기준으로 양호/주의/위험 구간을 옅게 칠함
    ax.axhspan(0, warn_th, color=STATUS_COLOR["위험"], alpha=0.08, zorder=0)
    ax.axhspan(warn_th, good_th, color=STATUS_COLOR["주의"], alpha=0.10, zorder=0)
    ax.axhspan(good_th, 100, color=STATUS_COLOR["양호"], alpha=0.08, zorder=0)

    # 손상 후보로 잡힌 연속 구간(run)마다 빨간 띠를 겹쳐서 "어디에" 있는지 강조
    damage_mask = df["outlier_type"] == "손상 후보"
    if damage_mask.any():
        idxs = df.index[damage_mask].tolist()
        runs, run = [], [idxs[0]]
        for cur in idxs[1:]:
            if df.loc[cur, "frame_idx"] - df.loc[run[-1], "frame_idx"] <= sample_every:
                run.append(cur)
            else:
                runs.append(run)
                run = [cur]
        runs.append(run)
        for i, run in enumerate(runs):
            start = df.loc[run[0], "frame_idx"]
            end = df.loc[run[-1], "frame_idx"]
            ax.axvspan(start - sample_every / 2, end + sample_every / 2,
                       color=STATUS_COLOR["위험"], alpha=0.35, zorder=1,
                       label="손상 후보 구간" if i == 0 else None)

    ax.plot(df["frame_idx"], df["condition_score"], color="#2E3A46", linewidth=1.8,
            zorder=3, label="종합 상태점수")

    # 스크래치/가로줄 감지 프레임을 그래프 아래쪽에 눈금(|)으로 표시 -> 점수엔 안 잡혀도
    # 위치는 바로 보임
    if "has_scratch" in df.columns and df["has_scratch"].any():
        sc = df[df["has_scratch"]]
        ax.scatter(sc["frame_idx"], [2] * len(sc), marker="|", color="#7A4FA3",
                   s=140, linewidths=2, zorder=4, label="세로줄/스크래치 의심")
    if "has_hline" in df.columns and df["has_hline"].any():
        hl = df[df["has_hline"]]
        ax.scatter(hl["frame_idx"], [6] * len(hl), marker="|", color="#1B6FA8",
                   s=140, linewidths=2, zorder=4, label="가로줄/수평선 의심")

    ax.set_ylim(0, 100)
    ax.set_xlabel("프레임 번호")
    ax.set_ylabel("종합 상태점수 (0~100)")
    ax.set_title(f"훼손 의심 구간 한눈에 보기 — 최종 판정: {summary['verdict']}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def save_timeline_chart(df, out_dir, sample_every=1):
    os.makedirs(out_dir, exist_ok=True)
    fig = build_timeline_figure(df, sample_every=sample_every)
    png_path = os.path.join(out_dir, "damage_timeline.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path


def save_report(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "frame_metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(df["frame_idx"], df["sharpness"], color="#2E75B6")
    axes[0].set_ylabel("선명도\n(Laplacian var)")
    axes[0].set_title("프레임별 상태평가 지표 추이")

    axes[1].plot(df["frame_idx"], df["noise"], color="#C0504D")
    axes[1].set_ylabel("노이즈\n(잔차 표준편차)")

    axes[2].plot(df["frame_idx"], df["condition_score"], color="#548235")
    axes[2].set_ylabel("종합 상태점수\n(0~100)")
    axes[2].set_ylim(0, 100)

    axes[3].plot(df["frame_idx"], df["frame_diff"], color="#7F7F7F")
    damage = df[df["outlier_type"] == "손상 후보"]
    sceneshift = df[df["outlier_type"] == "장면전환/동작 후보"]
    axes[3].scatter(damage["frame_idx"], damage["frame_diff"],
                     color="red", zorder=5, label="손상 후보(짧게 튐)")
    axes[3].scatter(sceneshift["frame_idx"], sceneshift["frame_diff"],
                     color="orange", marker="^", zorder=5, label="장면전환/동작 후보(길게 튐)")
    axes[3].set_ylabel("프레임 간 차이\n(이상치)")
    axes[3].set_xlabel("프레임 번호")
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    png_path = os.path.join(out_dir, "condition_report.png")
    plt.savefig(png_path, dpi=120)
    plt.close()

    return csv_path, png_path


def print_summary(df, sample_every=1, audio_summary=None):
    print("\n===== 상태평가 요약 =====")
    print(f"분석 프레임 수      : {len(df)}")
    print(f"평균 선명도         : {df['sharpness'].mean():.2f}")
    print(f"평균 노이즈         : {df['noise'].mean():.2f}")
    print(f"평균 밝기           : {df['brightness'].mean():.2f}")
    print(f"평균 대비           : {df['contrast'].mean():.2f}")
    print(f"평균 채도           : {df['saturation'].mean():.2f}")
    print(f"종합 상태점수(평균) : {df['condition_score'].mean():.1f} / 100")

    summary = summarize_condition(df, sample_every=sample_every, audio_summary=audio_summary)
    print("\n----- 다기준 판정 (평균 희석형 오류 방지) -----")
    for key, label in CRITERIA_LABELS.items():
        if key not in summary["values"]:
            continue  # 오디오 분석을 안 했거나 못 한 경우 audio_* 키는 아예 없음
        v = summary["values"][key]
        g = summary["grades"][key]
        print(f"  {label:24s}: {v:6.1f}  [{g}]")
    if audio_summary is None:
        print("  (오디오 분석 안 함 — ffmpeg 없음. pip install imageio-ffmpeg 후 다시 실행하면 포함됩니다)")
    print(f"  >> 최종 판정: {summary['verdict']}" +
          (f" (사유: {', '.join(summary['reasons'])})" if summary["reasons"] else ""))

    n_outliers = int(df["is_outlier"].sum())
    n_damage = int((df["outlier_type"] == "손상 후보").sum())
    n_scene = int((df["outlier_type"] == "장면전환/동작 후보").sum())
    n_scratch = int(df["has_scratch"].sum()) if "has_scratch" in df.columns else 0
    n_hline = int(df["has_hline"].sum()) if "has_hline" in df.columns else 0
    print(f"\n이상 프레임 후보 수 : {n_outliers}  (손상 후보 {n_damage} / 장면전환·동작 후보 {n_scene})")
    print(f"세로줄/스크래치 의심 프레임 수 : {n_scratch}")
    print(f"가로줄/수평선 의심 프레임 수   : {n_hline}")

    if n_damage > 0:
        damage_list = df.loc[df["outlier_type"] == "손상 후보", "frame_idx"].tolist()
        preview = damage_list[:15]
        suffix = " ..." if len(damage_list) > 15 else ""
        print(f"  손상 후보 프레임(일부)        : {preview}{suffix}")
    if n_scene > 0:
        scene_list = df.loc[df["outlier_type"] == "장면전환/동작 후보", "frame_idx"].tolist()
        preview = scene_list[:15]
        suffix = " ..." if len(scene_list) > 15 else ""
        print(f"  장면전환/동작 후보 프레임(일부) : {preview}{suffix}")
    print("=========================\n")
    print("* 점수는 고정된 절대 기준값 기준이라 다른 영상과도 직접 비교 가능합니다(영상마다 다시 계산되지 않음).")
    print("* 단, 기준값은 경험적 값이므로 실제 자료로 여러 건 검증 후 필요하면 --sharpness-range 등으로 보정하세요.")
    print("* '손상 후보'/'장면전환·동작 후보' 구분은 연속 길이 기준 휴리스틱입니다 — diff_check.py로 눈으로 재확인 권장.")


def main():
    parser = argparse.ArgumentParser(description="영상 상태평가 툴")
    parser.add_argument("video", help="분석할 영상 파일 경로 (예: sample_clip.mp4)")
    parser.add_argument("--every", type=int, default=1, help="N프레임마다 1장 분석 (기본 1, 속도 위해 늘릴 수 있음)")
    parser.add_argument("--out", default="condition_report", help="결과 저장 폴더명 (기본: condition_report)")
    parser.add_argument("--zthresh", type=float, default=2.5, help="이상 프레임 탐지 민감도 (기본 2.5, 낮출수록 더 많이 잡힘)")
    parser.add_argument("--damage-run", type=int, default=3,
                         help="연속 이상치 프레임 수가 이 값 이하면 '손상 후보', 초과면 '장면전환/동작 후보' (기본 3)")
    parser.add_argument("--sharpness-range", type=float, nargs=2, default=SHARPNESS_RANGE,
                         metavar=("LO", "HI"), help=f"선명도 절대 기준 범위 (기본 {SHARPNESS_RANGE})")
    parser.add_argument("--noise-range", type=float, nargs=2, default=NOISE_RANGE,
                         metavar=("LO", "HI"), help=f"노이즈 절대 기준 범위 (기본 {NOISE_RANGE})")
    parser.add_argument("--contrast-range", type=float, nargs=2, default=CONTRAST_RANGE,
                         metavar=("LO", "HI"), help=f"대비 절대 기준 범위 (기본 {CONTRAST_RANGE})")
    parser.add_argument("--no-audio", action="store_true",
                         help="음성(오디오) 분석 생략(속도 위해, 또는 ffmpeg 문제 회피용)")
    args = parser.parse_args()

    df = analyze_video(args.video, sample_every=args.every, outlier_z=args.zthresh,
                        max_damage_run=args.damage_run,
                        sharpness_range=tuple(args.sharpness_range),
                        noise_range=tuple(args.noise_range),
                        contrast_range=tuple(args.contrast_range))

    audio_summary = None
    if not args.no_audio:
        audio_result = analyze_audio(args.video)
        audio_summary = summarize_audio(audio_result)

    csv_path, png_path = save_report(df, args.out)
    timeline_path = save_timeline_chart(df, args.out, sample_every=args.every)
    print_summary(df, sample_every=args.every, audio_summary=audio_summary)
    print(f"[저장 완료] CSV: {csv_path}")
    print(f"[저장 완료] 그래프: {png_path}")
    print(f"[저장 완료] 훼손 구간 한눈에 보기: {timeline_path}")


if __name__ == "__main__":
    main()
