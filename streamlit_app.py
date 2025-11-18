import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.stats import t
import re

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v2.9", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v2.9 (선정표 검토 기능 강화)")

# --- 유틸리티 및 전역 상수 ---
SERIES_ORDER = ["XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32", "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185", "XRF215", "XRF255"]
STANDARD_MOTORS = [0.75, 1.5, 2.2, 3.7, 5.5, 7.5, 11, 15, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200]

def get_best_match_column(df, names):
    if df is None or df.empty: return None
    for n in names:
        for col in df.columns:
            if n in col.strip():
                return col
    return None

def calculate_efficiency(df, q_col, h_col, k_col):
    if not all(col and col in df.columns for col in [q_col, h_col, k_col]): return df
    df_copy = df.copy()
    hydraulic_power = 0.163 * df_copy[q_col] * df_copy[h_col]
    shaft_power = df_copy[k_col]
    df_copy['Efficiency'] = np.where(shaft_power > 0, (hydraulic_power / shaft_power) * 100, 0)
    return df_copy

def load_sheet(uploaded_file, sheet_name):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        mcol = get_best_match_column(df, ["모델명", "모델", "Model"])
        if not mcol: return None, pd.DataFrame()
        if 'Series' in df.columns: df = df.drop(columns=['Series'])
        
        df['Series'] = df[mcol].astype(str).str.extract(r"(XRF\d+)")
        df['Series'] = pd.Categorical(df['Series'], categories=SERIES_ORDER, ordered=True)
        df = df.sort_values('Series')
        return mcol, df
    except Exception:
        return None, pd.DataFrame()

def process_data(df, q_col, h_col, k_col):
    if df.empty: return df
    temp_df = df.copy()
    for col in [q_col, h_col, k_col]:
        if col and col in temp_df.columns:
            temp_df = temp_df.dropna(subset=[col])
            temp_df = temp_df[pd.to_numeric(temp_df[col], errors='coerce').notna()]
            temp_df[col] = pd.to_numeric(temp_df[col])
    return calculate_efficiency(temp_df, q_col, h_col, k_col)

# --- 분석 로직 (단일/소방/배치) ---

def analyze_operating_point(df, models, target_q, target_h, m_col, q_col, h_col, k_col):
    if target_h <= 0: return pd.DataFrame()
    results = []

    if target_q == 0:
        for model in models:
            model_df = df[df[m_col] == model].sort_values(q_col)
            if model_df.empty: continue
            churn_h = model_df.iloc[0][h_col]
            if churn_h >= target_h:
                churn_kw = model_df.iloc[0][k_col] if k_col and k_col in model_df.columns else np.nan
                churn_eff = np.interp(0, model_df[q_col], model_df['Efficiency']) if 'Efficiency' in model_df.columns else 0
                results.append({"모델명": model, "요구 유량": "0 (체절)", "요구 양정": target_h, "예상 양정": f"{churn_h:.2f}", "예상 동력(kW)": f"{churn_kw:.2f}", "예상 효율(%)": f"{churn_eff:.2f}", "선정 가능": "✅"})
        return pd.DataFrame(results)

    for model in models:
        model_df = df[df[m_col] == model].sort_values(q_col)
        if len(model_df) < 2 or not (model_df[q_col].min() <= target_q <= model_df[q_col].max()): continue
        interp_h = np.interp(target_q, model_df[q_col], model_df[h_col])
        
        if interp_h >= target_h:
            interp_kw = np.interp(target_q, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df.columns else np.nan
            interp_eff = np.interp(target_q, model_df[q_col], model_df['Efficiency']) if 'Efficiency' in model_df.columns else np.nan
            results.append({"모델명": model, "요구 유량": target_q, "요구 양정": target_h, "예상 양정": f"{interp_h:.2f}", "예상 동력(kW)": f"{interp_kw:.2f}", "예상 효율(%)": f"{interp_eff:.2f}", "선정 가능": "✅"})
        else:
            # 보정 가능 여부 확인 (역산)
            h_values_rev = model_df[h_col].values[::-1]
            q_values_rev = model_df[q_col].values[::-1]
            if target_h <= model_df[h_col].max() and target_h >= model_df[h_col].min():
                q_required = np.interp(target_h, h_values_rev, q_values_rev)
                if 0.95 * target_q <= q_required < target_q:
                    correction_pct = (1 - (q_required / target_q)) * 100
                    status_text = f"유량 {correction_pct:.1f}% 보정 전제 사용 가능"
                    interp_kw_corr = np.interp(q_required, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df.columns else np.nan
                    interp_eff_corr = np.interp(q_required, model_df[q_col], model_df['Efficiency']) if 'Efficiency' in model_df.columns else np.nan
                    results.append({"모델명": model, "요구 유량": target_q, "요구 양정": target_h, "예상 양정": f"{target_h:.2f} (at Q={q_required:.2f})", "예상 동력(kW)": f"{interp_kw_corr:.2f}", "예상 효율(%)": f"{interp_eff_corr:.2f}", "선정 가능": status_text})
    
    return pd.DataFrame(results)

def analyze_fire_pump_point(df, models, target_q, target_h, m_col, q_col, h_col, k_col):
    if target_q <= 0 or target_h <= 0: return pd.DataFrame()
    results = []
    for model in models:
        model_df = df[df[m_col] == model].sort_values(q_col)
        if len(model_df) < 2: continue
        
        res_dict = _batch_analyze_fire_point(model_df, target_q, target_h, q_col, h_col, k_col, STANDARD_MOTORS)
        
        row = {
            "모델명": model,
            "정격 예상 양정": res_dict['정격 예상 양정'],
            "체절 양정 (예상)": res_dict['체절 양정 (예상)'],
            "체절 양정 (기준)": res_dict['체절 양정 (기준)'],
            "최대운전 양정 (예상)": res_dict['최대운전 양정 (예상)'],
            "최대운전 양정 (기준)": res_dict['최대운전 양정 (기준)'],
            "예상 동력(kW)": f"{res_dict['정격 동력(kW)']:.2f}",
            "선정 가능": res_dict['선정 가능']
        }
        results.append(row)
        
    return pd.DataFrame(results)

def _calculate_motor(p_rated, p_overload, standard_motors):
    if pd.isna(p_rated) or pd.isna(p_overload):
        return np.nan
    for motor_kw in standard_motors:
        if (p_rated <= motor_kw * 1.05) and (p_overload <= motor_kw * 1.15):
            return motor_kw
    return np.nan

def _batch_analyze_fire_point(model_df, target_q, target_h, q_col, h_col, k_col, standard_motors):
    if target_q <= 0 or target_h <= 0: 
        return {
            "선정 가능": "❌ 사용 불가", "상세": "유량/양정 0",
            "정격 예상 양정": "N/A", "체절 양정 (예상)": "N/A", "체절 양정 (기준)": "N/A",
            "최대운전 양정 (예상)": "N/A", "최대운전 양정 (기준)": "N/A", 
            "정격 동력(kW)": np.nan, "최대 동력(kW)": np.nan, "선정 모터(kW)": np.nan,
            "보정률(%)": 0.0, "동력초과(100%)": 0.0, "동력초과(150%)": 0.0
        }
    
    h_churn_limit = 1.40 * target_h
    h_overload_limit = 0.65 * target_h
    h_churn = model_df.iloc[0][h_col]
    TOLERANCE_FACTOR = 0.97 

    correction_steps = np.linspace(0, 0.05, 51) 
    
    for correction_pct in correction_steps:
        q_corrected = target_q * (1 - correction_pct)
        interp_h_rated = np.interp(q_corrected, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
        q_overload_corr = 1.5 * q_corrected
        interp_h_overload_corr = np.interp(q_overload_corr, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
        
        p_corr = np.interp(q_corrected, model_df[q_col], model_df[k_col], left=np.nan, right=np.nan)
        p_overload_corr = np.interp(q_overload_corr, model_df[q_col], model_df[k_col], left=np.nan, right=np.nan)
        
        cond_rated = (not np.isnan(interp_h_rated)) and (interp_h_rated >= target_h)
        cond_churn = (h_churn <= h_churn_limit)
        cond_overload = (not np.isnan(interp_h_overload_corr)) and (interp_h_overload_corr >= h_overload_limit * TOLERANCE_FACTOR)
        
        if cond_rated and cond_churn and cond_overload:
            motor_corr = _calculate_motor(p_corr, p_overload_corr, standard_motors)
            p_ratio_100 = (p_corr / motor_corr * 100) if motor_corr and not pd.isna(motor_corr) else 0.0
            p_ratio_150 = (p_overload_corr / motor_corr * 100) if motor_corr and not pd.isna(motor_corr) else 0.0

            if correction_pct == 0:
                status_text = "✅"
                detail_text = "정격 유량 기준"
            else:
                status_text = f"⚠️ 보정 필요"
                detail_text = f"유량 {correction_pct*100:.1f}% 보정"
            
            return {
                "정격 예상 양정": f"{interp_h_rated:.2f}",
                "체절 양정 (예상)": f"{h_churn:.2f}",
                "체절 양정 (기준)": f"≤{h_churn_limit:.2f}",
                "최대운전 양정 (예상)": f"{interp_h_overload_corr:.2f}",
                "최대운전 양정 (기준)": f"≥{h_overload_limit:.2f}",
                "정격 동력(kW)": p_corr,
                "최대 동력(kW)": p_overload_corr,
                "선정 모터(kW)": motor_corr,
                "선정 가능": status_text,
                "상세": detail_text,
                "보정률(%)": correction_pct * 100,
                "동력초과(100%)": p_ratio_100,
                "동력초과(150%)": p_ratio_150
            }

    # 실패 시 원인 분석
    q_orig = target_q
    interp_h_orig = np.interp(q_orig, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
    q_over_orig = 1.5 * q_orig
    interp_h_over_orig = np.interp(q_over_orig, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
    
    fail_reason = ""
    if np.isnan(interp_h_orig) or interp_h_orig < target_h: fail_reason = "정격 양정 미달"
    elif h_churn > h_churn_limit: fail_reason = "체절 양정 초과"
    elif np.isnan(interp_h_over_orig) or interp_h_over_orig < h_overload_limit * TOLERANCE_FACTOR: fail_reason = "최대 운전 양정 미달"
    else: fail_reason = "3점 기준 미달"

    return {
        "정격 예상 양정": f"{interp_h_orig:.2f}",
        "체절 양정 (예상)": f"{h_churn:.2f}",
        "체절 양정 (기준)": f"≤{h_churn_limit:.2f}",
        "최대운전 양정 (예상)": f"{interp_h_over_orig:.2f}",
        "최대운전 양정 (기준)": f"≥{h_overload_limit:.2f}",
        "정격 동력(kW)": np.nan, "최대 동력(kW)": np.nan, "선정 모터(kW)": np.nan,
        "선정 가능": "❌ 사용 불가",
        "상세": fail_reason,
        "보정률(%)": 0.0, "동력초과(100%)": 0.0, "동력초과(150%)": 0.0
    }

def find_recommendation(df_r, m_r, q_col, h_col, k_col, target_q, target_h, assigned_model):
    # 현재 시리즈 파악 및 검색 범위 축소
    match = re.search(r"(XRF\d+)", str(assigned_model))
    target_series_subset = []
    
    if match:
        current_series = match.group(1)
        if current_series in SERIES_ORDER:
            curr_idx = SERIES_ORDER.index(current_series)
            start_idx = max(0, curr_idx - 2) # 이전 2개 시리즈부터 검색
            target_series_subset = SERIES_ORDER[start_idx:]
    
    if target_series_subset:
        candidate_models = df_r[df_r['Series'].isin(target_series_subset)][m_r].unique()
    else:
        candidate_models = df_r[m_r].unique()

    candidates = []

    for model in candidate_models:
        if model == assigned_model: continue
        
        model_df = df_r[df_r[m_r] == model].sort_values(q_col)
        if model_df.empty: continue
        
        # 대략적인 범위 필터링 (속도 향상)
        if not (model_df[q_col].max() * 1.1 >= target_q and model_df[h_col].max() >= target_h):
            continue

        res = _batch_analyze_fire_point(model_df, target_q, target_h, q_col, h_col, k_col, STANDARD_MOTORS)
        
        if "❌" not in res['선정 가능']:
            candidates.append({
                "모델명": model,
                "보정률": res['보정률(%)'],
                "모터": res['선정 모터(kW)']
            })
    
    if not candidates: return None
    
    # 보정률 낮은 순, 모터 용량 작은 순 정렬
    candidates.sort(key=lambda x: (x['보정률'], x['모터']))
    
    best = candidates[0]
    rec_str = f"{best['모델명']} ({best['보정률']:.1f}% 보정)" if best['보정률'] > 0 else best['모델명']
    return rec_str

def render_filters(df, mcol, prefix):
    if df is None or df.empty or mcol is None or 'Series' not in df.columns:
        st.warning("필터링할 데이터가 없습니다.")
        return pd.DataFrame()
    series_opts = df['Series'].dropna().unique().tolist()
    default_series = [series_opts[0]] if series_opts else []
    mode = st.radio("분류 기준", ["시리즈별", "모델별"], key=f"{prefix}_mode", horizontal=True)
    if mode == "시리즈별":
        sel = st.multiselect("시리즈 선택", series_opts, default=default_series, key=f"{prefix}_series")
        df_f = df[df['Series'].isin(sel)] if sel else pd.DataFrame()
    else:
        model_opts = df[mcol].dropna().unique().tolist()
        default_model = [model_opts[0]] if model_opts else []
        sel = st.multiselect("모델 선택", model_opts, default=default_model, key=f"{prefix}_models")
        df_f = df[df[mcol].isin(sel)] if sel else pd.DataFrame()
    return df_f

def parse_selection_table(df_selection_table):
    try:
        q_col_indices = list(range(4, df_selection_table.shape[1], 3))
        h_row_indices = list(range(15, df_selection_table.shape[0], 3))
        
        tasks = []
        q_values = {}
        h_values = {}

        # Q 파싱
        for c_idx in q_col_indices:
            q_val_raw = str(df_selection_table.iloc[10, c_idx])
            if pd.isna(q_val_raw) or q_val_raw == "": continue
            try:
                q_val_clean = q_val_raw.split('(')[0].strip()
                q_values[c_idx] = float(q_val_clean)
            except (ValueError, TypeError):
                continue 
        
        # H 파싱
        for r_idx in h_row_indices:
            h_val_raw = str(df_selection_table.iloc[r_idx, 1])
            if pd.isna(h_val_raw) or h_val_raw == "": continue
            try:
                h_val_clean = h_val_raw.split('\n')[0].split('(')[0].strip()
                h_values[r_idx] = float(h_val_clean)
            except (ValueError, TypeError):
                continue 
        
        # 교차 지점 파싱 (완전 탐색)
        for r_idx in h_values:
            for c_idx in q_values:
                raw_cell = df_selection_table.iloc[r_idx, c_idx]
                model_name = str(raw_cell).strip()
                
                # XRF 포함 여부로 모델 판단, 아니면 미선정
                if "XRF" in model_name:
                    pass
                else:
                    model_name = "미선정"
                
                tasks.append({
                    "모델명": model_name,
                    "요구 유량 (Q)": q_values[c_idx],
                    "요구 양정 (H)": h_values[r_idx],
                    "_source_cell": f"[Row {r_idx + 1}, Col {chr(65 + c_idx)}]"
                })
        
        return pd.DataFrame(tasks)
    
    except Exception as e:
        st.error(f"선정표 파싱 중 오류 발생: {e}")
        return pd.DataFrame()

# --- 시각화 함수 ---
def add_traces(fig, df, mcol, xcol, ycol, models, mode, line_style=None, name_suffix=""):
    for m in models:
        sub = df[df[mcol] == m].sort_values(xcol)
        if sub.empty or ycol not in sub.columns: continue
        fig.add_trace(go.Scatter(x=sub[xcol], y=sub[ycol], mode=mode, name=m + name_suffix, line=line_style or {}))

def add_bep_markers(fig, df, mcol, qcol, ycol, models):
    for m in models:
        model_df = df[df[mcol] == m]
        if not model_df.empty and 'Efficiency' in model_df.columns and not model_df['Efficiency'].isnull().all():
            bep_row = model_df.loc[model_df['Efficiency'].idxmax()]
            fig.add_trace(go.Scatter(x=[bep_row[qcol]], y=[bep_row[ycol]], mode='markers', marker=dict(symbol='star', size=15, color='gold'), name=f'{m} BEP'))

def add_guide_lines(fig, h_line, v_line):
    if h_line is not None and h_line > 0:
        fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=h_line, y1=h_line, yref="y", line=dict(color="gray", dash="dash"))
    if v_line is not None and v_line > 0:
        fig.add_shape(type="line", x0=v_line, x1=v_line, xref="x", y0=0, y1=1, yref="paper", line=dict(color="gray", dash="dash"))

def render_chart(fig, key):
    fig.update_layout(dragmode='pan', xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False}, key=key)

def perform_validation_analysis(df_r, df_d, m_r, m_d, q_r, q_d, y_r_col, y_d_col, test_id_col, models_to_validate, analysis_type):
    all_results = {}
    for model in models_to_validate:
        model_summary = []
        model_r_df = df_r[(df_r[m_r] == model) & (df_r[y_r_col].notna())].sort_values(by=q_r)
        model_d_df = df_d[(df_d[m_d] == model) & (df_d[y_d_col].notna())]
        if model_r_df.empty or model_d_df.empty: continue
        
        max_q = model_r_df[q_r].max()
        validation_q = np.linspace(0, max_q, 10)
        ref_y = np.interp(validation_q, model_r_df[q_r], model_r_df[y_r_col])
        test_ids = model_d_df[test_id_col].unique()
        interpolated_y_samples = {q: [] for q in validation_q}
        for test_id in test_ids:
            test_df = model_d_df[model_d_df[test_id_col] == test_id].sort_values(by=q_d)
            if len(test_df) < 2: continue
            interp_y = np.interp(validation_q, test_df[q_d], test_df[y_d_col])
            for i, q in enumerate(validation_q):
                interpolated_y_samples[q].append(interp_y[i])
        
        for i, q in enumerate(validation_q):
            samples = np.array(interpolated_y_samples[q])
            n = len(samples)
            base_col_name = f"기준 {analysis_type}"
            mean_col_name = "평균"
            if n < 2:
                model_summary.append({
                    "모델명": model, "검증 유량(Q)": q, base_col_name: ref_y[i], 
                    "시험 횟수(n)": n, mean_col_name: np.nan, "표준편차": np.nan, 
                    "95% CI 하한": np.nan, "95% CI 상한": np.nan, "유효성": "판단불가",
                    "_original_q": q
                })
                continue
            
            mean_y, std_dev = np.mean(samples), np.std(samples, ddof=1)
            std_err = std_dev / np.sqrt(n)
            t_critical = t.ppf(0.975, df=n-1)
            margin_of_error = t_critical * std_err
            ci_lower, ci_upper = mean_y - margin_of_error, mean_y + margin_of_error
            is_valid = "✅ 유효" if ci_lower <= ref_y[i] <= ci_upper else "❌ 벗어남"
            
            model_summary.append({
                "모델명": model, "검증 유량(Q)": f"{q:.2f}", base_col_name: f"{ref_y[i]:.2f}",
                "시험 횟수(n)": n, mean_col_name: f"{mean_y:.2f}", "표준편차": f"{std_dev:.2f}",
                "95% CI 하한": f"{ci_lower:.2f}", "95% CI 상한": f"{ci_upper:.2f}", "유효성": is_valid,
                "_original_q": q
            })
        
        all_results[model] = { 'summary': pd.DataFrame(model_summary), 'samples': interpolated_y_samples }
    return all_results

# --- 메인 애플리케이션 ---
uploaded_file = st.file_uploader("1. 기준 데이터 Excel 파일 업로드 (reference data 시트 포함)", type=["xlsx", "xlsm"])
if uploaded_file:
    m_r, df_r_orig = load_sheet(uploaded_file, "reference data"); m_c, df_c_orig = load_sheet(uploaded_file, "catalog data"); m_d, df_d_orig = load_sheet(uploaded_file, "deviation data")
    if df_r_orig.empty: st.error("오류: 'reference data' 시트를 찾을 수 없거나 '모델명' 관련 컬럼이 없습니다.")
    else:
        st.sidebar.title("⚙️ 분석 설정")
        all_columns_r = df_r_orig.columns.tolist()
        def safe_get_index(items, value, default=0):
            try: return items.index(value)
            except (ValueError, TypeError): return default
        q_auto_r = get_best_match_column(df_r_orig, ["토출량", "유량"]); h_auto_r = get_best_match_column(df_r_orig, ["토출양정", "전양정"]); k_auto_r = get_best_match_column(df_r_orig, ["축동력"])
        q_col_total = st.sidebar.selectbox("유량 (Flow) 컬럼", all_columns_r, index=safe_get_index(all_columns_r, q_auto_r))
        h_col_total = st.sidebar.selectbox("양정 (Head) 컬럼", all_columns_r, index=safe_get_index(all_columns_r, h_auto_r))
        k_col_total = st.sidebar.selectbox("축동력 (Power) 컬럼", all_columns_r, index=safe_get_index(all_columns_r, k_auto_r))
        q_c, h_c, k_c = (get_best_match_column(df_c_orig, ["토출량", "유량"]), get_best_match_column(df_c_orig, ["토출양정", "전양정"]), get_best_match_column(df_c_orig, ["축동력"]))
        q_d, h_d, k_d = (get_best_match_column(df_d_orig, ["토출량", "유량"]), get_best_match_column(df_d_orig, ["토출양정", "전양정"]), get_best_match_column(df_d_orig, ["축동력"]))
        test_id_col_d = get_best_match_column(df_d_orig, ["시험번호", "Test No", "Test ID"])
        if not df_d_orig.empty and test_id_col_d:
            df_d_orig[test_id_col_d] = df_d_orig[test_id_col_d].astype(str).str.strip().replace(['', 'nan'], np.nan).ffill()
        df_r = process_data(df_r_orig, q_col_total, h_col_total, k_col_total); df_c = process_data(df_c_orig, q_c, h_c, k_c); df_d = process_data(df_d_orig, q_d, h_d, k_d)
        
        tab_list = ["Total", "Reference", "Catalog", "Deviation", "Validation", "🔥 선정표 검토 (AI)"]
        tabs = st.tabs(tab_list)
        
        # [Total 탭]
        with tabs[0]:
            st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
            df_f = render_filters(df_r, m_r, "total")
            models = df_f[m_r].unique().tolist() if m_r and not df_f.empty else []
            
            with st.expander("운전점 분석 (Operating Point Analysis)"):
                analysis_mode = st.radio("분석 모드", ["기계", "소방"], key="analysis_mode", horizontal=True)
                op_col1, op_col2 = st.columns(2)
                with op_col1:
                    target_q = float(st.text_input("목표 유량 (Q, m³/min)", value="0.0"))
                with op_col2:
                    target_h = float(st.text_input("목표 양정 (H, m)", value="0.0"))
                
                if st.button("모델 검색 실행"):
                    if not models: st.warning("모델을 선택해주세요.")
                    else:
                        if analysis_mode == "소방": op_results_df = analyze_fire_pump_point(df_r, models, target_q, target_h, m_r, q_col_total, h_col_total, k_col_total)
                        else: op_results_df = analyze_operating_point(df_r, models, target_q, target_h, m_r, q_col_total, h_col_total, k_col_total)
                        st.dataframe(op_results_df, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 📥 모델별 개별 운전점 검토 (Batch)")
                if 'batch_df' not in st.session_state:
                    st.session_state.batch_df = pd.DataFrame([{"모델명": "XRF5-16", "요구 유량 (Q)": 0.06, "요구 양정 (H)": 35.0, "분석 모드": "기계"}])
                
                edited_df = st.data_editor(st.session_state.batch_df, num_rows="dynamic", use_container_width=True, key="batch_editor")
                st.session_state.batch_df = edited_df
                
                if st.button("🚀 개별 모델 검토 실행"):
                    results = []
                    for _, row in edited_df.iterrows():
                        model, q, h, mode = row['모델명'], row['요구 유량 (Q)'], row['요구 양정 (H)'], row['분석 모드']
                        if not model or model not in df_r[m_r].unique(): continue
                        if mode == "소방": res_df = analyze_fire_pump_point(df_r, [model], q, h, m_r, q_col_total, h_col_total, k_col_total)
                        else: res_df = analyze_operating_point(df_r, [model], q, h, m_r, q_col_total, h_col_total, k_col_total)
                        if not res_df.empty:
                            res_row = res_df.iloc[0]
                            results.append({'모델명': model, '요구 유량 (Q)': q, '요구 양정 (H)': h, '분석 모드': mode, '결과': res_row['선정 가능'], '상세': str(res_row.to_dict())})
                    st.session_state.batch_results_df = pd.DataFrame(results)

                if 'batch_results_df' in st.session_state and not st.session_state.batch_results_df.empty:
                    st.dataframe(st.session_state.batch_results_df, use_container_width=True)

            st.markdown("---")
            ref_show, cat_show, dev_show = st.checkbox("Reference", True), st.checkbox("Catalog"), st.checkbox("Deviation")
            fig_h = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_h, df_f, m_r, q_col_total, h_col_total, models, 'lines+markers'); add_bep_markers(fig_h, df_f, m_r, q_col_total, h_col_total, models)
            if cat_show and not df_c.empty: add_traces(fig_h, df_c, m_c, q_c, h_c, models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_h, df_d, m_d, q_d, h_d, models, 'markers')
            render_chart(fig_h, "total_qh")

        # [기본 데이터 탭들]
        for idx, sheet_name in enumerate(["Reference", "Catalog", "Deviation"]):
            with tabs[idx+1]:
                df, mcol = (df_r, m_r) if sheet_name == "Reference" else (df_c, m_c) if sheet_name == "Catalog" else (df_d, m_d)
                if df.empty: st.info("데이터 없음"); continue
                df_f_tab = render_filters(df, mcol, sheet_name)
                models_tab = df_f_tab[mcol].unique().tolist()
                if not models_tab: continue
                mode, style = ('markers', None) if sheet_name == "Deviation" else ('lines+markers', dict(dash='dot') if sheet_name == "Catalog" else None)
                q_col_tab = get_best_match_column(df_r_orig, ["토출량", "유량"]) # 임시: 탭별 컬럼 매핑 단순화
                h_col_tab = get_best_match_column(df_r_orig, ["토출양정", "전양정"])
                fig1 = go.Figure(); add_traces(fig1, df_f_tab, mcol, q_col_tab, h_col_tab, models_tab, mode, line_style=style); render_chart(fig1, f"{sheet_name}_qh")

        # [Validation 탭]
        with tabs[4]:
            st.subheader("🔬 Reference Data 통계적 유효성 검증")
            common_models = sorted(list(set(df_r[m_r].unique()) & set(df_d[m_d].unique())))
            models_to_validate = st.multiselect("검증할 모델 선택", common_models)
            if st.button("📈 통계 검증 실행") and models_to_validate:
                head_results = perform_validation_analysis(df_r, df_d, m_r, m_d, q_col_total, q_d, h_col_total, h_d, test_id_col_d, models_to_validate, "양정")
                for model in models_to_validate:
                    st.markdown(f"### {model}")
                    st.dataframe(head_results[model]['summary'], use_container_width=True)

        # [AI 선정표 검토 탭]
        with tabs[5]:
            st.subheader("🔥 XRF 모델 선정표 자동 검토 (AI)")
            if df_r.empty: st.error("Reference data가 로드되지 않았습니다.")
            else:
                review_excel_file = st.file_uploader("2. 선정표 Excel 파일 업로드", type=["xlsx", "xlsm"], key="review_excel")
                if review_excel_file:
                    try:
                        df_selection_excel = pd.read_excel(review_excel_file, header=None)
                    except:
                        st.error("파일 로드 실패")
                        df_selection_excel = None

                    if df_selection_excel is not None:
                        if 'task_list_df' not in st.session_state or st.session_state.get('review_file_name') != review_excel_file.name:
                             st.session_state.task_list_df = parse_selection_table(df_selection_excel)
                             st.session_state.review_file_name = review_excel_file.name
                        
                        task_df = st.session_state.task_list_df
                        if task_df.empty: st.error("유효한 검토 대상을 찾지 못했습니다.")
                        else:
                            st.markdown(f"총 {len(task_df)}개 검토 대상 발견")
                            
                            if st.button("🚀 소방 성능 기준 검토 실행"):
                                results = []
                                grouped_tasks = task_df.groupby('모델명')
                                for model_name, tasks in grouped_tasks:
                                    if model_name == "미선정":
                                        for _, row in tasks.iterrows():
                                            results.append({"선정 모델": "미선정", "요구 유량(Q)": row['요구 유량 (Q)'], "요구 양정(H)": row['요구 양정 (H)'], "결과": "❌ 선정 불가", "추천모델": ""})
                                        continue
                                    
                                    if model_name not in df_r[m_r].unique():
                                        for _, row in tasks.iterrows():
                                            results.append({"선정 모델": model_name, "요구 유량(Q)": row['요구 유량 (Q)'], "요구 양정(H)": row['요구 양정 (H)'], "결과": "❌ 모델 없음", "추천모델": ""})
                                        continue

                                    model_df = df_r[df_r[m_r] == model_name].sort_values(q_col_total)
                                    for _, row in tasks.iterrows():
                                        res = _batch_analyze_fire_point(model_df, row['요구 유량 (Q)'], row['요구 양정 (H)'], q_col_total, h_col_total, k_col_total, STANDARD_MOTORS)
                                        results.append({
                                            "선정 모델": model_name, "요구 유량(Q)": row['요구 유량 (Q)'], "요구 양정(H)": row['요구 양정 (H)'], 
                                            "결과": res['선정 가능'], "선정 모터(kW)": res['선정 모터(kW)'], "보정률(%)": res['보정률(%)'], 
                                            "동력초과(100%)": res['동력초과(100%)'], "동력초과(150%)": res['동력초과(150%)'], "추천모델": ""
                                        })
                                st.session_state.review_results_df = pd.DataFrame(results)
                                st.rerun()

                if 'review_results_df' in st.session_state:
                    st.markdown("---")
                    results_df = st.session_state.review_results_df
                    
                    if st.button("🕵️ 대안 모델 추천 실행"):
                        with st.spinner("최적 모델 탐색 중..."):
                            for idx, row in results_df.iterrows():
                                if "✅" in row['결과']: continue
                                rec_str = find_recommendation(df_r, m_r, q_col_total, h_col_total, k_col_total, row['요구 유량(Q)'], row['요구 양정(H)'], row['선정 모델'])
                                results_df.at[idx, '추천모델'] = rec_str if rec_str else "대안 없음"
                            st.session_state.review_results_df = results_df
                            st.success("추천 완료!")
                            st.rerun()

                    st.markdown("#### ✅ 전체 검토 결과 (피벗 테이블)")
                    if results_df.empty: st.info("결과 없음")
                    else:
                        def format_motor(kw):
                            if pd.isna(kw): return "(?kW)"
                            return f"({int(kw)}kW)" if kw == int(kw) else f"({kw}kW)"

                        def create_display_text(row):
                            model_val = row['선정 모델']
                            rec_val = row.get('추천모델', '')
                            
                            # [수정됨] 미선정 공란 처리 로직
                            if model_val == "미선정":
                                base_text = "❌ 선정불가"
                                if rec_val == "대안 없음": return base_text + "\n(대안모델 없음)"
                                elif rec_val: return base_text + f"\n💡 추천: {rec_val}"
                                else: return base_text
                            
                            base_text = f"{model_val} {format_motor(row.get('선정 모터(kW)', np.nan))}"
                            if "❌" in str(row['결과']): base_text = f"❌ {base_text}"
                            
                            extras = []
                            if row.get('보정률(%)', 0) > 0: extras.append(f"💧 보정:{row['보정률(%)']:.1f}%")
                            p_max = max(row.get('동력초과(100%)', 0), row.get('동력초과(150%)', 0))
                            if p_max > 100: extras.append(f"⚡ 초과:{p_max:.0f}%")
                            
                            if rec_val == "대안 없음": extras.append("(대안모델 없음)")
                            elif rec_val: extras.append(f"💡 추천: {rec_val}")
                            
                            return base_text + ("\n" + "\n".join(extras) if extras else "")

                        results_df['표시값'] = results_df.apply(create_display_text, axis=1)
                        
                        try:
                            pivot_df = pd.pivot_table(
                                results_df, values='표시값', index='요구 양정(H)', columns='요구 유량(Q)', 
                                aggfunc='first', fill_value="❌ 선정불가"
                            ).sort_index(ascending=False)
                            st.dataframe(pivot_df, use_container_width=True, height=800)
                        except Exception as e:
                            st.error(f"피벗 생성 오류: {e}")
                            st.dataframe(results_df)
