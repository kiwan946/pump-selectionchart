import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import t

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v1.0", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v1.0")

# --- 유틸리티 및 기본 분석 함수들 ---
SERIES_ORDER = ["XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32", "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185", "XRF215", "XRF255"]

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
    # Q(m³/min), H(m) 기준 축동력(kW) 계산 상수 0.163
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
        
        interp_h_rated = np.interp(target_q, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
        h_churn = model_df.iloc[0][h_col]
        q_overload = 1.5 * target_q
        interp_h_overload = np.interp(q_overload, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)

        if not np.isnan(interp_h_rated) and interp_h_rated >= target_h:
            cond1_ok = h_churn <= (1.40 * target_h)
            cond2_ok = (not np.isnan(interp_h_overload)) and (interp_h_overload >= (0.65 * target_h))
            if cond1_ok and cond2_ok:
                interp_kw = np.interp(target_q, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df.columns else np.nan
                results.append({"모델명": model, "정격 예상 양정": f"{interp_h_rated:.2f}", "체절 양정 (≤{1.4*target_h:.2f})": f"{h_churn:.2f}", "최대운전 양정 (≥{0.65*target_h:.2f})": f"{interp_h_overload:.2f}", "예상 동력(kW)": f"{interp_kw:.2f}", "선정 가능": "✅"})
                continue

        h_values_rev = model_df[h_col].values[::-1]
        q_values_rev = model_df[q_col].values[::-1]

        if target_h <= model_df[h_col].max() and target_h >= model_df[h_col].min():
            q_required = np.interp(target_h, h_values_rev, q_values_rev)
            if 0.95 * target_q <= q_required < target_q:
                q_overload_corr = 1.5 * q_required
                interp_h_overload_corr = np.interp(q_overload_corr, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
                
                cond1_ok = h_churn <= (1.40 * target_h)
                cond2_ok = (not np.isnan(interp_h_overload_corr)) and (interp_h_overload_corr >= (0.65 * target_h))

                if cond1_ok and cond2_ok:
                    correction_pct = (1 - (q_required / target_q)) * 100
                    status_text = f"유량 {correction_pct:.1f}% 보정 전제 사용 가능"
                    interp_kw_corr = np.interp(q_required, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df.columns else np.nan
                    results.append({"모델명": model, "정격 예상 양정": f"{target_h:.2f} (at Q={q_required:.2f})", "체절 양정 (≤{1.4*target_h:.2f})": f"{h_churn:.2f}", "최대운전 양정 (≥{0.65*target_h:.2f})": f"{interp_h_overload_corr:.2f}", "예상 동력(kW)": f"{interp_kw_corr:.2f}", "선정 가능": status_text})
    
    return pd.DataFrame(results)

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

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ [수정됨] parse_selection_table 함수 ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def parse_selection_table(df_selection_table):
    """
    사용자가 업로드한 'XRF 모델 선정표...' (CSV 또는 Excel) 파일의 특정 구조를 파싱합니다.
    - Q (유량)은 11행 (인덱스 10)에서 E, H, K... 열(3칸 간격)에서 가져옵니다.
    - H (양정)은 B열 (인덱스 1)에서 16, 19, 22... 행(3줄 간격)에서 가져옵니다.
    - Model은 위 Q, H가 교차하는 지점에서 가져옵니다.
    """
    try:
        # 유량(Q) 헤더가 있는 열 인덱스: 4(E), 7(H), 10(K), ...
        q_col_indices = list(range(4, df_selection_table.shape[1], 3))
        # 양정(H) 헤더가 있는 행 인덱스: 15(16행), 18(19행), 21(22행), ...
        h_row_indices = list(range(15, df_selection_table.shape[0], 3))
        
        tasks = []
        q_values = {}
        h_values = {}

        # 1. 유량(Q) 값 파싱 (11행, 인덱스 10)
        # iloc[10, c_idx]는 엑셀 기준 11행
        for c_idx in q_col_indices:
            q_val_raw = str(df_selection_table.iloc[10, c_idx])
            if pd.isna(q_val_raw) or q_val_raw == "": continue
            try:
                # '0.13 (7.8)' 형식에서 '0.13'만 추출
                q_val_clean = q_val_raw.split('(')[0].strip()
                q_values[c_idx] = float(q_val_clean)
            except (ValueError, TypeError):
                continue # 유효하지 않은 열 스킵
        
        # 2. 양정(H) 값 파싱 (B열, 인덱스 1)
        # iloc[r_idx, 1]는 엑셀 기준 B열
        for r_idx in h_row_indices:
            h_val_raw = str(df_selection_table.iloc[r_idx, 1])
            if pd.isna(h_val_raw) or h_val_raw == "": continue
            try:
                # [수정] '301\n(139.8)' 또는 '301 (139.8)' 형식에서 '301'만 추출
                h_val_clean = h_val_raw.split('\n')[0].split('(')[0].strip()
                h_values[r_idx] = float(h_val_clean)
            except (ValueError, TypeError):
                continue # 유효하지 않은 행 스킵
        
        # 3. 교차 지점의 모델명 파싱
        for r_idx in h_values:
            for c_idx in q_values:
                # iloc[r_idx, c_idx]는 엑셀 기준 [16행, E열], [16행, H열]...
                model_name = str(df_selection_table.iloc[r_idx, c_idx]).strip()
                
                # 'nan', '미선정...' 등이 아닌 유효한 모델명인지 확인
                if model_name and model_name.lower() != 'nan' and 'XRF' in model_name:
                    tasks.append({
                        "모델명": model_name,
                        "요구 유량 (Q)": q_values[c_idx],
                        "요구 양정 (H)": h_values[r_idx],
                        "_source_cell": f"[Row {r_idx + 1}, Col {chr(65 + c_idx)}]" # 디버깅용
                    })
        
        return pd.DataFrame(tasks)
    
    except Exception as e:
        st.error(f"선정표 파싱 중 심각한 오류 발생: {e}. (엑셀 행/열 구조가 예상과 다를 수 있습니다.)")
        return pd.DataFrame()
# ★ (수정 끝) ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

def display_validation_output(model, validation_data, analysis_type, df_r, df_d, m_r, m_d, q_r, q_d, y_r_col, y_d_col, test_id_col):
    if model not in validation_data or validation_data[model]['summary'].empty:
        st.warning(f"'{model}' 모델에 대한 {analysis_type} 분석 결과가 없습니다.")
        return

    model_data = validation_data[model]
    model_summary_df = model_data['summary']
    model_samples = model_data['samples']
    base_col_name = f"기준 {analysis_type}"
    
    st.markdown(f"#### 분석 결과 요약 ({analysis_type})")
    display_summary = model_summary_df.drop(columns=['_original_q']).set_index('모델명')
    st.dataframe(display_summary, use_container_width=True)
    
    st.markdown(f"#### 모델별 상세 결과 시각화 ({analysis_type})")
    fig_main = go.Figure()
    numeric_cols = ["검증 유량(Q)", base_col_name, "95% CI 하한", "95% CI 상한"]
    for col in numeric_cols: model_summary_df[col] = pd.to_numeric(model_summary_df[col], errors='coerce')
    
    fig_main.add_trace(go.Scatter(x=model_summary_df['검증 유량(Q)'], y=model_summary_df['95% CI 상한'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='95% CI 상한'))
    fig_main.add_trace(go.Scatter(x=model_summary_df['검증 유량(Q)'], y=model_summary_df['95% CI 상한'], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='95% CI 하한'))
    
    model_d_df_vis = df_d[(df_d[m_d] == model) & (df_d[y_d_col].notna())]; test_ids_vis = model_d_df_vis[test_id_col].unique()
    for test_id in test_ids_vis:
        test_df_vis = model_d_df_vis[model_d_df_vis[test_id_col] == test_id].sort_values(by=q_d)
        fig_main.add_trace(go.Scatter(x=test_df_vis[q_d], y=test_df_vis[y_d_col], mode='lines', line=dict(width=1, color='grey'), name=f'시험 {test_id}', opacity=0.5, showlegend=False))
    
    model_r_df_vis = df_r[(df_r[m_r] == model) & (df_r[y_r_col].notna())].sort_values(by=q_r)
    fig_main.add_trace(go.Scatter(x=model_r_df_vis[q_r], y=model_r_df_vis[y_r_col], mode='lines+markers', line=dict(color='blue', width=3), name='Reference Curve'))
    
    if analysis_type == '양정':
        upper_limit = model_summary_df[base_col_name] * 1.05
        lower_limit = model_summary_df[base_col_name] * 0.95
        fig_main.add_trace(go.Scatter(x=model_summary_df['검증 유량(Q)'], y=upper_limit, mode='lines', name='양정 상한 (+5%)', line=dict(color='orange', dash='dash')))
        fig_main.add_trace(go.Scatter(x=model_summary_df['검증 유량(Q)'], y=lower_limit, mode='lines', name='양정 하한 (-5%)', line=dict(color='orange', dash='dash')))

    valid_points = model_summary_df[model_summary_df['유효성'] == '✅ 유효']; invalid_points = model_summary_df[model_summary_df['유효성'] == '❌ 벗어남']
    fig_main.add_trace(go.Scatter(x=valid_points['검증 유량(Q)'], y=valid_points[base_col_name], mode='markers', marker=dict(color='green', size=10, symbol='circle'), name='유효 포인트'))
    fig_main.add_trace(go.Scatter(x=invalid_points['검증 유량(Q)'], y=invalid_points[base_col_name], mode='markers', marker=dict(color='red', size=10, symbol='x'), name='벗어남 포인트'))
    
    fig_main.update_layout(yaxis_title=analysis_type)
    st.plotly_chart(fig_main, use_container_width=True)

    with st.expander(f"검증 유량 지점별 {analysis_type} 데이터 분포표 보기"):
        for idx, row in model_summary_df.iterrows():
            q_point_original = row['_original_q']
            samples = model_samples.get(q_point_original, [])
            if not samples or row['시험 횟수(n)'] < 2: continue
            q_point_str, ref_y_point, mean_y, std_y, n_samples = row['검증 유량(Q)'], float(row[base_col_name]), float(row['평균']), float(row['표준편차']), int(row['시험 횟수(n)'])
            st.markdown(f"**Q = {q_point_str}**")
            st.markdown(f"<small>평균: {mean_y:.2f} | 표준편차: {std_y:.2f} | n: {n_samples}</small>", unsafe_allow_html=True)
            fig_dist = ff.create_distplot([samples], ['시험 데이터'], show_hist=False, show_rug=True)
            fig_dist.add_vline(x=ref_y_point, line_width=2, line_dash="dash", line_color="red")
            fig_dist.add_vline(x=mean_y, line_width=2, line_dash="dot", line_color="blue")
            fig_dist.update_layout(title_text=None, xaxis_title=analysis_type, yaxis_title="밀도", height=300, margin=dict(l=20,r=20,t=5,b=20), showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
            st.markdown("---")

# --- 메인 애플리케이션 로직 ---
uploaded_file = st.file_uploader("1. 기준 데이터 Excel 파일 업로드 (reference data 시트 포함)", type=["xlsx", "xlsm"])
if uploaded_file:
    m_r, df_r_orig = load_sheet(uploaded_file, "reference data"); m_c, df_c_orig = load_sheet(uploaded_file, "catalog data"); m_d, df_d_orig = load_sheet(uploaded_file, "deviation data")
    if df_r_orig.empty: st.error("오류: 'reference data' 시트를 찾을 수 없거나 '모델명' 관련 컬럼이 없습니다. 파일을 확인해주세요.")
    else:
        st.sidebar.title("⚙️ 분석 설정"); st.sidebar.markdown("### Total 탭 & 운전점 분석 컬럼 지정")
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
            df_d_orig[test_id_col_d] = df_d_orig[test_id_col_d].astype(str).str.strip()
            df_d_orig[test_id_col_d].replace(['', 'nan'], np.nan, inplace=True)
            df_d_orig[test_id_col_d] = df_d_orig[test_id_col_d].ffill()
        df_r = process_data(df_r_orig, q_col_total, h_col_total, k_col_total); df_c = process_data(df_c_orig, q_c, h_c, k_c); df_d = process_data(df_d_orig, q_d, h_d, k_d)
        
        # '탭 리스트' 수정 (맨 뒤에 "🔥 선정표 검토 (AI)" 추가)
        tab_list = ["Total", "Reference", "Catalog", "Deviation", "Validation", "🔥 선정표 검토 (AI)"]
        tabs = st.tabs(tab_list)
        
        # ★★★★★★★★★★★★★★★★★★★ 'Total' 탭 (원본 유지) ★★★★★★★★★★★★★★★★★★★
        with tabs[0]:
            st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
            df_f = render_filters(df_r, m_r, "total")
            models = df_f[m_r].unique().tolist() if m_r and not df_f.empty else []
            
            with st.expander("운전점 분석 (Operating Point Analysis)"):
                st.markdown("#### 🎯 단일 운전점 기준 모델 검색")
                analysis_mode = st.radio("분석 모드", ["기계", "소방"], key="analysis_mode", horizontal=True)
                op_col1, op_col2 = st.columns(2)

                with op_col1:
                    q_input_str = st.text_input("목표 유량 (Q, m³/min)", value="0.0")
                    try:
                        target_q = float(q_input_str)
                    except ValueError:
                        target_q = 0.0
                        st.warning("유량에 유효한 숫자를 입력해주세요.", icon="⚠️")
                
                with op_col2:
                    h_input_str = st.text_input("목표 양정 (H, m)", value="0.0")
                    try:
                        target_h = float(h_input_str)
                    except ValueError:
                        target_h = 0.0
                        st.warning("양정에 유효한 숫자를 입력해주세요.", icon="⚠️")

                if analysis_mode == "소방": st.info("소방 펌프 성능 기준 3점(정격, 체절, 최대)을 자동으로 분석합니다.")
                if st.button("모델 검색 실행"):
                    if not models: st.warning("먼저 분석할 시리즈나 모델을 선택해주세요.")
                    else:
                        with st.spinner("선택된 모델들을 분석 중입니다..."):
                            if analysis_mode == "소방": op_results_df = analyze_fire_pump_point(df_r, models, target_q, target_h, m_r, q_col_total, h_col_total, k_col_total)
                            else: op_results_df = analyze_operating_point(df_r, models, target_q, target_h, m_r, q_col_total, h_col_total, k_col_total)
                            
                            if not op_results_df.empty: st.success(f"총 {len(op_results_df)}개의 모델을 찾았습니다."); st.dataframe(op_results_df, use_container_width=True)
                            else: st.info("요구 성능을 만족하는 모델을 찾지 못했습니다.")

                st.markdown("---")
                st.markdown("#### 📥 모델별 개별 운전점 검토 (Batch)")
                
                st.info("엑셀에서 '모델명 | 유량(m³/min) | 양정(m)' 3개 열을 복사하여 아래 표에 붙여넣으세요.\n행 추가 버튼을 눌러 수동으로 입력할 수도 있습니다.")

                if 'batch_df' not in st.session_state:
                    st.session_state.batch_df = pd.DataFrame(
                        [{"모델명": "XRF5-16", "요구 유량 (Q)": 0.06, "요구 양정 (H)": 35.0, "분석 모드": "기계"}],
                        columns=["모델명", "요구 유량 (Q)", "요구 양정 (H)", "분석 모드"]
                    )
                
                st.markdown("##### 1. 검토할 데이터 입력 (붙여넣기/수정)")
                edited_df = st.data_editor(
                    st.session_state.batch_df,
                    column_config={
                        "모델명": st.column_config.TextColumn("모델명", width="medium"),
                        "요구 유량 (Q)": st.column_config.NumberColumn("요구 유량 (Q, m³/min)", format="%.3f", width="small"),
                        "요구 양정 (H)": st.column_config.NumberColumn("요구 양정 (H, m)", format="%.2f", width="small"),
                        "분석 모드": st.column_config.SelectboxColumn(
                            "분석 모드",
                            options=["기계", "소방"],
                            required=True,
                            width="small"
                        )
                    },
                    use_container_width=True,
                    num_rows="dynamic",
                    key="batch_editor"
                )
                
                st.session_state.batch_df = edited_df

                st.markdown("##### 2. 분석 실행")
                if st.button("🚀 개별 모델 검토 실행"):
                    results = []
                    if df_r.empty:
                        st.error("Reference data (df_r)가 로드되지 않았습니다. 파일 업로드를 확인하세요.")
                    elif edited_df.empty:
                        st.warning("검토할 데이터가 없습니다. 표에 데이터를 입력해주세요.")
                    else:
                        with st.spinner("개별 모델 검토 중..."):
                            for _, row in edited_df.iterrows():
                                model = row['모델명']
                                q = row['요구 유량 (Q)']
                                h = row['요구 양정 (H)']
                                mode = row['분석 모드']
                                
                                if not model or pd.isna(model):
                                    continue
                                
                                if model not in df_r[m_r].unique():
                                    results.append({
                                        '모델명': model, '요구 유량 (Q)': q, '요구 양정 (H)': h, '분석 모드': mode,
                                        '결과': '❌ 모델 없음',
                                        '상세': 'Reference 데이터에 해당 모델이 없습니다.'
                                    })
                                    continue

                                if mode == "소방":
                                    op_result = analyze_fire_pump_point(df_r, [model], q, h, m_r, q_col_total, h_col_total, k_col_total)
                                else: # "기계"
                                    op_result = analyze_operating_point(df_r, [model], q, h, m_r, q_col_total, h_col_total, k_col_total)
                                    
                                if not op_result.empty:
                                    status = op_result.iloc[0]['선정 가능']
                                    head_col_name = '정격 예상 양정' if mode == "소방" else '예상 양정'
                                    eff_str = f" | 예상 효율: {op_result.iloc[0]['예상 효율(%)']}" if '예상 효율(%)' in op_result.columns else ""
                                    head_val = op_result.iloc[0].get(head_col_name, 'N/A')
                                    power_val = op_result.iloc[0].get('예상 동력(kW)', 'N/A')
                                    details = f"예상 양정: {head_val} | 예상 동력: {power_val}{eff_str}"
                                        
                                    results.append({
                                        '모델명': model, '요구 유량 (Q)': q, '요구 양정 (H)': h, '분석 모드': mode,
                                        '결과': status,
                                        '상세': details
                                    })
                                else:
                                    results.append({
                                        '모델명': model, '요구 유량 (Q)': q, '요구 양정 (H)': h, '분석 모드': mode,
                                        '결과': '❌ 사용 불가',
                                        '상세': '요구 성능을 만족하는 운전점을 찾을 수 없습니다.'
                                    })
                            
                        st.session_state.batch_results_df = pd.DataFrame(results)
                        if 'batch_results_df' not in st.session_state or st.session_state.batch_results_df.empty:
                            st.info("분석 결과가 없습니다.")


                if 'batch_results_df' in st.session_state and not st.session_state.batch_results_df.empty:
                    st.markdown("##### 3. 분석 결과")
                    st.dataframe(st.session_state.batch_results_df.set_index('모델명'), use_container_width=True)

            with st.expander("차트 보조선 추가"):
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1: h_guide_h, v_guide_h = st.number_input("Q-H 수평선", value=0.0), st.number_input("Q-H 수직선", value=0.0)
                with g_col2: h_guide_k, v_guide_k = st.number_input("Q-kW 수평선", value=0.0), st.number_input("Q-kW 수직선", value=0.0)
                with g_col3: h_guide_e, v_guide_e = st.number_input("Q-Eff 수평선", value=0.0), st.number_input("Q-Eff 수직선", value=0.0)
            
            st.markdown("---")
            ref_show = st.checkbox("Reference 표시", value=True); cat_show = st.checkbox("Catalog 표시"); dev_show = st.checkbox("Deviation 표시")
            st.markdown(f"#### Q-H (유량-{h_col_total})")
            fig_h = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_h, df_f, m_r, q_col_total, h_col_total, models, 'lines+markers'); add_bep_markers(fig_h, df_f, m_r, q_col_total, h_col_total, models)
            if cat_show and not df_c.empty: add_traces(fig_h, df_c, m_c, q_c, h_c, models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_h, df_d, m_d, q_d, h_d, models, 'markers')
            
            if 'target_q' in locals() and target_q > 0 and target_h > 0:
                fig_h.add_trace(go.Scatter(x=[target_q], y=[target_h], mode='markers', marker=dict(symbol='cross', size=15, color='magenta'), name='정격 운전점 (단일)'))
                if analysis_mode == "소방":
                    churn_h_limit = 1.4 * target_h; fig_h.add_trace(go.Scatter(x=[0], y=[churn_h_limit], mode='markers', marker=dict(symbol='x', size=12, color='red'), name=f'체절점 상한'))
                    overload_q, overload_h_limit = 1.5 * target_q, 0.65 * target_h; fig_h.add_trace(go.Scatter(x=[overload_q], y=[overload_h_limit], mode='markers', marker=dict(symbol='diamond-open', size=12, color='blue'), name=f'최대점 하한'))
            
            if 'batch_results_df' in st.session_state and not st.session_state.batch_results_df.empty:
                batch_plot_df = st.session_state.batch_results_df
                fig_h.add_trace(go.Scatter(
                    x=batch_plot_df['요구 유량 (Q)'], 
                    y=batch_plot_df['요구 양정 (H)'],
                    mode='markers+text',
                    marker=dict(symbol='star', size=12, color='orange'),
                    text=batch_plot_df['모델명'] + " (" + batch_plot_df['결과'] + ")",
                    textposition="top right",
                    name='개별 검토 운전점'
                ))

            add_guide_lines(fig_h, h_guide_h, v_guide_h); render_chart(fig_h, "total_qh")
            
            st.markdown("#### Q-kW (유량-축동력)"); fig_k = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_k, df_f, m_r, q_col_total, k_col_total, models, 'lines+markers')
            if cat_show and not df_c.empty: add_traces(fig_k, df_c, m_c, q_c, k_c, models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_k, df_d, m_d, q_d, k_d, models, 'markers')
            add_guide_lines(fig_k, h_guide_k, v_guide_k); render_chart(fig_k, "total_qk")
            
            st.markdown("#### Q-Efficiency (유량-효율)"); fig_e = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_e, df_f, m_r, q_col_total, 'Efficiency', models, 'lines+markers'); add_bep_markers(fig_e, df_f, m_r, q_col_total, 'Efficiency', models)
            if cat_show and not df_c.empty: add_traces(fig_e, df_c, m_c, q_c, 'Efficiency', models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_e, df_d, m_d, q_d, 'Efficiency', models, 'markers')
            add_guide_lines(fig_e, h_guide_e, v_guide_e); render_chart(fig_e, "total_qe")
        # ★★★★★★★★★★★★★★★★★★★ 'Total' 탭 끝 ★★★★★★★★★★★★★★★★★★★

        for idx, sheet_name in enumerate(["Reference", "Catalog", "Deviation"]):
            with tabs[idx+1]:
                st.subheader(f"📊 {sheet_name} Data")
                df, mcol, df_orig = (df_r, m_r, df_r_orig) if sheet_name == "Reference" else (df_c, m_c, df_c_orig) if sheet_name == "Catalog" else (df_d, m_d, df_d_orig)
                if df.empty: st.info(f"'{sheet_name.lower()}' 시트의 데이터가 없거나 처리할 수 없습니다."); continue
                q_col_tab = get_best_match_column(df_orig, ["토출량", "유량"]); h_col_tab = get_best_match_column(df_orig, ["토출양정", "전양정"]); k_col_tab = get_best_match_column(df_orig, ["축동력"])
                df_f_tab = render_filters(df, mcol, sheet_name)
                models_tab = df_f_tab[mcol].unique().tolist() if not df_f_tab.empty else []
                if not models_tab: st.info("차트를 보려면 모델을 선택해주세요."); continue
                mode, style = ('markers', None) if sheet_name == "Deviation" else ('lines+markers', dict(dash='dot') if sheet_name == "Catalog" else None)
                if h_col_tab: st.markdown(f"#### Q-H ({h_col_tab})"); fig1 = go.Figure(); add_traces(fig1, df_f_tab, mcol, q_col_tab, h_col_tab, models_tab, mode, line_style=style); render_chart(fig1, key=f"{sheet_name}_qh")
                if k_col_tab in df_f_tab.columns: st.markdown("#### Q-kW (축동력)"); fig2 = go.Figure(); add_traces(fig2, df_f_tab, mcol, q_col_tab, k_col_tab, models_tab, mode, line_style=style); render_chart(fig2, key=f"{sheet_name}_qk")
                if 'Efficiency' in df_f_tab.columns: st.markdown("#### Q-Efficiency (효율)"); fig3 = go.Figure(); add_traces(fig3, df_f_tab, mcol, q_col_tab, 'Efficiency', models_tab, mode, line_style=style); fig3.update_layout(yaxis_title="효율 (%)", yaxis=dict(range=[0, 100])); render_chart(fig3, key=f"{sheet_name}_qe")
                st.markdown("#### 데이터 확인"); st.dataframe(df_f_tab, use_container_width=True)
        
        with tabs[4]:
            st.subheader("🔬 Reference Data 통계적 유효성 검증")
            power_cols_exist = k_col_total and k_d
            if not power_cols_exist: st.info("축동력 분석을 위해서는 Reference와 Deviation 시트 양쪽에 '축동력' 관련 컬럼이 필요합니다.")
            if df_d_orig.empty or test_id_col_d is None: st.warning("유효성 검증을 위해 'deviation data' 시트와 '시험번호' 컬럼이 필요합니다.")
            else:
                with st.expander("Deviation 데이터 확인하기"): st.dataframe(df_d_orig)
                common_models = sorted(list(set(df_r[m_r].unique()) & set(df_d[m_d].unique())))
                if not common_models: st.info("Reference와 Deviation 데이터에 공통으로 존재하는 모델이 없습니다.")
                else:
                    models_to_validate = st.multiselect("검증할 모델 선택", common_models, default=common_models[:1])
                    if st.button("📈 통계 검증 실행"):
                        with st.spinner("통계 분석을 진행 중입니다..."):
                            head_results = perform_validation_analysis(df_r, df_d, m_r, m_d, q_col_total, q_d, h_col_total, h_d, test_id_col_d, models_to_validate, "양정")
                            if power_cols_exist: power_results = perform_validation_analysis(df_r, df_d, m_r, m_d, q_col_total, q_d, k_col_total, k_d, test_id_col_d, models_to_validate, "축동력")
                        st.success("통계 분석 완료!")
                        for model in models_to_validate:
                            st.markdown("---"); st.markdown(f"### 모델: {model}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("📈 양정(Head) 유효성 검증")
                                display_validation_output(model, head_results, "양정", df_r, df_d, m_r, m_d, q_col_total, q_d, h_col_total, h_d, test_id_col_d)
                            with col2:
                                if power_cols_exist:
                                    st.subheader("⚡ 축동력(Power) 유효성 검증")
                                    display_validation_output(model, power_results, "축동력", df_r, df_d, m_r, m_d, q_col_total, q_d, k_col_total, k_d, test_id_col_d)
                        st.markdown("---"); st.header("📊 표준성능 곡선 제안 (Reference vs. 실측 평균)")
                        fig_col1, fig_col2 = st.columns(2)
                        with fig_col1:
                            st.subheader("Q-H Curve (양정)")
                            fig_h_proposal = go.Figure()
                            for model in models_to_validate:
                                if model in head_results and not head_results[model]['summary'].empty:
                                    summary_df = head_results[model]['summary']
                                    summary_df['평균'] = pd.to_numeric(summary_df['평균'], errors='coerce')
                                    fig_h_proposal.add_trace(go.Scatter(x=summary_df['검증 유량(Q)'], y=summary_df['평균'], mode='lines+markers', name=f'{model} (제안)'))
                                    model_r_df = df_r[df_r[m_r] == model].sort_values(q_col_total)
                                    fig_h_proposal.add_trace(go.Scatter(x=model_r_df[q_col_total], y=model_r_df[h_col_total], mode='lines', name=f'{model} (기존)', line=dict(dash='dot'), opacity=0.7))
                            st.plotly_chart(fig_h_proposal, use_container_width=True)
                        with fig_col2:
                            if power_cols_exist:
                                st.subheader("Q-kW Curve (축동력)")
                                fig_k_proposal = go.Figure()
                                for model in models_to_validate:
                                    if model in power_results and not power_results[model]['summary'].empty:
                                        summary_df = power_results[model]['summary']
                                        summary_df['평균'] = pd.to_numeric(summary_df['평균'], errors='coerce')
                                        fig_k_proposal.add_trace(go.Scatter(x=summary_df['검증 유량(Q)'], y=summary_df['평균'], mode='lines+markers', name=f'{model} (제안)'))
                                        model_r_df = df_r[df_r[m_r] == model].sort_values(q_col_total)
                                        fig_k_proposal.add_trace(go.Scatter(x=model_r_df[q_col_total], y=model_r_df[k_col_total], mode='lines', name=f'{model} (기존)', line=dict(dash='dot'), opacity=0.7))
                                st.plotly_chart(fig_k_proposal, use_container_width=True)

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ 3. '선정표 검토 (AI)' 탭 로직 (신규 추가) ★
        # ★   (시트 이름 'XRF 모델 선정표_품질검토본_20250110'을 먼저 시도하도록 수정됨) ★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        with tabs[5]:
            st.subheader("🔥 XRF 모델 선정표 자동 검토 (AI)")
            st.warning("이 기능은 'reference data'가 (첫번째 업로드로) 정상 로드되었을 때만 동작합니다.")
            
            # (1) 기준 데이터(df_r)가 로드되었는지 확인
            if df_r.empty or m_r is None:
                st.error("가장 먼저 'reference data'가 포함된 원본 Excel 파일을 업로드해야 합니다.")
            
            # (2) 기준 데이터가 있을 경우, 검토 파일 업로더 표시
            else:
                st.info("검토 대상인 'XRF 모델 선정표...' 엑셀 파일을 업로드하세요.")
                
                review_excel_file = st.file_uploader("2. 선정표 Excel 파일 업로드 (.xlsx, .xlsm)", type=["xlsx", "xlsm"], key="review_excel")
                
                if review_excel_file:
                    
                    # [수정] 사용자가 언급한 특정 시트 이름을 먼저 시도합니다.
                    sheet_to_try = 'XRF 모델 선정표_품질검토본_20250110'
                    try:
                        df_selection_excel = pd.read_excel(review_excel_file, sheet_name=sheet_to_try, header=None)
                        st.success(f"'{sheet_to_try}' 시트를 성공적으로 로드했습니다.")
                    except Exception:
                        st.warning(f"'{sheet_to_try}' 시트를 찾을 수 없습니다. 엑셀 파일의 첫 번째 시트를 대신 읽습니다.")
                        try:
                            # 특정 시트가 없으면 첫 번째 시트로 Fallback
                            df_selection_excel = pd.read_excel(review_excel_file, sheet_name=0, header=None)
                            st.info("첫 번째 시트를 로드했습니다.")
                        except Exception as e_first:
                            st.error(f"Excel 파일 로딩에 실패했습니다: {e_first}")
                            df_selection_excel = None

                    if df_selection_excel is not None:
                        # (3) 엑셀 파싱 (기존 parse_selection_table 함수 재사용)
                        if 'task_list_df' not in st.session_state or st.session_state.get('review_file_name') != review_excel_file.name:
                            with st.spinner("선정표(Excel) 파일을 분석하여 검토 목록을 생성 중입니다..."):
                                st.session_state.task_list_df = parse_selection_table(df_selection_excel)
                                st.session_state.review_file_name = review_excel_file.name # 새 파일 감지용
                        
                        task_df = st.session_state.task_list_df
                        
                        if task_df.empty:
                            st.error("Excel 파일에서 유효한 검토 대상(모델명, Q, H)을 찾지 못했습니다. 파일 형식이나 시트 이름을 확인하세요.")
                        else:
                            st.markdown(f"**총 {len(task_df)}개**의 검토 대상을 찾았습니다.")
                            with st.expander("파싱된 검토 목록 확인 (Excel 파일 기준)"):
                                st.dataframe(task_df, use_container_width=True)

                            # (4) 검토 실행 버튼
                            if st.button("🚀 소방 성능 기준 검토 실행"):
                                with st.spinner(f"{len(task_df)}개 항목을 'reference data'와 비교 검토 중입니다... (1~2분 소요)"):
                                    results = []
                                    all_ref_models = df_r[m_r].unique() # 모델 존재 여부 확인용 (빠른 속도)
                                    
                                    for _, row in task_df.iterrows():
                                        model = row['모델명']
                                        q = row['요구 유량 (Q)']
                                        h = row['요구 양정 (H)']
                                        
                                        # 기준 데이터(df_r)에 모델이 없는 경우
                                        if model not in all_ref_models:
                                            result_detail = {
                                                "결과": "❌ 모델 없음",
                                                "상세": "Reference 데이터에 해당 모델명이 없습니다."
                                            }
                                        else:
                                            # 소방 성능 분석 실행
                                            op_result_df = analyze_fire_pump_point(df_r, [model], q, h, m_r, q_col_total, h_col_total, k_col_total)
                                            
                                            if not op_result_df.empty:
                                                # 분석 성공 (기준 통과)
                                                res_row = op_result_df.iloc[0]
                                                result_detail = {
                                                    "결과": res_row['선정 가능'], # "✅" 또는 "유량 X% 보정..."
                                                    "정격 양정": res_row['정격 예상 양정'],
                                                    "체절 양정": res_row[f'체절 양정 (≤{1.4*h:.2f})'],
                                                    "최대 양정": res_row[f'최대운전 양정 (≥{0.65*h:.2f})'],
                                                    "예상 동력": res_row['예상 동력(kW)']
                                                }
                                            else:
                                                # 분석 실패 (모델은 있으나 3점 기준 미달 또는 유량 범위 이탈)
                                                # '기계' 모드로 단순 체크하여 힌트 제공
                                                mech_check_df = analyze_operating_point(df_r, [model], q, h, m_r, q_col_total, h_col_total, k_col_total)
                                                if not mech_check_df.empty:
                                                    details = f"정격점은 만족하나 3점(체절/최대) 기준 미달. (예상양정: {mech_check_df.iloc[0]['예상 양정']})"
                                                else:
                                                    details = "요구 성능을 만족하는 운전점을 찾을 수 없음 (유량 범위 이탈 등)"
                                                    
                                                result_detail = {
                                                    "결과": "❌ 사용 불가",
                                                    "상세": details
                                                }
                                    
                                        base_info = {
                                            "선정 모델": model,
                                            "요구 유량(Q)": q,
                                            "요구 양정(H)": h
                                        }
                                        base_info.update(result_detail)
                                        results.append(base_info)
                                    
                                st.session_state.review_results_df = pd.DataFrame(results)
                                st.success("선정표 검토 완료!")

                # (5) 결과 표시
                if 'review_results_df' in st.session_state:
                    st.markdown("---")
                    st.markdown("### 📊 검토 결과 요약")
                    results_df = st.session_state.review_results_df
                    
                    # 결과 필터링
                    failed_df = results_df[results_df['결과'].str.contains("❌")]
                    warning_df = results_df[~results_df['결과'].str.contains("❌|✅")] # "보정" 등
                    success_df = results_df[results_df['결과'] == "✅"]
                    
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                    res_col1.metric("총 검토 항목", len(results_df))
                    res_col2.metric("❌ 선정 오류", len(failed_df), delta_color="inverse")
                    res_col3.metric("⚠️ 보정 필요", len(warning_df), delta_color="off")
                    res_col4.metric("✅ 선정 가능", len(success_df))
                    
                    st.markdown("#### ❌ 선정 오류 목록")
                    if failed_df.empty:
                        st.info("선정 오류로 판단된 항목이 없습니다.")
                    else:
                        st.dataframe(failed_df.set_index("선정 모델"), use_container_width=True)
                    
                    st.markdown("#### ⚠️ 보정 필요 목록")
                    if warning_df.empty:
                        st.info("유량 보정이 필요한 항목이 없습니다.")
                    else:
                        st.dataframe(warning_df.set_index("선정 모델"), use_container_width=True)
                        
                    with st.expander("✅ 전체 검토 결과 보기 (성공/실패/보정 포함)"):
                        st.dataframe(results_df.set_index("선정 모델"), use_container_width=True)
