# ... (이전 코드 생략: warning_df 표시 부분 이후) ...

                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★ [수정됨] 전체 검토 결과 피벗 테이블 (공란/미선정/대안없음 처리 강화) ★
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    st.markdown("#### ✅ 전체 검토 결과 (피벗 테이블)")
                    
                    display_pivot_source = results_df
                    
                    if display_pivot_source.empty:
                        st.info("피벗 테이블에 표시할 항목이 없습니다.")
                    else:
                        try:
                            def format_motor(kw):
                                if pd.isna(kw): return "(?kW)"
                                if kw == int(kw): return f"({int(kw)}kW)"
                                return f"({kw}kW)"
                            
                            def create_display_text(row):
                                model_val = row['선정 모델']
                                rec_val = row.get('추천모델', '')
                                result_val = str(row['결과'])

                                # [Case 1] 엑셀 공란 (미선정)인 경우
                                if model_val == "미선정":
                                    base_text = "❌ 선정불가"
                                    
                                    # 대안 모델 탐색 결과에 따른 텍스트 분기
                                    if rec_val == "대안 없음":
                                        return base_text + "\n(대안모델 없음)"
                                    elif rec_val: # 추천 모델이 존재할 경우
                                        return base_text + f"\n💡 추천: {rec_val}"
                                    else: # 아직 추천 버튼을 누르기 전
                                        return base_text

                                # [Case 2] 모델이 기입되어 있는 경우
                                else:
                                    base_text = f"{model_val} {format_motor(row['선정 모터(kW)'])}"
                                    
                                    # '❌ 사용 불가' 등의 결과가 있으면 앞에 표시
                                    if "❌" in result_val:
                                        base_text = f"❌ {base_text}"

                                    extras = []
                                    
                                    # 유량 보정 표시
                                    corr = row.get('보정률(%)', 0)
                                    if corr > 0:
                                        extras.append(f"💧 유량보정: {corr:.1f}%")
                                    
                                    # 동력 초과 표시
                                    p100 = row.get('동력초과(100%)', 0)
                                    p150 = row.get('동력초과(150%)', 0)
                                    if p100 > 100 or p150 > 100:
                                        p_str = f"{max(p100, p150):.0f}%"
                                        extras.append(f"⚡ 동력초과: {p_str}")
                                    
                                    # 추천 정보
                                    if rec_val == "대안 없음":
                                        extras.append("(대안모델 없음)")
                                    elif rec_val:
                                        extras.append(f"💡 추천: {rec_val}")

                                    if extras:
                                        return base_text + "\n" + "\n".join(extras)
                                    return base_text

                            # 표시값 컬럼 생성
                            display_pivot_source['표시값'] = display_pivot_source.apply(create_display_text, axis=1)

                            # 피벗 테이블 생성
                            pivot_df = pd.pivot_table(
                                display_pivot_source, 
                                values='표시값', 
                                index='요구 양정(H)', 
                                columns='요구 유량(Q)', 
                                aggfunc='first', 
                                # [핵심 수정] 데이터가 없는 구간도 '선정불가'로 표시 (파싱 누락 대비)
                                fill_value="❌ 선정불가" 
                            )
                            
                            # 양정 기준 내림차순 정렬
                            pivot_df = pivot_df.sort_index(ascending=False)
                            
                            # 테이블 표시 (높이 지정으로 스크롤 가능하게)
                            st.dataframe(pivot_df, use_container_width=True, height=800)
                        
                        except Exception as e_pivot:
                            st.error(f"피벗 테이블 생성 중 오류 발생: {e_pivot}")
                            st.dataframe(display_pivot_source.set_index("선정 모델"), use_container_width=True)
