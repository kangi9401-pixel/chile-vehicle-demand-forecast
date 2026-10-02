# 저장소 이전 버전 감사 결과

감사일: 2026-09-06  
범위: `chile_forecast` 수요예측 코드, 합성 데이터 생성기, 실행 진입점, 테스트, README 및 기존 기술보고서

## 요약

기존 저장소는 seed가 고정된 합성 데이터, XGBoost·DeepAR, 3개 rolling origin, 역-MAE 앙상블, SHAP, 테스트 21개를 갖고 있었다. 단일 holdout만 쓰던 초기 형태보다 개선돼 있었지만, 최종 평가로 사용하기에는 전처리 누출과 미래 설명변수 가용성, 약한 기준모형, 짧은 평가기간, 제한된 지표라는 중대한 문제가 남아 있었다. 기존 테스트 21개는 변경 전 모두 통과했다.

## 항목별 감사

| 점검항목 | 기존 상태 | 위험/판정 | 조치 |
|---|---|---|---|
| 데이터 생성 | 2015-01~2025-05, 125개월 합성 자료, NumPy seed 42 | 보안상 적절. 실제 자료 아님이 명시됨 | 유지 |
| 분할 순서 | 전체 특징 생성 후 holdout/backtest mask 적용 | 전처리 통계가 분할보다 먼저 계산됨 | fold 내부 `fit`으로 이동 |
| 원자재 표준화 | 전체 기간 평균·표준편차로 z-score | 평가기간 정보가 훈련 특징에 누출 | 훈련구간 통계만 사용 |
| 결측 처리 | 특징 전체에 forward fill·0 fill | fold 밖 정보가 통계에 관여할 수 있음 | 훈련 중앙값과 마지막 관측값 정책 분리 |
| 불안정 비율 | z-score인 `avg_resource_price`를 여러 비율의 분모로 사용 | 0 근처에서 값 폭발 후 0 치환 | 해당 비율 제거 |
| 미래 거시변수 | 과거 backtest에서 평가기간의 실제 거시값 사용 | 실제 예측시점에 알 수 없는 값 사용 | 마지막 훈련값 고정 |
| XGBoost | 거시 특징 기반 회귀, seed 42 | 시계열 target lag는 없으나 전처리 누출 영향 | 새 안전 특징으로 재학습 |
| DeepAR | 단일 시계열 + 동적 특징, PyTorch/GluonTS | 평가기간 실제 거시값 사용, seed가 전 라이브러리에 고정되지 않음 | 안전 특징, Python/NumPy/Torch seed, deterministic trainer 적용 |
| 앙상블 | 같은 segment의 전체 rolling 결과로 역-MAE 가중치 | 평가하려는 시점 이후 origin이 가중치에 포함될 수 있음 | 각 origin보다 과거 결과만 사용 |
| rolling-origin | 3개월 holdout × 최대 3 origins | 분포 추정과 horizon 비교에 부족 | 8 origins × 3/6/12개월 |
| 기준모형 | 없음 | 복잡모형 우위 판단 불가 | Last, Seasonal, MA6 추가 |
| 지표 | MAE 중심, holdout에 MAPE/RMSE/R² | 0에서 MAPE 불안정, horizon/segment 비교 제한 | MAE, RMSE, WAPE, sMAPE, MASE |
| 산출물 | Excel 중심, 장기 DeepAR·SHAP 그림 | origin별 원자료와 의사결정 연결 부족 | long-format CSV, 요약 CSV, 7개 그래프 |
| 테스트 | 특징·앙상블·튜닝 위주 21개 | 누출·공통분할·최적화·통합실행 미검증 | 의사결정 테스트 10개 추가 |
| README 일치 | 3 origins·3개월 holdout·기존 장기예측 설명 | 완료 기준 및 새 목적과 불일치 | 새 기본 CLI와 확정 결과로 개편 |

## 미래 참조 가능 변수 목록

1. 전체 기간 평균·표준편차로 계산한 세 원자재 z-score
2. 평가기간의 실제 금리·소득·원자재 값을 사용한 XGBoost 설명변수
3. 평가기간의 실제 동적 거시변수를 사용한 DeepAR
4. 현재 origin의 오차 또는 이후 origin의 오차를 포함해 계산할 수 있었던 앙상블 가중치
5. 전체 origin 결과로 선택한 모형을 최신 의사결정 예측에 사용하는 경우

새 기본 경로는 1~5를 모두 차단한다. 최적화용 최신 3개월 예측의 모형도 앞선 7개 origin만으로 선택한다.

## 재현성과 테스트 기준선

- 변경 전: `pytest -q` → 21 passed
- 난수: 합성 생성기, XGBoost, Python, NumPy, Torch 모두 42
- DeepAR 동일 입력 2회 실행: 예측 배열 완전 일치, 최대 절대차 0
- 새 테스트는 누출 불변성, Seasonal Naive, 공통 origin/horizon, 예측행 정렬, WAPE, 모든 최적화 제약, 규제 0배분, 비딥/DeepAR 재현성, 전체 smoke run을 포함한다.

## 결론

기존 성능표는 전처리 누출 수정 전 결과이므로 최종 성능으로 사용하지 않는다. 새 최종 결과는 `outputs/decision_system/forecast_metrics_summary.csv`만을 기준으로 하며, 기존 Excel 결과와 git 이력에 남은 이전 버전의 `docs/REPORT.md`는 감사 추적용 과거 실험으로 취급한다.

## 2차 감사

감사일: 2026-10-02  
범위: `src/chile_forecast/evaluation.py`의 `LeakageSafeEnsemble` 가중치

### 발견

1차 수정 후 origin (k)의 가중치는 origin (1,…,k-1)의 MAE로 계산했지만, 저장한 값이 각 origin의 **12개월 평가창 전체** MAE였다. origin 간격이 3개월이므로 앞선 origin의 평가창이 현재 cutoff 이후로 최대 9개월 겹친다. 예를 들어 origin 7(cutoff 2024-02-29, 평가창 2024-03~2025-02)의 전체 오차가 origin 8(cutoff 2024-05-31)의 가중치에 들어가, 2024-06~2025-02의 실측값이 가중치에 영향을 줬다.

### 조치

- origin마다 (예측 날짜, DeepAR 절대오차, XGBoost 절대오차)를 저장하고, `_visible_history(records, cutoff)`로 날짜가 현재 cutoff 이하인 오차만 골라 origin별 MAE를 계산한다. 관측된 날짜가 없는 origin은 제외하고, 기록이 없으면 가중치 0.5를 쓴다.
- 오차는 기존과 같이 클리핑 전 예측값으로 계산한다. 하이퍼파라미터, seed, origin 수·간격, horizon과 레거시 모듈은 변경하지 않았다.
- 회귀 테스트 `tests/test_ensemble_weights.py` 3개를 추가했다: cutoff 이후 날짜에 매우 큰 오차를 넣어도 결과 불변, cutoff 이전 날짜만 반영되고 관측 날짜가 없는 origin은 제외, 빈 기록이면 0.5.

### 영향

DeepAR·XGBoost·기준모형 예측값은 수정 전과 완전히 같고, 앙상블 예측만 바뀌었다.

| 4개 차급 평균 앙상블 WAPE | 3개월 | 6개월 | 12개월 |
|---|---:|---:|---:|
| 수정 전 | 13.80 | 14.48 | 14.98 |
| 수정 후 | 13.90 | 14.63 | 15.13 |

- 앙상블은 여전히 세 horizon 모두 평균 1위다. 12개월 Seasonal Naive 대비 개선율은 18.9%에서 18.1%로 낮아졌다.
- 12개월 차급별 앙상블 WAPE는 B-Sedan 12.17→12.21, B_HB 12.59→12.64, SUV-A 16.32→16.46, SUV-B 18.85→19.22이며 차급별 최선 모형은 바뀌지 않았다.
- 6개월 SUV-B의 결론이 바뀌었다. 수정 전에는 앙상블(18.66)이 최선이었지만, 수정 후에는 DeepAR(18.78)가 앙상블(18.92)보다 낫다. 3개월에서는 앙상블이 여전히 네 차급 모두 1위이며, B-Sedan에서 XGBoost와의 차이는 0.06%p다.
- 판촉 입력 모형 선택에서 SUV-A가 앙상블에서 DeepAR로 바뀌었다(앞선 7개 origin의 3개월 WAPE: DeepAR 13.149, 앙상블 13.051→13.168). 이에 따라 SUV-A 기준수요가 3809.7에서 3870.0으로 바뀌었다.
- 최적화 결과는 유지됐다. 최적 배분(275 MCLP)과 순증분이익 148.57 MCLP, 균등배분 대비 +7.45%는 같고, 예측수요 비중 전략의 이익은 121.28에서 121.22 MCLP로, 상방 시나리오는 184.11 MCLP·124.59 MCLP에서 185.56 MCLP·125.29 MCLP로 바뀌었다. 모든 제약검사는 계속 참이다.

### 추가 점검

판촉 입력 모형 선택(`promotion_response._latest_forecast_by_target`)에도 같은 문제가 있는지 확인했다. 이 선택은 최신 origin을 제외한 7개 origin의 **3개월** 예측만 사용하고, origin 간격도 3개월이므로 마지막 평가창(origin 7, 2024-03~2024-05)이 최신 cutoff(2024-05-31) 안에서 끝난다. 따라서 cutoff 이후 실측값을 참조하지 않는다.

### 테스트 수

- 2차 감사 전: 23개 (레거시 10개, `test_metrics.py` 3개, `test_decision_system.py` 10개)
- 2차 감사 후: 26개 (`test_ensemble_weights.py` 3개 추가), 모두 통과
