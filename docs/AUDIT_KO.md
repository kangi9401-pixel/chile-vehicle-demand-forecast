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
