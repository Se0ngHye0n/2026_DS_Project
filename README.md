# 2025 NESS Statathon  
## Insurance Fraud Detection using Feature Engineering

---

## Project Overview

보험 사기(Fraudulent Claim)는 보험사에 직접적인 재무 손실을 초래하며,  
조기 탐지 실패 시 손실 규모가 크게 증가한다.

또한 사기 조사는 인력과 시간이 많이 소요되기 때문에,  
**효율적인 탐지 모델을 통해 조사 우선순위를 설정하는 것이 핵심 과제**이다.

---

## Project Objective

본 프로젝트의 목표는 다음과 같다:

- 차량 보험 데이터를 기반으로 **사기 여부 예측 모델 구축**
- Fraud에 영향을 미치는 **핵심 변수(Key Drivers) 분석**
- 실제 운영 환경을 고려한 **Top-K 기반 탐지 성능 개선**
- 보험사의 의사결정에 활용 가능한 **해석 가능한 모델 구축**

---

## Key Research Questions

- Feature engineering이 실제 성능 향상에 기여하는가?
- 어떤 feature 유형이 fraud detection에 가장 중요한가?
- 제한된 조사 리소스 환경에서 Recall@TopK를 어떻게 개선할 수 있는가?

---

## Dataset

- Source: Travelers Insurance Claim Data (Kaggle Statathon)
- 기간: 2015 ~ 2016
- 데이터 규모: 약 18,000건
- 구조:
  - `train_2025.csv` (fraud 포함)
  - `test_2025.csv`
  - `sample_submission.csv`

- Target:
  - `fraud = 1` → 사기 청구
  - `fraud = 0` → 정상 청구

### 주요 변수

- 운전자 정보: age, gender, income
- 사고 정보: accident_site, claim_est_payout
- 차량 정보: vehicle_price, weight
- 행동 정보: past_num_of_claims, witness 여부 등

---

## EDA Summary

### 1. 데이터 품질
- 일부 결측 존재 (marital_status, witness)
- 구조적 이상 데이터 존재 (0.018%)

### 2. 클래스 불균형
- 정상: 15,152 / 사기: 2,848  
- 약 **15% 사기율 (현실보다 높음)**

👉 대응 전략:
- class_weight / SMOTE 고려
- threshold 조정 필요

---

## Data Preprocessing Strategy

- Missing:
  - Numeric → 그대로 유지 + indicator
  - Categorical → Unknown 처리
- Outlier:
  - Clipping (log transform 사용 X)
- Encoding:
  - Label Encoding
  - K-Fold Target Encoding (Leakage 방지)
- Leakage 방지:
  - 사고 이후 생성 변수 제거

👉 금융 데이터 특성상 **결측 자체도 중요한 signal로 판단**

---

## Feature Engineering Strategy

### Stage 1

| Dataset | Feature 수 | 특징 | ROC-AUC |
|--------|----------|------|--------|
| V0 | 24 | 기본 변수 | 0.6570 |
| V1 | 64 | Aggregation, ratio | 0.6726 |
| V2 | 92 | interaction + behavioral | 0.6937 |

👉 Feature 추가는 성능 향상에 기여

---

### Stage 2

#### V3
- 비율 / scale 기반 feature 추가
- 변수 조합 강화

#### V4
- Top-K 최적화 feature
- rank / percentile / flag 중심

| Dataset | ROC-AUC | Recall@Top10% |
|--------|--------|--------------|
| V2 | 0.6937 | 0.2156 |
| V3 | 0.6889 | 0.2072 |
| V4 | 0.6908 | 0.2177 |

👉 성능 개선은 제한적 → Feature Cleaning 필요

---

## Feature Cleaning

- 중요도 < 1.0 제거
- Feature 수: 152 → 131

| Dataset | ROC-AUC | PR-AUC |
|--------|--------|--------|
| V4 | 0.6908 | 0.2767 |
| Cleaning | 0.6921 | 0.2820 |

👉 Noise 제거 → 성능 소폭 개선

---

## Model Comparison

| Model | ROC-AUC | PR-AUC | Recall@Top10% |
|------|--------|--------|--------------|
| CatBoost | **0.7018** | **0.2928** | **0.2261** |
| LightGBM | 0.6934 | 0.2830 | 0.2219 |
| RandomForest | 0.6829 | 0.2719 | 0.2145 |
| XGBoost | 0.6464 | 0.2408 | 0.1847 |

👉 CatBoost 최종 선택

---

### Statistical Validation

- DeLong Test 수행
- p-value = 0.0011 < 0.05

👉 **CatBoost 성능 향상은 통계적으로 유의미**

---

## Feature Selection

| Feature Set | ROC-AUC | Recall@Top10% |
|------------|--------|--------------|
| Top20 | 0.7065 | 0.2293 |
| Top30 | **0.7075** | **0.2310** |
| Top40 | 0.7066 | 0.2275 |

👉 최종: **Top30 Feature 선택**

---

## Final Model

- Dataset: Top30
- Model: CatBoost
- Threshold 조정

| Threshold | Recall |
|----------|--------|
| 0.5 | 0.6598 |
| 0.45 | **0.7763** |

👉 운영 환경 고려 시 **threshold tuning 매우 중요**

---

## Modeling

- Model: CatBoost
- CV: Stratified K-Fold (5-fold)
- Evaluation: OOF 기반

### 주요 설정
```python
iterations = 5000
learning_rate = 0.03
depth = 4
l2_leaf_reg = 5
subsample = 1.0
```

---

## Evaluation Metrics
- ROC-AUC (Primary)
- PR-AUC
- F1-score
- Recall@Top10%

---

## Key Insights

### 중요 변수

- accident_parking → 사고 유형 핵심
- age_of_driver → 특정 구간 영향
- witness_present_ind → fraud 판단 핵심
- high_education_ind → 사회적 signal
- unstable_lifestyle → 종합 리스크

### 핵심 결론

👉 **“행동 + 환경 + 변수 조합”이 fraud 판단의 핵심**

---

## SHAP Interpretation

- 젊은 운전자 → fraud ↑
- 목격자 없음 → fraud ↑
- 고학력 → fraud ↓

👉 모델은 단순 rule이 아닌 **복합 패턴 기반 구조**

---

## Scorecard (Business Application)

- 모델 결과를 점수로 변환
- 고객별 fraud 위험도 직관적 비교
- 실무 의사결정 지원

---

## Limitations & Future Work

### 한계
1. 비현실적인 fraud 비율 (약 15%)
2. SHAP → 인과관계 설명 불가

### 개선 방향
- 실제 fraud 비율로 재샘플링
- Threshold 재설정
- Causal Analysis (DAG)
- Rule 기반 해석 강화

---

## Project Structure

```bash
📁 2025-ness-statathon
│
├── 📁 .devcontainer/
│
├── 📁 Data/
│   ├── 📄 train_2025 / test_2025 / sample_submission.csv
│   ├── 📄 train_baseline.csv
│   ├── 📄 train_V1 ~ V4 / test_V1 ~ V4.csv
│   ├── 📄 train_cleaning / test_cleaning.csv
│   ├── 📄 train_selection_top* / test_selection_top*.csv
│   ├── 📄 test_pred_*.csv
│   └── 📄 submission.csv
│
├── 📁 Feature/
│   ├── 📜 EDA.ipynb
│   ├── 📜 Feature_Baseline.ipynb
│   ├── 📜 Feature_V1.ipynb
│   ├── 📜 Feature_V2.ipynb
│   ├── 📜 Feature_V3.ipynb
│   ├── 📜 Feature_V4.ipynb
│   ├── 📜 Feature_Cleaning.ipynb
│   └── 📜 Feature_Selection.ipynb
│
├── 📁 Modeling/
│   ├── 📜 Modeling_Baseline.ipynb
│   ├── 📜 Modeling_V1.ipynb
│   ├── 📜 Modeling_V2.ipynb
│   ├── 📜 Modeling_V3.ipynb
│   ├── 📜 Modeling_V4.ipynb
│   ├── 📜 Modeling_Cleaning.ipynb
│   ├── 📜 Modeling_Final.ipynb
│   ├── 📜 Model_Comparison.ipynb
│   ├── 📜 Model_Tuning.ipynb
│   └── 📜 Model_Ensemble.ipynb
│
├── 📄 Dashboard.py
├── 📜 Score_Card.ipynb
├── 📄 QR_Dashboard.png
│
├── 📄 requirements.txt
├── 📄 runtime.txt
└── 📄 README.md
```

## Project Flow

### 1. EDA

- EDA.ipynb

### 2. Feature Engineering

- Feature_Baseline.ipynb
- Modeling_Baseline.ipynb
- Feature_V1.ipynb
- Modeling_V1.ipynb
- Feature_V2.ipynb
- Modeling_V2.ipynb
- Feature_V3.ipynb
- Modeling_V3.ipynb
- Feature_V4.ipynb
- Modeling_V4.ipynb

### 3. Feature Cleaning

- Feature_Cleaning.ipynb
- Modeling_Cleaning.ipynb

### 4. Model Comparison

- Feature_Cleaning.ipynb
- Modeling_Cleaning.ipynb

### 5. Feature Selection

- Feature_Selection.ipynb

### 6. Model Tuning

- Modeling_Tuning.ipynb

### 7. Model Ensemble

- Modeling_Ensemble.ipynb

### 8. Final Modeling

- Modeling_Final.ipynb

### 9. Visualization

- Score_Card.ipynb
- Dashboard.py

---

## Team
- 문성현 (2020320012)
- 이은서 (2026951021)
- 전철우 (2022320070)

---

## References
[Kaggle Competition/2025-ness-statathon](https://www.kaggle.com/competitions/2025-ness-statathon)
