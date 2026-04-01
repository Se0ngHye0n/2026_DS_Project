# 2025 NESS Statathon  
## Insurance Fraud Detection using Feature Engineering

---

## Objective

보험 사기(Fraudulent Claim)는 보험사에 직접적인 재무 손실을 초래하며,  
효율적인 탐지 시스템 구축은 비용 절감 및 운영 효율성 측면에서 매우 중요하다.

본 프로젝트의 목표는:

- 차량 보험 청구 데이터를 기반으로 사기 여부를 예측하는 모델 구축
- Feature Engineering이 모델 성능에 미치는 영향 분석
- 실제 운영 환경을 고려한 **Recall@TopK 기반 탐지 성능 개선**

---

## Key Research Questions

- Feature engineering이 실제 성능 향상에 기여하는가?
- 어떤 feature 유형이 fraud detection에 가장 중요한가?
- 제한된 조사 리소스 환경에서 Recall@TopK를 어떻게 개선할 수 있는가?

---

## Dataset

- Source: Travelers Insurance Claim Data (Kaggle Statathon)
- 기간: 2015 ~ 2016
- 구조:
  - `train_2025.csv` (fraud 포함)
  - `test_2025.csv`
  - `sample_submission.csv`

- Target:
  - `fraud = 1` → 사기 청구
  - `fraud = 0` → 정상 청구

---

## Project Structure
```bash
📁 2025-ness-statathon
┣ 📜 EDA.ipynb
┣ 📜 Feature_Baseline.ipynb
┣ 📜 Feature_V1.ipynb
┣ 📜 Feature_V2.ipynb
┣ 📜 Modeling_Baseline.ipynb
┣ 📜 Modeling_V1.ipynb
┣ 📜 Modeling_V2.ipynb
┣ 📄 train_2025.csv
┣ 📄 test_2025.csv
┣ 📄 sample_submission.csv
┗ 📄 README.md
```

---

## Data Preprocessing

- **Missing Value Handling**
  - Categorical → "Unknown"
  - Numeric → Median + Missing Indicator

- **Outlier Handling**
  - Log Transform
  - Clipping

- **Encoding**
  - Label Encoding
  - K-Fold Target Encoding (Leakage 방지)

- **Leakage Prevention**
  - 사고 이후 생성된 변수 제거
  - K-Fold 기반 encoding 적용

---

## Feature Engineering Strategy

### Baseline (24 features)
- 기본 demographic / policy 정보

---

### V1 (64 features)
- Claim 관련 feature
- Aggregation
- 일부 ratio feature

---

### V2 (92 features)

- **Ratio Features**
  - claim / income
  - claim / policy duration

- **Behavior Features**
  - claim frequency
  - 사고 발생 패턴

- **Interaction Features**
  - 변수 간 결합

---

## Modeling

- Model: LightGBM
- CV: Stratified K-Fold (5-fold)
- Evaluation: OOF 기반

### 주요 설정
```python
learning_rate = 0.03
num_leaves = 64
n_estimators = 10000
scale_pos_weight = neg / pos
```

---

## Evaluation Metrics
- ROC-AUC (Primary)
- PR-AUC
- F1-score
- Recall@Top10%

---

## How to Run

### 1. EDA

- EDA.ipynb

### 2. Feature Engineering

- Feature_Baseline.ipynb
- Feature_V1.ipynb
- Feature_V2.ipynb

### 3. Modeling

- Modeling_Baseline.ipynb
- Modeling_V1.ipynb
- Modeling_V2.ipynb

---

## Team
- 문성현 (2020320012)
- 이은서 (2026951021)
- 전철우 (2022320070)

---

## References
[Kaggle Competition/2025-ness-statathon](https://www.kaggle.com/competitions/2025-ness-statathon)
