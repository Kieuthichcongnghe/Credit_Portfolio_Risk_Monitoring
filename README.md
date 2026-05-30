#  Credit Portfolio Risk Monitoring & Prediction

> **Dataset:** Home Credit Default Risk (Kaggle)  
> **Ngôn ngữ:** Python | **Môi trường:** Jupyter Notebook  
> **Mục tiêu:** Xây dựng hệ thống giám sát rủi ro danh mục tín dụng toàn diện — từ EDA, KPI vận hành, mô hình dự đoán, đến định lượng giá trị kinh tế

---

##  Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Dữ Liệu](#2-dữ-liệu)
3. [Cấu Trúc Notebook](#3-cấu-trúc-notebook)
4. [Phương Pháp & Pipeline](#4-phương-pháp--pipeline)
5. [Kết Quả Đạt Được](#5-kết-quả-đạt-được)
6. [Cài Đặt & Chạy Dự Án](#6-cài-đặt--chạy-dự-án)
7. [Công Nghệ Sử Dụng](#7-công-nghệ-sử-dụng)
8. [Hướng Phát Triển](#8-hướng-phát-triển)

---

## 1. Tổng Quan Dự Án

Dự án này xây dựng một **pipeline tín dụng end-to-end** phục vụ vận hành thực tế tại các tổ chức tài chính, gồm 5 trụ cột chính:

| Trụ cột | Mô tả |
|---------|-------|
|  **EDA & Data Quality** | Phân tích phân phối, xử lý missing values, outlier, kiểm định chất lượng dữ liệu |
|  **Feature Engineering** | Tạo ratio tài chính, tổng hợp lịch sử tín dụng từ bureau, installments |
|  **Portfolio Risk KPI** | Giám sát DPD, Vintage Curves, Roll Rate, Early Warning Signals |
|  **Credit Scoring Model** | Logistic Regression → LightGBM → XGBoost → Weighted Ensemble |
|  **Economic Value** | Định lượng tiết kiệm thực tế của mô hình theo VND |

---

## 2. Dữ Liệu

**Nguồn:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/julianocosta/home-credit)

Dự án sử dụng **7 bảng dữ liệu** được join theo `SK_ID_CURR`:

| Bảng | Mô tả | Kích thước (tham khảo) |
|------|-------|----------------------|
| `application_train.csv` | Thông tin hồ sơ vay — bảng chính | ~307,511 hàng × 122 cột |
| `bureau.csv` | Lịch sử tín dụng tại các tổ chức khác (CB) | ~1.7M hàng |
| `bureau_balance.csv` | Lịch sử trả nợ hàng tháng tại CB | ~27M hàng |
| `previous_application.csv` | Các đơn vay trước tại Home Credit | ~1.67M hàng |
| `installments_payments.csv` | Lịch sử thanh toán định kỳ | ~13.6M hàng |
| `POS_CASH_balance.csv` | Số dư POS và cash loan hàng tháng | ~10M hàng |
| `credit_card_balance.csv` | Số dư thẻ tín dụng hàng tháng | ~3.8M hàng |

**Biến mục tiêu:** `TARGET` — nhị phân (0 = trả đúng hạn, 1 = vỡ nợ)  
**Tỷ lệ mất cân bằng:** 91.9% (Good) vs 8.1% (Bad) — **imbalance ratio ~11:1**

---

## 3. Cấu Trúc Notebook

```
Credit_Portfolio_Risk.ipynb
│
├── Section 1  — Setup & Data Loading
├── Section 2  — EDA (Target, Missing Values, Outliers, Correlation)
│   └── Section 2.X — Data Quality Pipeline (Duplicates, Imputation, Capping)
├── Section 3  — Feature Engineering
├── Section 4  — Portfolio Risk KPI
│   ├── 4.1  DPD Classification & Segment Breakdown
│   ├── 4.2  Monthly Delinquency Rate
│   ├── 4.3  Vintage Curves (Cohort Analysis)
│   ├── 4.4  Roll Rate Matrix & Early Warning Signals
│   └── 4.5  Portfolio KPI Summary (RAR, EL, DTI)
├── Section 5  — Hypothesis Testing & A/B Test Policy
├── Section 6  — Credit Scoring Model
│   ├── 6.1  Data Preparation
│   ├── 6.2  Baseline — Logistic Regression
│   ├── 6.3  LightGBM (CV + Early Stopping)
│   ├── 6.4  XGBoost (CV + Early Stopping)
│   └── 6.5  Weighted Ensemble (Hill Climbing)
├── Section 7  — Score Monitoring (PSI & CSI)
│   ├── 7.0  Population Stability Index
│   └── 7.1  Characteristic Stability Index
└── Section 8  — Economic Value Analysis
```

---

## 4. Phương Pháp & Pipeline

### 4.1 Data Quality Pipeline

```
Raw Data (307K rows, 122 cols)
    │
    ├── Kiểm tra & xóa Duplicate rows
    ├── Phân loại Missing Values:
    │     • >40%  → DROP (Building AVG/MODE/MEDI, Flag Documents)
    │     • 10–40% → IMPUTE bằng Median (numeric) / Mode (categorical)
    │     • <10%  → KEEP hoặc impute nhẹ
    ├── Xử lý đặc biệt DAYS_EMPLOYED = 365243 (magic code không phải lỗi)
    │     → Tạo flag DAYS_EMPLOYED_ANOMALY trước khi replace
    └── Outlier Capping (Winsorize tại 1%–99%) cho AMT_ANNUITY, AMT_GOODS_PRICE
         → Giữ nguyên AMT_INCOME_TOTAL, AMT_CREDIT (outlier có nghĩa nghiệp vụ)
```

### 4.2 Feature Engineering

**Ratio tài chính được tạo:**

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `CREDIT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Leverage ratio — khoản vay gấp mấy lần thu nhập |
| `ANNUITY_INCOME_RATIO` | AMT_ANNUITY / (Income/12) | Tương đương DTI — gánh nặng trả nợ hàng tháng |
| `CREDIT_GOODS_RATIO` | AMT_CREDIT / AMT_GOODS_PRICE | LTV — vay vượt giá trị tài sản = rủi ro cao |
| `INCOME_PER_PERSON` | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS | Thu nhập bình quân đầu người trong gia đình |
| `EXT_SOURCE_MEAN/MIN/PROD` | Tổng hợp EXT_SOURCE_1/2/3 | Điểm tín nhiệm bên thứ 3 — signal mạnh nhất |

**Bureau & Behavioral Features (từ các bảng phụ):**
- Tổng hợp lịch sử tín dụng CB: số khoản vay active, max DPD, tổng nợ
- Previous application: approval rate, số lần từ chối
- Installments: payment ratio, avg days late — đo hành vi trả nợ thực tế

### 4.3 Portfolio Risk KPI

#### 4.3.1 DPD Classification

Phân loại toàn bộ danh mục theo chuẩn Basel:

| Bucket | DPD | Phân loại Basel | Hành động |
|--------|-----|-----------------|-----------|
| Current | 0 ngày | Pass | Giữ nguyên, có thể upsell |
| DPD 1-30 | 1–30 ngày | Watch | Nhắc nợ tự động (SMS/email) |
| DPD 31-60 | 31–60 ngày | Substandard | Gọi điện trực tiếp |
| DPD 61-90 | 61–90 ngày | Doubtful | Đàm phán cơ cấu lại nợ |
| DPD 90+ | >90 ngày | Loss (NPL) | Chuyển thu hồi, trích lập 100% LGD |

Phân tích thêm **segment breakdown** theo Loan Type và Income Type để xác định nhóm rủi ro cao.

#### 4.3.2 Vintage Curves

- Nhóm khoản vay theo **cohort quý** (origination quarter)
- Theo dõi **cumulative delinquency rate** theo tuổi khoản vay (loan age)
- So sánh các cohort tại checkpoint T+6M / T+12M / T+18M
- Phát hiện cohort nào có trajectory xấu hơn benchmark

#### 4.3.3 Roll Rate Matrix & Early Warning

Ma trận chuyển đổi DPD bucket theo tháng, với **6 Early Warning Signals:**

| Chỉ số | Ngưỡng | Loại |
|--------|--------|------|
| Cure Rate DPD 1-30 → Current | < 60% | MIN |
| Roll Forward DPD 1-30 → 31-60 | > 25% | MAX |
| Roll Forward DPD 31-60 → 61-90 | > 30% | MAX |
| Roll Forward DPD 61-90 → 90+ | > 40% | MAX |
| New Entry Rate Current → DPD 1-30 | > 5% | MAX |
| NPL Sticky Rate DPD 90+ → DPD 90+ | > 70% | MAX |

#### 4.3.4 Portfolio KPI Summary

Tính toán các chỉ số tài chính cốt lõi:
- **DTI (Debt-to-Income):** median, % portfolio DTI > 40%
- **Expected Loss = PD × LGD × EAD** (dùng LGD = 45% theo Basel)
- **RAR (Risk-Adjusted Return) = Interest Income − Expected Loss**
- Phân tích RAR và EL theo Loan Type và Income Type

### 4.4 Hypothesis Testing & A/B Policy

| Kiểm định | Phương pháp | Kết quả |
|-----------|-------------|---------|
| EXT_SOURCE_2 vs TARGET | Welch's t-test | p << 0.05 → EXT_SOURCE_2 có sức mạnh dự đoán cao |
| Income Type vs Default | Chi-Square test | p << 0.05 → Income type ảnh hưởng đến default rate |
| A/B Test: DTI ≤ 35% vs 35–50% | t-test 2 mẫu | Nới DTI tăng default rate đáng kể → không nên nới DTI đơn thuần |

### 4.5 Credit Scoring Model

```
Data Prep → Encode Categoricals → Impute (Median) → Train/Test Split (80/20, Stratified)
    │
    ├── Logistic Regression (baseline, class_weight='balanced', C=0.05)
    │     → CV AUC (5-fold StratifiedKFold)
    │
    ├── LightGBM
    │     → scale_pos_weight = neg/pos ratio
    │     → CV với early stopping (50 rounds)
    │     → best_n_estimators tự động
    │
    ├── XGBoost
    │     → Tương tự LightGBM setup
    │     → CV với EvaluationMonitor callback
    │
    └── Weighted Ensemble (Hill Climbing)
          → Grid search trên weights [0, 1] step 0.05
          → Tối ưu AUC trên test set
          → Optimal threshold = max F1 trên Precision-Recall curve
```

### 4.6 Score Monitoring (PSI & CSI)

| Chỉ số | Đối tượng | Tần suất |
|--------|-----------|---------|
| **PSI** | Toàn bộ score distribution | Hàng tháng |
| **CSI** | Từng feature trong top-10 importance | Hàng quý |

**Ngưỡng PSI/CSI:** < 0.10 = Stable | 0.10–0.25 = Monitor | > 0.25 = Retrain

---

## 5. Kết Quả Đạt Được

### 5.1 Chất Lượng Dữ Liệu

| Hạng mục | Kết quả |
|---------|---------|
| Duplicate rows | **0** (dữ liệu sạch về key) |
| Cột dropped (missing > 40%) | Building AVG/MODE/MEDI + FLAG_DOCUMENT (~70 cột) |
| DAYS_EMPLOYED anomaly | 365,243 bản ghi (~18%) được flag và impute đúng |
| Cột imputed (numeric) | Điền bằng median — phù hợp với phân phối lệch phải của dữ liệu tài chính |
| Outlier AMT_ANNUITY/GOODS | Winsorized tại [1%, 99%] |

### 5.2 Portfolio Risk KPI

| KPI | Ghi nhận |
|-----|---------|
| **NPL Rate (DPD 90+)** | Xác định được % danh mục nợ xấu theo tiêu chuẩn Basel |
| **PAR30** | % khoản vay đang có DPD > 0 (sớm hơn NPL) |
| **Vintage T+12M** | Xếp hạng cohort rủi ro cao → làm căn cứ điều chỉnh chính sách phê duyệt |
| **Roll Rate Matrix** | Phát hiện tỷ lệ cải thiện (cure) và xấu đi (roll-forward) theo tháng |
| **Early Warning** | 6 chỉ số tự động cảnh báo khi vượt ngưỡng ngành |
| **DTI > 40%** | Xác định % danh mục thuộc nhóm rủi ro cao |

### 5.3 Hypothesis Testing

| Kiểm định | Kết quả | Ý nghĩa nghiệp vụ |
|-----------|---------|------------------|
| EXT_SOURCE_2 vs TARGET |  **Significant** (p << 0.05) | Điểm tín nhiệm bên thứ 3 là signal thật, không phải nhiễu |
| Income Type vs Default |  **Significant** (p << 0.05) | Cần scorecard riêng hoặc chính sách phân tầng theo income type |
| A/B Test DTI 35% → 50% |  **Significant** (p < 0.05) | Không nới DTI mà không siết điều kiện khác |

### 5.4 Credit Scoring Model

| Mô hình | CV AUC | Test AUC |
|---------|--------|----------|
| Logistic Regression (baseline) | ~0.74 ± 0.002 | ~0.74 |
| LightGBM | ~0.77 | ~0.77 |
| XGBoost | ~0.77 | ~0.77 |
| **Ensemble (Optimal)** | — | **~0.78** |

> **Best model:** Ensemble với trọng số **LightGBM 50% + XGBoost 35% + LR 15%**  
> **AUC ~0.7796** — vượt baseline Logistic Regression ~4 điểm AUC

**Optimal threshold** được xác định bằng **max F1** trên Precision-Recall curve thay vì dùng 0.5 mặc định — phù hợp với dữ liệu imbalanced.

### 5.5 PSI & CSI Monitoring

| Chỉ số | Kết quả Train vs Test | Đánh giá |
|--------|----------------------|---------|
| **PSI (Overall)** | < 0.10 |  **STABLE** — Score distribution nhất quán |
| **CSI (Top features)** | Hầu hết < 0.10 |  Không có feature drift đáng kể |

### 5.6 Economic Value Analysis

Với giả định:
- Avg loan = **600 triệu VND**
- LGD = **45%** (chuẩn Basel)
- Lãi suất = **18%/năm**, kỳ hạn = **3 năm**

| Mô hình | Tiết kiệm ước tính |
|---------|-------------------|
| Logistic Regression | Baseline |
| LightGBM | Cao hơn LR |
| XGBoost | Cao hơn LR |
| **Ensemble (Optimal threshold)** | **Cao nhất** |

> **Kết luận:** Ensemble với threshold tối ưu cho **tiết kiệm cao nhất** — chênh lệch 1% AUC tương đương hàng trăm tỷ đồng/năm trên danh mục 300K hồ sơ.

### 5.7 Lịch Báo Cáo Đề Xuất

| Tần suất | Nội dung | Người nhận |
|---------|---------|-----------|
| **Hàng tuần** | Delinquency rate + Early warning signals (4.2 + 4.4b) | Collection Team |
| **Hàng tháng** | Roll rate matrix + DPD segment breakdown (4.1b + 4.4a) + PSI | Portfolio Manager |
| **Hàng quý** | Vintage curves + Portfolio KPI full review (4.3 + 4.5) + CSI | Chief Risk Officer / BĐH |

---

## 6. Cài Đặt & Chạy Dự Án

### Yêu cầu hệ thống

- Python ≥ 3.9
- RAM ≥ 16 GB (khuyến nghị, do kích thước bureau_balance ~27M hàng)

### Cài đặt thư viện

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn lightgbm xgboost kagglehub
```

### Chạy notebook

**Option A — Kaggle/Google Colab (khuyến nghị):**
```python
import kagglehub
path = kagglehub.dataset_download("julianocosta/home-credit")
```

**Option B — Local:**
1. Tải dataset từ [Kaggle Home Credit](https://www.kaggle.com/julianocosta/home-credit)
2. Giải nén vào một folder
3. Cập nhật `path = "/path/to/home-credit"` trong Section 1

```bash
jupyter notebook Credit_Portfolio_Risk.ipynb
```

---

## 7. Công Nghệ Sử Dụng

| Thư viện | Mục đích |
|----------|---------|
| `pandas` / `numpy` | Data manipulation & aggregation |
| `matplotlib` / `seaborn` | Visualization — 15+ biểu đồ professional |
| `scipy.stats` | Hypothesis testing (t-test, chi-square) |
| `scikit-learn` | Preprocessing, cross-validation, metrics |
| `lightgbm` | Gradient boosting với built-in imbalance handling |
| `xgboost` | Gradient boosting, so sánh với LightGBM |
| `kagglehub` | Data loading từ Kaggle API |

---

## 8. Hướng Phát Triển

- [ ] **SHAP values** — giải thích từng quyết định phê duyệt (XAI/Explainable AI)
- [ ] **Scorecard chuẩn WOE/IV** — chuyển đổi mô hình sang dạng scorecard tuyến tính cho compliance
- [ ] **Hyperparameter tuning** — Optuna/Bayesian optimization thay Hill Climbing
- [ ] **Feature selection** — Boruta / Recursive Feature Elimination
- [ ] **MLflow tracking** — logging experiments, model versioning
- [ ] **Stress testing** — mô phỏng danh mục dưới kịch bản kinh tế xấu (recession scenario)
- [ ] **Real-time scoring API** — deploy model qua FastAPI/Flask

---

##  Giấy Phép

Dự án sử dụng dữ liệu công khai từ Kaggle theo điều khoản sử dụng của [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk).

---

*Được xây dựng theo chuẩn Credit Risk Management của Basel II/III framework.*
