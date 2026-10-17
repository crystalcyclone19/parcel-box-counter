# สถาปัตยกรรมระบบ Automated Training & Model Selection (Multi-class Classification)

เอกสารฉบับนี้อธิบายรายละเอียดการออกแบบและโครงสร้างของระบบ **End-to-End Automated Training & Model Selection Pipeline** สำหรับโปรเจกต์ตรวจนับกล่องพัสดุ (Parcel Box Counter) ในเวอร์ชัน **Multi-class Classification** ซึ่งรวมกระบวนการ **Data Augmentation**, **Image Preprocessing**, **SVD Feature Extraction**, **Statistical Edge Features**, **Hyperparameter Grid Search** และ **Model Registry & Deployment** เข้าด้วยกันเป็นระบบอัตโนมัติ 100%

---

## 1. วัตถุประสงค์และภาพรวมระบบ (System Objectives & Architecture)

### 1.1 วัตถุประสงค์
1. **Multi-class Classification Paradigm**: เปลี่ยนเป้าหมายจากการประมาณค่าต่อเนื่อง (Regression) ซึ่งมักให้คำตอบแบบจุดทศนิยม ($2.4$ กล่อง) มาเป็นการจำแนกคลาสจำนวนเต็ม **1 ถึง 9 กล่อง** ด้วย Softmax + Cross-Entropy Loss ซึ่งสอดคล้องกับธรรมชาติของจำนวนกล่องจริง
2. **Hybrid Feature Representation (67 มิติ)**: ผสานเวกเตอร์ SVD 64 มิติ เข้ากับฟีเจอร์ทางสถิติและเรขาคณิตของเส้นขอบ (Edge Statistics) 3 มิติ ได้แก่ สัดส่วนพื้นที่ขอบ (Total Edge Area), จำนวนคอนทัวร์ (Log-Contour Count) และอัตราส่วนโครงร่าง (Bounding Box Aspect Ratio)
3. **Data Augmentation Integration**: ขยายชุดข้อมูล Train Set ด้วย 9 รูปแบบอิสระ (ตัดการหมุน $\pm 15^\circ$ ออก คงเหลือ พลิกภาพแนวนอน, ปรับแสง $\pm 25\%$, ปรับ Contrast $\pm 25\%$, Shear แกน X/Y $\pm 10\%$) รวมเป็น 10 เท่า เพื่อลดปัญหา Overfitting โดยคงชุด Validation และ Test ไว้สะอาด 100% ปราศจาก Augmentation
4. **Automated Grid Search & Checkpointing**: สำรวจพื้นที่พารามิเตอร์แบบ Full Grid Search (108 Trials) ครอบคลุม `clip_percentile`, `learning_rate`, `batch_size` และ `hidden_dim` พร้อมระบบ Early Stopping และ Checkpoint อัตโนมัติ
5. **Multi-Metric Evaluation & Model Registry**: วัดผลด้วย Cross-Entropy Loss, Exact Accuracy, $\pm 1$ Box Tolerance Accuracy, Macro F1-Score, และ Confusion Matrix ($9 \times 9$) พร้อมส่งออก Champion Model Bundle ที่พร้อมใช้งานทันที

---

### 1.2 แผนภาพสถาปัตยกรรมระบบ (End-to-End Pipeline Flow)

```mermaid
flowchart TD
    subgraph S1["Stage 1: Dataset Pairing & Stratified Split"]
        A["Image Files (1_box - 9_box) + labels.csv"] --> B["Data Validation & Matching"]
        B --> C["Stratified Split (70% Train, 15% Val, 15% Test)"]
    end

    subgraph S2["Stage 2: Augmentation & Feature Extraction Pipeline"]
        C -->|"Train Set Only"| D["Data Augmentation (9x Independent Variants) -> 10x Samples"]
        C -->|"Val & Test Sets"| E["Clean / No Augmentation"]
        D --> F["Preprocessing: Grayscale -> Resize 512 -> Blur 5x5 -> Laplacian Edge (3x3)"]
        E --> F
        F --> G1["SVD Features: Resize 128 -> Centering (Train Mean) -> Project (Train Vt) -> 64 dims"]
        F --> G2["Stat Features: Edge Area + Log Contours + Aspect Ratio -> Z-Score Normalization -> 3 dims"]
        G1 & G2 --> H["Combined Feature Matrix Z (67 dims)"]
    end

    subgraph S3["Stage 3: Automated Multi-Model Training Engine (Grid Search)"]
        H --> I["Search Space Definition (108 Configurations)"]
        I --> J["2-Layer MLP Classifier (Input: 67 -> Hidden: H + ReLU + Dropout -> Output: 9 + Softmax)"]
        J --> K["Mini-batch SGD with Gradient Clipping (+-5.0) & Early Stopping (Patience: 30)"]
        K --> L["Validation Cross-Entropy & Metric Computation"]
    end

    subgraph S4["Stage 4: Evaluation, Model Selection & Registry"]
        L --> M["Leaderboard Ranking (Select Best Champion Model)"]
        M --> N["Final Test Set Verification (Unseen Data)"]
        N --> O["Save Model Bundle (best_weights.json, svd_model.npz, pipeline_config.json)"]
        N --> P["Generate Reports (manifest.json, leaderboard.csv, training_report.md)"]
    end

    subgraph S5["Stage 5: Production Inference (predict.py)"]
        Q["User Input Image"] --> R["Full Edge -> SVD+Stat Extraction"]
        S5["Production Inference"]
        R --> S["Champion 2-Layer MLP Forward Pass"]
        S --> T["Predicted Box Count + Probability Distribution + Theoretical Values"]
    end
```

---

## 2. โครงสร้างโฟลเดอร์และไฟล์ของระบบ (Project Layout)

```
parcel-box-counter/
├── dataset/
│   ├── dataset/                           # โฟลเดอร์เก็บภาพจริงแยกตาม 1_box ถึง 8_box
│   └── labels.csv                         # ไฟล์ Label (filename, count) รวม 397 ภาพ
├── documents/
│   ├── model_training_pipeline.md         # สรุปขั้นตอน Pipeline ทางทฤษฎีและคณิตศาสตร์
│   └── auto_training_pipeline_design.md   # เอกสารการออกแบบสถาปัตยกรรมฉบับนี้
├── model/
│   ├── augmentation/                      # [NEW] โมดูล Data Augmentation 11 รูปแบบ
│   │   ├── __init__.py
│   │   └── augment_image.py               # ฟังก์ชันการแปลงภาพแบบอิสระ
│   ├── preprocessing/                     # โมดูล Preprocessing พื้นฐาน
│   │   ├── config.py
│   │   ├── data_ingestion.py
│   │   ├── edge_processing.py             # Laplacian Edge & Percentile Clipping
│   │   ├── svd_features.py
│   │   └── geometric_transform.py         # รองรับ Web UI ในอนาคต
│   ├── mlp/                               # สถาปัตยกรรมโมเดล Classification
│   │   ├── mlp_classifier.py              # [NEW] 2-Layer MLP (ReLU, Dropout, Softmax, Cross-Entropy)
│   │   ├── mlp_regression.py              # โมเดล Regression เดิม (เก็บไว้เป็น Reference)
│   │   └── reset_model.py
│   ├── training/                          # Core Training & Evaluation Engine
│   │   ├── __init__.py
│   │   ├── create_labels.py               # สร้าง labels.csv อัตโนมัติจากโฟลเดอร์ *_box
│   │   ├── dataset.py                     # โหลดข้อมูล, ทำ Augmentation และสกัดฟีเจอร์สถิติ 3 มิติ
│   │   ├── trainer.py                     # Mini-batch SGD, Xavier Init, Gradient Clip, Early Stopping
│   │   ├── tuner.py                       # Full Grid Search Tuner & Feature Cache
│   │   ├── evaluator.py                   # Classification Metrics, Confusion Matrix, Macro F1
│   │   ├── model_registry.py              # จัดการบันทึก Champion Model Bundle
│   │   └── test_imports.py                # ชุด Unit & Integration Test
│   ├── trained_models/                    # [OUTPUT] ที่เก็บผลการทดลองแยกตาม Session
│   │   └── session_<timestamp>/           # โฟลเดอร์บันทึก Artifacts โมเดลที่ดีที่สุด
│   │       ├── best_weights.json          # W1, b1, W2, b2, architecture
│   │       ├── svd_model.npz              # mean, Vt, S, stat_mean, stat_std
│   │       ├── pipeline_config.json       # พารามิเตอร์ที่ใช้เทรน
│   │       ├── manifest.json              # ผลการประเมิน Val/Test และ Metadata
│   │       ├── leaderboard.csv            # ตารางคะแนนทุก Trial
│   │       └── training_report.md         # รายงานสรุปผลอย่างเป็นทางการ
│   ├── run_auto_train.py                  # CLI Master Script สั่งเทรนอัตโนมัติ
│   └── predict.py                         # Wrapper สำหรับเรียกใช้งาน predict.py
├── predict.py                             # [ROOT] Inference Engine สำหรับทำนายภาพเดี่ยว
└── tools/
    └── augment_image.py                   # CLI Tool สำหรับทดสอบดูภาพ Augmentation
```

---

## 3. รายละเอียดการทำงานของแต่ละส่วนประกอบ (Component Specifications)

### 3.1 การเตรียมข้อมูล และ Data Augmentation ([dataset.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/training/dataset.py))
* **การแบ่งชุดข้อมูล**:
  * ข้อมูลจริง 397 ภาพ ถูกแบ่งเป็น Train 278 ภาพ ($70\%$), Validation 60 ภาพ ($15\%$), และ Test 59 ภาพ ($15\%$)
* **Data Leakage Isolation**:
  * การคำนวณ Mean และ Right Singular Vectors ($V^T$) ของ SVD รวมถึงค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของฟีเจอร์สถิติ ($\boldsymbol{\mu}_{\text{stat}}, \boldsymbol{\sigma}_{\text{stat}}$) จะคำนวณจาก **Train Set เท่านั้น**
* **การทำ Data Augmentation**:
  * ใช้กับ Train Set เท่านั้น โดยนำแต่ละภาพมาผ่านการดัดแปลง 11 รูปแบบอิสระ ขยายจำนวนตัวอย่างฝึกฝนเป็น $278 \times 12 = 3,336$ ตัวอย่าง ช่วยป้องกันการ Overfitting ต่อมุมกล้องและแสงในห้องแล็บได้อย่างมีนัยสำคัญ

---

### 3.2 การสกัดฟีเจอร์ผสมผสาน (Hybrid Feature Extraction: 67 มิติ)
ฟีเจอร์เวกเตอร์ที่นำเข้าโมเดลประกอบด้วย:
1. **SVD Latent Features (64 มิติ)**: สกัดโครงสร้างเส้นขอบโดยรวมของภาพผ่าน Randomized SVD บนภาพขนาด $128 \times 128$ พิกเซล
2. **Statistical Edge Features (3 มิติ)**: สกัดลักษณะทางสถิติและรูปทรงเรขาคณิตจาก Edge Map ขนาด $512 \times 512$ พิกเซล:
   * **Total Edge Area**: สัดส่วนพิกเซลเส้นขอบ สะท้อนความหนาแน่นของวัตถุ
   * **Log-Contour Count**: ปริมาณคอนทัวร์ที่ตรวจพบ สะท้อนจำนวนชิ้นพัสดุ
   * **Aspect Ratio**: อัตราส่วนความกว้างต่อความยาวของกล่องรวมทั้งหมด

$$\mathbf{z}_{\text{combined}} = \big[\, \mathbf{z}_{\text{svd}}\,(64) \;\big|\; \mathbf{z}_{\text{stat}}\,(3) \,\big] \quad \in \mathbb{R}^{67}$$

---

### 3.3 สถาปัตยกรรมโมเดลและเครื่องมือค้นหา Hyperparameter ([mlp_classifier.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/mlp/mlp_classifier.py), [tuner.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/training/tuner.py))

#### 1. Search Space สำหรับการค้นหาโมเดลที่ดีที่สุด:
| Parameter Group | Hyperparameter | ค่าที่ทดสอบ (Search Space) | จำนวนค่า | เหตุผล |
|---|---|---|:---:|---|
| **Preprocessing** | `clip_percentile` | `[95.0, 98.0, 99.0]` | 3 | ปรับระดับความคมชัดและการตัดแสงสะท้อนบนเทปกาว |
| **Optimization** | `learning_rate` | `[0.001, 0.005, 0.01, 0.05]` | 4 | ควบคุมอัตราการเรียนรู้ของ SGD |
| | `batch_size` | `[8, 16, 32]` | 3 | ขนาดของ Mini-batch เพื่อสมดุลของ Gradient Variance |
| **Architecture** | `hidden_dim` | `[32, 64, 128]` | 3 | ความจุในการเรียนรู้ความสัมพันธ์ของโครงข่ายประสาทเทียม |

> **จำนวน Configuration ทั้งหมด**: $3 \times 4 \times 3 \times 3 = 108\text{ Trials}$ (Full Grid Search)

#### 2. พารามิเตอร์คงที่ (Fixed Constants):
| Parameter | ค่าที่กำหนด | เหตุผล |
|---|:---:|---|
| `n_svd_features` | **64** | ขนาดมาตรฐานของ Feature Map |
| `n_stat_features` | **3** | Edge Area, Log-Contours, Aspect Ratio |
| `input_dim` | **67** | $64 + 3 = 67$ |
| `n_classes` | **8** | จำแนกกล่องจำนวน 1 ถึง 8 กล่อง |
| `dropout_rate` | **0.3** | ป้องกัน Co-adaptation ของ Hidden Neurons |
| `blur_ksize` | **5** | Gaussian Blur เพื่อลดสัญญาณรบกวน |
| `max_epochs` | **500** | จำนวนรอบสูงสุดในการเทรนต่อ 1 Trial |
| `patience` | **30** | เกณฑ์ Early Stopping เมื่อ Val Cross-Entropy ไม่ลดลง |

---

### 3.4 เกณฑ์การประเมินและจัดอันดับโมเดล ([evaluator.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/training/evaluator.py))

1. **เกณฑ์หลัก (Primary Metric - ค่าน้อยที่สุดชนะ)**:
   $$\text{Cross-Entropy Loss} = -\frac{1}{M} \sum_{i=1}^M \ln\big(P(y_i)\big)$$
2. **เกณฑ์เสริม (Secondary Metrics)**:
   * **Exact Accuracy**: ความแม่นยำของการทายจำนวนกล่องถูกต้องตรง $100\%$
   * **$\pm 1$ Box Tolerance Accuracy**: ความแม่นยำเมื่อยอมรับความคลาดเคลื่อนได้ไม่เกิน 1 กล่อง ($|y_{\text{pred}} - y_{\text{true}}| \le 1$)
   * **Macro F1-Score**: ค่าเฉลี่ย Harmonic Mean ของ Precision และ Recall ทุกคลาส เพื่อป้องกันการลำเอียงจาก Class Imbalance
   * **Confusion Matrix ($8 \times 8$)**: ตรวจสอบรูปแบบความผิดพลาดว่าโมเดลมักสับสนระหว่างกล่องจำนวนใด

---

### 3.5 Model Registry & Artifacts Packaging ([model_registry.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/training/model_registry.py))

เมื่อได้ Champion Model ระบบจะประเมินบน Test Set ครั้งสุดท้าย และส่งออกแพ็กเกจที่ประกอบด้วย:
- `best_weights.json`: ค่าน้ำหนัก $W_1, b_1, W_2, b_2$ พร้อม Metadata โครงสร้าง (input_dim=67, hidden_dim, n_classes=9)
- `svd_model.npz`: เมทริกซ์การแปลงฟีเจอร์ (`mean`, $V^T$, `stat_mean`, `stat_std`)
- `pipeline_config.json`: พารามิเตอร์ของระบบ
- `manifest.json`: สรุปข้อมูลผลทดสอบเชิงตัวเลขทั้งหมด
- `leaderboard.csv`: อันดับของทุก Configuration ที่เข้าร่วมการทดลอง
- `training_report.md`: รายงานสรุปผลอย่างละเอียดในรูปแบบ Markdown

---

## 4. ตัวอย่างคำสั่งและการใช้งานจริง (Execution Commands & CLI)

### 4.1 สั่งรันการเทรนอัตโนมัติ (Automated Training)

```bash
# 1. Full Grid Search (108 Trials พร้อม Data Augmentation 10x):
python model/run_auto_train.py \
    --images-dir dataset/dataset \
    --labels dataset/labels.csv \
    --output-dir model/trained_models

# 2. Quick Mode (ทดสอบด้วย Search Space ขนาดเล็ก):
python model/run_auto_train.py \
    --images-dir dataset/dataset \
    --labels dataset/labels.csv \
    --clip-percentiles 98.0 \
    --learning-rates 0.01 \
    --batch-sizes 16 \
    --hidden-dims 64 \
    --no-augment
```

### 4.2 สั่งทำนายภาพเดี่ยว (Inference with predict.py)

```bash
# ระบบจะค้นหา Session ล่าสุดใน model/trained_models อัตโนมัติ:
python predict.py --image dataset/dataset/1_box/IMG_6510.jpg

# ส่งออกผลลัพธ์เป็นไฟล์ JSON สำหรับเชื่อมต่อ Backend / API:
python predict.py --image test.jpg --output-json result.json

# บันทึกภาพขั้นตอนย่อย (Original, Gray, Blur, Edge, SVD):
python predict.py --image test.jpg --save-steps debug_output/
```

### 4.3 ตัวอย่างผลลัพธ์การทำนายบนหน้าจอ:

```
[Predict] โหลดภาพสำเร็จ: IMG_6510.jpg (4284×5712)
[ModelParameters] โหลด Session ล่าสุดอัตโนมัติ: session_20261017_131440
────────────────────────────────────────────────────────────
  PARCEL BOX COUNTER — PREDICTION RESULT
────────────────────────────────────────────────────────────
  ภาพ     : IMG_6510.jpg
  Session : session_20261017_131440

  ┌─ PREDICTION ──────────────────────────────┐
  │  จำนวนกล่อง (Predicted) : 1 กล่อง       │
  │  Confidence             : 76.2%            │
  └───────────────────────────────────────────┘

  Probability Distribution (กล่อง 1-9):
    กล่อง 1: [██████████████████████░░░░░░░░]  76.2% ◄ Predicted
    กล่อง 2: [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   9.3%
    กล่อง 3: [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   3.4%
    กล่อง 4: [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   9.5%
    กล่อง 5: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.7%
    กล่อง 6: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.1%
    กล่อง 7: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.3%
    กล่อง 8: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.3%
    กล่อง 9: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.2%

  ─ Theoretical Values ─────────────────────────
  Edge Density      : 86.01%
  Edge Total Energy : 903221312.00
  SVD Latent Energy : 228.1082
  Stat Features     : EdgeArea=0.8601, LogContours=0.6931, AR=1.0000
  Feature Dim       : 67  (SVD=64 + Stat=3)

  Pre-Softmax Logits (per class 1-9):
    +3.215, +1.109, +0.119, +1.128, -1.215, -3.516, -2.191, -2.252, -2.510
────────────────────────────────────────────────────────────
```
