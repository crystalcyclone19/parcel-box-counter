# Pipeline สรุปกระบวนการเทรนโมเดล (End-to-End Multi-class Classification Pipeline)

เอกสารฉบับนี้สรุปสถาปัตยกรรมและกระบวนการทำงานทั้งหมดของระบบ **Parcel Box Counter** ในเวอร์ชันล่าสุด ซึ่งได้ปรับปรุงจากการประมาณค่าแบบต่อเนื่อง (Regression) สู่ **Multi-class Classification (จำแนก 1–9 กล่อง)** พร้อมผสาน **Statistical Edge Features** และ **Data Augmentation** โดยอ้างอิงจาก Logic ที่เกิดขึ้นจริงใน:
- [model/preprocessing/](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/preprocessing) (`data_ingestion.py`, `edge_processing.py`, `svd_features.py`)
- [model/augmentation/augment_image.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/augmentation/augment_image.py)
- [model/mlp/mlp_classifier.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/mlp/mlp_classifier.py)
- [model/training/](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/model/training) (`dataset.py`, `trainer.py`, `tuner.py`, `evaluator.py`, `model_registry.py`)
- [predict.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/predict.py)

---

## แผนภาพรวมระบบ (Pipeline Architecture)

```
[ Input Images + labels.csv (1-9 กล่อง) ]
                    │
                    ▼
   (1) Ingestion, Validation & Stratified Split (70% Train / 15% Val / 15% Test)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 [ Train Set (70%) ]    [ Val & Test Sets (30%) ]
        │                       │
   (2) Data Augmentation        │ (Clean / No Augmentation)
   (9x Independent Variants)    │
   → 10x Effective Samples      │
        │                       │
        └───────────┬───────────┘
                    ▼
   (3) Image Preprocessing & Edge Convolution
       - Grayscale (BGR → Gray)
       - Downscale (512×512 px) & Gaussian Blur (5×5, σ=0)
       - Laplacian Edge Convolution (3×3 Kernel) + Percentile Clipping (98%)
                    │
                    ▼
   (4) Hybrid Feature Extraction (Combined Dimension = 67)
       ├─ Component A (64 dims): Resize 128×128 → Centering (Train Mean) → SVD Projection (Train Vt)
       └─ Component B (3 dims) : Statistical Edge Features (from 512×512 Edge Map)
            1. Total Edge Area  (สัดส่วนพิกเซลขอบต่อพื้นที่ภาพทั้งหมด)
            2. Log-Contour Count (log(1 + จำนวน Contour จาก cv2.findContours))
            3. Aspect Ratio     (อัตราส่วน Bounding Box ของคอนทัวร์ทั้งหมด)
            → Z-Score Normalized โดยใช้ค่า Mean และ Std ของ Train Set
                    │
                    ▼
   (5) Multi-class Classification Model (2-Layer MLP Architecture)
       - Input Layer  : 67 โหนด (64 SVD + 3 Stat Features)
       - Hidden Layer : hidden_dim (32, 64 หรือ 128) + ReLU Activation + Dropout (rate = 0.3)
       - Output Layer : 9 โหนด (แทนจำนวนกล่อง 1 ถึง 9) + Softmax Activation
                    │
                    ▼
   (6) Training Loop with Mini-batch SGD & Early Stopping
       - Categorical Cross-Entropy Loss
       - Gradient Clipping (±5.0) ป้องกัน Gradient Exploding
       - Validation Checkpoint & Early Stopping (Patience = 30 epochs)
                    │
                    ▼
   (7) Multi-metric Evaluation & Champion Model Selection
       - Primary Metric : Validation Cross-Entropy Loss (ค่าน้อยที่สุดชนะ)
       - Secondary      : Exact Accuracy, ±1 Box Tolerance Accuracy, Macro F1, Confusion Matrix
                    │
                    ▼
   (8) Champion Model Packaging & Registry Export
       - best_weights.json   (W1, b1, W2, b2, architecture metadata)
       - svd_model.npz       (mean, Vt, S, stat_mean, stat_std)
       - pipeline_config.json, manifest.json, leaderboard.csv, training_report.md
                    │
                    ▼
   (9) Production Inference Engine (predict.py)
       - โหลดรูปภาพ → Edge → SVD+Stats → 2-Layer MLP → Softmax
       - รายงาน: จำนวนกล่องที่ทำนาย, Confidence (%), Probability Distribution ทั้ง 9 คลาส
         และค่า Theoretical Values (Edge Density, Energy, Latent Energy, Logits)
```

---

## ช่วงที่ 1: Data Preparation, Splitting & Data Augmentation
*(Logic หลักตาม `model/training/dataset.py` และ `model/augmentation/augment_image.py`)*

### ขั้นตอนที่ 1: Dataset Pairing & Stratified Random Split
- **การโหลดและจับคู่ภาพ:**
  - สแกนภาพทั้งหมดในโฟลเดอร์ภาพ เช่น `dataset/dataset/` (`1_box` ถึง `9_box`)
  - อ่าน Label จาก `labels.csv` (ประกอบด้วยคอลัมน์ `filename` และ `count`) และจับคู่กับไฟล์จริง
- **การแบ่งชุดข้อมูล (Data Split):**
  - แบ่งเป็น **Train 70%**, **Validation 15%**, และ **Test 15%**
  - ใช้ Random Permutation แบบกำหนด Seed เพื่อให้ผลลัพธ์สามารถทำซ้ำได้ (Reproducibility)
  - **Data Leakage Prevention:** พารามิเตอร์การ Normalize ทั้งหมด (Mean ของภาพ, เมทริกซ์ $V^T$ ของ SVD, ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของฟีเจอร์สถิติ) จะต้องถูกคำนวณ**เฉพาะบน Train Set เท่านั้น** ห้ามนำข้อมูล Validation หรือ Test มาร่วมคำนวณเป็นอันขาด

### ขั้นตอนที่ 2: Data Augmentation เฉพาะบน Train Set (ตัด Rotation ออก)
เพื่อเพิ่มความหลากหลายของข้อมูลและแก้ปัญหาโมเดลจำเฉพาะภาพมุมมองเดียว ระบบได้นำโมดูล Augmentation เข้ามาทำงานร่วมกันในขั้นตอนสกัดฟีเจอร์ โดย**ตัดกระบวนการหมุนภาพ (Rotation $\pm 15^\circ$) ออก** เพื่อคงมุมมองแนวตั้ง/แนวนอนที่แท้จริงของพัสดุ:
- **เฉพาะ Train Set เท่านั้น** ที่จะเข้าสู่กระบวนการสร้างภาพดัดแปลง 9 รูปแบบอิสระ (ไม่ทำซ้อนทับกัน):
  1. พลิกภาพแนวนอน (Horizontal Flip)
  2. เพิ่มความสว่าง $+25\%$
  3. ลดความสว่าง $-25\%$
  4. เพิ่ม Contrast $+25\%$
  5. ลด Contrast $-25\%$
  6. Shear แกน X $+10\%$ (Crop เอาเฉพาะพื้นที่ภาพจริง)
  7. Shear แกน X $-10\%$ (Crop เอาเฉพาะพื้นที่ภาพจริง)
  8. Shear แกน Y $+10\%$ (Crop เอาเฉพาะพื้นที่ภาพจริง)
  9. Shear แกน Y $-10\%$ (Crop เอาเฉพาะพื้นที่ภาพจริง)
- **อัตราการขยายข้อมูล:** 1 ภาพต้นฉบับ + 9 ภาพ Augmentation = **10 เท่า** (เช่น ภาพ Train 278 ภาพ ขยายเป็น 2,780 ตัวอย่าง)
- **Validation และ Test Set:** **ห้ามทำ Augmentation เด็ดขาด** เพื่อให้การวัดผลสะท้อนภาพถ่ายจริงในสภาพแวดล้อมจริงแบบ Unseen Data 100%

---

## ช่วงที่ 2: Edge Processing & Hybrid Feature Extraction
*(Logic หลักตาม `model/preprocessing/edge_processing.py` และ `model/training/dataset.py`)*

### ขั้นตอนที่ 3: Grayscale Conversion & Downscaling
- แปลงภาพ BGR เป็น Grayscale ด้วย `cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)`
- ย่อขนาดภาพเป็น $512 \times 512$ พิกเซล ด้วย `cv2.resize(..., interpolation=cv2.INTER_AREA)` เพื่อลดสัญญาณรบกวนความละเอียดสูง (Sensor Micro-noise) และปรับขนาดเส้นขอบให้สอดคล้องกับขนาด Kernel

### ขั้นตอนที่ 4: Gaussian Blur & Laplacian Edge Convolution
- **Gaussian Blur:** กรองสัญญาณรบกวนบนพื้นผิวลังกระดาษและเงาสะท้อนด้วย Kernel $5 \times 5$:
  ```python
  cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
  ```
- **Laplacian Edge Convolution:** ทำ 2D Convolution ด้วย Kernel $3 \times 3$:
  $$\begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$
- **Percentile Clipping (98%):** ตัดค่า Outlier ที่สว่างเกินไป (เช่น แสงไฟสะท้อนบนเทปกาวใส) โดยนำค่าสัมบูรณ์ของ Laplacian มาตัดขอบบนที่เปอร์เซ็นไทล์ 98.0 แล้ว Normalize สเกลค่าให้อยู่ในช่วง $[0, 255]$ ได้เป็น **Edge Map ขนาด $512 \times 512$**

### ขั้นตอนที่ 5: Hybrid Feature Extraction (67 มิติ)
ฟีเจอร์เวกเตอร์ที่ป้อนเข้าสู่โมเดลประกอบด้วย 2 องค์ประกอบรวมกัน:

$$\mathbf{z}_{\text{combined}} = \big[\, \mathbf{z}_{\text{svd}}\,(64\text{ dims}) \;\big|\; \mathbf{z}_{\text{stat}}\,(3\text{ dims}) \,\big] \quad \in \mathbb{R}^{67}$$

#### ส่วนที่ 1: SVD Latent Features (64 มิติ)
1. นำ Edge Map $512 \times 512$ มาย่อเป็นภาพขนาดเล็ก $128 \times 128$ พิกเซล และสเกลค่าพิกเซลเป็น $[0, 1]$
2. Flatten เป็นเวกเตอร์ 1D ขนาด $128^2 = 16,384$ มิติ
3. ลบด้วยค่าเฉลี่ยของ Train Set: $\mathbf{x}_{\text{centered}} = \mathbf{x} - \boldsymbol{\mu}_{\text{train}}$
4. ฉายลงบน Right Singular Vectors ($V^T \in \mathbb{R}^{64 \times 16384}$):
   $$\mathbf{z}_{\text{svd}} = \mathbf{x}_{\text{centered}} \cdot V^T \quad \in \mathbb{R}^{64}$$

#### ส่วนที่ 2: Statistical Edge Features (3 มิติ)
สกัดค่าสถิติเชิงโครงสร้างเรขาคณิตโดยตรงจาก Edge Map ขนาด $512 \times 512$:
1. **Total Edge Area:** สัดส่วนจำนวนพิกเซลที่เป็นเส้นขอบต่อจำนวนพิกเซลทั้งหมดของภาพ:
   $$\text{Total Edge Area} = \frac{\sum \mathbb{I}(\text{Edge} > 0)}{512 \times 512}$$
2. **Log-Contour Count:** ตรวจจับจำนวน Contour ของวัตถุผ่าน `cv2.findContours` แล้วแปลงเป็นสเกล Logarithm เพื่อให้ค่ามีการกระจายตัวที่เสถียร:
   $$\text{Log Contours} = \ln(1 + \text{Count})$$
3. **Bounding Box Aspect Ratio:** หา Bounding Box ครอบคลุม Contour ทั้งหมด แล้วคำนวณอัตราส่วน กว้าง/ยาว:
   $$\text{Aspect Ratio} = \frac{\text{Width}_{\text{total}}}{\text{Height}_{\text{total}} + \epsilon}$$
4. **Z-Score Normalization:** นำทั้ง 3 ค่ามาทำ Standardization โดยใช้ $\boldsymbol{\mu}_{\text{stat}}$ และ $\boldsymbol{\sigma}_{\text{stat}}$ ที่คำนวณจาก Train Set:
   $$\mathbf{z}_{\text{stat}} = \frac{\mathbf{x}_{\text{stat}} - \boldsymbol{\mu}_{\text{stat}}}{\boldsymbol{\sigma}_{\text{stat}} + 10^{-8}} \quad \in \mathbb{R}^{3}$$

---

## ช่วงที่ 3: Multi-class MLP Architecture & Training Loop
*(Logic หลักตาม `model/mlp/mlp_classifier.py` และ `model/training/trainer.py`)*

### ขั้นตอนที่ 6: สถาปัตยกรรม 2-Layer Multi-class Classifier
เปลี่ยนผ่านจากการทำ Regression 1 โหนด มาเป็นสถาปัตยกรรมจำแนกประเภทแบบเต็มรูปแบบ (1–9 กล่อง):

```
Input (67) ──► Linear(67 → H) ──► ReLU ──► Dropout(0.3) ──► Linear(H → 9) ──► Softmax ──► Probs (9)
```

1. **Layer 1 (Feature Transformation):**
   $$\mathbf{z}_1 = \mathbf{X} W_1 + \mathbf{b}_1 \quad (W_1 \in \mathbb{R}^{67 \times H}, \; \mathbf{b}_1 \in \mathbb{R}^{H})$$
   $$\mathbf{h} = \text{ReLU}(\mathbf{z}_1) = \max(0, \mathbf{z}_1)$$
   $$\mathbf{h}_{\text{drop}} = \mathbf{h} \odot \mathbf{m}_{\text{dropout}} \quad \text{(เฉพาะระหว่าง Training, } p=0.3\text{)}$$
2. **Layer 2 (Class Logits & Softmax):**
   $$\mathbf{z}_2 = \mathbf{h}_{\text{drop}} W_2 + \mathbf{b}_2 \quad (W_2 \in \mathbb{R}^{H \times 9}, \; \mathbf{b}_2 \in \mathbb{R}^{9})$$
   $$P(\text{class } c) = \text{Softmax}(\mathbf{z}_2)_c = \frac{e^{z_{2, c} - \max(\mathbf{z}_2)}}{\sum_{j=1}^{9} e^{z_{2, j} - \max(\mathbf{z}_2)}}$$
3. **Weight Initialization:** ใช้ **Xavier/Glorot Normal Initialization**:
   $$\text{Std}(W_1) = \sqrt{\frac{2}{67 + H}}, \quad \text{Std}(W_2) = \sqrt{\frac{2}{H + 9}}$$

### ขั้นตอนที่ 7: Loss Function & Backpropagation
- **Categorical Cross-Entropy Loss:**
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \ln\big(P(y_i)\big)$$
  *(โดย $y_i \in \{0, 1, \dots, 8\}$ คือ 0-indexed Class Index แทนจำนวนกล่อง 1 ถึง 9)*
- **Backward Gradients:**
  1. Derivative ของ Softmax ร่วมกับ Cross-Entropy:
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_2} = \frac{1}{N} (\mathbf{P} - \mathbf{Y}_{\text{one-hot}})$$
  2. Gradient ของ Layer 2:
     $$\frac{\partial \mathcal{L}}{\partial W_2} = \mathbf{h}_{\text{drop}}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{z}_2}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \sum \frac{\partial \mathcal{L}}{\partial \mathbf{z}_2}$$
  3. Derivative ย้อนกลับผ่าน Dropout และ ReLU:
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}_2} \cdot W_2^T \odot \mathbf{m}_{\text{dropout}}\right) \odot \mathbb{I}(\mathbf{z}_1 > 0)$$
  4. Gradient ของ Layer 1:
     $$\frac{\partial \mathcal{L}}{\partial W_1} = \mathbf{X}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \sum \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1}$$
- **Gradient Clipping:** ทำ Clip Gradients ในช่วง $[-5.0, 5.0]$ ก่อนอัปเดตน้ำหนัก เพื่อความเสถียรของ SGD
- **Weight Update:** อัปเดตด้วย Mini-batch SGD:
  $$W \leftarrow W - \eta \cdot \nabla_W \mathcal{L}, \quad \mathbf{b} \leftarrow \mathbf{b} - \eta \cdot \nabla_{\mathbf{b}} \mathcal{L}$$

### ขั้นตอนที่ 8: Early Stopping & Model Checkpointing
- ติดตามค่า **Validation Cross-Entropy** ทุก Epoch โดยปิด Dropout (`dropout_rate = 0.0`)
- หาก Validation Loss ลดลงจนทำ New Best จะทำการบันทึก Checkpoint สำเนา $W_1, b_1, W_2, b_2$
- หากไม่ดีขึ้นติดต่อกันเกิน `patience = 30` Epochs ระบบจะตัดจบการเทรนของ Trial นั้นอัตโนมัติ เพื่อประหยัดเวลาและป้องกัน Overfitting

---

## ช่วงที่ 4: Evaluation Metrics & Model Registry
*(Logic หลักตาม `model/training/evaluator.py` และ `model/training/model_registry.py`)*

### ขั้นตอนที่ 9: ชุด Classification Metrics สำหรับการคัดเลือกโมเดล
ระบบคำนวณและรายงาน Metrics ครบถ้วนสำหรับการวิเคราะห์โมเดลจำแนกประเภท:
1. **Cross-Entropy Loss (Primary Metric):** ชี้วัดความมั่นใจและความถูกต้องของ Probability Distribution ยิ่งต่ำยิ่งดี
2. **Exact Accuracy:** สัดส่วนที่ทำนายจำนวนกล่องถูกต้องตรง $100\%$
3. **$\pm 1$ Box Tolerance Accuracy (Ordinal Metric):** สัดส่วนที่ทำนายคลาดเคลื่อนไม่เกิน $\pm 1$ กล่อง ซึ่งสะท้อนความสามารถในการประมาณขนาดกองพัสดุในทางปฏิบัติ
4. **Macro F1-Score:** ค่าเฉลี่ย F1-score ของแต่ละคลาส ชี้วัดความแม่นยำในกรณีที่จำนวนภาพแต่ละคลาสใน Dataset ไม่เท่ากัน (Class Imbalance)
5. **Confusion Matrix ($9 \times 9$):** แจกแจงจำนวนการทำนายจริงเทียบกับ Ground Truth ในทุกๆ คลาส (1 ถึง 9)
6. **Generalization Gap:** $|\text{Train Loss} - \text{Val Loss}|$ สำหรับตรวจสอบความเสี่ยง Overfitting

### ขั้นตอนที่ 10: Model Registry & Artifacts Packaging
เมื่อเสร็จสิ้น Grid Search ระบบจะประเมิน Champion Model บน Test Set (Unseen Data) แล้วส่งออกชุดไฟล์ Artifacts 6 ไฟล์ในโฟลเดอร์ `model/trained_models/session_<timestamp>/`:

```
session_<timestamp>/
├── best_weights.json      # W1, b1, W2, b2, hidden_dim, n_classes=9, input_dim=67
├── svd_model.npz          # mean, Vt, S, stat_mean, stat_std, image_size=128
├── pipeline_config.json   # ค่า config พารามิเตอร์ของ preprocessing และ training
├── manifest.json          # Metadata สรุปผล metrics บน val/test, hyperparameters, วันที่
├── leaderboard.csv        # ตารางสรุปผลการแข่งขันทุก Trial เรียงตาม Cross-Entropy
└── training_report.md     # รายงานสรุปผลอย่างเป็นทางการ พร้อม Confusion Matrix Markdown
```

---

## ช่วงที่ 5: Inference Pipeline (`predict.py`)

เมื่อนำภาพใหม่เข้ามาทำนาย สคริปต์ [predict.py](file:///c:/Users/DELL/Downloads/CNNs/parcel-box-counter/predict.py) จะรัน Pipeline แบบ End-to-End โดยอัตโนมัติ:

```
[ New Image (JPG, PNG, HEIC) ]
               │
               ▼
   1. Ingestion & Preprocessing: Grayscale → Resize(512) → Blur(5) → Laplacian Edge(98%)
               │
               ▼
   2. Feature Extraction:
      - SVD Component : Resize(128) → Centering (svd_mean) → Project (svd_vt) → 64 dims
      - Stat Component: Compute [EdgeArea, LogContours, AR] → Normalize (stat_mean, stat_std) → 3 dims
      - Combine       : z_combined (1 × 67)
               │
               ▼
   3. 2-Layer MLP Forward (no dropout):
      - h = ReLU(z_combined @ W1 + b1)
      - Logits = h @ W2 + b2
      - Probs = Softmax(Logits)
               │
               ▼
   4. Output Formatting:
      - Predicted Box Count = argmax(Probs) + 1
      - Confidence = max(Probs)
      - Probability Distribution Bar Chart (คลาส 1–9)
      - Theoretical Metrics: Edge Density (%), Edge Energy, SVD Latent Energy, Logits
```
