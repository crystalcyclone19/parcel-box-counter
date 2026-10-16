# Pipeline สรุปกระบวนการเทรนโมเดล (End-to-End Pipeline Overview)

เอกสารฉบับนี้สรุป pipeline ทุกกระบวนการของการทำงาน ตั้งแต่การนำเข้าข้อมูล (Data Ingestion), การเตรียมและแปลงรูปภาพ (Image Preprocessing & Augmentation), การสกัดและลดมิติของฟีเจอร์ด้วย SVD จนถึงกระบวนการเทรนและปรับแต่งค่าน้ำหนักโมเดล (Fine-tuning & Weight Persistence) โดยอ้างอิงจาก Logic ที่เกิดขึ้นจริงในไฟล์ `model/preprocessing/run_preprocessing.py`, `model/preprocessing/svd_features.py` และ `model/mlp/mlp_regression.py`

---

## แผนภาพรวมระบบ (Pipeline Architecture)

```
[ Input Images / Zip / Colab ]
              │
              ▼
    (1) System Setup & Output Dirs Initialization
              │
              ▼
    (2) Data Ingestion & Image Reading
              │
              ▼
    (3) Geometric Transformation (Pre-aligned by Data Ingestor / Bypassed in Training)
        *หมายเหตุ: ในการเทรนจริงจะข้ามขั้นตอนนี้ เพราะผู้เตรียมข้อมูลจะปรับแต่งภาพมาแล้ว
        (ฟังก์ชัน transform จะถูกนำไปใช้ใน Web App ให้ User ปรับแต่งภาพก่อนส่งทำนาย)*
              │
              ▼
    (4) Grayscale Conversion (BGR → Grayscale)
              │
              ▼
    (5) Downscaling & Gaussian Blur (Resize 512x512 → Blur 5x5 เพื่อลด Noise)
              │
              ▼
    (6) Edge Convolution (Laplacian 3×3 + Percentile Clipping)
              │
              ▼
    (7) SVD Feature Extraction (Resize 128x128 → Centering → SVD → Matrix 8x8)
              │
              ▼
    (8) Model State Loading & Architecture Setup (from model_store.json)
              │
              ▼
    (9) Forward Pass with Scaled Sigmoid Regression (Prediction: y_hat ∈ [0, k])
              │
              ▼
    (10) Loss Calculation (MSE) & Backward Pass (Backpropagation / Gradients)
              │
              ▼
    (11) Parameter Update (Gradient Descent) & Checkpoint Persistence (Fine-Tuning)
```

---

## ช่วงที่ 1: Data Preparation & Preprocessing Pipeline
*(logic หลักตาม `run_preprocessing.py`, `image_pipeline.py`, และ `svd_features.py`)*

### ขั้นตอนที่ 1 : System Setup & Directory Initialization
- **สิ่งที่ทำ:** 
  - ตรวจสอบและสร้างโครงสร้างโฟลเดอร์สำหรับจัดเก็บ Artifacts ในการประมวลผล ได้แก่:
    - `input_images/` (ภาพต้นฉบับนำเข้า)
    - `Grayscale/` (ภาพหลังแปลงขาวดำ)
    - `Resized/` (ภาพหลังย่อขนาดเพื่อลด Micro-noise)
    - `Blurred/` (ภาพหลังกรอง Noise ด้วย Gaussian Blur)
    - `Convolution/` (ภาพขอบพัสดุหลังทำ Edge Map)
    - `output_svd/` (ผลลัพธ์เวกเตอร์ฟีเจอร์ SVD และโมเดล SVD)
    - `output_svd/matrices/` (เมทริกซ์ 8×8 ของแต่ละรูป)
  - มีฟังก์ชันตัวเลือกสำหรับล้างไฟล์ในโฟลเดอร์เก่า (`clear_pipeline_folders()`) ก่อนเริ่มการประมวลผลรอบใหม่

### ขั้นตอนที่ 2 : Data Ingestion & Input Verification
- **สิ่งที่ทำ:**
  - รองรับการนำเข้าไฟล์ภาพหลากหลายช่องทาง เช่น ระบุผ่าน Local Directory, ไฟล์ `.zip`, ระบุรายการชื่อไฟล์โดยตรง, หรือผ่าน Interactive Uploader บน Colab (`load_from_local_paths()`, `load_from_zip_path()`)
  - แตกไฟล์ zip (ถ้ามี) สแกนกรองเฉพาะนามสกุลไฟล์ภาพที่รองรับ (`.jpg`, `.jpeg`, `.png`) แล้วคัดลอกไฟล์ทั้งหมดลงใน `INPUT_DIR`
  - ตรวจสอบความถูกต้องของไฟล์รูปภาพผ่าน `cv2.imread(..., cv2.IMREAD_COLOR)` พร้อมข้ามไฟล์ที่เสียหายหรือไฟล์ว่าง

### ขั้นตอนที่ 3 : Geometric Transformation (Data Ingestor Task & Future Web App Integration)
- **สถานะในระบบการเทรนจริง (Training Pipeline):**
  - **ข้ามขั้นตอนการแปลงอัตโนมัติ (Bypassed in Automated Training):** ในกระบวนการเทรนโมเดลจริง ระบบจะ**ข้าม (Skip)** การ Transform อัตโนมัติด้วยพารามิเตอร์คงที่ (+50px, +30px, 1.2×, 25°) เนื่องจากรูปภาพแต่ละรูปที่ถ่ายมามีสภาพแวดล้อม มุมมอง องศา และระยะห่างที่แตกต่างกัน การใช้ค่า Transform ชุดเดียวกับทุกภาพจะทำให้พัสดุหลุดเฟรมหรือเสียสัดส่วน
  - **งานของผู้นำเข้าข้อมูล (Data Ingestor Responsibility):** ขั้นตอนการจัดมุม จัดกึ่งกลาง ย่อ/ขยาย หรือหมุนภาพให้ตรง เป็นหน้าที่ของ**ผู้ที่นำเข้าและเตรียมข้อมูล (Human-in-the-loop Ingestion / Annotation Team)** ที่จะทำการ Transform หรือ Crop/Align แต่ละรูปภาพอย่างเหมาะสมมาก่อนแล้วก่อนนำเข้าโฟลเดอร์สำหรับเทรน
- **การนำไปใช้งานจริงในอนาคตบน Web Application (Inference & Production Role):**
  - ฟังก์ชันในโมดูล `geometric_transform.py` (`translate_image`, `scale_image`, `rotate_image`, `transform_image`) จะถูกนำไปใช้งานจริงในระบบ **Web Application**:
    1. **User Interaction on Web:** ผู้ใช้งานถ่ายรูปหรืออัปโหลดภาพพัสดุผ่านหน้าเว็บแอปพลิเคชัน
    2. **Interactive Adjustment:** หน้าเว็บมีเครื่องมือให้ผู้ใช้ปรับแต่งรูปภาพ (เช่น เลื่อนตำแหน่ง X/Y, ซูมเข้า/ออก, หมุนองศา) เพื่อให้กล่องพัสดุอยู่ในกรอบตรวจจับที่สมบูรณ์
    3. **Parameter Passing & Backend Prediction:** Web Frontend จะส่งภาพพร้อมค่า Parameters ที่ผู้ใช้ปรับแต่ง (`shift_x`, `shift_y`, `scale_factor`, `rotation_deg`) ไปยัง Backend API เพื่อเรียกใช้ฟังก์ชัน `transform_image()` ก่อนส่งเข้าสู่กระบวนการ Edge Processing $\rightarrow$ SVD $\rightarrow$ MLP Prediction ต่อไป
- **ตรรกะของฟังก์ชันทางคณิตศาสตร์ (เมื่อถูกเรียกใช้):**
  - 1. **Translation:** เลื่อนตำแหน่งภาพตามแกน X (`shift_x`) และแกน Y (`shift_y`) ด้วย `cv2.warpAffine`
  - 2. **Scaling:** ขยายขนาดภาพ (`scale_factor`) โดยยึดจุดศูนย์กลางภาพ (`width/2`, `height/2`)
  - 3. **Rotation:** หมุนภาพ (`rotation_deg`) จากจุดศูนย์กลางภาพ
  - เติมขอบส่วนที่เกิดจากการเลื่อน/หมุนด้วยสีดำ (`cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)`) และรักษาขนาดภาพต้นฉบับไว้

### ขั้นตอนที่ 4 : Grayscale Conversion, Resizing & Noise Filtering
- **สิ่งที่ทำ:**
  - แปลงภาพ BGR ที่ผ่านการ Transform แล้วให้เป็นภาพขาวดำ Grayscale ระดับความสว่าง 0–255 ด้วย `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`
  - **Resize (Downscaling):** ย่อขนาดภาพให้มีขนาดเหมาะสม (`512 × 512 px`) ผ่าน `resize_image()` เพื่อลด Micro-noise จากเซนเซอร์ และปรับสเกลขอบกล่องให้เข้ากับขนาด Receptive field ของ Kernel $3\times3$
  - **Gaussian Blur:** กรองสัญญาณรบกวนความถี่สูงบนพื้นผิวลังกระดาษและพื้นโต๊ะด้วย Gaussian filter (`apply_gaussian_blur(..., kernel_size=5)`)
  - บันทึกผลลัพธ์แยกตามโฟลเดอร์: `Grayscale/gray_<name>`, `Resized/resized_<name>`, และ `Blurred/blur_<name>`

### ขั้นตอนที่ 5 : Edge Convolution & Percentile Normalization
- **สิ่งที่ทำ:**
  - สกัดขอบและโครงร่างพัสดุ (Edge Extraction) จากภาพที่ผ่านการ Blur แล้ว ด้วย 2D Convolution (`cv2.filter2D`) โดยใช้ Kernel สไตล์ **Laplacian 3×3**:
    $$\begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$
  - นำค่าผลลัพธ์ convolution มาหาค่าสัมบูรณ์ (`np.abs`) และใช้ **Percentile Clipping (ตัดที่ 98.0%)** แทน Min-Max ธรรมดา เพื่อป้องกันไม่ให้จุดสะท้อนแสงหลอดไฟดึงสเกลภาพให้มืดสนิท
  - บันทึกผลลัพธ์ลงในโฟลเดอร์ `Convolution/conv_<filename>` พร้อมบันทึกภาพเปรียบเทียบทุกขั้นตอน 6 คอลัมน์ side-by-side (`visualize_all_steps()`)

### ขั้นตอนที่ 6 : SVD Dimensionality Reduction & Feature Representation
- **สิ่งที่ทำ:**
  - นำภาพ Grayscale มา Resize เป็นขนาดมาตรฐาน $128 \times 128$ พิกเซล และแปลงค่าพิกเซลให้เป็น `float32` อยู่ในช่วงสเกล $[0, 1]$
  - Flatten แต่ละภาพให้เป็น 1D Vector ขนาด $128^2 = 16,384$ มิติ แล้วรวมเป็น Matrix ข้อมูล $X \in \mathbb{R}^{N \times 16384}$
  - ลบค่าเฉลี่ยของข้อมูล (Centering): $X_{\text{centered}} = X - \mu$
  - ทำการลดขนาดมิติด้วย **Randomized SVD** เพื่อสกัด 64 องค์ประกอบเด่น ($n\_features = 64$):
    $$Z = X_{\text{centered}} \cdot V^T \quad \in \mathbb{R}^{N \times 64}$$
  - บันทึก Feature Vector ของแต่ละภาพให้อยู่ในรูปเมทริกซ์ **$8 \times 8$** (`.npy` และ `.csv`) ใน `output_svd/matrices/` และบันทึก `svd_model.npz` สำหรับใช้ฉายภาพใหม่ในขั้นตอน Inference

---

## ช่วงที่ 2: MLP Training & Iterative Fine-Tuning
*(logic หลักตาม `mlp_regression.py` และ `model_store.json`)*

### ขั้นตอนที่ 7 : Model State Loading & Architecture Setup
- **สิ่งที่ทำ:**
  - โหลด Metadata, Hyperparameters และสถานะโมเดลล่าสุดจาก `model_store.json` ด้วย `load_model_store()`
  - กำหนดขนาด Input Vector เป็น $m\_dim = 64$ (สอดคล้องกับขนาด Feature $8 \times 8$ จาก SVD) และตัวแปรสเกลสูงสุด $k = 10.0$
  - ฟังก์ชัน `get_params()` จะตรวจสอบสถานะ:
    - หากเป็นการเริ่มเทรนใหม่ (W, b มีค่าเป็น null): จะสุ่มค่าน้ำหนักเริ่มต้น $W \sim \mathcal{N}(0, 0.01^2)$ และกำหนด Bias $b = 0$
    - **หากเป็นการ Fine-tuning หรือเทรนต่อ:** จะโหลดค่าน้ำหนัก ($W$) และ Bias ($b$) จาก Version และ Epoch ล่าสุดที่ถูกบันทึกไว้ใน checkpoint

### ขั้นตอนที่ 8 : Forward Pass with Scaled Sigmoid Regression
- **สิ่งที่ทำ:**
  - รับชุดข้อมูล Batch ($X$ มิติ `[Batch, 64]`)
  - คำนวณ Linear Combination:
    $$z = X \cdot W + b$$
  - ทำนายผลลัพธ์ผ่าน Activation Function แบบ **Scaled Sigmoid**:
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
    $$\hat{y} = k \cdot \sigma(z)$$
  - ตัวคูณ $k = 10.0$ ทำหน้าที่บีบและจำกัดขอบเขตการทำนายจำนวนกล่องพัสดุให้อยู่ในช่วง $[0, 10]$

### ขั้นตอนที่ 9 : Loss Evaluation & Backward Pass (Backpropagation)
- **สิ่งที่ทำ:**
  - คำนวณความคลาดเคลื่อนด้วยฟังก์ชัน Mean Squared Error (MSE):
    $$\mathcal{L} = \frac{1}{2 \cdot \text{batch\_size}} \sum (\hat{y} - y)^2$$
  - คำนวณอนุพันธ์ย้อนกลับ (Gradients) ด้วย Chain Rule:
    1. อนุพันธ์เทียบกับผลลัพธ์ทำนาย: $\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\hat{y} - y}{\text{batch\_size}}$
    2. อนุพันธ์เทียบกับผลรวมเชิงเส้น $z$: $\frac{\partial \hat{y}}{\partial z} = k \cdot \sigma(z) \cdot (1 - \sigma(z))$
    3. Error Term รวม: $\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$
    4. Gradient เทียบกับค่าน้ำหนัก $W$ และไบแอส $b$:
       $$\frac{\partial \mathcal{L}}{\partial W} = X^T \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial b} = \sum \frac{\partial \mathcal{L}}{\partial z}$$

### ขั้นตอนที่ 10 : Parameter Update & Model Checkpointing (Fine-tuning Persistence)
- **สิ่งที่ทำ:**
  - ปรับปรุงค่าน้ำหนักและไบแอสด้วยอัลกอริทึม **Gradient Descent**:
    $$W_{\text{new}} = W - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W}$$
    $$b_{\text{new}} = b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$
    *(โดย $\alpha$ คือ Learning Rate เช่น 0.05)*
  - อัปเดตสถานะโมเดลผ่าน `update_store()`:
    - เพิ่มลำดับเวอร์ชัน `version += 1` และนับรอบ `epoch += 1`
    - บันทึกประวัติ `loss` และ Timestamp ลงใน `history`
    - เซฟค่าน้ำหนักและไบแอสชุดใหม่กลับลงไฟล์ `model_store.json` ทันที รองรับการนำโมเดลไปเทรนต่อยอด (Fine-tuning) ใน batch หรือรอบถัดไปได้อย่างต่อเนื่อง
