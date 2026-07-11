# Face Recognition Authentication System

**Course:** CYB302 - Biometrics Security  
**Project:** Lab Assessment — Face Recognition for University Secure Lab Access  
**Technology:** FastAPI + Python + OpenCV + dlib + SQLite  

---

## Table of Contents

1. [Scenario & Problem Statement](#1-scenario--problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Lab Task 1 — Biometric Data Capture](#3-lab-task-1--biometric-data-capture)
4. [Lab Task 2 — Image Preprocessing](#4-lab-task-2--image-preprocessing)
5. [Lab Task 3 — Feature Extraction](#5-lab-task-3--feature-extraction)
6. [Lab Task 4 — Matching](#6-lab-task-4--matching)
7. [Lab Task 5 — Threshold Selection](#7-lab-task-5--threshold-selection)
8. [Lab Task 6 — Performance Evaluation](#8-lab-task-6--performance-evaluation)
9. [Lab Task 7 — Multimodal Biometrics](#9-lab-task-7--multimodal-biometrics)
10. [Lab Task 8 — Security & Privacy](#10-lab-task-8--security--privacy)
11. [API Endpoints Reference](#11-api-endpoints-reference)
12. [Running the System](#12-running-the-system)
13. [Presentation Guide](#13-presentation-guide)
---



## 1. Scenario & Problem Statement

A university wants to introduce a new access system for staff entering secure research labs. The system must:

- Reduce impersonation risks
- Eliminate password sharing
- Verify identity using biometric traits

**Solution:** A face recognition authentication system that:
1. Enrolls staff by capturing their face images
2. Extracts unique facial features (128D embeddings)
3. Stores templates securely (encrypted)
4. Verifies staff by comparing live captures against stored templates
5. Grants or denies access based on similarity score

**Why Face Recognition?**
- Contactless and hygienic
- No special hardware required (uses standard webcams)
- Hard to spoof compared to passwords/ID cards
- Natural and intuitive for users

---

## 2. System Architecture

```
                  User (Staff Member)
                          |
                          ▼
                   FastAPI Backend
                          |
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     Enrollment       Verification     Metrics/Reports
          |               |               |
          └───────► Face Service ◄────────┘
                      |       |
          ┌───────────┘       └───────────┐
          ▼                               ▼
   Preprocessing                    Feature Extraction
   (OpenCV)                          (dlib)
          |                               |
          ▼                               ▼
   Grayscale                        128D Embedding
   Resize (160x160)                     |
   Gaussian Blur                        ▼
   Histogram Equalization          Encrypted Storage
   Normalization                  (Fernet AES)
                                          |
                                          ▼
                                    SQLite Database
```

### Technology Stack

| Component          | Technology              |
|--------------------|-------------------------|
| Backend Framework  | FastAPI                 |
| Language           | Python 3.12+            |
| Face Detection     | dlib (HOG + CNN)        |
| Face Recognition   | dlib ResNet (128D embeddings) |
| Image Processing   | OpenCV                  |
| Database           | SQLite (SQLAlchemy ORM) |
| Encryption         | Fernet (AES-128)        |
| Metrics & Plots    | scikit-learn, Matplotlib |
| Numerical Ops      | NumPy                   |

### Folder Structure

```
face-auth-system/
│
├── app/                          # Application code
│   ├── main.py                   # FastAPI entry point
│   ├── config.py                 # Configuration & paths
│   ├── database.py               # SQLAlchemy engine & session
│   ├── models.py                 # Database models (User)
│   ├── schemas.py                # Pydantic request/response schemas
│   ├── utils.py                  # Utility functions (file save)
│   │
│   ├── routes/                   # API route handlers
│   │   ├── enroll.py             # POST /enroll, GET /enroll/users, DELETE /enroll/users/{id}
│   │   ├── verify.py             # POST /verify
│   │   ├── metrics.py            # GET /metrics, POST /metrics/evaluate, GET /metrics/threshold-sweep
│   │   └── preprocess.py         # POST /preprocess
│   │
│   └── services/                 # Business logic
│       ├── preprocessing.py      # Image preprocessing pipeline
│       ├── feature_extraction.py # Face detection & embedding extraction
│       ├── matching.py           # Cosine similarity & Euclidean distance
│       ├── performance.py        # FAR, FRR, EER, ROC, DET curves
│       └── encryption.py         # Fernet encryption for templates
│
├── database/                     # SQLite database file
│   └── biometric.db
│
├── models/                       # dlib model files
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── images/                       # Uploaded face images
│   ├── enrollment/               # Enrollment photos
│   └── test/                     # Test photos
│
├── reports/                      # Generated plots
│   ├── ROC.png
│   └── DET.png
│
├── venv/                         # Python virtual environment
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 3. Lab Task 1 — Biometric Data Capture

### Objective
Enroll staff members by capturing their face images, creating the enrollment dataset that the system will use for future verification.

### Implementation

**Endpoint:** `POST /enroll`

When a staff member is enrolled:
1. The user provides their name and a face image
2. The image is saved to `images/enrollment/` with a unique filename
3. A face embedding is extracted using dlib
4. The embedding is encrypted with Fernet and stored in the database
5. The user record is created with their name, image path, and encrypted template

```
Request:  POST /enroll  (name="John", file=photo.jpg)
Response: {"message": "User 'John' enrolled successfully"}
```

**Endpoint:** `GET /enroll/users`

Lists all enrolled users with their IDs, names, and enrollment dates.

```
Response: [{"id": 1, "name": "John", "created_at": "2024-01-01T12:00:00"}]
```

**Endpoint:** `DELETE /enroll/users/{id}`

Removes an enrolled user and their data from the system.

### Quality Problems in Acquisition

| Problem       | Effect on Recognition                                    |
|---------------|----------------------------------------------------------|
| Poor lighting | Low contrast, shadows obscure facial features             |
| Blur          | Loss of fine details (edges, texture)                     |
| Pose variation| Off-angle faces reduce feature extraction accuracy        |
| Occlusion     | Glasses, masks, or hands covering parts of the face       |
| Low resolution| Insufficient pixels for reliable feature extraction       |
| Expression    | Extreme expressions distort facial geometry              |

### Raw Image vs Biometric Template

| Aspect          | Raw Biometric Image              | Biometric Template                     |
|-----------------|----------------------------------|----------------------------------------|
| What is stored  | JPEG/PNG of the face             | Array of 128 floating-point numbers    |
| Size            | ~50-500 KB                       | ~1 KB (1024 bytes)                     |
| Reversible?     | Can be displayed                 | Cannot be reconstructed into a face    |
| Privacy risk    | High (original image exposed)    | Low (only mathematical features)       |
| Storage         | Image files on disk              | Encrypted in database                  |

---

## 4. Lab Task 2 — Image Preprocessing

### Objective
Improve input image quality before feature extraction to maximize recognition accuracy.

### Implementation

**Endpoint:** `POST /preprocess`

Returns each preprocessing stage as a base64-encoded image for visualization.

**Pipeline (in order):**

```
Original (BGR)
    │
    ▼
1. Grayscale Conversion
   - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   - Reduces 3 channels to 1, lowers computational cost
    │
    ▼
2. Resize to 160x160
   - cv2.resize(gray, (160, 160))
   - Ensures all images have consistent dimensions
    │
    ▼
3. Gaussian Blur (5x5 kernel)
   - cv2.GaussianBlur(resized, (5, 5), 0)
   - Removes high-frequency noise and fine grain
    │
    ▼
4. Histogram Equalization
   - cv2.equalizeHist(blurred)
   - Stretches pixel intensity distribution
   - Improves contrast, especially in dark/uneven lighting
    │
    ▼
5. Normalization [0, 1]
   - equalized / 255.0
   - Scaling pixel values for numerical stability
    │
    ▼
Enhanced Image (ready for feature extraction)
```

### Why Preprocessing Helps

1. **Grayscale:** Face recognition algorithms work on intensity patterns, not color. Removing color reduces data by 66% with minimal accuracy loss.
2. **Resize:** Standardizing input size ensures consistent feature extraction regardless of camera distance.
3. **Gaussian Blur:** Eliminates sensor noise that could be mistaken for facial features.
4. **Histogram Equalization:** Dramatically improves recognition under varying lighting conditions — one of the biggest challenges in real-world face recognition.
5. **Normalization:** Prevents large pixel values from dominating the embedding calculation.

---

## 5. Lab Task 3 — Feature Extraction

### Objective
Convert the preprocessed face image into a unique numerical representation (biometric template) that can be stored and compared.

### Implementation

**Technology:** dlib `face_recognition_model_v1` (ResNet-34 architecture)

**Process:**
1. Face detection using dlib's HOG-based detector
2. Facial landmark detection (68 points) for alignment
3. 128D embedding extraction via the ResNet model
4. The embedding is a compact numerical signature of the face

**What is a Face Embedding?**

A face embedding is a vector of 128 floating-point numbers that uniquely represents a person's face. The numbers encode facial characteristics such as:
- Distance between eyes
- Jawline shape
- Nose width and length
- Cheekbone structure
- Chin shape

```
Example embedding (128 numbers):
[0.124, -0.056, 0.231, 0.089, ..., 0.012]
  ↑                               ↑
 feature 1                     feature 128
```

### Key Properties of Good Biometric Templates

| Property     | Description                                              |
|--------------|----------------------------------------------------------|
| Uniqueness   | Different faces produce very different embeddings        |
| Stability    | Same face produces similar embeddings across conditions  |
| Compactness  | 128 numbers is efficient for storage and comparison      |
| Non-reversible| Cannot reconstruct a face image from the embedding      |

### Comparison: Raw Image vs Template

| Aspect        | Raw Image                     | Biometric Template              |
|---------------|-------------------------------|----------------------------------|
| Content       | Visual pixels                 | Mathematical features            |
| Size          | 50-500 KB                    | ~1 KB (1024 bytes)               |
| Privacy       | Identifiable by human         | Not visually interpretable       |
| Reversible    | Yes (can be viewed)          | No (cannot reconstruct face)     |
| Storage       | Disk as files                 | Encrypted in database            |

---

## 6. Lab Task 4 — Matching

### Objective
Compare a probe face (from a person seeking access) against stored gallery templates to determine identity.

### Implementation

**Endpoint:** `POST /verify`

Two matching modes are supported:

| Mode    | Description                      | Use Case                        |
|---------|----------------------------------|----------------------------------|
| 1:1 Verification | Compare probe against one claimed identity | PIN + face at ATM       |
| 1:N Identification | Compare probe against all enrolled users | Door access system    |

This system uses **1:N Identification** — the probe is compared against all enrolled users, and the best match is returned if it exceeds the threshold.

### Distance Metrics

**Cosine Similarity** (default, recommended):
```
similarity = dot(A, B) / (||A|| * ||B||)
Range: [-1, 1]
Higher = more similar
```

**Euclidean Distance** (alternative):
```
distance = sqrt(sum((A - B)^2))
Range: [0, ∞)
Lower = more similar
```

### Matching Algorithm

```
For each enrolled user:
    Compute similarity(probe_embedding, user_embedding)
    Track best score and corresponding user

If best_score >= threshold:
    Return "Access Granted" + identity + similarity
Else:
    Return "Access Denied"
```

### Example Verification Response

```json
{
  "identity": "John",
  "similarity": 0.91,
  "decision": "Access Granted"
}
```

```json
{
  "identity": null,
  "similarity": 0.34,
  "decision": "Access Denied"
}
```

---

## 7. Lab Task 5 — Threshold Selection

### Objective
Find the optimal similarity threshold that balances security (low FAR) with usability (low FRR).

### Implementation

**Endpoint:** `GET /metrics/threshold-sweep`

Tests multiple thresholds and reports FAR and FRR for each.

### How Threshold Affects Performance

| Threshold | Effect on Security              | Effect on Usability              |
|-----------|---------------------------------|----------------------------------|
| 0.3 (low) | High FAR (many intruders accepted) | Low FRR (authorized users rarely rejected) |
| 0.5       | Moderate FAR                     | Moderate FRR                     |
| 0.6       | Good balance                     | Good balance                     |
| 0.7       | Low FAR (few intruders accepted)  | High FRR (authorized users often rejected) |
| 0.9 (high)| Very low FAR                    | Very high FRR                    |

### Security vs Usability Trade-off

- **Low threshold** (e.g., 0.3): Convenient for users but insecure — impostors will be accepted.
- **High threshold** (e.g., 0.8): Very secure but frustrating — legitimate users will be rejected frequently, especially in poor lighting.
- **Optimal threshold** (e.g., 0.6): Equal Error Rate (EER) point where FAR = FRR.

### Threshold Sweep Results (Example)

```json
[
  {"threshold": 0.3, "far": 0.15, "frr": 0.01},
  {"threshold": 0.4, "far": 0.08, "frr": 0.03},
  {"threshold": 0.5, "far": 0.04, "frr": 0.06},
  {"threshold": 0.6, "far": 0.02, "frr": 0.10},
  {"threshold": 0.7, "far": 0.01, "frr": 0.18},
  {"threshold": 0.8, "far": 0.00, "frr": 0.30}
]
```

---

## 8. Lab Task 6 — Performance Evaluation

### Objective
Quantitatively measure how accurate the system is using standard biometric performance metrics.

### Metrics Defined

**False Acceptance Rate (FAR):**
```
FAR = Number of impostors accepted / Total number of impostor attempts
```
- The percentage of unauthorized people who are incorrectly granted access
- **Security risk:** High FAR means the system lets intruders in

**False Rejection Rate (FRR):**
```
FRR = Number of genuine users rejected / Total number of genuine attempts
```
- The percentage of authorized people who are incorrectly denied access
- **Usability risk:** High FRR means legitimate users are locked out

**Equal Error Rate (EER):**
```
EER = Point where FAR = FRR
```
- Lower EER = More accurate system
- Used as a single-number summary of system accuracy

**Area Under Curve (AUC):**
```
AUC = Area under the ROC curve
Range: [0.5, 1.0]
```
- 0.5 = Random guessing
- 1.0 = Perfect classification

### Endpoints

**`GET /metrics`** — Returns FAR, FRR, EER at the configured threshold.

**`POST /metrics/evaluate`** — Full evaluation with all metrics plus generated ROC/DET plots.

**`GET /metrics/threshold-sweep`** — FAR and FRR across multiple thresholds.

### ROC Curve

The ROC (Receiver Operating Characteristic) curve plots:
- **X-axis:** False Positive Rate (FPR) = FAR
- **Y-axis:** True Positive Rate (TPR) = 1 - FRR

A good system has a curve that rises steeply toward the top-left corner. The Area Under the Curve (AUC) quantifies overall performance.

### DET Curve

The DET (Detection Error Trade-off) curve plots:
- **X-axis:** False Acceptance Rate (FAR)
- **Y-axis:** False Rejection Rate (FRR)

The closer the curve is to the origin (0,0), the better. The intersection with the diagonal line (FAR = FRR) marks the EER.

### Reports Generated

After running evaluation, plots are saved to `reports/`:
- `reports/ROC.png`
- `reports/DET.png`

These can be accessed at `http://localhost:8000/reports/ROC.png` and `http://localhost:8000/reports/DET.png`.

---

## 9. Lab Task 7 — Multimodal Biometrics

### Objective
Implement score-level fusion using two feature sets from the same face image to demonstrate how multimodal systems improve accuracy.

### Implementation

This system uses **multialgorithmic fusion** — two different feature extractors on the same face image:

| Modality | Algorithm       | Description                          | Vector Size |
|----------|-----------------|--------------------------------------|-------------|
| Face     | dlib ResNet-34  | Deep learning 128D face embedding    | 128 floats  |
| Face     | HOG             | Handcrafted histogram-of-gradients   | 81,252 floats |

**Fusion level:** Score-level fusion (weighted sum)

**Code:** `app/services/multimodal.py`

### How It Works

```
Probe Image
    │
    ├──► dlib feature extractor ──► 128D embedding ──► cosine similarity ──┐
    │                                                                       │
    │                                                                       ▼
    │                                                          Weighted Fusion
    │                                                          (score = w·dlib + (1-w)·hog)
    │                                                                       │
    │                                                                       ▼
    └──► HOG feature extractor ──► 81K feature vector ──► cosine similarity ┘
                                                                       │
                                                                       ▼
                                                              Decision (threshold)
```

### Endpoints

**`POST /multimodal/verify`** — Verify using fused dlib + HOG scores
```
Query params: threshold (default 0.6), weight (default 0.7)
```

**`POST /multimodal/evaluate`** — Compare unimodal vs multimodal performance on ORL dataset
```
Query params: threshold, weight, enroll_count
Returns: FAR, FRR, EER, AUC for dlib-only, HOG-only, and fused
```

### Example Evaluation Response

```json
{
  "unimodal_dlib":  {"far": 0.02, "frr": 0.10, "eer": 0.06, "auc": 0.97},
  "unimodal_hog":   {"far": 0.08, "frr": 0.22, "eer": 0.15, "auc": 0.88},
  "multimodal_fused": {"far": 0.01, "frr": 0.07, "eer": 0.04, "auc": 0.99}
}
```

### Why Multimodal Improves Accuracy

| Aspect               | Unimodal (dlib) | Unimodal (HOG) | Multimodal (fused) |
|----------------------|-----------------|----------------|--------------------|
| EER (lower is better)| 0.06            | 0.15           | **0.04**           |
| AUC (higher is better)| 0.97           | 0.88           | **0.99**           |
| Robustness           | Degrades with pose | Degrades with lighting | Better in both |

The fusion exploits the fact that dlib (deep learning) and HOG (handcrafted) fail on different types of degraded images — their errors are uncorrelated, so combining them cancels out individual weaknesses.

---

## 10. Lab Task 8 — Security & Privacy

### Objective
Protect biometric data through encryption, access control, and privacy-aware design.

### Security Measures Implemented

#### 1. Template Encryption

All stored face embeddings are encrypted using **Fernet (symmetric AES-128)** before being saved to the database.

```
Raw Embedding (128D numpy array)
    │
    ▼
Fernet.encrypt() with key derived from SECRET_KEY
    │
    ▼
Encrypted bytes stored in SQLite
```

When verifying, the encrypted template is decrypted in memory, compared, and discarded.

#### 2. Secure Secret Key

The encryption key is derived from `SECRET_KEY` (configured in `.env`). In production, this should be:
- Generated randomly (e.g., `openssl rand -hex 32`)
- Stored securely (environment variable, not hardcoded)
- Rotated periodically

#### 3. Database Security

- SQLite file should be protected with filesystem permissions
- Database contains encrypted templates, not raw images
- Raw enrollment images can be deleted after enrollment to minimize exposure

### Privacy Considerations

#### Cancelable Biometrics

Cancelable biometrics transform a biometric template using a non-invertible function. If the transformed template is compromised, a new transformation can be applied (similar to changing a password).

**How it would work:**
```
Original Embedding → Apply User-Specific Transform → Store Transformed Template
```

If compromised:
```
Generate New Transform → Apply to Original Embedding → Store New Template
```

#### Template Revocation

When a user leaves the university:
1. Delete their user record from the database
2. Delete their enrollment images
3. All their biometric data is permanently removed

This is supported via `DELETE /enroll/users/{id}`.

#### Data Protection Principles

| Principle         | Application                                      |
|-------------------|--------------------------------------------------|
| Minimization      | Only store essential features (128D embedding), not raw images |
| Encryption        | All templates encrypted at rest                   |
| Access Control    | API endpoints restrict who can enroll/delete     |
| Transparency      | Users are informed of what data is collected      |
| Deletion          | Users can request deletion of their biometric data |
| Limited Retention | Data kept only while user is affiliated with university |

### Ethical Concerns

1. **Informed Consent:** Users must understand what biometric data is collected and how it is used.
2. **Function Creep:** Biometric data should not be repurposed (e.g., for surveillance) without consent.
3. **Data Breach Risk:** Unlike passwords, biometric traits cannot be changed if compromised.
4. **Bias:** Face recognition may have demographic biases — the system should be tested across diverse populations.
5. **Surveillance:** The system should only be used for access control, not continuous monitoring.

---

## 11. API Endpoints Reference

| Method   | Endpoint                           | Description                                   | Request                          | Response                  |
|----------|------------------------------------|-----------------------------------------------|----------------------------------|---------------------------|
| `POST`   | `/enroll`                          | Enroll a new staff member                     | Form: name + image file         | `{"message": "..."}`     |
| `GET`    | `/enroll/users`                    | List all enrolled users                       | —                                | `[UserResponse]`          |
| `DELETE` | `/enroll/users/{id}`              | Delete an enrolled user                       | Path: user_id                    | `{"message": "..."}`     |
| `POST`   | `/verify`                          | Verify a face against enrolled users          | Form: image file + query threshold | `VerifyResponse`       |
| `GET`    | `/metrics`                         | Get FAR, FRR, EER                             | Query: threshold                  | `MetricsResponse`        |
| `POST`   | `/metrics/evaluate`                | Full evaluation with plots                    | Query: threshold                  | Full metrics + plot paths|
| `POST`   | `/metrics/evaluate-dataset`        | Evaluation using real ORL dataset scores      | Query: threshold, enroll_count    | Full metrics + sample scores|
| `GET`    | `/metrics/threshold-sweep`         | FAR/FRR across multiple thresholds            | —                                | `[{threshold, far, frr}]`|
| `POST`   | `/preprocess`                      | View preprocessing stages of an image         | Form: image file                  | Stages with base64 images|
| `GET`    | `/`                                | API root information                          | —                                | API documentation        |

---

## 12. Running the System

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
# 1. Navigate to project
cd face-auth-system

# 2. Activate virtual environment
source venv/Scripts/activate        # Git Bash
# OR
venv\Scripts\activate               # Windows CMD

# 3. Install dependencies (if not already done)
pip install -r requirements.txt

# 4. Download dlib models (if not already done)
python -c "from app.services.feature_extraction import download_models; download_models()"

# 5. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Quick Test

```bash
# Enroll a user (file upload)
curl -X POST http://localhost:8000/enroll \
  -F "name=John" \
  -F "file=@path/to/john_photo.jpg"

# Verify (unimodal)
curl -X POST http://localhost:8000/verify \
  -F "file=@path/to/test_photo.jpg"

# Verify (multimodal — dlib + HOG fused)
curl -X POST "http://localhost:8000/multimodal/verify?weight=0.7" \
  -F "file=@path/to/test_photo.jpg"

# View metrics (synthetic scores)
curl http://localhost:8000/metrics

# Live webcam capture (open in browser)
# http://localhost:8000/capture
```

### ORL Dataset & Multimodal Evaluation

The system supports the **ORL (Olivetti Research Lab) face dataset** — 40 subjects, 10 images each (92×112 grayscale).

```bash
# 1. Bulk-enroll the dataset (first 5 images per subject, stores dlib + HOG features)
python scripts/bulk_enroll_orl.py --enroll-count 5

# 2. Real evaluation using ORL test images against enrolled gallery
curl -X POST "http://localhost:8000/metrics/evaluate-dataset?threshold=0.6&enroll_count=5"

# 3. Compare unimodal (dlib / HOG) vs multimodal (fused) performance
curl -X POST "http://localhost:8000/multimodal/evaluate?weight=0.7&enroll_count=5"
```

This replaces the random-synthetic scores in the demo routes with actual genuine vs impostor comparisons from the ORL dataset for proper FAR/FRR/EER calculation.

### API Documentation

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---



### Key Terms to Used

| Term | Meaning |
|------|---------|
| 128D embedding | 128-dimensional face feature vector |
| Cosine similarity | Measures angle between two vectors; higher = more similar |
| FAR | False Acceptance Rate — impostors let in |
| FRR | False Rejection Rate — legitimate users locked out |
| EER | Equal Error Rate — where FAR = FRR; lower is better |
| AUC | Area Under the ROC Curve; 1.0 = perfect |
| Fernet | Symmetric AES-128 encryption |
| Template | Mathematical representation of biometric data |
| 1:N identification | Comparing one face against many enrolled users |

### What Makes This Project Stand Out

1. **Full pipeline** — covers capture → preprocessing → extraction → matching → evaluation → security
2. **REST API** — not a script, but a production-ready system with Swagger docs
3. **Encrypted templates** — addresses real privacy concerns
4. **Performance metrics** — goes beyond "it works" to quantify accuracy
5. **Modular architecture** — separate services for each concern, easy to extend

---

## References

1. dlib C++ Library - http://dlib.net
2. FastAPI - https://fastapi.tiangolo.com
3. OpenCV - https://opencv.org
4. scikit-learn - https://scikit-learn.org
5. Jain, A. K., Ross, A., & Prabhakar, S. (2004). An introduction to biometric recognition. IEEE Transactions on Circuits and Systems for Video Technology.
6. ISO/IEC 19795-1:2006 - Biometric performance testing and reporting.
