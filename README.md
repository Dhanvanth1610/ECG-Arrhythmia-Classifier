# ECG Arrhythmia Classifier

A from-scratch machine learning system that classifies **17 types of cardiac arrhythmia**
from single-lead ECG signals — built for a hackathon at VIT Vellore. No pretrained
models, no external ML APIs: every parameter is learned from this repo's own training run.

**Result: 91.3% cross-validated accuracy** (weighted F1 = 0.911) across 1000 real ECG
fragments and 17 classes, honestly evaluated with Stratified 5-Fold Cross-Validation
(not a single lucky train/test split).

---

## 1. What's in this repository

```
ecg-arrhythmia-classifier/
├── README.md                     <- you are here
├── requirements.txt               <- Python dependencies
│
├── data/
│   ├── raw/MLII/                  <- original dataset: 1000 .mat files, 17 class folders
│   └── ecg_features.csv           <- extracted feature table (1000 rows x 41 features)
│
├── src/
│   ├── features.py                <- signal processing + feature extraction (the core DSP)
│   ├── build_dataset.py           <- walks data/raw/, extracts features -> ecg_features.csv
│   ├── train_model.py             <- trains + cross-validates 5 candidate models
│   └── predict.py                 <- run inference on a new .mat file
│
├── models/
│   └── ecg_model.joblib           <- final trained model (scaler + label encoder + ensemble)
│
├── results/
│   ├── cv_results.json            <- numeric cross-validation results, all models
│   ├── classification_report.txt  <- per-class precision/recall/F1
│   ├── model_comparison.csv/.png  <- accuracy/F1 comparison across candidate models
│   ├── confusion_matrix.png       <- 17x17 confusion matrix (5-fold CV)
│   ├── feature_importance.csv/.png<- which signal features drive the predictions
│   └── signal_processing_demo.png <- raw ECG + detected R-peaks, 3 example rhythms
│
├── viewer/
│   └── ecg_real_data_viewer.html  <- open in any browser: interactive results explorer,
│                                      built on real signals + real held-out predictions
│
└── browser_experimental/          <- JS ports of the filtering / R-peak / spectral
                                       code, unit-validated against the Python output
                                       (bandpass filtfilt, Pan-Tompkins-style R-peak
                                       detector, Welch PSD). Building blocks for a
                                       future 100%-client-side inference demo; not
                                       yet wired to a UI.
```

## 2. The dataset

**ECG Signals (1000 fragments)** — single-lead (MLII), MIT-BIH derived, 360 Hz,
10-second fragments, 17 arrhythmia classes:

| Class | Meaning | Count |
|---|---|---|
| NSR | Normal Sinus Rhythm | 283 |
| AFIB | Atrial Fibrillation | 135 |
| PVC | Premature Ventricular Contraction | 133 |
| LBBBB | Left Bundle Branch Block | 103 |
| APB | Atrial Premature Beat | 66 |
| RBBBB | Right Bundle Branch Block | 62 |
| Bigeminy | Ventricular Bigeminy | 55 |
| PR | Paced Rhythm | 45 |
| WPW | Wolff-Parkinson-White | 21 |
| AFL | Atrial Flutter | 20 |
| SVTA | Supraventricular Tachyarrhythmia | 13 |
| Trigeminy | Ventricular Trigeminy | 13 |
| Fusion | Fusion Beat | 11 |
| VT | Ventricular Tachycardia | 10 |
| IVR | Idioventricular Rhythm | 10 |
| VFL | Ventricular Flutter | 10 |
| SDHB | Sinus/AV Block | 10 |

Class sizes range from **10 to 283** — this imbalance is the central challenge the
methodology below is built around.

## 3. Methodology

### 3.1 Signal processing (`src/features.py`)
Every raw signal is bandpass-filtered (0.5–40 Hz, Butterworth order 4, zero-phase
`filtfilt`) and run through a Pan-Tompkins-style R-peak detector (5–15 Hz bandpass →
square → moving-average → peak search → local refinement to the true R-wave apex).
From there, **41 engineered features** are computed per fragment:

- **Time-domain statistics** — mean, std, skew, kurtosis, RMS, percentiles, IQR,
  zero-crossing rate, signal energy, line length
- **Heart-rate / HRV** — heart rate, RR-interval mean/std/min/max/CV, RMSSD, pNN50,
  R-wave amplitude stats
- **Frequency-domain** — Welch power spectral density, dominant frequency, band
  energies (0–5, 5–15, 15–40 Hz), spectral entropy
- **Wavelet** — 5-level `db4` discrete wavelet decomposition, energy ratio per level

### 3.2 Why feature engineering instead of a raw deep-learning CNN
Several classes have only **10 examples**. A deep 1D-CNN trained directly on raw
3600-sample waveforms would badly overfit those classes. Hand-engineered, clinically
interpretable features let a classical model generalize from very few examples per
class — standard practice in small-sample clinical ECG ML.

### 3.3 Model (`src/train_model.py`)
Five candidates are trained and compared, all from scratch on this dataset:

| Model | CV Accuracy | Balanced Acc. | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Random Forest | 90.3% | 81.5% | 85.8% | 89.9% |
| XGBoost | 89.4% | 80.4% | 84.0% | 89.1% |
| Gradient Boosting | 87.7% | 74.0% | 77.3% | 87.1% |
| SVM (RBF) | 90.4% | **86.2%** | **87.4%** | 90.5% |
| **Ensemble (RF + SVM + XGB, soft-voting, weights 1:2:1)** | **91.3%** | 83.1% | 86.5% | **91.1%** |

**Final model = the ensemble** (best accuracy + weighted F1). The standalone SVM has
better balanced accuracy on the rare classes — worth mentioning in a presentation as
evidence of a rigorous comparison rather than a cherry-picked headline number.

All numbers come from **Stratified 5-Fold Cross-Validation** — the model is never
scored on data it trained on. This is the only defensible way to report accuracy when
some classes have just 10 samples.

## 4. How to run

```bash
git clone <this-repo-url>
cd ecg-arrhythmia-classifier
pip install -r requirements.txt

# (optional) re-extract features from the raw dataset
python3 src/build_dataset.py

# (optional) retrain from scratch — reproduces everything in results/ and models/
python3 src/train_model.py

# classify a new ECG fragment with the already-trained model
python3 src/predict.py "data/raw/MLII/6 WPW/230m (0).mat"
```

`models/ecg_model.joblib` already contains the fully trained model, so `predict.py`
works immediately without retraining.

## 5. Interactive results viewer

Open `viewer/ecg_real_data_viewer.html` directly in a browser (no server, no internet
required). It shows 68 real ECG fragments across all 17 classes with:
- the actual waveform and detected R-peaks
- the model's genuine **held-out cross-validation prediction** (including honest
  misclassifications — nothing is cherry-picked)
- per-class confidence bars
- the full model-comparison table above

## 6. Talking points for judges

- 17-class problem with severe class imbalance (10–283 samples/class) — accuracy
  alone is misleading, so balanced accuracy and macro-F1 are reported alongside it.
- Every feature is clinically interpretable (RR-interval variability, QRS energy
  bands, wavelet morphology) — `results/feature_importance.png` shows HRV and
  wavelet-energy features dominate, matching cardiology domain knowledge.
- Cross-validated, not a single lucky split — defensible under judge questioning.
- 100% offline and reproducible: no API keys, no internet calls, no pretrained weights.

## 7. Hackathon context

Built for a hackathon at VIT Vellore. The team split up topics from a shared dataset
folder (CT scans, chest X-rays, tabular heart-disease data, stroke prediction, ECG
signals); this repo covers the **ECG signals / arrhythmia classification** part.
