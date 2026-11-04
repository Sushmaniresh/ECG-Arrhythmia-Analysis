# ECG Arrhythmia Analysis: Signal Processing, Machine Learning and Deep Learning

An end-to-end ECG arrhythmia pipeline on the MIT-BIH Arrhythmia Database, built step by step: signal processing and heart-rate analysis, then classical ML classifiers, then deep learning with interpretability.

Coursework for **EEE 598: Biomedical Machine Learning, Arizona State University (Fall 2025)**.

## Data
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet): 48 half-hour, two-lead ECG recordings sampled at 360 Hz, with beat-level annotations from cardiologists. Records are streamed directly with `wfdb`, so no data is stored in this repo.

## 1. Signal processing and HR/HRV analysis
`ECG_analysis.ipynb` · PDF export: `ECG-01.pdf`
- **Loading:** reads records and beat annotations with `wfdb`, and plots raw ECG with the annotated beats.
- **Preprocessing:** a Butterworth band-pass filter removes baseline drift (from breathing and movement) and high-frequency muscle noise; a notch filter removes power-line interference.
- **Transforms:** frequency-domain views (FFT and spectrogram), compared across 10 randomly selected records.
- **Features:** heart rate and heart-rate variability in both the time and frequency domains, compared across records.

## 2. Classifying normal vs arrhythmic segments with ML
`ECG-02-ML.pdf`
- Band-pass filtering (0.5–40 Hz), segmentation and feature extraction.
- Compared Logistic Regression, KNN, SVM (RBF), Random Forest and XGBoost, with nested cross-validation and hyperparameter tuning.
- Scored each model on sensitivity, specificity, precision, recall, F1, AUC-ROC and AUC-PR, to look at the trade-offs rather than accuracy alone.
- XGBoost performed best overall.

## 3. Beat classification with deep learning, and interpretability
`ECG-03-DL.pdf`
- **Preprocessing:** lead II (MLII), band-pass filtered 5–40 Hz, z-score normalized per record, polarity corrected, and R-peaks detected with PhysioNet's XQRS. Each beat becomes a segment centered on its R-peak.
- **Models:** a 1D CNN and a two-layer bidirectional LSTM.
- **Training:** tuned with cross-validation, selecting models on **macro-F1** to handle heavy class imbalance (most beats are normal).
- **Interpretability:** saliency maps show which parts of each heartbeat drive the prediction.

## Takeaways
- Filtering choices (the band-pass range, notch filtering and polarity correction) change downstream features and model results a lot.
- With imbalanced medical data, macro-F1 and AUC-PR are more honest than accuracy.
- Saliency maps help check that a model looks at physiologically meaningful parts of the beat, such as the QRS complex.

## Run it
```bash
pip install wfdb numpy scipy pandas matplotlib scikit-learn xgboost tensorflow
jupyter notebook ECG_analysis.ipynb
```
