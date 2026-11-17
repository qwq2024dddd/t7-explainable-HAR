# Explainable AI for Human Activity Recognition: SHAP vs. Integrated Gradients

An empirical comparison of two explanation methods — SHAP (TreeSHAP on Random Forest) and
Integrated Gradients (on a 1D-CNN) — on the PAMAP2 dataset, with deletion-test validation
of the explanations.

## Key findings
- Random Forest (Macro-F1 0.892) generalizes across subjects much better than
  the 1D-CNN (0.609), which overfits training subjects.
- SHAP's sensor ranking (ankle most important) is confirmed by deletion tests
  under two replacement methods (zeroing and mean imputation).
- Per-activity deletion confirms SHAP's claims: walking relies on the ankle;
  Nordic walking needs both hand and ankle; the ankle distinguishes ironing
  from vacuuming.
- Integrated Gradients does not yield a clear, validated sensor ranking for
  the CNN.

## Repository contents
- `T7.ipynb` — full pipeline: data download, preprocessing, windowing,
  RF + CNN training, SHAP/IG explanations, deletion tests, figures.
- `figures/` — paper figures (heatmaps, deletion comparison, confusion matrices).
- `cnn_seed0_v2.pt` — trained 1D-CNN weights (seed 0).
- `norm_stats.npz` — per-channel standardization statistics (training set).

## Reproduce
1. Open `T7.ipynb` in Google Colab (T4 GPU runtime recommended).
2. Run all cells. The notebook downloads PAMAP2 (~656 MB) from UCI,
   preprocesses it (8 subjects, 12 activities, 7,412 windows of 5.12 s
   @ 100 Hz, 50% overlap), trains both models (3 seeds), and regenerates
   all tables and figures.
3. Runtime: about 30–45 minutes on a Colab T4.

## Data
PAMAP2 (Reiss & Stricker, ISWC 2012):
https://archive.ics.uci.edu/dataset/231/pamap2-physical-activity-monitoring
- Subject-independent split: train on subjects 101–106, test on 107–108.
  Subject 109 excluded (only 23 windows, data mostly missing).
- 27 channels: accelerometer (±16 g), gyroscope, magnetometer ×
  hand/chest/ankle @ 100 Hz.

## Environment
- Python 3.13, PyTorch 2.14, scikit-learn, SHAP 0.54, Captum 0.9
  (see notebook install cell).

## References
- Lundberg & Lee (2017), "A Unified Approach to Interpreting Model
  Predictions" (SHAP), NeurIPS 2017.
- Sundararajan, Taly & Yan (2017), "Axiomatic Attribution for Deep
  Networks" (Integrated Gradients), ICML 2017.
- Hooker et al. (2019), "A Benchmark for Interpretability Methods in Deep
  Neural Networks" (ROAR), NeurIPS 2019.
- Reiss & Stricker (2012), "Introducing a New Benchmarked Dataset for
  Activity Monitoring", ISWC 2012.
- Reiss & Stricker (2012), "Creating and Benchmarking a New Dataset for
  Physical Activity Monitoring", ABRA 2012.
