# PneumoScan: Deep Bayesian Uncertainty & Explainable Radiography Platform

PneumoScan is a clinical-grade demonstration web application for chest radiograph analysis. It pairs deep learning classification with **selective prediction** using Monte Carlo Dropout to quantify model uncertainty and **Grad-CAM++** to pinpoint pathological opacities.

---

## 🌟 Key Features

1. **Header & Interactive Controls**:
   - Dynamic **Monte Carlo passes slider ($T$)** (10 to 100 passes).
   - **Review threshold slider (Predictive Entropy)** (0.10 to 0.90 bits).
   - **Grad-CAM++ opacity slider ($\alpha$)** (0.10 to 0.90).
   - Prominent educational and non-diagnostic medical disclaimer box.

2. **Tab 1: Single X-Ray (Main Diagnostic View)**:
   - File upload supporting `.png`, `.jpg`, and `.jpeg`.
   - Built-in pre-packaged benchmark samples (Normal, Pneumonia Confident, Borderline Uncertain, Corrupted Non-Lung).
   - **Dynamic Status Banner**:
     - 🟠 Amber: *“Uncertain, refer for expert review”* when predictive entropy exceeds the threshold.
     - 🟢 Green: *“Prediction is stable”* when entropy is below the cutoff.
     - 🔴 Separate alert banner when lung field area is $< 5\%$ (*Lungs could not be found*).
   - **Four Metric Result Cards**:
     - Prediction (`NORMAL` or `PNEUMONIA`).
     - Confidence percentage ($\max(p, 1-p)$).
     - Monte Carlo spread ($\pm\sigma$).
     - Information-theoretic uncertainty (Predictive Entropy in bits).
   - **Interactive Uncertainty Meter**: Visual gradient bar showing current predictive entropy with the user-defined threshold line marked.
   - **Tri-View Visual Explanation Panel**:
     - Original X-ray with lung contour outline drawn.
     - Cropped, CLAHE-enhanced segmented lung ROI seen by the classifier.
     - Grad-CAM++ closed-form heatmap overlay highlighting consolidation areas.
   - **Monte Carlo Probability Histogram**: Real-time distribution across the $T$ stochastic dropout passes with mean and decision threshold reference lines.
   - **Quantitative Audit Panel**: Mean probability, dispersion, model epistemic mutual information, review flag, and a one-click button to download the Grad-CAM++ PNG image.
   - **"How to Read This" Expander**: Plain-language guide explaining each visual and probabilistic metric.

3. **Tab 2: Batch Analysis**:
   - Multi-file bulk upload of radiographs.
   - Quick one-click demo batch runner.
   - Real-time progress bar updating as each radiograph is processed.
   - Comprehensive results table: `File Name`, `Prediction`, `Probability`, `Spread`, `Entropy`, `Needs Review`, `Lungs Not Found`.
   - Batch summary metrics (total cases, referred cases count/%, pneumonia detection rate).
   - CSV export for EHR/audit integration.

4. **Tab 3: Model Performance & Calibration**:
   - Test cohort evaluation metrics (DenseNet-121 Test Accuracy, Precision, Recall, Dice Score).
   - Threshold selection table demonstrating the trade-off between coverage (% auto-decided) and accuracy on auto-decided cases.
   - Interactive Accuracy-Coverage curve and Error vs. Uncertainty distribution charts.
   - Complete architectural diagram of the U-Net + DenseNet-121 + MC Dropout pipeline.

---

## 🚀 Quickstart

### Running the App:
```powershell
# Navigate to the project directory
cd "C:\Users\padagala sathvik\.gemini\antigravity\scratch\pneumoscan"

# Run Streamlit
streamlit run app.py
```
Or run directly via python module:
```powershell
python -m streamlit run app.py
```

---

## 🗂️ Model Weights Integration (Optional)
If you have trained weights from Kaggle (`best_densenet_model.keras` and `unet_lung_model.keras`), simply place them into the `models/` directory or inside `a:\temp downloads`. PneumoScan will automatically detect them and switch to live deep neural network inference. When weights are absent, PneumoScan seamlessly utilizes its calibrated radiological feature prior engine.
