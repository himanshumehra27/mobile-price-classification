# Mobile Price Classification

## Problem Statement

In the competitive mobile phone market, understanding the relationship between mobile phone specifications and their price range is crucial for both manufacturers and consumers. This project aims to develop machine learning models that can accurately classify mobile phones into different price ranges (Low, Medium, High, Very High) based on their technical specifications and features.

The classification problem involves predicting which price segment a mobile phone belongs to, given 20 different features including battery power, RAM, camera specifications, display properties, and connectivity options. This enables:
- Manufacturers to price their products competitively based on specifications
- Consumers to make informed purchasing decisions
- Market analysts to understand pricing trends in the mobile industry

## Dataset Description

**Dataset Name:** Mobile Price Classification Dataset  
**URL:** https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification  
**Source:** Kaggle  
**Total Instances:** 2000  
**Total Features:** 20 (+ 1 target variable)  
**Task Type:** Multi-class Classification (4 classes)

### Features Description

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| `battery_power` | Total energy a battery can store (mAh) | Continuous | 500-2000 |
| `blue` | Bluetooth availability | Binary | 0 (No), 1 (Yes) |
| `clock_speed` | Speed at which microprocessor executes (GHz) | Continuous | 0.5-3.0 |
| `dual_sim` | Dual SIM support | Binary | 0 (No), 1 (Yes) |
| `fc` | Front Camera megapixels | Discrete | 0-19 |
| `four_g` | 4G network support | Binary | 0 (No), 1 (Yes) |
| `int_memory` | Internal Memory (GB) | Discrete | 2-64 |
| `m_dep` | Mobile depth (cm) | Continuous | 0.1-1.0 |
| `mobile_wt` | Mobile weight (grams) | Discrete | 80-200 |
| `n_cores` | Number of processor cores | Discrete | 1-8 |
| `pc` | Primary Camera megapixels | Discrete | 0-20 |
| `px_height` | Pixel Resolution Height | Discrete | 0-1960 |
| `px_width` | Pixel Resolution Width | Discrete | 500-1998 |
| `ram` | Random Access Memory (MB) | Discrete | 256-3998 |
| `sc_h` | Screen Height (cm) | Discrete | 5-19 |
| `sc_w` | Screen Width (cm) | Discrete | 0-18 |
| `talk_time` | Longest time battery lasts during calls (hours) | Discrete | 2-20 |
| `three_g` | 3G network support | Binary | 0 (No), 1 (Yes) |
| `touch_screen` | Touch screen availability | Binary | 0 (No), 1 (Yes) |
| `wifi` | WiFi availability | Binary | 0 (No), 1 (Yes) |

### Target Variable

| Variable | Description | Values |
|----------|-------------|--------|
| `price_range` | Price category | 0 (Low), 1 (Medium), 2 (High), 3 (Very High) |

### Dataset Statistics

- **Training Set:** 1600 samples (80%)
- **Test Set:** 400 samples (20%)
- **Class Distribution:**
  - Low (0): 262 samples (13.1%)
  - Medium (1): 414 samples (20.7%)
  - High (2): 710 samples (35.5%)
  - Very High (3): 614 samples (30.7%)

## Models Used

### Model Comparison Table

| | Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 0 | **Logistic Regression** | 0.965000 | 0.998700 | 0.965000 | 0.965000 | 0.965000 | 0.953400 |
| 1 | **Decision Tree** | 0.820000 | 0.880200 | 0.824100 | 0.820000 | 0.820800 | 0.760700 |
| 2 | **K-Nearest Neighbors** | 0.500000 | 0.769700 | 0.521100 | 0.500000 | 0.505400 | 0.335000 |
| 3 | **Naive Bayes** | 0.810000 | 0.950600 | 0.811300 | 0.810000 | 0.810500 | 0.746800 |
| 4 | **Random Forest** | 0.892500 | 0.980600 | 0.891500 | 0.892500 | 0.891700 | 0.856800 |
| 5 | **XGBoost** | 0.932500 | 0.992900 | 0.932700 | 0.932500 | 0.932200 | 0.910200 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
| :--- | :--- |
| **Logistic Regression** | **Top Performer:** Achieved the highest accuracy (**96.50%**) and a near-perfect AUC (**0.9987**). This indicates that the relationship between specifications like RAM and price range is highly linear and well-separated. |
| **Decision Tree** | **Moderate Performance:** Captured non-linear patterns with an accuracy of **82.00%**. While interpretable, it lagged behind the ensemble methods and linear regression, likely due to a lack of depth optimization. |
| **K-Nearest Neighbors** | **Poorest Performance:** Struggled significantly with an accuracy of only **50.00%**. This suggests that distance-based classification is ineffective here, possibly due to high dimensionality or feature scaling issues. |
| **Naive Bayes** | **Strong Probabilistic Classifier:** Despite lower accuracy (**81.00%**), it maintained a high AUC (**0.9506**), showing it is very capable of ranking the classes correctly. |
| **Random Forest (Ensemble)** | **Robust & Balanced:** Delivered a strong accuracy of **89.25%**. By aggregating multiple trees, it effectively reduced variance and outperformed the single Decision Tree. |
| **XGBoost (Ensemble)** | **Excellent Gradient Boosting:** Produced the second-best results with **93.25%** accuracy and an exceptional AUC of **0.9929**, proving highly reliable for complex data patterns. |

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/himanshumehra27/mobile-price-classification.git
cd mobile-price-classification
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

4. **Access the application:**
Open your browser and navigate to `http://localhost:8501`

## Usage Guide

### Using the Streamlit App

1. **Select a Model:**
   - Choose from 6 different ML models in the sidebar dropdown
   - Each model has pre-trained weights loaded automatically

2. **Upload Test Data:**
   - Click "Browse files" in the sidebar
   - Upload a CSV file with mobile specifications
   - Ensure the file includes all 20 features plus the `price_range` target

3. **View Results:**
   - **Evaluation Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC
   - **Confusion Matrix:** Visual representation of prediction performance
   - **Classification Report:** Detailed per-class metrics
   - **Distribution Plots:** Compare actual vs predicted price ranges

4. **Download Predictions:**
   - Click "Download Predictions as CSV" to save results
   - File includes original data, predictions, and correctness indicator

### Training New Models

To retrain models on new data:

```bash
python model/model_training.py
```

This will:
- Load the dataset from `data/train.csv`
- Train all 6 models
- Save models to the `model/` directory
- Generate performance metrics
- Create test data samples

## Deployment on Streamlit Community Cloud

**App URL:**  https://mobile-prices-classification.streamlit.app  

### Streamlit App Features

✅ **Dataset Upload Option** - Easy CSV file upload with drag-and-drop  
✅ **Model Selection Dropdown** - Choose from 6 trained ML models  
✅ **Evaluation Metrics Display** - Comprehensive metrics dashboard  
✅ **Confusion Matrix Visualization** - Interactive heatmap of predictions  
✅ **Classification Report** - Detailed per-class performance metrics  
✅ **Prediction Distribution** - Visual comparison of actual vs predicted  
✅ **Download Results** - Export predictions as CSV file  
✅ **Model Comparison** - Side-by-side performance visualization  

---

**Last Updated:** 15, February 2026  
