import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Dataset source: Kaggle - Mobile Price Classification
# URL - https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

# Features: battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
#           m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w,
#           talk_time, three_g, touch_screen, wifi

# Target: price_range (0: Low, 1: Medium, 2: High, 3: Very High)

def load_and_preprocess_data(filepath):
    """Load and preprocess the mobile price dataset"""
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop('price_range', axis=1)
    y = df['price_range']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'model/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Calculate all required evaluation metrics"""
    
    # For multiclass, we need to specify average method
    accuracy = accuracy_score(y_true, y_pred)
    
    # AUC Score - for multiclass, use ovr (one-vs-rest)
    try:
        auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    # Precision, Recall, F1 - use weighted average for multiclass
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # MCC Score
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    }
    
    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and evaluate them"""
    
    results = []
    models_dict = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, 'Logistic Regression')
    results.append(metrics)
    models_dict['Logistic Regression'] = lr_model
    joblib.dump(lr_model, 'model/logistic_regression.pkl')
    
    # 2. Decision Tree Classifier
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    y_pred_proba = dt_model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, 'Decision Tree')
    results.append(metrics)
    models_dict['Decision Tree'] = dt_model
    joblib.dump(dt_model, 'model/decision_tree.pkl')
    
    # 3. K-Nearest Neighbors
    print("Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    y_pred_proba = knn_model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, 'K-Nearest Neighbors')
    results.append(metrics)
    models_dict['K-Nearest Neighbors'] = knn_model
    joblib.dump(knn_model, 'model/knn.pkl')
    
    # 4. Naive Bayes (Gaussian)
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    y_pred_proba = nb_model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, 'Naive Bayes')
    results.append(metrics)
    models_dict['Naive Bayes'] = nb_model
    joblib.dump(nb_model, 'model/naive_bayes.pkl')
    
    # 5. Random Forest (Ensemble)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, 'Random Forest')
    results.append(metrics)
    models_dict['Random Forest'] = rf_model
    joblib.dump(rf_model, 'model/random_forest.pkl')
    
    # 6. XGBoost (Using Gradient Boosting as alternative since xgboost might not be available)
    print("Training Gradient Boosting (XGBoost Alternative)...")
    from sklearn.ensemble import GradientBoostingClassifier
    xgb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, 'XGBoost')
    results.append(metrics)
    models_dict['XGBoost'] = xgb_model
    joblib.dump(xgb_model, 'model/xgboost.pkl')
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    return results_df, models_dict

def main():
    """Main execution function"""
    
    # Create model directory
    import os
    os.makedirs('model', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data('data/train.csv')
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results_df, models_dict = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_df.to_csv('model/model_results.csv', index=False)
    
    # Display results
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Save test data for streamlit app
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_df['price_range'] = y_test.values
    X_test_df.to_csv('data/test_data_sample.csv', index=False)
    
    print("\nAll models trained and saved successfully!")
    print("Results saved to: model/model_results.csv")
    print("Test data saved to: test_data_sample.csv")

if __name__ == "__main__":
    main()
