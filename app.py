"""
Mobile Price Classification - Streamlit Web Application
Author: Himanshu Kumar Mehra
BITS ID: 2025AA05048
Date: 15 February, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_auc_score)
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mobile Price Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Mobile Price Classification System")
st.markdown("""
This application predicts mobile phone price ranges using machine learning models.
Upload your test data and select a model to get predictions and evaluation metrics.
""")

# Sidebar
st.sidebar.header("Model Configuration")
st.sidebar.markdown("---")

# Model selection dropdown
model_options = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'K-Nearest Neighbors': 'model/knn.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Random Forest': 'model/random_forest.pkl',
    'XGBoost': 'model/xgboost.pkl'
}

selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    options=list(model_options.keys()),
    help="Choose a machine learning model for prediction"
)

# Load model results
@st.cache_data
def load_model_results():
    """Load pre-computed model results"""
    try:
        results_df = pd.read_csv('model/model_results.csv')
        return results_df
    except:
        return None

# Load model and scaler
@st.cache_resource
def load_model_and_scaler(model_path):
    """Load the selected model and scaler"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load('model/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# File upload section
st.sidebar.markdown("---")
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with mobile specifications"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Expected Features:**
- battery_power, blue, clock_speed
- dual_sim, fc, four_g, int_memory
- m_dep, mobile_wt, n_cores, pc
- px_height, px_width, ram
- sc_h, sc_w, talk_time
- three_g, touch_screen, wifi
- price_range (target)
""")

# Main content
if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Data uploaded successfully! ({len(df)} samples)")
        
        # Display data preview
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check if target column exists
        if 'price_range' not in df.columns:
            st.error("❌ Error: 'price_range' column not found in the dataset!")
            st.stop()
        
        # Separate features and target
        X_test = df.drop('price_range', axis=1)
        y_test = df['price_range']
        
        # Load model and scaler
        model, scaler = load_model_and_scaler(model_options[selected_model_name])
        
        if model is not None and scaler is not None:
            # Scale the features
            X_test_scaled = scaler.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Display model information
            st.header(f"Model: {selected_model_name}")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            except:
                auc = 0.0
            
            # Display metrics in columns
            st.subheader("Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
            
            with col2:
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
            
            with col3:
                st.metric("AUC Score", f"{auc:.4f}")
                st.metric("MCC Score", f"{mcc:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Low', 'Medium', 'High', 'Very High'],
                       yticklabels=['Low', 'Medium', 'High', 'Very High'],
                       ax=ax)
            ax.set_xlabel('Predicted Price Range')
            ax.set_ylabel('Actual Price Range')
            ax.set_title(f'Confusion Matrix - {selected_model_name}')
            
            st.pyplot(fig)
            plt.close()
            
            # Classification Report
            st.subheader("Classification Report")
            
            report = classification_report(y_test, y_pred, 
                                          target_names=['Low', 'Medium', 'High', 'Very High'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            st.dataframe(report_df.style.background_gradient(cmap='YlOrRd', subset=['f1-score']),
                        use_container_width=True)
            
            # Prediction distribution
            st.subheader("Prediction Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                pd.Series(y_test).value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Actual Price Range Distribution')
                ax.set_xlabel('Price Range')
                ax.set_ylabel('Count')
                ax.set_xticklabels(['Low', 'Medium', 'High', 'Very High'], rotation=45)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                pd.Series(y_pred).value_counts().sort_index().plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title('Predicted Price Range Distribution')
                ax.set_xlabel('Price Range')
                ax.set_ylabel('Count')
                ax.set_xticklabels(['Low', 'Medium', 'High', 'Very High'], rotation=45)
                st.pyplot(fig)
                plt.close()
            
            # Download predictions
            st.subheader("Download Predictions")
            
            results_df = df.copy()
            results_df['predicted_price_range'] = y_pred
            results_df['prediction_correct'] = (y_test == y_pred)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        st.info("Please ensure your CSV file has the correct format and column names.")

else:
    # Show instructions when no file is uploaded
    st.info("Please upload a test dataset CSV file from the sidebar to begin.")
    
    # Display all model results if available
    results_df = load_model_results()
    
    if results_df is not None:
        st.header("All Models Performance Comparison")
        
        st.dataframe(
            results_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC', 'F1']),
            use_container_width=True
        )
        
        # Visualize model comparison
        st.subheader("Model Performance Visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel('')
            ax.set_ylabel(metric)
            ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Best model
        best_model_idx = results_df['Accuracy'].idxmax()
        best_model = results_df.loc[best_model_idx, 'Model']
        best_accuracy = results_df.loc[best_model_idx, 'Accuracy']
        
        st.success(f"**Best Performing Model:** {best_model} with {best_accuracy:.4f} accuracy")
    
    # Sample data format
    st.header("Sample Data Format")
    
    sample_data = {
        'battery_power': [842, 1021, 563],
        'blue': [0, 1, 1],
        'clock_speed': [2.2, 0.5, 0.5],
        'dual_sim': [0, 1, 1],
        'fc': [1, 0, 2],
        'four_g': [0, 1, 1],
        'int_memory': [7, 53, 41],
        'm_dep': [0.6, 0.7, 0.9],
        'mobile_wt': [188, 136, 145],
        'n_cores': [2, 3, 5],
        'pc': [2, 6, 6],
        'px_height': [20, 905, 1263],
        'px_width': [756, 1988, 1716],
        'ram': [2549, 2631, 2603],
        'sc_h': [9, 17, 11],
        'sc_w': [7, 3, 2],
        'talk_time': [19, 7, 9],
        'three_g': [0, 1, 1],
        'touch_screen': [0, 1, 1],
        'wifi': [1, 0, 0],
        'price_range': [1, 2, 2]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    
    # Download sample data
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="Download Sample Data",
        data=csv,
        file_name="data/sample_mobile_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Mobile Price Classification System</p>
</div>
""", unsafe_allow_html=True)
