import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model, scaler, and PCA
model = joblib.load('ifm.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')  

# Streamlit app
st.title("Credit Card Fraud Detection")

# File upload
uploaded_file = st.file_uploader("Upload a file containing credit card transactions", type="excel")

if uploaded_file is not None:
    # Read the uploaded file
    new_data = pd.read_excel(uploaded_file)
    
    # Detect fraud
    frauds = detect_fraud(new_data, model, scaler, pca)
    
    st.write(f"Detected {len(frauds)} fraudulent transactions.")
    
    # Show detected frauds
    st.dataframe(frauds)
    
    # Visualize
    if not frauds.empty:
        st.write("Visualizing the anomalies:")
        pca_df = pd.DataFrame(pca.transform(scaler.transform(new_data.drop(columns=['Time', 'Class']))),
                              columns=['PC1', 'PC2'])  # Adjust columns as needed
        plot_anomalies(pca_df, frauds['Fraudulent'], title="Detected Anomalies")