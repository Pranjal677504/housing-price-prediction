import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_housing_data, preprocess_data, prepare_train_test_data
from src.model_utils import train_models, evaluate_models, save_model, load_model
import os
import joblib

# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.header("🏠 Housing Price Prediction")

# Load and train models automatically on startup if not already in session state
if "models" not in st.session_state:
    with st.spinner("Initializing system, loading data and training models..."):
        df = load_housing_data()
        X, y, feature_names = preprocess_data(df)
        X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(X, y)

        models = train_models(X_train, y_train)
        metrics = evaluate_models(models, X_train, X_test, y_train, y_test)

        st.session_state.models = models
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names
        st.session_state.metrics = metrics

        best_model_name = max(metrics, key=lambda x: metrics[x]['test_r2'])
        save_model(models[best_model_name], 'models/best_model.pkl')
        st.session_state.best_model_name = best_model_name

# Display the success message automatically to the recruiter
if "best_model_name" in st.session_state:
    st.success(f"✅ Models trained! Best model: {st.session_state.best_model_name}")

if st.session_state.models is not None:
    st.subheader("Enter Property Features:")
    
    coll, col2, col3 = st.columns(3)
    
    with coll:
        crim = st.slider("Crime Rate (CRIM)", min_value=0.0, max_value=100.0, value=3.0)
        zn = st.slider("Zoned Land (ZN)", min_value=0.0, max_value=100.0, value=20.0)
        indus = st.slider("Industrial (INDUS)", min_value=0.0, max_value=30.0, value=10.0)
        
    with col2:
        rm = st.slider("Avg Rooms (RM)", min_value=3.0, max_value=9.0, value=6.0)
        age = st.slider("Age of Building (AGE)", min_value=0.0, max_value=100.0, value=50.0)
        dis = st.slider("Distance to CBD (DIS)", min_value=1.0, max_value=13.0, value=5.0)

    with col3:
        rad = st.slider("Radial Highway Index (RAD)", min_value=1, max_value=24, value=5)
        tax = st.slider("Tax Rate (TAX)", min_value=180, max_value=720, value=400)
        ptratio = st.slider("Pupil-Teacher Ratio (PTRATIO)", min_value=12.0, max_value=23.0, value=16.0)

    nox = st.slider("NO2 Concentration (NOX)", min_value=0.3, max_value=1.0, value=0.55)
    lstat = st.slider("Lower Status % (LSTAT)", min_value=0.0, max_value=40.0, value=10.0)
    chas = st.selectbox("Near Charles River (CHAS)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    b = st.slider("Black Population % (B)", min_value=0.0, max_value=400.0, value=300.0)

    if st.button("🎯 Predict Price"):
        input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
        input_scaled = st.session_state.scaler.transform(input_data)

        prediction_results = {}
        for model_name, model in st.session_state.models.items():
            pred = model.predict(input_scaled)[0]
            prediction_results[model_name] = pred

        best_model_name = st.session_state.best_model_name
        best_prediction = prediction_results[best_model_name]
        average_prediction = np.mean(list(prediction_results.values()))

        st.markdown("### Prediction Results")
        st.write(f"**Best model:** {best_model_name}")
        st.write(f"**Predicted price:** ${best_prediction:,.2f}")
        st.write(f"**Average prediction across models:** ${average_prediction:,.2f}")

        cols = st.columns(2)
        with cols[0]:
            st.subheader("All model predictions")
            for model_name, pred in prediction_results.items():
                st.metric(model_name, f"${pred:,.2f}")

        with cols[1]:
            metrics_df = pd.DataFrame(st.session_state.metrics).T
            st.subheader("Model performance")
            st.dataframe(metrics_df[['test_r2', 'test_rmse', 'test_mae']].round(4))
