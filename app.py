"""
Streamlit application for housing price prediction
"""
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
        h1 {
            color: #1f77b4;
        }
        .metric-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Sidebar
st.sidebar.title("🎛️ Control Panel")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Predict Price",
    "📊 Data & Models",
    "📈 Model Comparison",
    "ℹ️ About"
])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    st.header("🏠 Housing Price Prediction")
    
    # Load and train models automatically on startup if not already in session state
if "models" not in st.session_state:
    with st.spinner("Initializing system, loading data and training models..."):
        # # Load data (Your original logic)
        df = load_housing_data()
        X, y, feature_names = preprocess_data(df)
        X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(X, y)

        # # Train models (Your original logic)
        models = train_models(X_train, y_train)
        metrics = evaluate_models(models, X_train, X_test, y_train, y_test)

        # # Save to session state (Your original logic)
        st.session_state.models = models
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names
        st.session_state.metrics = metrics

        # # Save best model (Your original logic)
        best_model_name = max(metrics, key=lambda x: metrics[x]['test_r2'])
        save_model(models[best_model_name], 'models/best_model.pkl')
        st.session_state.best_model_name = best_model_name

# Display the success message automatically to the recruiter
if "best_model_name" in st.session_state:
    st.success(f"✅ Models trained! Best model: {st.session_state.best_model_name}")
    # Prediction interface
    if st.session_state.models is not None:
        st.subheader("Enter Property Features:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            crim = st.slider("Crime Rate (CRIM)", min_value=0.0, max_value=100.0, value=3.0)
            zn = st.slider("Zoned Land (ZN)", min_value=0.0, max_value=100.0, value=20.0)
            indus = st.slider("Industrial (INDUS)", min_value=0.0, max_value=30.0, value=10.0)
        
        with col2:
            rm = st.slider("Avg Rooms (RM)", min_value=3.0, max_value=9.0, value=6.0)
            age = st.slider("Age of Building (AGE)", min_value=0.0, max_value=100.0, value=50.0)
            dis = st.slider("Distance to CBD (DIS)", min_value=1.0, max_value=13.0, value=5.0)
        
        with col3:
            ptratio = st.slider("Pupil-Teacher Ratio (PTRATIO)", min_value=12.0, max_value=23.0, value=16.0)
            lstat = st.slider("Lower Status % (LSTAT)", min_value=0.0, max_value=40.0, value=10.0)
            nox = st.slider("NO2 Concentration (NOX)", min_value=0.3, max_value=1.0, value=0.55)
        
        # Additional features
        chas = st.selectbox("Near Charles River (CHAS)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        rad = st.slider("Radial Highway Index (RAD)", min_value=1, max_value=24, value=5)
        tax = st.slider("Tax Rate (TAX)", min_value=187, max_value=711, value=400)
        b = st.slider("Black Population % (B)", min_value=0.0, max_value=400.0, value=300.0)
        
        # Make prediction
        if st.button("🎯 Predict Price", key="predict_btn"):
            # Prepare input
            input_data = np.array([[
                crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat
            ]])
            
            # Scale input
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Get predictions from all models
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Predictions by Model:")
                predictions = {}
                for model_name, model in st.session_state.models.items():
                    pred = model.predict(input_scaled)[0]
                    predictions[model_name] = pred
                    metric_value = f"${pred:,.2f}" if pred > 0 else "Error"
                    st.metric(model_name, metric_value)
            
            with col2:
                st.subheader("📈 Average Prediction:")
                avg_pred = np.mean(list(predictions.values()))
                st.metric("Ensemble Average", f"${avg_pred:,.2f}", delta=f"±${np.std(list(predictions.values())):,.0f}")
                
                # Model performance
                best_model = max(st.session_state.metrics, key=lambda x: st.session_state.metrics[x]['test_r2'])
                st.info(f"Best performing model: **{best_model}**")
    else:
        st.warning("⚠️ Please train models first using the button in the sidebar!")


# ==================== TAB 2: DATA & MODELS ====================
with tab2:
    st.header("📊 Data Exploration & Model Information")
    
    if st.session_state.models is None:
        st.info("Train models first to see data exploration and model details.")
    else:
        # Data statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            df = load_housing_data()
            st.write(df.describe().round(2))
        
        with col2:
            st.subheader("Dataset Info")
            st.write(f"**Total Samples:** {len(df)}")
            st.write(f"**Total Features:** {len(df.columns) - 1}")
            st.write(f"**Target Variable:** PRICE")
            st.write(f"**Data Types:** {df.dtypes.to_dict()}")
        
        # Feature correlations
        st.subheader("Feature Correlations with Price")
        X, y, _ = preprocess_data(df)
        correlations = X.corrwith(y).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        correlations.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Correlation with Price')
        ax.set_title('Feature Correlation Analysis')
        st.pyplot(fig)
        
        # Model information
        st.subheader("Trained Models")
        for model_name, model in st.session_state.models.items():
            with st.expander(f"📋 {model_name}"):
                st.write(f"**Type:** {type(model).__name__}")
                st.write(f"**Parameters:** {model.get_params()}")


# ==================== TAB 3: MODEL COMPARISON ====================
with tab3:
    st.header("📈 Model Performance Comparison")
    
    if st.session_state.metrics is None:
        st.info("Train models first to see performance metrics.")
    else:
        # Create comparison dataframe
        metrics_df = pd.DataFrame(st.session_state.metrics).T
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("R² Score Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(metrics_df))
            width = 0.35
            ax.bar(x - width/2, metrics_df['train_r2'], width, label='Train', color='skyblue')
            ax.bar(x + width/2, metrics_df['test_r2'], width, label='Test', color='orange')
            ax.set_ylabel('R² Score')
            ax.set_title('R² Score by Model')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            st.subheader("RMSE Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - width/2, metrics_df['train_rmse'], width, label='Train', color='lightcoral')
            ax.bar(x + width/2, metrics_df['test_rmse'], width, label='Test', color='darkred')
            ax.set_ylabel('RMSE ($)')
            ax.set_title('RMSE by Model')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
            ax.legend()
            st.pyplot(fig)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(metrics_df.round(4), use_container_width=True)
        
        # Best model
        best_model_r2 = metrics_df['test_r2'].idxmax()
        best_model_rmse = metrics_df['test_rmse'].idxmin()
        
        st.success(f"🏆 Best R² Score: **{best_model_r2}**")
        st.success(f"🏆 Best RMSE: **{best_model_rmse}**")


# ==================== TAB 4: ABOUT ====================
with tab4:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🏠 Housing Price Prediction ML Project
    
    This is a comprehensive machine learning project designed to predict housing prices based on property characteristics.
    
    ### 🎯 Project Goals
    - Build and train multiple regression models
    - Compare model performance
    - Create an interactive prediction interface
    - Demonstrate ML best practices
    
    ### 📚 What's Included
    - **Data Loading & Preprocessing:** Handles data cleaning and normalization
    - **Model Training:** Trains 4 different regression models
    - **Evaluation:** Compares models using R², RMSE, and MAE
    - **Deployment:** Interactive Streamlit web application
    - **Documentation:** Complete project documentation
    
    ### 🛠️ Technologies Used
    - **Python 3.8+**
    - **Scikit-learn:** Machine learning models
    - **Pandas & NumPy:** Data manipulation
    - **Streamlit:** Web application framework
    - **Matplotlib & Seaborn:** Data visualization
    
    ### 📁 Project Structure
    ```
    project2/
    ├── data/                  # Datasets
    ├── notebooks/             # Jupyter notebooks
    ├── src/                   # Python modules
    ├── models/                # Saved models
    ├── app.py                 # Main Streamlit app
    ├── requirements.txt       # Dependencies
    └── README.md              # Documentation
    ```
    
    ### 🚀 Next Steps
    1. Train models using the "Train Models" button
    2. Explore data in the "Data & Models" tab
    3. Compare model performance in the "Model Comparison" tab
    4. Make predictions in the "Predict Price" tab
    
    ### 💡 Tips for Internship Interviews
    - **Explain your models:** Understand why you chose each model
    - **Discuss trade-offs:** Linear vs. complex models
    - **Share improvements:** What would you do differently?
    - **Show your process:** Include exploratory analysis
    
    ### 📖 Learn More
    - [Scikit-learn Docs](https://scikit-learn.org/)
    - [Streamlit Docs](https://docs.streamlit.io/)
    - [ML Best Practices](https://github.com/microsoft/ML-For-Beginners)
    
    ---
    
    **Created with ❤️ for aspiring data scientists**
    """)
