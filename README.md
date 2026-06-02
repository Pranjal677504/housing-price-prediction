# Housing Price Prediction ML Project

A machine learning project for predicting housing prices using regression models. Perfect for demonstrating ML skills for internship applications.

## 📋 Project Overview

This project builds and deploys a **housing price prediction model** that takes property characteristics as input and predicts their market value. It showcases:

- ✅ Data exploration and preprocessing
- ✅ Multiple regression model training
- ✅ Model evaluation and comparison
- ✅ Interactive Streamlit web application
- ✅ Production-ready code structure

## 🏗️ Project Structure

```
housing-price-prediction/
├── data/                          # Data files
├── notebooks/                     # Jupyter notebooks for exploration
│   └── training.ipynb
├── src/                          # Python modules
│   ├── data_loader.py            # Data loading utilities
│   └── model_utils.py            # Model training utilities
├── models/                       # Saved trained models
├── app.py                        # Streamlit application
├── requirements.txt              # Project dependencies
├── .gitignore
└── README.md
```







## Live Demo

[Open Streamlit App](https://housing-price-prediction-001.streamlit.app/)

## 📊 Models Included

The project trains and compares 4 regression models:

1. **Linear Regression** - Baseline model
2. **Ridge Regression** - L2 regularization
3. **Random Forest** - Ensemble method
4. **Gradient Boosting** - Advanced ensemble

## 📈 Dataset

The project uses a sample housing dataset with features including:
- Crime rate (CRIM)
- Average rooms per dwelling (RM)
- Pupil-teacher ratio (PTRATIO)
- Status of the population (LSTAT)
- And 10 more features


## 🖼️ Application Preview

### Input Dashboard

![Input Dashboard](images/housing%20price%20prediction.png)

### Prediction Results

![Prediction Results](images/prediction%20results.png)



## 🎯 Performance Metrics

Models are evaluated using:
- **R² Score** - Coefficient of determination
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error

## 🖥️ Web Application Features

The Streamlit app includes:

- 📊 Model selection and comparison
- 🔍 Feature input interface for predictions
- 📈 Performance visualization
- 📉 Feature importance analysis
- 💾 Model information and metrics


