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
project2/
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

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- pip or conda

### 2. Installation

Clone the repository and install dependencies:

```bash
cd project2
pip install -r requirements.txt
```

### 3. Train the Model

Run the training notebook:

```bash
jupyter notebook notebooks/training.ipynb
```

Or train via Python:

```bash
python train_model.py
```

### 4. Launch the Web App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

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

## 💡 How to Customize

### Use Your Own Dataset

1. Place your CSV file in the `data/` folder
2. Update the `data_loader.py` to match your column names
3. Modify `app.py` to use your dataset path

### Add More Models

In `src/model_utils.py`, add to the `train_models()` function:

```python
from sklearn.svm import SVR

models['SVM'] = SVR(kernel='rbf')
models['SVM'].fit(X_train, y_train)
```

### Customize the Web App

Edit `app.py` to add:
- Custom CSS styling
- Additional visualizations
- More input features
- Deployment configurations

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (FREE)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your repository
4. Done! Your app is live

### Option 2: Heroku

1. Create `Procfile`:
```
web: streamlit run app.py --logger.level=error
```

2. Deploy:
```bash
git push heroku main
```

### Option 3: AWS/GCP

Use containerization (Docker) for cloud deployment.

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ML Best Practices](https://github.com/microsoft/ML-For-Beginners)

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new features
- Improve model performance
- Create additional visualizations
- Add unit tests




## 📞 Questions?

Review the code comments and documentation in each file for detailed explanations.

---


