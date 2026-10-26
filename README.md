# Housing Price Prediction

An end-to-end machine learning project that predicts housing prices from property and neighbourhood characteristics. It covers data exploration, training and comparing four regression models, and serving predictions through an interactive Streamlit app.

**[Live Demo](https://housing-price-prediction-001.streamlit.app/)**

![Input Dashboard](images/housing%20price%20prediction.png)

## Highlights

- Exploratory data analysis and preprocessing in a Jupyter notebook
- Four regression models trained and compared on the same train/test split
- Evaluation with R², RMSE and MAE
- Streamlit app for live predictions, model comparison and feature importance
- Modular code structure: data loading and model utilities live in `src/`, separate from the app and notebook

## Models

| Model | Type | Role |
|---|---|---|
| Linear Regression | Linear | Baseline |
| Ridge Regression | Linear, L2 regularization | Reduces overfitting versus the baseline |
| Random Forest | Ensemble (bagging) | Captures non-linear relationships |
| Gradient Boosting | Ensemble (boosting) | Typically the strongest of the four on tabular data |

## Results

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | – | – | – |
| Ridge Regression | – | – | – |
| Random Forest | – | – | – |
| Gradient Boosting | – | – | – |

- **R²**: share of variance in price explained by the model
- **RMSE**: root mean squared error, which penalizes large misses more heavily
- **MAE**: mean absolute error, the average size of a miss in the target's units

## Dataset

The project uses a sample housing dataset. Example features include:

- `CRIM`: per-capita crime rate
- `RM`: average number of rooms per dwelling
- `PTRATIO`: pupil-teacher ratio
- `LSTAT`: percentage of lower-status population
- 10 additional features

## Web Application

The Streamlit app lets you:

- Select a model and compare it against the others
- Enter property features and get an instant price prediction
- View performance metrics and visualizations
- Explore feature importance

![Prediction Results](images/prediction%20results.png)

## Project Structure

```
housing-price-prediction/
├── data/                  # Dataset files
├── images/                # Screenshots used in this README
├── models/                # Saved trained models
├── notebooks/
│   └── training.ipynb     # EDA, training and evaluation
├── src/
│   ├── data_loader.py     # Data loading utilities
│   └── model_utils.py     # Model training utilities
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Pranjal677504/housing-price-prediction.git
cd housing-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

To retrain the models, open `notebooks/training.ipynb` and run all cells.

## Tech Stack

Python, scikit-learn, pandas, NumPy, Streamlit
