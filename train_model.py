"""
Script to train and save ML models
Run this to train models without using Streamlit
"""
import sys
import os
from src.data_loader import load_housing_data, preprocess_data, prepare_train_test_data
from src.model_utils import train_models, evaluate_models, save_model


def main():
    """Train and evaluate models"""
    print("🚀 Starting model training...")
    print("-" * 50)
    
    # Load and prepare data
    print("\n📊 Loading data...")
    df = load_housing_data()
    print(f"Dataset shape: {df.shape}")
    
    print("\n🔧 Preprocessing data...")
    X, y, feature_names = preprocess_data(df)
    print(f"Features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    
    print("\n✂️ Splitting data...")
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(X, y)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train models
    print("\n🤖 Training models...")
    models = train_models(X_train, y_train)
    print(f"Trained {len(models)} models: {list(models.keys())}")
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    metrics = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Print results
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"  Train R²:     {model_metrics['train_r2']:.4f}")
        print(f"  Test R²:      {model_metrics['test_r2']:.4f}")
        print(f"  Train RMSE:   ${model_metrics['train_rmse']:,.2f}")
        print(f"  Test RMSE:    ${model_metrics['test_rmse']:,.2f}")
        print(f"  Train MAE:    ${model_metrics['train_mae']:,.2f}")
        print(f"  Test MAE:     ${model_metrics['test_mae']:,.2f}")
    
    # Find and save best model
    best_model_name = max(metrics, key=lambda x: metrics[x]['test_r2'])
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test R² Score: {metrics[best_model_name]['test_r2']:.4f}")
    
    # Save best model
    best_model = models[best_model_name]
    save_model(best_model, 'models/best_model.pkl')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n✅ Scaler saved to models/scaler.pkl")
    
    print("\n" + "=" * 70)
    print("✅ Model training complete!")
    print("=" * 70)
    print("\n🚀 To run the Streamlit app: streamlit run app.py")


if __name__ == "__main__":
    main()
