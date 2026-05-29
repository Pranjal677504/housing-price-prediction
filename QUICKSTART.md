# Quick Start Guide - Housing Price Prediction ML Project

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
cd /Users/pranjalprajapati/Desktop/project2
pip install -r requirements.txt
```

### Step 2: Option A - Run the Web App (Easiest!)
```bash
streamlit run app.py
```
Then click "🚀 Train Models" button in the sidebar. The app will open at `http://localhost:8501`

### Step 3: Option B - Run the Training Notebook
```bash
jupyter notebook notebooks/training.ipynb
```
Run all cells from top to bottom to see the complete ML workflow.

### Step 4: Option C - Train via Command Line
```bash
python train_model.py
```

---

## 📁 Project Files Overview

| File | Purpose |
|------|---------|
| `app.py` | Interactive Streamlit web application |
| `train_model.py` | Standalone training script |
| `notebooks/training.ipynb` | Jupyter notebook with complete walkthrough |
| `src/data_loader.py` | Data loading and preprocessing utilities |
| `src/model_utils.py` | Model training and evaluation utilities |
| `requirements.txt` | Python dependencies |
| `README.md` | Full project documentation |

---

## 🎯 What You'll Learn

✅ Data exploration and visualization  
✅ Data preprocessing and scaling  
✅ Training multiple ML models  
✅ Model evaluation and comparison  
✅ Creating an interactive web interface  
✅ Saving and loading models  
✅ Deployment strategies  

---

## 📊 Project Features

**Models Included:**
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting

**Evaluation Metrics:**
- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

**Interactive Dashboard:**
- Real-time price predictions
- Feature input sliders
- Model comparison charts
- Feature correlation analysis

---

## 🌐 Deploy Your Project (Next Steps)

### To Heroku (Takes 5 minutes)
```bash
# Install Heroku CLI, then:
heroku login
heroku create your-app-name
git push heroku main
# Your app is live at: your-app-name.herokuapp.com
```

### To Streamlit Cloud (Easiest!)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Select your repository and click "Deploy"
5. Your app is live in seconds!

### To AWS/Google Cloud/Azure
See `README.md` for Docker containerization instructions.

---

## 💡 Interview Tips

When talking about this project:
- **Explain why** you chose each model
- **Discuss trade-offs** (simple vs complex, accuracy vs speed)
- **Mention improvements** (what would you do next?)
- **Show understanding** of metrics (R², RMSE, why they matter)
- **Demo the app** live to your interviewer

---

## ❓ Common Questions

**Q: Do I need my own dataset?**  
A: No! The project includes a sample dataset generator. Later, replace it with your own data.

**Q: Can I add more models?**  
A: Yes! Edit `src/model_utils.py` and add to the `train_models()` function.

**Q: How do I customize the web app?**  
A: Edit `app.py` to add features, visualizations, or styling.

**Q: Can I use this for production?**  
A: The structure is production-ready. For real-world use, add error handling, logging, and unit tests.

---

## 🎉 You're Ready!

This is a complete, portfolio-quality ML project. Good luck with your internship applications!

**Questions?** Check the code comments - they're detailed and educational.
