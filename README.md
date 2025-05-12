# 📦 Demand Forecast App

This Streamlit app predicts product demand for the upcoming week using weather and holiday data.

## 🚀 Features

- 🔁 7-day demand forecasting
- 🌤️ Weather and public holiday integration
- 📊 Trend visualization (Plotly)
- 🧠 Auto-retraining after threshold met
- 🌍 Multilingual interface (EN/DE)

---

## 🧠 Model Overview

This app uses a **Random Forest Regressor** from scikit-learn, implemented with the following setup:

### 🔧 Preprocessing
- `StandardScaler` is applied before training.

### 🔍 Hyperparameter Tuning
- GridSearchCV is used to optimize:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]

### 🔄 Cross-validation
- **Nested cross-validation** strategy:
  - Inner CV: `KFold(n_splits=3, shuffle=True, random_state=42)`
  - Outer CV: `KFold(n_splits=5, shuffle=True, random_state=42)`

### 📈 Scoring
- Model performance is evaluated using **negative root mean squared error** (`neg_root_mean_squared_error`).

---

## 📦 How to Run Locally

```bash
git clone https://github.com/yourusername/demand-forecast-app.git
cd demand-forecast-app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🌐 Deployment on Streamlit Cloud

1. Upload this project to a GitHub repository.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud).
3. Click “New app”.
4. Choose your repo and set `streamlit_app.py` as the entry point.
5. (Optional) Include `setup.sh` to set server configs.

---

## 📁 File Overview

| File                  | Purpose                              |
|-----------------------|--------------------------------------|
| `streamlit_app.py`    | Main app logic                       |
| `train_demand_model.py` | Model training script               |
| `rf_demand_model.pkl` | Pre-trained model                    |
| `demand_prediction_log.csv` | Log of past predictions and actuals |
| `requirements.txt`    | Python dependencies                  |
| `setup.sh`            | Streamlit Cloud config (optional)    |

---

## ✨ Credits

Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org/), and [Visual Crossing Weather API](https://www.visualcrossing.com/).
