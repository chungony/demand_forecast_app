# 📦 Demand Forecast App

This Streamlit app predicts product demand for the upcoming week using weather and holiday data in Baden-Württemberg (Germany).

## 🚀 Features

- 🔁 7-day demand forecasting
- 🌤️ Weather and public holiday integration
- 📊 Trend visualization (Plotly)
- 🧠 Auto-retraining after threshold met
- 🌍 Multilingual interface (EN/DE)

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
4. Choose your repo and set `app.py` as the entry point.
5. (Optional) Include `setup.sh` to set server configs.

---

## 📁 File Overview

| File                  | Purpose                              |
|-----------------------|--------------------------------------|
| `app.py`              | Main app logic                       |
| `train_demand_model.py` | Model training script               |
| `rf_demand_model.pkl` | Pre-trained model                    |
| `demand_prediction_log.csv` | Log of past predictions and actuals |
| `requirements.txt`    | Python dependencies                  |
| `setup.sh`            | Streamlit Cloud config (optional)    |

---

## ✨ Credits

Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org/), and [Visual Crossing Weather API](https://www.visualcrossing.com/).
