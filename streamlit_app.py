import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import base64
import hashlib
import json
from train_demand_model import retrain as run_training_logic
from translations import translations


# --- Password protection ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input(
            "🔒 Enter password / Passwort eingeben",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.stop()
    elif not st.session_state["authenticated"]:
        st.error("❌ Incorrect password / Falsches Passwort")
        st.stop()


check_password()


# --- Parameters ---
MODEL_FILE = "rf_demand_model.pkl"
LOG_FILE_NAME = "demand_prediction_log.csv"
RETRAIN_MARKER_FILE = "last_retrain_marker.txt"
RETRAIN_THRESHOLD = 24
VC_API_KEY = st.secrets["api_key"]
GITHUB_TOKEN = st.secrets["github_token"]
GITHUB_USERNAME = st.secrets["github_username"]
GITHUB_REPO = "demand_forecast_log"
GITHUB_BRANCH = "main"


# --- Functions ---
def fetch_from_github(
    path, repo=GITHUB_REPO, branch=GITHUB_BRANCH, filename=LOG_FILE_NAME
):
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{path}?ref={branch}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = base64.b64decode(response.json()["content"])
        with open(filename, "wb") as f:
            f.write(content)
        return filename
    else:
        st.error("❌ Failed to fetch log file from GitHub")
        st.stop()


def upload_to_github(df, path, repo=GITHUB_REPO, branch=GITHUB_BRANCH):
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    new_content = df.to_csv(index=False)
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()

    # Get current file content and compare hash
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        current_content = base64.b64decode(response.json()["content"]).decode()
        current_hash = hashlib.sha256(current_content.encode()).hexdigest()
        if current_hash == new_hash:
            return  # No change, skip upload
        sha = response.json().get("sha")
    else:
        sha = None

    encoded_content = base64.b64encode(new_content.encode()).decode()
    data = {
        "message": "Update demand prediction log",
        "content": encoded_content,
        "branch": branch,
    }
    if sha:
        data["sha"] = sha

    put_response = requests.put(url, headers=headers, json=data)
    if put_response.status_code in [200, 201]:
        st.success("✅ Log file automatically pushed to GitHub!")
    else:
        st.error(f"❌ GitHub upload failed: {put_response.json()}")


def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        st.error("Model not found.")
        st.stop()


def is_holiday_in_bw(date_str):
    try:
        year = int(date_str.split("-")[0])
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/DE"
        response = requests.get(url)
        holidays = response.json()
        return any(
            h["date"] == date_str and (not h["counties"] or "DE-BW" in h["counties"])
            for h in holidays
        )
    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError):
        return False


def fetch_weather(date_str):
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Stuttgart,DE/{date_str}?unitGroup=metric&key={VC_API_KEY}&include=days"
        response = requests.get(url)
        day = response.json().get("days", [{}])[0]
        return (
            day.get("temp", 22.0),
            day.get("precip", 1.0),
            day.get("conditions", "Clear"),
        )
    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError):
        return 22.0, 1.0, "Clear"


def prepare_new_entry(
    model, forecast_date_str, is_holiday_bw, temperature, rainfall, condition
):
    condition_map = {"Clear": 0, "Rain": 1, "Partially cloudy": 2, "Overcast": 3}
    condition_code = condition_map.get(condition, 0)
    features = {
        "Holiday": int(is_holiday_bw),
        "Temperature": temperature,
        "Rainfall": rainfall,
        "Condition": condition_code,
    }
    prediction = model.predict(pd.DataFrame([features]))[0]
    return pd.DataFrame(
        [
            {
                "prediction_for_date": forecast_date_str,
                **features,
                "PredictedDemand": prediction,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    )


def load_logs():
    try:
        df = pd.read_csv(LOG_FILE)
        df["prediction_for_date"] = pd.to_datetime(
            df["prediction_for_date"], errors="coerce", dayfirst=True, format="mixed"
        ).dt.strftime("%Y-%m-%d")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def save_logs(logs):
    logs.to_csv(LOG_FILE, index=False)


def should_retrain(logs):
    verified_logs = logs.dropna(subset=["ActualDemand"])
    current_count = len(verified_logs)
    if os.path.exists(RETRAIN_MARKER_FILE):
        with open(RETRAIN_MARKER_FILE, "r") as f:
            last_count = int(f.read().strip())
    else:
        last_count = 0
    return current_count - last_count >= RETRAIN_THRESHOLD


def retrain_model():
    st.info(_["retraining_info"])
    try:
        success, msg = run_training_logic()
        if success:
            st.success(_["retraining_success"])
        else:
            st.warning(msg)
    except Exception as e:
        st.error(_["retraining_error"].format(e))


def get_sorted_logs_by_date(
    logs: pd.DataFrame, date_col: str = "prediction_for_date"
) -> pd.DataFrame:
    """
    Ensures consistent date format and returns logs sorted by the given date column.
    """
    logs[date_col] = pd.to_datetime(logs[date_col], errors="coerce").dt.date
    return logs.dropna(subset=[date_col]).sort_values(date_col)


# --- Fetch logs from GitHub---
GITHUB_PATH = f"logs/{LOG_FILE_NAME}"
LOG_FILE = fetch_from_github(GITHUB_PATH, filename=LOG_FILE_NAME)


# --- Streamlit UI ---
st.set_page_config(page_title="Demand Forecast / Nachfrageprognose", layout="centered")

lang = st.selectbox("🌐 Language / Sprache", options=["en", "de"], index=1)
_ = translations[lang]

st.title(_["title"])


# --- Forecast Form ---
with st.form("forecast_form"):
    actual_date = st.date_input(_["select_date"], value=datetime.today().date())
    actual_demand = st.number_input(_["actual_demand"], min_value=0)
    overwrite_confirm = st.checkbox(_["overwrite"])
    submitted = st.form_submit_button(_["submit"])

logs = load_logs()

if submitted:
    if actual_date.weekday() != 3:
        st.warning(_["not_thursday"])
        st.stop()

    model = load_model()
    forecast_date = actual_date + timedelta(days=7)
    forecast_date_str = forecast_date.strftime("%Y-%m-%d")

    is_holiday_bw = is_holiday_in_bw(forecast_date_str)
    temperature, rainfall, condition = fetch_weather(forecast_date_str)

    new_entry = prepare_new_entry(
        model, forecast_date_str, is_holiday_bw, temperature, rainfall, condition
    )

    logs["prediction_for_date"] = pd.to_datetime(logs["prediction_for_date"]).dt.date
    matched = logs[logs["prediction_for_date"] == actual_date]
    if not matched.empty:
        if pd.isna(matched.iloc[0].get("ActualDemand")) or overwrite_confirm:
            logs.loc[matched.index[0], "ActualDemand"] = actual_demand
            forecasted = float(logs.at[matched.index[0], "PredictedDemand"])
            error = forecasted - float(actual_demand)
            st.success(_["actual_recorded"].format(error))
        else:
            st.warning(_["entry_exists"])
    else:
        st.info(_["no_forecast"])

    if not (logs["prediction_for_date"] == forecast_date).any():
        logs = pd.concat([logs, new_entry], ignore_index=True)
        st.success(
            _["forecasted_demand"].format(
                forecast_date_str, new_entry["PredictedDemand"].values[0]
            )
        )
    else:
        predicted_row = logs[logs["prediction_for_date"] == forecast_date]
        predicted_value = predicted_row["PredictedDemand"].values[0]
        st.warning(_["already_exists"].format(forecast_date_str, predicted_value))

    # Translate condition to local language
    condition_key = "condition_" + condition.lower().replace(" ", "_")
    translated_condition = _.get(condition_key, condition)

    # Show forecast context
    st.markdown(f"### {_['forecast_context_title']}")
    formatted_date = forecast_date.strftime(_["date_format"])
    st.write(f"{_['forecast_date_label']}: {formatted_date}")
    st.write(f"{_['forecast_holiday_label']}: {_['yes'] if is_holiday_bw else _['no']}")
    st.write(f"{_['forecast_temp_label']}: {temperature:.1f} °C")
    st.write(f"{_['forecast_rain_label']}: {rainfall:.1f} mm")
    st.write(f"{_['forecast_cond_label']}: {translated_condition}")

    save_logs(logs)

    # Update retraining status after submission
    verified_logs = logs.dropna(subset=["ActualDemand"])
    current_count = len(verified_logs)
    if os.path.exists(RETRAIN_MARKER_FILE):
        with open(RETRAIN_MARKER_FILE, "r") as f:
            last_count = int(f.read().strip())
    else:
        last_count = 0
    entries_remaining = RETRAIN_THRESHOLD - (current_count - last_count)
    if entries_remaining > 0:
        st.info(_["retraining_remaining"].format(entries_remaining))
    else:
        st.info(_["retraining_reached"])


# if should_retrain(logs):
#    retrain_model()


# --- Show last 12 weeks of predictions based on the most recent prediction date ---
if not logs.empty and "PredictedDemand" in logs.columns:
    logs_sorted = get_sorted_logs_by_date(logs)
    last_date = logs_sorted["prediction_for_date"].max()
    start_date = last_date - timedelta(weeks=12)
    logs_recent = logs_sorted[logs_sorted["prediction_for_date"] >= start_date]

    fig = go.Figure()

    # Predicted Demand
    fig.add_trace(
        go.Scatter(
            x=logs_recent["prediction_for_date"],
            y=logs_recent["PredictedDemand"],
            mode="lines+markers",
            name=_["label_predicted"],
            line=dict(color="royalblue"),
        )
    )

    # Actual Demand
    if "ActualDemand" in logs_recent.columns:
        fig.add_trace(
            go.Scatter(
                x=logs_recent["prediction_for_date"],
                y=logs_recent["ActualDemand"],
                mode="lines+markers",
                name=_["label_actual"],
                line=dict(color="darkorange"),
            )
        )

    fig.update_layout(
        title=_["trend_chart_title"],
        xaxis_title=_["label_date"],
        yaxis_title=_["label_demand"],
        xaxis=dict(
            tickmode="linear",
            tick0=logs_recent["prediction_for_date"].min(),
            dtick=7 * 24 * 60 * 60 * 1000,  # 1 week in milliseconds
            tickformat=_["date_format"],
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


# --- Auto-upload logs at the end ---
if not logs.empty:
    upload_to_github(logs, GITHUB_PATH)
