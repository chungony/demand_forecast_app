import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import os
from train_demand_model import retrain as run_training_logic
import plotly.graph_objects as go

MODEL_FILE = "rf_demand_model.pkl"
LOG_FILE = "demand_prediction_log.csv"
RETRAIN_MARKER_FILE = "last_retrain_marker.txt"
RETRAIN_THRESHOLD = 5
api_key = "H9T4DCAATLPQ5MT7XW9NRXGY8"  # API key for Visual Crossing

translations = {
    "en": {
        "title": "📦 Demand Forecast for the Coming Week (Step)",
        "select_date": "Select a Thursday",
        "actual_demand": "Actual Demand for selected date",
        "overwrite": "✅ Overwrite existing actual demand if it exists?",
        "submit": "Submit and Predict",
        "not_thursday": "⚠️ Please select a Thursday.",
        "actual_recorded": "✅ Actual demand recorded. Forecast error: {:.0f} units.",
        "entry_exists": "⚠️ Entry already exists for this date. Check the box to overwrite actual demand.",
        "no_forecast": "ℹ️ No forecast to match this actual demand yet.",
        "forecasted_demand": "📈 Forecasted demand for {}: {:.0f}",
        "already_exists": "⚠️ Forecast for {} already exists. Forecasted demand: {:.0f}",
        "retraining_info": "🔁 Retraining threshold reached. Calling training script...",
        "retraining_success": "✅ Model retrained successfully.",
        "retraining_fail": "❌ Retraining script failed.",
        "retraining_error": "⚠️ Could not run training script: {}",
        "retraining_remaining": "🔄 {} more entries with actual demand before the model is retrained.",
        "retraining_reached": "✅ Retraining threshold reached. Model will be updated after this submission.",
        "trend_chart_title": "Demand Prediction Trend",
        "label_predicted": "Predicted Demand",
        "label_actual": "Actual Demand",
        "label_date": "Date",
        "label_demand": "Demand",
        "forecast_context_title": "🌤️ Forecast Context",
        "forecast_date_label": "📅 Date",
        "forecast_holiday_label": "🏖️ Public Holiday in BW",
        "forecast_temp_label": "🌡️ Temperature",
        "forecast_rain_label": "🌧️ Precipitation",
        "forecast_cond_label": "☁️ Weather Condition",
        "condition_clear": "Clear",
        "condition_rain": "Rain",
        "condition_partially_cloudy": "Partially cloudy",
        "condition_overcast": "Overcast",
        "yes": "Yes",
        "no": "No",
        "date_format": "%b %d, %Y",
    },
    "de": {
        "title": "📦 Nachfrageprognose für die kommende Woche (Step)",
        "select_date": "Wähle einen Donnerstag",
        "actual_demand": "Tatsächliche Nachfrage für das ausgewählte Datum",
        "overwrite": "✅ Vorhandene tatsächliche Nachfrage überschreiben?",
        "submit": "Absenden und Prognostizieren",
        "not_thursday": "⚠️ Bitte wählen Sie einen Donnerstag.",
        "actual_recorded": "✅ Tatsächliche Nachfrage erfasst. Prognosefehler: {:.0f} Einheiten.",
        "entry_exists": "⚠️ Eintrag für dieses Datum existiert bereits. Aktivieren Sie das Kontrollkästchen, um zu überschreiben.",
        "no_forecast": "ℹ️ Keine Prognose vorhanden, um dieser Nachfrage zu entsprechen.",
        "forecasted_demand": "📈 Prognostizierte Nachfrage für {}: {:.0f}",
        "already_exists": "⚠️ Prognose für {} existiert bereits. Prognostizierte Nachfrage: {:.0f}",
        "retraining_info": "🔁 Schwelle für erneutes Training erreicht. Trainingsskript wird aufgerufen...",
        "retraining_success": "✅ Modell erfolgreich neu trainiert.",
        "retraining_fail": "❌ Trainingsskript fehlgeschlagen.",
        "retraining_error": "⚠️ Trainingsskript konnte nicht ausgeführt werden: {}",
        "retraining_remaining": "🔄 Noch {} Einträge mit tatsächlicher Nachfrage, bevor das Modell neu trainiert wird.",
        "retraining_reached": "✅ Schwelle für erneutes Training erreicht. Das Modell wird nach dieser Eingabe aktualisiert.",
        "trend_chart_title": "Prognosetrend der Nachfrage",
        "label_predicted": "Prognostizierte Nachfrage",
        "label_actual": "Tatsächliche Nachfrage",
        "label_date": "Datum",
        "label_demand": "Nachfrage",
        "forecast_context_title": "🌤️ Prognosekontext",
        "forecast_date_label": "📅 Datum",
        "forecast_holiday_label": "🏖️ Feiertag in BW",
        "forecast_temp_label": "🌡️ Temperatur",
        "forecast_rain_label": "🌧️ Niederschlag",
        "forecast_cond_label": "☁️ Wetterzustand",
        "condition_clear": "Klar",
        "condition_rain": "Regen",
        "condition_partially_cloudy": "Teilweise bewölkt",
        "condition_overcast": "Bedeckt",
        "yes": "Ja",
        "no": "Nein",
        "date_format": "%d.%m.%Y",
    },
}


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
    except:
        return False


def fetch_weather(date_str):
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Stuttgart,DE/{date_str}?unitGroup=metric&key={api_key}&include=days"
        response = requests.get(url)
        day = response.json().get("days", [{}])[0]
        return (
            day.get("temp", 22.0),
            day.get("precip", 1.0),
            day.get("conditions", "Clear"),
        )
    except:
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
            df["prediction_for_date"],
            format="mixed",
            dayfirst=True,
            errors="coerce",  # Optional: set to 'raise' if you want to see bad formats
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


# --- Streamlit UI ---
st.set_page_config(page_title="7-Day Demand Forecast (DE-BW)", layout="centered")
logs = load_logs()

lang = st.selectbox("🌐 Language / Sprache", options=["en", "de"], index=1)
_ = translations[lang]

st.title(_["title"])


# --- Forecast Form ---
with st.form("forecast_form"):
    actual_date = st.date_input(_["select_date"], value=datetime.today().date())
    actual_demand = st.number_input(_["actual_demand"], min_value=0)
    overwrite_confirm = st.checkbox(_["overwrite"])
    submitted = st.form_submit_button(_["submit"])


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
            forecasted = logs.loc[matched.index[0], "PredictedDemand"]
            error = forecasted - actual_demand
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


if should_retrain(logs):
    retrain_model()


# Show last 12 weeks of predictions based on the most recent prediction date
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
    )

    st.plotly_chart(fig, use_container_width=True)
