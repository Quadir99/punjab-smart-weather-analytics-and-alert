from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable

import folium
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium


BASE_URL = "https://api.openweathermap.org/data/2.5"
DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "weather_history.csv"
GEOJSON_FILE = DATA_DIR / "punjab_districts.geojson"
TELEGRAM_STATE_FILE = DATA_DIR / "telegram_alert_state.json"
HISTORY_COLUMNS = [
    "Fetched_At",
    "City",
    "Crop_Focus",
    "Temp",
    "Humidity",
    "Pressure",
    "Visibility",
    "Wind_Speed",
    "Clouds",
    "Weather_Desc",
    "Lat",
    "Lon",
    "Forecast_Rain_Events",
    "Forecast_Min_Temp",
    "Forecast_Max_Temp",
    "Forecast_Min_Visibility",
    "Forecast_Note",
    "Risk_Score",
    "Alert_Band",
    "Smart_Alerts",
    "Advisory",
]

LOCATIONS = {
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "crop_focus": "Wheat"},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723, "crop_focus": "Wheat"},
    "Bathinda": {"lat": 30.2110, "lon": 74.9455, "crop_focus": "Cotton"},
    "Patiala": {"lat": 30.33, "lon": 76.40, "crop_focus": "Paddy"},
    "Jalandhar": {"lat": 31.33, "lon": 75.57, "crop_focus": "Maize"},
    "Mansa": {"lat": 29.99, "lon": 75.40, "crop_focus": "Cotton"},
    "Barnala": {"lat": 30.37, "lon": 75.55, "crop_focus": "Wheat"},
}

CROP_GUIDANCE = {
    "Wheat": {
        "heat_temp": 33,
        "message": "Protect grain filling by shifting irrigation to evening hours.",
    },
    "Paddy": {
        "heat_temp": 35,
        "message": "Watch water availability and avoid fertilizer application before rainfall.",
    },
    "Cotton": {
        "heat_temp": 36,
        "message": "Monitor leaf stress and postpone spraying during strong afternoon winds.",
    },
    "Maize": {
        "heat_temp": 34,
        "message": "Check tasseling-stage moisture demand and avoid moisture shock.",
    },
}


@dataclass
class ForecastSummary:
    rain_events: int
    min_temp: float | None
    max_temp: float | None
    min_visibility: float | None
    forecast_note: str


def format_visibility(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value)} m"


def _safe_get(url: str, timeout: int = 20) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_current_weather(api_key: str) -> pd.DataFrame:
    records: list[dict] = []

    for city, meta in LOCATIONS.items():
        url = (
            f"{BASE_URL}/weather?lat={meta['lat']}&lon={meta['lon']}"
            f"&appid={api_key}&units=metric"
        )
        payload = _safe_get(url)
        weather_desc = payload.get("weather", [{}])[0].get("description", "")
        records.append(
            {
                "City": city,
                "Crop_Focus": meta["crop_focus"],
                "Temp": payload.get("main", {}).get("temp"),
                "Humidity": payload.get("main", {}).get("humidity"),
                "Pressure": payload.get("main", {}).get("pressure"),
                "Visibility": payload.get("visibility"),
                "Wind_Speed": payload.get("wind", {}).get("speed"),
                "Clouds": payload.get("clouds", {}).get("all"),
                "Weather_Desc": weather_desc,
                "Lat": meta["lat"],
                "Lon": meta["lon"],
            }
        )

    return pd.DataFrame(records)


def summarize_forecast(api_key: str, city: str, lat: float, lon: float) -> ForecastSummary:
    url = f"{BASE_URL}/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    payload = _safe_get(url)
    items = payload.get("list", [])
    if not items:
        return ForecastSummary(0, None, None, None, "Forecast unavailable")

    temps = [item.get("main", {}).get("temp") for item in items if item.get("main")]
    temps = [temp for temp in temps if temp is not None]
    visibilities = [item.get("visibility") for item in items if item.get("visibility") is not None]
    rain_events = 0

    for item in items[:8]:
        description = item.get("weather", [{}])[0].get("description", "").lower()
        rain_volume = item.get("rain", {}).get("3h", 0)
        if "rain" in description or rain_volume:
            rain_events += 1

    max_temp = max(temps) if temps else None
    min_temp = min(temps) if temps else None
    min_visibility = min(visibilities) if visibilities else None

    if rain_events >= 3:
        note = f"{city} may see recurring rainfall in the next 24 hours."
    elif max_temp is not None and max_temp >= 36:
        note = f"{city} shows a high heat build-up in the next 24 hours."
    elif min_visibility is not None and min_visibility < 1000:
        note = f"{city} may face poor visibility in the next 24 hours."
    else:
        note = f"{city} has relatively stable conditions in the next 24 hours."

    return ForecastSummary(rain_events, min_temp, max_temp, min_visibility, note)


def assess_conditions(
    *,
    temp: float | None,
    pressure: float | None,
    visibility: float | None,
    humidity: float | None,
    weather_desc: str,
    wind_speed: float | None,
    crop_focus: str,
    forecast: ForecastSummary,
) -> tuple[int, str, str, str]:
    score = 10
    alerts: list[str] = []
    description = (weather_desc or "").lower()
    crop_profile = CROP_GUIDANCE.get(crop_focus, CROP_GUIDANCE["Wheat"])

    if temp is not None:
        temp_excess = temp - crop_profile["heat_temp"]
        if temp_excess > 0:
            score += min(30, int(round(temp_excess * 8)))
            alerts.append("Heat stress risk")
        elif temp >= crop_profile["heat_temp"] - 1:
            score += 6
            alerts.append("Near heat threshold")

    if forecast.max_temp is not None:
        forecast_excess = forecast.max_temp - crop_profile["heat_temp"]
        if forecast_excess > 0:
            score += min(20, int(round(forecast_excess * 4)))
            alerts.append("Forecast heat spike")
        elif forecast.max_temp >= crop_profile["heat_temp"] - 1:
            score += 4
            alerts.append("Forecast warming")

    if pressure is not None and pressure < 1008 and "rain" in description:
        score += 18
        alerts.append("Storm pressure signal")
    elif pressure is not None and pressure < 1010 and "rain" in description:
        score += 10
        alerts.append("Rain pressure signal")

    if visibility is not None:
        if visibility < 500:
            score += 20
            alerts.append("Severe low visibility")
        elif visibility < 1000:
            score += 14
            alerts.append("Low visibility")
        elif visibility < 4000:
            score += 8
            alerts.append("Reduced visibility")

    if humidity is not None:
        if humidity > 90:
            score += 12
            alerts.append("High humidity disease risk")
        elif humidity > 80:
            score += 8
            alerts.append("Elevated humidity")
        elif humidity < 25 and temp is not None and temp > crop_profile["heat_temp"]:
            score += 6
            alerts.append("Dry heat stress")

    if wind_speed is not None:
        if wind_speed > 10:
            score += 10
            alerts.append("High wind exposure")
        elif wind_speed > 7:
            score += 6
            alerts.append("Wind exposure")

    if forecast.rain_events >= 4:
        score += 12
        alerts.append("Rain buildup")
    elif forecast.rain_events >= 2:
        score += 6
        alerts.append("Possible rainfall window")

    if forecast.min_visibility is not None:
        if forecast.min_visibility < 1000:
            score += 8
            alerts.append("Forecast visibility drop")
        elif forecast.min_visibility < 4000:
            score += 4
            alerts.append("Forecast haze risk")

    if temp is not None and humidity is not None and temp > 34 and humidity > 70:
        score += 8
        alerts.append("Heat-humidity stress")

    if temp is not None and forecast.max_temp is not None and forecast.max_temp - temp >= 3:
        score += 6
        alerts.append("Rapid heat escalation")

    score = max(0, min(100, score))
    if score >= 70:
        band = "High"
    elif score >= 40:
        band = "Moderate"
    else:
        band = "Low"

    alerts_text = "Stable weather window" if not alerts else " | ".join(alerts)
    advisory_parts = [crop_profile["message"], forecast.forecast_note]
    if visibility is not None and visibility < 1000:
        advisory_parts.insert(0, "Transport and spraying should be delayed until visibility improves.")
    advisory = " ".join(advisory_parts)
    return score, band, alerts_text, advisory


def generate_crop_recommendation(
    *,
    crop_focus: str,
    temp: float | None,
    humidity: float | None,
    visibility: float | None,
    wind_speed: float | None,
    forecast: ForecastSummary,
) -> str:
    crop_profile = CROP_GUIDANCE.get(crop_focus, CROP_GUIDANCE["Wheat"])
    recommendations: list[str] = []

    if temp is not None and temp >= crop_profile["heat_temp"]:
        recommendations.append("Shift irrigation to late evening to reduce heat stress.")
    if forecast.max_temp is not None and forecast.max_temp >= crop_profile["heat_temp"] + 2:
        recommendations.append("Prepare for hotter conditions over the next 24 hours.")
    if visibility is not None and visibility < 1000:
        recommendations.append("Delay transport and spraying until visibility improves.")
    if wind_speed is not None and wind_speed > 7:
        recommendations.append("Avoid chemical spraying during strong winds.")
    if humidity is not None and humidity > 80:
        recommendations.append("Monitor fungal and leaf-disease pressure in humid fields.")
    if forecast.rain_events >= 2:
        recommendations.append("Hold fertilizer application if rainfall is likely soon.")

    if not recommendations:
        recommendations.append(crop_profile["message"])
    return " ".join(recommendations)


def build_enriched_dataset(current_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    enriched_rows: list[dict] = []

    for _, row in current_df.iterrows():
        forecast = summarize_forecast(api_key, row["City"], row["Lat"], row["Lon"])
        risk_score, alert_band, smart_alerts, advisory = assess_conditions(
            temp=row["Temp"],
            pressure=row["Pressure"],
            visibility=row["Visibility"],
            humidity=row["Humidity"],
            weather_desc=row["Weather_Desc"],
            wind_speed=row["Wind_Speed"],
            crop_focus=row["Crop_Focus"],
            forecast=forecast,
        )
        enriched_rows.append(
            {
                **row.to_dict(),
                "Forecast_Rain_Events": forecast.rain_events,
                "Forecast_Min_Temp": forecast.min_temp,
                "Forecast_Max_Temp": forecast.max_temp,
                "Forecast_Min_Visibility": forecast.min_visibility,
                "Forecast_Note": forecast.forecast_note,
                "Risk_Score": risk_score,
                "Alert_Band": alert_band,
                "Smart_Alerts": smart_alerts,
                "Advisory": advisory,
                "Crop_Recommendation": generate_crop_recommendation(
                    crop_focus=row["Crop_Focus"],
                    temp=row["Temp"],
                    humidity=row["Humidity"],
                    visibility=row["Visibility"],
                    wind_speed=row["Wind_Speed"],
                    forecast=forecast,
                ),
            }
        )

    return pd.DataFrame(enriched_rows)


def persist_snapshot(df: pd.DataFrame) -> None:
    if df.empty:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_df = df.copy()
    snapshot_df.insert(0, "Fetched_At", datetime.now(timezone.utc).isoformat())
    snapshot_df.to_csv(HISTORY_FILE, mode="a", index=False, header=not HISTORY_FILE.exists())


def _read_history_csv() -> pd.DataFrame:
    rows: list[dict] = []

    with HISTORY_FILE.open("r", encoding="utf-8", newline="") as history_file:
        reader = csv.reader(history_file)
        header = next(reader, None)
        if not header:
            return pd.DataFrame(columns=HISTORY_COLUMNS)

        for raw_row in reader:
            if not raw_row:
                continue
            trimmed_row = raw_row[: len(HISTORY_COLUMNS)]
            if len(trimmed_row) < len(HISTORY_COLUMNS):
                trimmed_row += [""] * (len(HISTORY_COLUMNS) - len(trimmed_row))
            rows.append(dict(zip(HISTORY_COLUMNS, trimmed_row)))

    history_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    numeric_columns = [
        "Temp",
        "Humidity",
        "Pressure",
        "Visibility",
        "Wind_Speed",
        "Clouds",
        "Lat",
        "Lon",
        "Forecast_Rain_Events",
        "Forecast_Min_Temp",
        "Forecast_Max_Temp",
        "Forecast_Min_Visibility",
        "Risk_Score",
    ]
    for column in numeric_columns:
        history_df[column] = pd.to_numeric(history_df[column], errors="coerce")
    return history_df


def load_history(limit: int = 250) -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    history_df = _read_history_csv()
    if history_df.empty:
        return history_df

    history_df["Fetched_At"] = pd.to_datetime(history_df["Fetched_At"], errors="coerce")
    history_df = history_df.dropna(subset=["Fetched_At"]).sort_values("Fetched_At")
    if limit > 0:
        history_df = history_df.groupby("City", group_keys=False).tail(limit)
    return history_df


def get_city_trend(history_df: pd.DataFrame, city: str) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()
    return history_df[history_df["City"] == city].copy().sort_values("Fetched_At")


def latest_summary_metrics(df: pd.DataFrame) -> dict[str, float | int | str]:
    if df.empty:
        return {"avg_temp": 0.0, "avg_risk": 0.0, "high_risk_count": 0, "top_city": "N/A"}

    top_row = df.sort_values("Risk_Score", ascending=False).iloc[0]
    return {
        "avg_temp": float(df["Temp"].mean()),
        "avg_risk": float(df["Risk_Score"].mean()),
        "high_risk_count": int((df["Alert_Band"] == "High").sum()),
        "top_city": str(top_row["City"]),
    }


def exportable_columns() -> Iterable[str]:
    return (
        "City",
        "Crop_Focus",
        "Temp",
        "Humidity",
        "Pressure",
        "Visibility",
        "Wind_Speed",
        "Weather_Desc",
        "Risk_Score",
        "Alert_Band",
        "Smart_Alerts",
        "Advisory",
        "Crop_Recommendation",
        "Forecast_Rain_Events",
        "Forecast_Max_Temp",
        "Forecast_Note",
        "Predicted_Risk_24h",
        "Yield_Protection_Index",
        "Prediction_Confidence",
        "Prediction_Note",
    )


def predict_city_outlook(current_row: pd.Series, city_history: pd.DataFrame) -> dict[str, float | str]:
    recent_history = city_history.tail(8).copy()
    history_count = len(recent_history)

    temp_values = recent_history["Temp"].dropna() if "Temp" in recent_history else pd.Series(dtype=float)
    risk_values = recent_history["Risk_Score"].dropna() if "Risk_Score" in recent_history else pd.Series(dtype=float)
    visibility_values = recent_history["Visibility"].dropna() if "Visibility" in recent_history else pd.Series(dtype=float)

    temp_delta = 0.0
    risk_delta = 0.0
    visibility_penalty = 0.0

    if len(temp_values) >= 2:
        temp_delta = float(temp_values.iloc[-1] - temp_values.iloc[0])
    if len(risk_values) >= 2:
        risk_delta = float(risk_values.iloc[-1] - risk_values.iloc[0])
    if len(visibility_values) >= 2 and visibility_values.min() < 1500:
        visibility_penalty = 8.0

    predicted_risk = float(current_row["Risk_Score"])
    predicted_risk += temp_delta * 2.2
    predicted_risk += risk_delta * 0.35
    predicted_risk += visibility_penalty
    predicted_risk += float(current_row.get("Forecast_Rain_Events", 0)) * 2.0

    forecast_max_temp = current_row.get("Forecast_Max_Temp")
    if pd.notna(forecast_max_temp) and pd.notna(current_row.get("Temp")):
        predicted_risk += max(0.0, float(forecast_max_temp) - float(current_row["Temp"])) * 1.5

    predicted_risk = max(0.0, min(100.0, predicted_risk))
    yield_protection_index = max(0.0, min(100.0, 100.0 - predicted_risk + 8.0))

    if history_count >= 8:
        confidence = "High"
    elif history_count >= 4:
        confidence = "Moderate"
    else:
        confidence = "Low"

    if predicted_risk >= 70:
        note = "Protective actions should be prioritized in the next 24 hours."
    elif predicted_risk >= 40:
        note = "Conditions suggest moderate stress buildup over the next 24 hours."
    else:
        note = "Short-range trend looks relatively stable for field operations."

    return {
        "Predicted_Risk_24h": round(predicted_risk, 1),
        "Yield_Protection_Index": round(yield_protection_index, 1),
        "Prediction_Confidence": confidence,
        "Prediction_Note": note,
    }


def add_prediction_features(current_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty:
        return current_df

    enriched_rows: list[dict] = []
    for _, row in current_df.iterrows():
        city_history = pd.DataFrame()
        if not history_df.empty:
            city_history = history_df[history_df["City"] == row["City"]].sort_values("Fetched_At")
        enriched_rows.append({**row.to_dict(), **predict_city_outlook(row, city_history)})

    return pd.DataFrame(enriched_rows)


def get_telegram_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Alert_Band" not in df.columns:
        return pd.DataFrame()
    return df[df["Alert_Band"].isin(["Moderate", "High"])].copy()


def build_telegram_message(row: pd.Series) -> str:
    return (
        "Punjab Weather Alert\n"
        f"District: {row['City']}\n"
        f"Alert Band: {row['Alert_Band']}\n"
        f"Risk Score: {row['Risk_Score']}/100\n"
        f"Temperature: {row['Temp']:.1f} C\n"
        f"Visibility: {format_visibility(row['Visibility'])}\n"
        f"Alerts: {row['Smart_Alerts']}\n"
        f"Advisory: {row['Advisory']}\n"
        f"Recommendation: {row['Crop_Recommendation']}"
    )


def _build_alert_signature(row: pd.Series) -> str:
    temp_value = row.get("Temp")
    rounded_temp = round(float(temp_value), 1) if pd.notna(temp_value) else "na"
    return "|".join(
        [
            str(row.get("Alert_Band", "")),
            str(row.get("Risk_Score", "")),
            str(row.get("Forecast_Rain_Events", "")),
            str(rounded_temp),
            str(row.get("Smart_Alerts", "")),
        ]
    )


def _load_telegram_state() -> dict[str, str]:
    if not TELEGRAM_STATE_FILE.exists():
        return {}
    try:
        return json.loads(TELEGRAM_STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_telegram_state(state: dict[str, str]) -> None:
    TELEGRAM_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def send_telegram_alerts(df: pd.DataFrame, bot_token: str | None, chat_id: str | None) -> dict[str, object]:
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    candidates = get_telegram_candidates(df)
    if candidates.empty:
        return {"sent": 0, "skipped": 0, "status": "No moderate/high alerts to send."}
    if not bot_token or not chat_id:
        return {"sent": 0, "skipped": len(candidates), "status": "Telegram bot settings are missing."}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    previous_state = _load_telegram_state()
    current_state = previous_state.copy()
    sent = 0
    skipped = 0

    for _, row in candidates.iterrows():
        signature = _build_alert_signature(row)
        city = str(row["City"])
        if previous_state.get(city) == signature:
            skipped += 1
            continue

        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": build_telegram_message(row)},
            timeout=20,
        )
        response.raise_for_status()
        current_state[city] = signature
        sent += 1

    _save_telegram_state(current_state)
    return {
        "sent": sent,
        "skipped": skipped,
        "status": f"Telegram alerts sent: {sent}, skipped duplicates: {skipped}.",
    }


def send_telegram_test_message(bot_token: str | None, chat_id: str | None) -> str:
    bot_token = (bot_token or "").strip()
    chat_id = (chat_id or "").strip()
    if not bot_token or not chat_id:
        return "Telegram bot settings are missing."

    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": (
                "Punjab Weather Alert Test\n"
                "Your Telegram bot is connected successfully.\n"
                "Future moderate/high alert bands will trigger weather warnings here."
            ),
        },
        timeout=20,
    )
    response.raise_for_status()
    return "Telegram test message sent successfully."


def print_snapshot_report(api_key: str) -> None:
    current_df = fetch_current_weather(api_key)
    enriched_df = build_enriched_dataset(current_df, api_key)
    history_df = load_history(limit=120)
    enriched_df = add_prediction_features(enriched_df, history_df)
    persist_snapshot(enriched_df)

    display_df = enriched_df[list(exportable_columns())].copy()
    display_df["Visibility"] = display_df["Visibility"].apply(format_visibility)
    print("--- Punjab Weather Advisory Snapshot ---")
    print(display_df.to_string(index=False))


load_dotenv()
API_KEY = (os.getenv("API_KEY") or "").strip()
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")

if "--snapshot" in sys.argv:
    print_snapshot_report(API_KEY)
    raise SystemExit(0)

st.set_page_config(
    page_title="Punjab Smart Weather Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, rgba(248,250,252,0.78) 0%, rgba(248,250,252,0.92) 100%),
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900'><defs><linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'><stop offset='0%25' stop-color='%23dbeafe'/><stop offset='55%25' stop-color='%23f8fafc'/><stop offset='100%25' stop-color='%23fef3c7'/></linearGradient></defs><rect width='1600' height='900' fill='url(%23sky)'/><circle cx='1260' cy='150' r='95' fill='%23fde68a' fill-opacity='0.85'/><ellipse cx='250' cy='170' rx='170' ry='55' fill='white' fill-opacity='0.75'/><ellipse cx='370' cy='150' rx='120' ry='42' fill='white' fill-opacity='0.82'/><ellipse cx='540' cy='185' rx='150' ry='50' fill='white' fill-opacity='0.68'/><path d='M0 630 C180 560 320 560 500 625 S860 700 1080 625 S1400 560 1600 630 L1600 900 L0 900 Z' fill='%23bbf7d0' fill-opacity='0.8'/><path d='M0 690 C200 620 420 640 620 700 S980 760 1240 700 S1460 650 1600 685 L1600 900 L0 900 Z' fill='%2384cc16' fill-opacity='0.32'/><path d='M0 760 C220 710 430 735 650 790 S1100 845 1600 770 L1600 900 L0 900 Z' fill='%2365a30d' fill-opacity='0.24'/><g stroke='%2394a3b8' stroke-opacity='0.18' stroke-width='2' fill='none'><path d='M118 640 l28 -22 l38 30 l44 -40 l52 26'/><path d='M1260 610 l32 -18 l34 26 l48 -36 l40 20'/></g></svg>");
        background-attachment: fixed;
        background-size: cover;
        background-position: center top;
    }
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(239,246,255,0.92) 100%);
        backdrop-filter: blur(10px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_dashboard_data(api_key: str) -> pd.DataFrame:
    current_df = fetch_current_weather(api_key)
    return build_enriched_dataset(current_df, api_key)


def refresh_dashboard_data() -> None:
    df = fetch_dashboard_data(API_KEY)
    history_df = load_history(limit=120)
    df = add_prediction_features(df, history_df)
    st.session_state.weather_df = df
    st.session_state.last_refresh = datetime.now()
    persist_snapshot(df)
    try:
        alert_result = send_telegram_alerts(df, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        st.session_state.telegram_alert_status = str(alert_result["status"])
    except requests.RequestException as exc:
        st.session_state.telegram_alert_status = f"Telegram alert send failed: {exc}"


def build_pdf_report(df: pd.DataFrame, generated_at: datetime | None) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas
    except ImportError:
        return None

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    avg_temp = df["Temp"].mean() if not df.empty else 0
    avg_risk = df["Risk_Score"].mean() if not df.empty else 0
    high_risk_count = int((df["Alert_Band"] == "High").sum()) if not df.empty else 0
    top_city = df.sort_values("Risk_Score", ascending=False).iloc[0]["City"] if not df.empty else "N/A"

    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, height - 95, width, 95, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 42, "Punjab Smart Weather Advisory Report")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, height - 62, "District-level weather intelligence, risk scoring, and crop guidance")

    y = height - 115
    pdf.setFillColor(colors.black)
    timestamp_text = generated_at.strftime("%Y-%m-%d %H:%M:%S") if generated_at else "N/A"
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generated at: {timestamp_text}")
    y -= 20

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Executive Summary")
    y -= 14
    pdf.setFont("Helvetica", 10)
    summary_lines = [
        f"Average Temperature: {avg_temp:.1f} C",
        f"Average Risk Score: {avg_risk:.0f}/100",
        f"High-Risk Districts: {high_risk_count}",
        f"Most Exposed District: {top_city}",
    ]
    for line in summary_lines:
        pdf.drawString(48, y, line)
        y -= 12
    y -= 6

    pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
    pdf.line(40, y, width - 40, y)
    y -= 20

    for _, row in df.iterrows():
        if y < 120:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(40, y, "District Advisory Details")
            y -= 20

        if row["Alert_Band"] == "High":
            band_color = colors.HexColor("#b91c1c")
            band_bg = colors.HexColor("#fee2e2")
        elif row["Alert_Band"] == "Moderate":
            band_color = colors.HexColor("#c2410c")
            band_bg = colors.HexColor("#ffedd5")
        else:
            band_color = colors.HexColor("#166534")
            band_bg = colors.HexColor("#dcfce7")

        pdf.setFillColor(band_bg)
        pdf.roundRect(36, y - 42, width - 72, 56, 8, fill=1, stroke=0)
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, y, f"{row['City']} | Risk {row['Risk_Score']} | {row['Alert_Band']}")
        pdf.setFillColor(band_color)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawRightString(width - 44, y, row["Alert_Band"].upper())
        y -= 14
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 9)
        lines = [
            f"Crop: {row['Crop_Focus']} | Temp: {row['Temp']:.1f} C | Visibility: {format_visibility(row['Visibility'])}",
            f"Alerts: {row['Smart_Alerts']}",
            f"Recommendation: {row['Crop_Recommendation']}",
        ]
        for line in lines:
            pdf.drawString(48, y, line[:110])
            y -= 12
        y -= 6

    pdf.save()
    return buffer.getvalue()


def create_map(df: pd.DataFrame) -> folium.Map:
    map_punjab = folium.Map(location=[31.0, 75.7], zoom_start=7, tiles="CartoDB positron")

    for _, row in df.iterrows():
        if row["Alert_Band"] == "High":
            color = "red"
        elif row["Alert_Band"] == "Moderate":
            color = "orange"
        else:
            color = "green"

        popup_text = (
            f"<b>{row['City']}</b><br>"
            f"Crop Focus: {row['Crop_Focus']}<br>"
            f"Temperature: {row['Temp']:.1f} C<br>"
            f"Humidity: {row['Humidity']}%<br>"
            f"Visibility: {format_visibility(row['Visibility'])}<br>"
            f"Risk Score: {row['Risk_Score']}<br>"
            f"Forecast: {row['Forecast_Note']}<br>"
            f"Advisory: {row['Advisory']}"
        )

        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=10 + (row["Risk_Score"] / 20),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_text, max_width=320),
            tooltip=f"{row['City']} | {row['Alert_Band']} risk",
        ).add_to(map_punjab)

    return map_punjab


def create_choropleth(df: pd.DataFrame) -> folium.Map:
    district_map = folium.Map(location=[31.0, 75.7], zoom_start=7, tiles="CartoDB positron")
    with GEOJSON_FILE.open("r", encoding="utf-8") as geojson_file:
        geojson_data = json.load(geojson_file)

    choropleth_df = df[["City", "Risk_Score", "Predicted_Risk_24h"]].copy()
    choropleth_df = choropleth_df.rename(columns={"City": "district"})

    folium.Choropleth(
        geo_data=geojson_data,
        name="Current Risk",
        data=choropleth_df,
        columns=["district", "Risk_Score"],
        key_on="feature.properties.district",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.5,
        legend_name="District Risk Score",
    ).add_to(district_map)

    for _, row in choropleth_df.iterrows():
        city_meta = LOCATIONS[row["district"]]
        tooltip_html = (
            f"<b>{row['district']}</b><br>"
            f"Current risk: {row['Risk_Score']:.0f}<br>"
            f"Predicted 24h risk: {row['Predicted_Risk_24h']:.0f}"
        )
        folium.Marker(
            location=[city_meta["lat"], city_meta["lon"]],
            icon=folium.DivIcon(
                html=(
                    "<div style=\"font-size:10px;font-weight:700;color:#1f2937;"
                    "background:#ffffffd9;padding:2px 4px;border-radius:4px;\">"
                    f"{row['district']}</div>"
                )
            ),
            tooltip=tooltip_html,
        ).add_to(district_map)

    folium.LayerControl().add_to(district_map)
    return district_map


st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 45%, #fef3c7 100%);
        border: 1px solid rgba(191, 219, 254, 0.9);
        border-radius: 28px;
        padding: 28px 24px 24px 24px;
        margin-bottom: 18px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        text-align: center;
    ">
        <h1 style="
            margin: 0 0 10px 0;
            color: #0f172a;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.1;
        ">
            Punjab Smart Weather and Agri Intelligence
        </h1>
        <p style="
            max-width: 860px;
            margin: 0 auto;
            color: #334155;
            font-size: 1.02rem;
            line-height: 1.7;
        ">
            A decision-support dashboard for district weather risk, crop advisories,
            visibility conditions, and short-term forecast intelligence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "weather_df" not in st.session_state:
    st.session_state.weather_df = pd.DataFrame()
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "telegram_alert_status" not in st.session_state:
    st.session_state.telegram_alert_status = "Telegram alerts not sent yet."

with st.sidebar:
    st.header("Control Panel")
    selected_cities = st.multiselect(
        "Districts",
        options=list(LOCATIONS.keys()),
        default=list(LOCATIONS.keys()),
    )
    selected_risk_bands = st.multiselect(
        "Risk Bands",
        options=["High", "Moderate", "Low"],
        default=["High", "Moderate", "Low"],
    )
    if st.button("Refresh Intelligence", use_container_width=True):
        fetch_dashboard_data.clear()
        refresh_dashboard_data()
    if st.button("Send Test Telegram Alert", use_container_width=True):
        try:
            st.session_state.telegram_alert_status = send_telegram_test_message(
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
            )
        except requests.RequestException as exc:
            st.session_state.telegram_alert_status = f"Telegram test failed: {exc}"

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.success("Telegram alerts are enabled for moderate/high alert bands.")
    else:
        st.info("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable Telegram alerts.")


if st.session_state.weather_df.empty:
    with st.spinner("Building the weather intelligence snapshot..."):
        refresh_dashboard_data()

weather_df = st.session_state.weather_df.copy()
history_df = load_history(limit=60)

if selected_cities:
    weather_df = weather_df[weather_df["City"].isin(selected_cities)]
weather_df = weather_df[weather_df["Alert_Band"].isin(selected_risk_bands)]

if st.session_state.last_refresh:
    st.write(
        f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
    )
st.caption(f"Telegram status: {st.session_state.telegram_alert_status}")

if weather_df.empty:
    st.warning("No districts match the selected filters.")
    st.stop()

ranked_df = weather_df.sort_values(["Risk_Score", "Forecast_Rain_Events"], ascending=[False, False])
summary = latest_summary_metrics(weather_df)
st.markdown(
    """
    <style>
    .kpi-band {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 10px 8px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }
    .kpi-tile {
        padding: 12px 14px;
        min-height: 102px;
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }
    .kpi-last {
        border-right: none;
    }
    .kpi-tile .label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
        opacity: 0.72;
    }
    .kpi-tile .value {
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-tile .subtext {
        font-size: 0.82rem;
        line-height: 1.3;
        opacity: 0.8;
    }
    .kpi-tile .icon {
        font-size: 1.2rem;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

summary_cards = [
    {
        "icon": "🌡️",
        "label": "Average Temperature",
        "value": f"{summary['avg_temp']:.1f} C",
        "subtext": "Live district average",
        "color": "#9a3412",
    },
    {
        "icon": "⚠️",
        "label": "Average Risk Score",
        "value": f"{summary['avg_risk']:.0f}/100",
        "subtext": "Combined weather stress",
        "color": "#1d4ed8",
    },
    {
        "icon": "🚨",
        "label": "High Risk Districts",
        "value": str(summary["high_risk_count"]),
        "subtext": "Active alert bracket count",
        "background": "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)",
        "color": "#b91c1c",
    },
    {
        "icon": "📍",
        "label": "Most Exposed District",
        "value": str(summary["top_city"]),
        "subtext": "Highest current risk",
        "color": "#6d28d9",
    },
    {
        "icon": "🔮",
        "label": "Avg Predicted 24h Risk",
        "value": f"{weather_df['Predicted_Risk_24h'].mean():.0f}/100",
        "subtext": "Short-range projected stress",
        "color": "#be123c",
    },
    {
        "icon": "🌾",
        "label": "Avg Yield Protection Index",
        "value": f"{weather_df['Yield_Protection_Index'].mean():.0f}/100",
        "subtext": "Estimated crop safety buffer",
        "color": "#047857",
    },
    {
        "icon": "✅",
        "label": "High Confidence Districts",
        "value": str(int((weather_df["Prediction_Confidence"] == "High").sum())),
        "subtext": "Prediction support strength",
        "color": "#334155",
    },
]

st.markdown("### Executive Insights")
first_row = summary_cards[:4]
second_row = summary_cards[4:]

for row_cards in (first_row, second_row):
    row_cols = st.columns(len(row_cards))
    for idx, card in enumerate(row_cards):
        with row_cols[idx]:
            tile_class = "kpi-tile kpi-last" if idx == len(row_cards) - 1 else "kpi-tile"
            st.markdown(
                f"""
                <div class="kpi-band">
                    <div class="{tile_class}" style="color:{card['color']};">
                        <div class="icon">{card['icon']}</div>
                        <div class="label">{card['label']}</div>
                        <div class="value">{card['value']}</div>
                        <div class="subtext">{card['subtext']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("### District Status Cards")
card_cols = st.columns(3)
for idx, (_, row) in enumerate(weather_df.sort_values("Risk_Score", ascending=False).iterrows()):
    card_col = card_cols[idx % 3]
    if row["Alert_Band"] == "High":
        accent = "#b91c1c"
        bg = "#fee2e2"
    elif row["Alert_Band"] == "Moderate":
        accent = "#c2410c"
        bg = "#ffedd5"
    else:
        accent = "#166534"
        bg = "#dcfce7"
    with card_col:
        st.markdown(
            f"""
            <div style="background:{bg}; border-left:6px solid {accent}; padding:12px; border-radius:10px; margin-bottom:12px;">
                <div style="font-weight:700; font-size:18px;">{row['City']}</div>
                <div style="margin-top:4px;">Temp: {row['Temp']:.1f} C</div>
                <div>Risk Score: {row['Risk_Score']}</div>
                <div>Band: {row['Alert_Band']}</div>
                <div style="margin-top:6px; font-size:13px;">{row['Advisory']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Executive Snapshot")
left_col, right_col = st.columns([1.15, 1.0])

with left_col:
    score_df = ranked_df[
        ["City", "Crop_Focus", "Risk_Score", "Alert_Band", "Smart_Alerts", "Forecast_Note"]
    ]
    st.dataframe(score_df, use_container_width=True, hide_index=True)

with right_col:
    top_city = ranked_df.iloc[0]
    top_city_forecast_temp = (
        "N/A"
        if pd.isna(top_city["Forecast_Max_Temp"])
        else f"{top_city['Forecast_Max_Temp']:.1f} C"
    )
    st.markdown(
        f"""
        #### Priority Advisory
        **District:** {top_city['City']}  
        **Crop Focus:** {top_city['Crop_Focus']}  
        **Risk Band:** {top_city['Alert_Band']}  
        **Advisory:** {top_city['Advisory']}
        """
    )
    st.markdown(
        f"""
        #### Forecast Watch
        **District:** {top_city['City']}  
        Next-24h rain events: **{int(top_city['Forecast_Rain_Events'])}**  
        Forecast maximum temperature: **{top_city_forecast_temp}**
        """
    )

st.markdown("---")
st.markdown("### District Trend Explorer")
trend_city = st.selectbox("Choose a district for historical trend view", list(weather_df["City"]))
trend_df = get_city_trend(history_df, trend_city)

trend_left, trend_right = st.columns(2)
with trend_left:
    if trend_df.empty:
        st.info("Trend history will appear after more refresh cycles are stored.")
    else:
        temp_chart = trend_df.set_index("Fetched_At")[["Temp", "Humidity"]]
        st.line_chart(temp_chart, use_container_width=True)

with trend_right:
    if trend_df.empty:
        st.info("Visibility and risk history will populate automatically over time.")
    else:
        visibility_chart = trend_df.set_index("Fetched_At")[["Visibility", "Risk_Score"]]
        st.line_chart(visibility_chart, use_container_width=True)

st.markdown("### Predictive Intelligence")
prediction_left, prediction_right = st.columns([1.15, 1.0])
with prediction_left:
    prediction_df = weather_df.sort_values("Predicted_Risk_24h", ascending=False)[
        [
            "City",
            "Predicted_Risk_24h",
            "Yield_Protection_Index",
            "Prediction_Confidence",
            "Prediction_Note",
        ]
    ]
    st.dataframe(prediction_df, use_container_width=True, hide_index=True)

with prediction_right:
    top_prediction = weather_df.sort_values("Predicted_Risk_24h", ascending=False).iloc[0]
    st.markdown(
        f"""
        #### 24-Hour Risk Outlook
        **District:** {top_prediction['City']}  
        **Predicted Risk:** {top_prediction['Predicted_Risk_24h']:.0f}/100  
        **Yield Protection Index:** {top_prediction['Yield_Protection_Index']:.0f}/100  
        **Model Confidence:** {top_prediction['Prediction_Confidence']}  
        **Interpretation:** {top_prediction['Prediction_Note']}
        """
    )

st.markdown("### Crop Recommendation Panel")
rec_city = st.selectbox(
    "Choose a district for detailed crop recommendation",
    list(weather_df.sort_values("City")["City"]),
    key="crop_recommendation_city",
)
rec_row = weather_df[weather_df["City"] == rec_city].iloc[0]
rec_cols = st.columns([1, 1])
with rec_cols[0]:
    st.markdown(
        f"""
        #### Advisory Summary
        **District:** {rec_row['City']}  
        **Crop Focus:** {rec_row['Crop_Focus']}  
        **Temperature:** {rec_row['Temp']:.1f} C  
        **Visibility:** {format_visibility(rec_row['Visibility'])}  
        **Risk Band:** {rec_row['Alert_Band']}
        """
    )
with rec_cols[1]:
    st.markdown(
        f"""
        #### Recommended Actions
        {rec_row['Crop_Recommendation']}
        """
    )

st.markdown("---")
st.markdown("### Regional Map Intelligence")
map_df = weather_df.sort_values("Risk_Score", ascending=False)
st_folium(create_map(map_df), width=1100, height=520)

st.markdown("### District Choropleth")
st.caption(
    "District polygons are stored locally for a stable offline demo. "
    "Colors represent current district risk score."
)
st_folium(create_choropleth(map_df), width=1100, height=520)

st.markdown("### District Intelligence Table")
display_df = ranked_df[
    list(exportable_columns())
].copy()
display_df["Visibility"] = display_df["Visibility"].apply(format_visibility)
display_df["Forecast_Max_Temp"] = display_df["Forecast_Max_Temp"].apply(
    lambda value: "N/A" if pd.isna(value) else f"{value:.1f} C"
)
display_df["Predicted_Risk_24h"] = display_df["Predicted_Risk_24h"].apply(
    lambda value: f"{value:.1f}"
)
display_df["Yield_Protection_Index"] = display_df["Yield_Protection_Index"].apply(
    lambda value: f"{value:.1f}"
)
st.dataframe(display_df, use_container_width=True, hide_index=True)

csv_data = display_df.to_csv(index=False).encode("utf-8")
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #eff6ff 0%, #fef3c7 100%);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 18px 16px 10px 16px;
        margin-bottom: 12px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    ">
        <div style="text-align:center; font-size:1.05rem; font-weight:700; color:#1e3a8a; margin-bottom:6px;">
            Export Your Advisory Snapshot
        </div>
        <div style="text-align:center; color:#475569; font-size:0.95rem; margin-bottom:4px;">
            Save the live advisory snapshot as a spreadsheet or a presentation-ready PDF.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

export_cols = st.columns([1.2, 2.4, 2.4, 1.2])
with export_cols[0]:
    st.empty()
with export_cols[1]:
    st.download_button(
        "📊 Download CSV Report",
        data=csv_data,
        file_name="punjab_weather_advisory_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
with export_cols[2]:
    pdf_data = build_pdf_report(weather_df.sort_values("Risk_Score", ascending=False), st.session_state.last_refresh)
    if pdf_data is not None:
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_data,
            file_name="punjab_weather_advisory_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Install `reportlab` to enable PDF export.")
with export_cols[3]:
    st.empty()
