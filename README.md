# Punjab Smart Weather and Agri Intelligence

This project is a Streamlit dashboard for Punjab district weather monitoring with:

- live current-weather intelligence
- short-range forecast summaries
- crop-focused district advisories
- risk scoring for agricultural planning
- local CSV history for trend analysis

## Run

```cmd
cd C:\Users\LUCKY ASHER\punjab_weather_analytics
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Environment

Create a `.env` file with:

```env
API_KEY=your_openweather_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```
