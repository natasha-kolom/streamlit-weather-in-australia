# 🌦 Streamlit Weather in Australia

This project predicts whether it will rain tomorrow in a selected Australian city based on user-input weather data. The app allows users to enter today’s weather conditions and receive a rain forecast for tomorrow, powered by a trained **Random Forest** model. The model was built using 10 years of daily weather observations from various Australian meteorological stations.

📍 **Test the application**: [Streamlit App - Weather in Australia](https://app-weather-in-australia-ich6hkv3r3obudttgtmft6.streamlit.app/)

> **Note**: If you see the message, “This app has gone to sleep due to inactivity,” click **Yes, get this app back up!** and wait about 30 seconds.

---

## 📁 Project Structure

- **`weatherAUS.csv`**: Dataset containing 10 years of weather observations from Australia.
- **`weather.png`**: Image used in the app.
- **`aussie_rain_2.joblib`**: File containing the trained ML model.
- **`app.py`**: Main Streamlit application file.
- **`requirements.txt`**: List of required Python packages.

---

## 🚀 Running the Streamlit Application

To test the app, visit: [Streamlit App - Weather in Australia](https://app-weather-in-australia-ich6hkv3r3obudttgtmft6.streamlit.app/)

---

## ✏️ Input Fields

Upon opening the app, enter the following weather details to receive a rain forecast:

- **Location**: Name of the location of the weather station.
- **MinTemp**: Minimum temperature (°C).
- **MaxTemp**: Maximum temperature (°C).
- **Rainfall**: Rainfall recorded for the day (mm).
- **Evaporation**: Class A pan evaporation (mm) in the 24 hours to 9am.
- **Sunshine**: Number of hours of sunshine in the day.
- **WindGustDir**: Direction of the strongest wind gust in the past 24 hours.
- **WindGustSpeed**: Speed of the strongest wind gust (km/h).
- **WindDir9am**: Wind direction at 9am.
- **WindDir3pm**: Wind direction at 3pm.
- **WindSpeed9am**: Average wind speed prior to 9am (km/h).
- **WindSpeed3pm**: Average wind speed prior to 3pm (km/h).
- **Humidity9am**: Humidity at 9am (%).
- **Humidity3pm**: Humidity at 3pm (%).
- **Pressure9am**: Atmospheric pressure at 9am (hPa).
- **Pressure3pm**: Atmospheric pressure at 3pm (hPa).
- **Cloud9am**: Fraction of sky covered by clouds at 9am (measured in "oktas").
- **Cloud3pm**: Fraction of sky covered by clouds at 3pm ("oktas").
- **Temp9am**: Temperature at 9am (°C).
- **Temp3pm**: Temperature at 3pm (°C).
- **RainToday**: Boolean: 1 if rainfall >1mm in the last 24 hours, otherwise 0.

