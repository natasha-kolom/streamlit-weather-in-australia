import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict(result):
    model = joblib.load('models/aussie_rain.joblib')
    data = np.expand_dims(np.array([sepal_l, sepal_w, petal_l, petal_w]), axis=0)
    predictions = model.predict(data)
    return predictions[0]

https://drive.google.com/file/d/1p9-QWeAIJWO6WmkTbTFEeJSZb8rUWqWm/view?usp=sharing
aussie_rain = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}


# Заголовок застосунку
st.title('Прогноз погоди на завтра в Австралії')
st.markdown('Ця модель передбачає чи буде завтра дощ на основі введених Вами даних, та аналізу попередніх даних за 10 років')
st.image('images/weather.png')


df = pd.read_csv("data/weatherAUS.csv")


# Заголовок секції з характеристиками рослини
st.header("Дані про погоду")
col1, col2 = st.columns(2)
locations = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
config = {
    'Mintemp' : st.column_config.NumberColumn('Min temp', min_value=-20, max_value=60),
    'Maxtemp' : st.column_config.NumberColumn('Max temp', min_value=-20, max_value=60),
    'Location' : st.column_config.SelectboxColumn('Location name', options=df.Location.unique())
}

result = st.data_editor(df, column_config = config, num_rows=1)


# Кнопка для прогнозування
if st.button("Прогнозувати погоду"):
    # Викликаємо функцію прогнозування
    result = predict(sepal_l, sepal_w, petal_l, petal_w)
    st.write(f"Прогнозований тип ірису: {result}")