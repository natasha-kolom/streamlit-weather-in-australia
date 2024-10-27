import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def predict(result):
    model = joblib.load('aussie_rain_2.joblib')
    #data = np.expand_dims(np.array(result), axis=0)
    predictions = model['model'].predict(result.iloc[[0]])
    return predictions[0]

def preprocess_data(data, df):
    data = pd.DataFrame([data])
 
    input_cols = list(df.columns)[1:-1]
    train_inputs = df[input_cols].copy()
    
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

    imputer = SimpleImputer(strategy = 'mean')
    imputer.fit(df[numeric_cols])

    scaler = MinMaxScaler()
    scaler.fit(df[numeric_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    data[numeric_cols] = imputer.transform(data[numeric_cols])
    data[numeric_cols] = scaler.transform(data[numeric_cols])
    data[encoded_cols] = encoder.transform(data[categorical_cols])

    X_new_input = data[numeric_cols + encoded_cols]

    return X_new_input
 

# Заголовок застосунку
st.title('Прогноз погоди на завтра в Австралії')
st.markdown('Ця модель передбачає чи буде завтра дощ на основі введених Вами даних, та аналізу попередніх даних за 10 років')
st.image('weather.png')


df = pd.read_csv("weatherAUS.csv")
df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
# Введення характеристик чашолистків

# Введення характеристик пелюсток

# Заголовок секції з характеристиками рослини
st.header("Дані про погоду")
col1, col2, col3 = st.columns(3)
result = {}
with col1:
    for i in range(1,8):
        if df[df.columns[i]].dtype=='object':
            result[df.columns[i]] = st.selectbox(df.columns[i], df[df.columns[i]].unique() )
        else:
            result[df.columns[i]] = st.slider(df.columns[i], df[df.columns[i]].min(), df[df.columns[i]].max(), 0.5)
                        
                        
with col2:
    for i in range(8,15):
        if df[df.columns[i]].dtype=='object':
           result[df.columns[i]] = st.selectbox(df.columns[i], df[df.columns[i]].unique() )
        else:
            result[df.columns[i]] = st.slider(df.columns[i], df[df.columns[i]].min(), df[df.columns[i]].max(), 0.5)

with col3:
    for i in range(15,22):
        if df[df.columns[i]].dtype=='object':
            result[df.columns[i]] = st.selectbox(df.columns[i], df[df.columns[i]].unique() )
        else:
            result[df.columns[i]] = st.slider(df.columns[i], df[df.columns[i]].min(), df[df.columns[i]].max(), 0.5)
                        
result['Date'] = pd.to_datetime('now')

# Кнопка для прогнозування
if st.button("Прогнозувати погоду на завтра"):
    # Викликаємо функцію прогнозування
    r = preprocess_data(result, df)
    res = predict(r)
    st.write(f"Чи буде завтра дощ: {res}")