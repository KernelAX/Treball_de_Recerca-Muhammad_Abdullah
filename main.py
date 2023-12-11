import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import joblib
from tensorflow.keras.models import load_model


def train_model():
    # cargamos el csv
    file_path = "/Users/abdullah/PycharmProjects/weather_prediction/open-meteo.csv"
    data = pd.read_csv(file_path)
    # imprimimos unas filas para verificar
    print(data.head())

    # convertimos la columna 'time' a tiempo y la ponemos como referencia de índice
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)
    print(data.head())

    hourly_temp = data["temp_c"]
    hourly_relh = data["rel_hum_percent"]
    hourly_rain = data["rain_mm"]
    hourly_wind = data["wind_speed_km"]
    hourly_wind_gusts = data["wind_gusts_km"]
    hourly_data = pd.DataFrame(
        {"Temperature": hourly_temp, "Humidity": hourly_relh, "Rain": hourly_rain, "Wind": hourly_wind,
         "Wind_Gusts": hourly_wind_gusts})

    # Normalizamos los datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hourly_data)
    scaled_data = pd.DataFrame(scaled_data, columns=hourly_data.columns, index=hourly_data.index)
    print(scaled_data.head())

    # Definimos una funcián que convertirá el DataFrame a un formato apto para entrenar el modelo LSTM
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(n_steps, len(data)):
            X.append(data[i - n_steps:i])
            y.append(data[i, 1])
        return np.array(X), np.array(y)

    # Definimos el numero de pasos n_steps(n_steps son el numero de filas(o horas) que el modelo recibirá para predecir la siguiente fila
    n_steps = 50
    # Generamos secuencias de entrada (X) y salida (y) para el LSTM a partir de datos escalados, con cada secuencia abarcando 'n_steps' intervalos de tiempo
    X, y = create_sequences(scaled_data.values, n_steps)

    # Dividimos los datos en conjuntos de entrenamiento (80%) y prueba (20%) con un estado aleatorio fijo para reproducibilidad
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos el modelo LSTM con 3 capas(1 entrada, 1 procesamiento, 1 salida)usando Keras
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Entrenamos el modelo
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Guardamos el modelo
    model.save("/Users/abdullah/PycharmProjects/WeatherPrediction/Model/model.h5")
    # Guardamos la referencia de normalización
    joblib.dump(scaler, "/Users/abdullah/PycharmProjects/WeatherPrediction/Model/scaler.pkl")
    # Guardamos el historial de entrenamiento
    with open("/Users/abdullah/PycharmProjects/WeatherPrediction/Model/training_history.txt", "w") as f:
        for key in history.history.keys():
            f.write(f"{key}: {history.history[key]}\n")

    # hacemos una predicción de prueba
    y_pred = model.predict(X_test)
    # Rescalamos la predicción para interpretarla
    y_pred_rescaled = scaler.inverse_transform(
        np.hstack((y_pred, np.zeros((y_pred.shape[0], scaled_data.shape[1] - 1)))))
    y_test_rescaled = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1)))))
    # Calculamos el error
    mse = mean_squared_error(y_test_rescaled[:, 0], y_pred_rescaled[:, 0])
    print(f'Mean Squared Error: {mse}')


def load_and_use_model():
    model_path = "/Users/abdullah/PycharmProjects/WeatherPrediction/Model/model.h5"
    model = load_model(model_path)

    scaler_path = "/Users/abdullah/PycharmProjects/WeatherPrediction/Model/scaler.pkl"
    scaler = joblib.load(scaler_path)

    new_data_path = "/Users/abdullah/PycharmProjects/loadModel/11-17.csv"  # data 10Nov-17Nov
    data = pd.read_csv(new_data_path)
    # imprimimos unas filas para verificar
    print(data.head())

    # convertimos la columna 'valid' a tiempo y la ponemos como referencia de índice
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)
    print(data.head())

    hourly_temp = data["Temperature"]
    hourly_relh = data["Humidity"]
    hourly_rain = data["Rain"]
    hourly_wind = data["Wind"]
    hourly_wind_gusts = data["Wind_Gusts"]

    hourly_data = pd.DataFrame(
        {"Temperature": hourly_temp, "Humidity": hourly_relh, "Rain": hourly_rain, "Wind": hourly_wind,
         "Wind_Gusts": hourly_wind_gusts})
    new_data_scaled = scaler.transform(hourly_data)

    n_steps = 50
    X_new = []
    for i in range(n_steps, len(new_data_scaled)):
        X_new.append(new_data_scaled[i - n_steps:i])
    X_new = np.array(X_new)

    predictions = model.predict(X_new)

    predictions_rescaled = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], new_data_scaled.shape[1] - 1)))))
    # Extract the rescaled predictions
    rescaled_predictions = predictions_rescaled[:, 0]
    print(rescaled_predictions)

    first_24h = rescaled_predictions[0:24]
    max_temp = float(max(first_24h))
    min_temp = float(min(first_24h))
    avg_temp = (max_temp + min_temp) / 2
    print(avg_temp)