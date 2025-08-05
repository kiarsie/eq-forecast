import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_and_evaluate_lstm():
    """
    Loads earthquake data, trains an LSTM model, and plots predicted vs actual magnitudes.
    Returns:
        model: The trained LSTM model.
    """
    # Resolve the absolute path to the dataset relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, '..', '..', 'data', 'eq_catalog.csv')

    print(f"Reading CSV from: {filepath}")

    # Load and preprocess the data
    df = pd.read_csv(filepath)
    df = df[['N_Lat', 'E_Long', 'Depth', 'Mag']].dropna()

    # Normalize the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Prepare sequences
    sequence_length = 10
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i, -1])  # Target is the magnitude

    X, y = np.array(X), np.array(y)

    # Define the LSTM model
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    # Predict and inverse transform the result
    predicted = model.predict(X)
    predicted_inverse = scaler.inverse_transform(
        np.hstack((scaled_data[sequence_length:, :-1], predicted))
    )[:, -1]

    actual_inverse = df['Mag'].values[sequence_length:]

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 5))
    plt.plot(actual_inverse, label='Actual Magnitude')
    plt.plot(predicted_inverse, label='Predicted Magnitude')
    plt.title("Earthquake Magnitude Forecast")
    plt.xlabel("Time Step")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model
