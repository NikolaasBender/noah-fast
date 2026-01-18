import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

def build_lstm_model(input_shape):
    """
    Builds a simple LSTM model to predict max sustainable power
    for the next window based on current state.
    """
    model = Sequential()
    # Masking layer to handle variable length sequences if we pad them
    # For now, we assume fixed window inputs
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear')) # Output: Predicted Max Watts
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_sequences(df, history_window=60, future_window=300):
    """
    Prepares sliding window data for the LSTM.
    
    Args:
        df: DataFrame with columns ['watts', 'heartrate', 'cadence', 'w_prime_bal']
        history_window: How many seconds of history to look at (e.g., 60s)
        future_window: The target window to predict max power for (e.g., next 5 mins)
        
    Returns:
        X: (num_samples, history_window, num_features)
        y: (num_samples, ) -> Max power sustained in future_window
    """
    data = df[['watts', 'heartrate', 'cadence', 'w_prime_bal']].fillna(0).values
    
    X = []
    y = []
    
    # Slide over the data
    # Step size can be > 1 to reduce data redundancy
    step = 30 
    
    for i in range(0, len(data) - history_window - future_window, step):
        # Input: History window
        X.append(data[i : i + history_window])
        
        # Target: Max average power in the FUTURE window
        # We want to know "What is the max steady state I can hold?" effectively.
        # However, purely 'max power' might just pick up a sprint.
        # Maybe we want "Average Power" of the next segment?
        # Let's try predicting Average Power of the next N minutes.
        future_data = data[i + history_window : i + history_window + future_window]
        future_watts = future_data[:, 0] # 0 index is watts
        y.append(np.mean(future_watts))
        
    return np.array(X), np.array(y)
