import os
import glob
import pandas as pd
import numpy as np
import joblib
from modeling.physiology import extract_power_profile, calculate_cp_and_w_prime, calculate_w_prime_balance
from modeling.fatigue_model import build_lstm_model, prepare_sequences

DATA_DIR = 'data/raw'
MODEL_DIR = 'data/models'

def ensure_dirs():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def load_all_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Basic cleaning
            if 'watts' in df.columns:
                 dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return dfs

def main():
    ensure_dirs()
    
    print("Loading data...")
    activities = load_all_data()
    print(f"Loaded {len(activities)} activities.")
    
    if not activities:
        print("No activities found.")
        return

    # 1. Calculate Physiology Baseline
    print("Extracting Power Profile...")
    # Combine all activities into one DF for the physiology model
    combined_activities = pd.concat(activities)
    durations, powers = extract_power_profile(combined_activities)
    
    print("Calculating CP and W'...")
    cp, w_prime = calculate_cp_and_w_prime(durations, powers)
    
    if cp:
        print(f"Estimated CP: {cp:.1f} W")
        print(f"Estimated W': {w_prime:.0f} J")
        
        # Save baseline
        joblib.dump({'cp': cp, 'w_prime': w_prime}, os.path.join(MODEL_DIR, 'physiology.pkl'))
    else:
        print("Could not calculate CP/W'. Using defaults.")
        cp, w_prime = 250, 20000

    # 2. Prepare Data for ML
    print("Preparing Training Data...")
    X_all = []
    y_all = []
    
    for df in activities:
        # Augment with W' Balance
        if 'watts' not in df.columns: continue
        
        df['w_prime_bal'] = calculate_w_prime_balance(df['watts'], cp, w_prime)
        
        # Clean other cols
        for col in ['heartrate', 'cadence']:
            if col not in df.columns:
                df[col] = 0
                
        # Create sequences
        # Predict average power for next 5 minutes (300s) based on last 1 minute (60s)
        X, y = prepare_sequences(df, history_window=60, future_window=300)
        
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            
    if not X_all:
        print("No training sequences generated.")
        return
        
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    
    print(f"Training Data Shape: {X_train.shape}")
    
    # 3. Train Model
    print("Training LSTM...")
    # input_shape = (time_steps, features)
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    
    # Save
    model.save(os.path.join(MODEL_DIR, 'fatigue_lstm.h5'))
    print("Model saved!")

if __name__ == "__main__":
    main()
