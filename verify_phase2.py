import os
import joblib
import numpy as np
import tensorflow as tf
from modeling.nutrition import carb_needs_per_hour, calculate_metabolic_cost

MODEL_DIR = 'data/models'

def verify():
    print("Loading models...")
    phys = joblib.load(os.path.join(MODEL_DIR, 'physiology.pkl'))
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'fatigue_lstm.h5'), compile=False)
    
    print(f"Loaded Physiology: CP={phys['cp']:.1f}W, W'={phys['w_prime']:.0f}J")
    
    # Create a dummy scenario: Rider has been riding for 60 mins at 200W
    # We need a sequence of 60s (history window)
    # Let's say steady state 200W, 140bpm, 90rpm, and some W' balance drop
    
    # Calculate W' balance usage for 200W vs CP
    cp = phys['cp']
    current_watts = 200
    
    # Logic: if 200 < CP, W' is full (approx). If 200 > CP, it drains.
    w_prime_bal = phys['w_prime'] # Start full
    
    # Create input vector (60 samples)
    # Features: [watts, heartrate, cadence, w_prime_bal]
    X_input = np.zeros((1, 60, 4))
    
    for i in range(60):
        X_input[0, i, 0] = current_watts
        X_input[0, i, 1] = 140
        X_input[0, i, 2] = 90
        X_input[0, i, 3] = w_prime_bal
        
    print("\nPredicting future capacity...")
    predicted_max_watts = model.predict(X_input)
    print(f"Scenario: Riding at {current_watts}W (steady).")
    print(f"Model predicts average sustainble power for next 5 mins: {predicted_max_watts[0][0]:.1f} W")
    
    print("\nCalculating Nutrition...")
    kcal = calculate_metabolic_cost(current_watts)
    carbs = carb_needs_per_hour(current_watts)
    print(f"At {current_watts}W:")
    print(f" - Burn Rate: {kcal:.0f} kcal/hr")
    print(f" - Recommended Carbs: {carbs:.0f} g/hr")

if __name__ == "__main__":
    verify()
