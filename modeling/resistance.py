import glob
import pandas as pd
import numpy as np
import json
import os
from scipy.optimize import curve_fit

DATA_DIR = 'data/raw'
MODEL_DIR = 'data/models'

def load_all_data():
    files = glob.glob(os.path.join(DATA_DIR, '*.parquet'))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Add gear_id from attrs to column
            gear_id = df.attrs.get('gear_id', 'Unknown')
            df['gear_id'] = gear_id
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def train_physics_model():
    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} points.")
    
    # Constants
    MASS = 85.0 # kg
    G = 9.81
    RHO = 1.225
    
    # Physics Equation:
    # Power = (Mass * G * (sin + cos*Crr)) * v + 0.5 * Rho * CdA * v^3
    # Approximating sin(theta) ~ grade, cos(theta) ~ 1 for small angles
    # Force = Mass*G*grade + Mass*G*Crr + 0.5*Rho*CdA*v^2
    # Power = Force * v
    # Power = Mass*G*grade*v + Mass*G*Crr*v + 0.5*Rho*CdA*v^3
    
    # We want to find Crr and CdA.
    # y = Power - Mass*G*grade*v
    # X1 = Mass*G*v  (Coeff for Crr)
    # X2 = 0.5*Rho*v^3 (Coeff for CdA)
    # y = Crr*X1 + CdA*X2
    
    # Filter Data for "Good" Physics
    # - Moving
    # - Watts > 0
    # - No braking (deceleration?) - hard to filter without accel stream
    # - Grade between -15% and 15%
    mask = (df['moving'] == True) & (df['watts'] > 50) & (df['velocity_smooth'] > 2.0)
    df_clean = df[mask].copy()
    
    # Calculate components
    # Grade is percentage in data? usually yes (e.g. 5.2 for 5.2%)
    # Strava grade_smooth is percent.
    grad_decimal = df_clean['grade_smooth'] / 100.0
    v = df_clean['velocity_smooth']
    
    # y = P - m*g*grade*v
    # Note: Accel term (ma) is missing!
    # P_rider = P_grav + P_roll + P_aero + P_accel
    # We ignore P_accel, so this is noisy. We assume avg accel is 0 over huge dataset.
    
    df_clean['y_check'] = df_clean['watts'] - (MASS * G * grad_decimal * v)
    
    # Regressors
    df_clean['X_roll'] = MASS * G * v
    df_clean['X_aero'] = 0.5 * RHO * (v**3)
    
    profiles = {}
    
    # Group by Gear
    for gear_id, group in df_clean.groupby('gear_id'):
        if len(group) < 1000:
            print(f"Skipping {gear_id}: Not enough data ({len(group)})")
            continue
            
        print(f"Training for Gear: {gear_id} ({len(group)} pts)...")
        
        # Define function to fit
        def power_func(X, crr, cda):
            # X is [X_roll, X_aero]
            return crr * X[0] + cda * X[1]
            
        X_data = np.stack([group['X_roll'].values, group['X_aero'].values])
        y_data = group['y_check'].values
        
        # Bounds: Crr (0.001, 0.05), CdA (0.1, 1.0)
        try:
            popt, pcov = curve_fit(power_func, X_data, y_data, 
                                   p0=[0.005, 0.32], 
                                   bounds=([0.001, 0.1], [0.05, 1.0]))
            
            crr, cda = popt
            print(f"  -> Crr: {crr:.4f}, CdA: {cda:.3f}")
            
            profiles[gear_id] = {
                'crr': float(crr),
                'cda': float(cda),
                'name': gear_id # Could fetch real name if we had it, using ID for now
            }
        except Exception as e:
            print(f"  -> Failed to fit: {e}")
            
    # Save
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    out_path = os.path.join(MODEL_DIR, 'bike_profiles.json')
    with open(out_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved profiles to {out_path}")

if __name__ == "__main__":
    train_physics_model()
