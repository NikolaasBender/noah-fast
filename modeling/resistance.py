import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def learn_bike_physics(df, gear_map=None):
    """
    Learns physics parameters (Crr, CdA) for each gear_id in the provided DataFrame.
    Args:
        df: DataFrame with activities
        gear_map: Optional dict mapping gear_id -> gear_name
    Returns: dict of profiles
    """
    if df.empty:
        return {}
        
    # ... (Constants Omitted for brevity, but I should be careful not to delete them if I use replace_file_content)
    # Actually, I should use a smaller chunk if I can, but the function start is changing.
    # Let me include the constants to be safe or use a targeted replacement.
    
    # Constants
    MASS = 85.0 # kg
    G = 9.81
    RHO = 1.225
    
    # Filter Data for "Good" Physics
    mask = (df['moving'] == True) & (df['watts'] > 50) & (df['velocity_smooth'] > 2.0)
    df_clean = df[mask].copy()
    
    if df_clean.empty:
        return {}

    # Calculate components
    grad_decimal = df_clean['grade_smooth'] / 100.0
    v = df_clean['velocity_smooth']
    
    # y = P - m*g*grade*v
    df_clean['y_check'] = df_clean['watts'] - (MASS * G * grad_decimal * v)
    
    # Regressors
    df_clean['X_roll'] = MASS * G * v
    df_clean['X_aero'] = 0.5 * RHO * (v**3)
    
    profiles = {}
    
    # Group by Gear
    if 'gear_id' not in df_clean.columns:
         groups = [('Unknown', df_clean)]
    else:
         groups = df_clean.groupby('gear_id')

    for gear_id, group in groups:
        if len(group) < 500: # Lower threshold for faster testing
            continue
            
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
            
            # Resolve Name
            name = gear_id
            if gear_map and gear_id in gear_map:
                name = gear_map[gear_id]
            
            profiles[gear_id] = {
                'crr': float(crr),
                'cda': float(cda),
                'name': name
            }
        except Exception as e:
            # print(f"  -> Failed to fit {gear_id}: {e}")
            pass
            
    return profiles

if __name__ == "__main__":
    pass
