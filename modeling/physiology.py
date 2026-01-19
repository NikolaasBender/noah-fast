import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def get_best_power_for_duration(power_stream, duration_seconds):
    """
    Finds the maximum average power for a given duration in a ride.
    """
    if len(power_stream) < duration_seconds:
        return 0
    
    # Calculate rolling average
    rolling_power = power_stream.rolling(window=duration_seconds).mean()
    return rolling_power.max()

def extract_power_profile(combined_df):
    """
    Given a single DataFrame with multiple activities (distinguished by 'activity_id'), 
    extracts the global Mean Maximal Power (MMP) curve.
    Returns: (durations, max_powers)
    """
    # Define durations of interest: 1s, 5s, 10s, 30s, 1m, 2m, 3m, 5m, 10m, 20m, 30m, 40m, 60m
    durations = [1, 5, 10, 30, 60, 120, 180, 300, 600, 1200, 1800, 2400, 3600]
    global_mmp = {d: 0 for d in durations}

    if combined_df.empty:
        return np.array([]), np.array([])
        
    if 'activity_id' not in combined_df.columns:
        # Single activity without ID? Treat as one
        grouped = [('single', combined_df)]
    else:
        grouped = combined_df.groupby('activity_id')

    for _, df in grouped:
        if 'watts' not in df.columns:
            continue
            
        # Optimization: Don't compute rolling for every duration if ride is short
        p = df['watts']
        
        for d in durations:
            best_p = get_best_power_for_duration(p, d)
            if best_p > global_mmp[d]:
                global_mmp[d] = best_p
                
    # Sort by duration
    sorted_durations = np.array(sorted(global_mmp.keys()))
    sorted_powers = np.array([global_mmp[d] for d in sorted_durations])
    
    # Filter out 0s if any duration wasn't found
    mask = sorted_powers > 0
    return sorted_durations[mask], sorted_powers[mask]

def cp_model(t, cp, w_prime):
    """
    2-parameter Critical Power Model: Power(t) = CP + W'/t
    """
    return cp + w_prime / t

def calculate_cp_and_w_prime(durations, powers):
    """
    Fits the CP model to the MMP data.
    """
    # We typically fit CP model on durations between 3 minutes and 20-30 minutes
    # Short durations are anaerobic dominant (Pmax), long durations have fatigue factors
    mask = (durations >= 180) & (durations <= 1200) 
    
    if np.sum(mask) < 2:
        # Fallback to wider range if not enough points
        mask = (durations >= 60) & (durations <= 2400)
        
    fit_durations = durations[mask]
    fit_powers = powers[mask]
    
    try:
        popt, pcov = curve_fit(cp_model, fit_durations, fit_powers, p0=[200, 20000], bounds=([0, 0], [1000, 100000]))
        return popt[0], popt[1] # CP, W'
    except Exception as e:
        print(f"Error calculating CP: {e}")
        return None, None

def calculate_w_prime_balance(power_stream, cp, w_prime):
    """
    Calculates the detailed W' balance for a ride.
    Formula: Differential model or Integral model (Skiba).
    
    Integral Model (simplified):
    W_bal(t) = W' - integral( (P(u) - CP) * exp(-(t-u)/tau) )
    
    For simplicity in this V1, we use the simple bucket model:
    - Above CP: Drain W' (linear)
    - Below CP: Recharge W' (exponential or linear? Skiba uses exponential)
    
    Let's use the standard Skiba implementation:
    W_bal(t) = W_bal(t-1) * exp(-dt/tau) - (P(t) - CP)*dt  <- this is not quite right.
    
    Correct iterative approximation:
    If P > CP:
       W_exp = (P - CP) * dt
       W_bal_new = W_bal_old - W_exp
    If P < CP:
       Recharge. Tau varies. 
       Skiba 2012: tau = 546 * exp(-0.01 * (CP - P)) + 316
       
    """
    w_bal = np.zeros(len(power_stream))
    current_w_bal = w_prime
    
    for i, p in enumerate(power_stream):
        if np.isnan(p):
            p = 0
            
        if p > cp:
            # Deplete
            current_w_bal -= (p - cp)
        else:
            # Recharge
            # Dynamic Tau (Skiba 2012)
            d_cp = cp - p
            tau = 546 * np.exp(-0.01 * d_cp) + 316
            
            # W_bal recovers towards W_prime
            # Asymptotic recovery: W_bal(t) = W_prime - (W_prime - W_bal(0)) * exp(-t/tau)
            # Iterative step:
            # Recover amount in 1 sec
            max_recovery = w_prime - current_w_bal
            recovery = max_recovery * (1 - np.exp(-1.0 / tau))
            current_w_bal += recovery
            
        # Clamp
        if current_w_bal > w_prime:
            current_w_bal = w_prime
        # Allow negative W' (indicates "in the red" / failure zone)
        
        w_bal[i] = current_w_bal
        
    return w_bal
