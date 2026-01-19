import pandas as pd
import numpy as np
from modeling.nutrition import calculate_metabolic_cost, carb_needs_per_hour
from modeling.physiology import calculate_w_prime_balance


import json
import os

# Physics Constants (Defaults)
GRAVITY = 9.81
RHO = 1.225

# Load Profiles
BIKE_PROFILES = {}
try:
    with open('data/models/bike_profiles.json', 'r') as f:
        BIKE_PROFILES = json.load(f)
except:
    pass

def get_speed(power, gradient_percent, crr=0.005, cda=0.32, rider_mass=85.0):
    """
    Estimates speed (m/s) for a given power and gradient using a simplified physics model.
    Robust to descents (Newton-Raphson on Force Balance or Power Balance).
    """

    grad_decimal = gradient_percent / 100.0
    # Small angle approx is usually fine, but let's keep trig
    # But beware gradient_percent large inputs.
    
    # Precompute constants
    # mg * ...
    # F_grav = mg sin(theta)
    # F_roll = mg cos(theta) * Crr
    # We define F_resist_constant (Gravity + Rolling)
    # BE CAREFUL SIGNS: 
    # Standard physics: P_total = P_aero + P_roll + P_grav
    # P_grav = mg * v * sin(theta). (Positive if climbing).
    # P_roll = mg * v * cos(theta) * Crr
    
    # Angles
    theta = np.arctan(grad_decimal)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Forces
    weight = rider_mass * GRAVITY
    F_grav = weight * sin_theta
    F_roll = weight * cos_theta * crr
    
    # F_static = F_grav + F_roll
    # P_req = (0.5*rho*cda*v^2 + F_static) * v
    #       = K*v^3 + F_static*v
    
    K = 0.5 * RHO * cda
    F_static = F_grav + F_roll
    
    # If Power > 0, we can use Newton Raphson
    # f(v) = K v^3 + F_static v - Power = 0
    # f'(v) = 3 K v^2 + F_static
    
    # Initial guessing is critical. 
    # If F_static is negative (Descent) and Power is small, v can be large.
    # Terminal velocity (Power=0) -> K v^2 + F_static = 0 => v = sqrt(-F_static / K)
    
    if F_static < 0 and power == 0:
         # Coasting on descent?
         # Check if gravity overcomes rolling: F_grav + F_roll < 0 ?
         # i.e. F_grav is negative enough (large descent) 
         if F_static < 0:
             return np.sqrt(-F_static / K)
         else:
             # Gravity not enough to overcome rolling, and no power -> Stop
             return 0.1

    v = 10.0 # Default guess
    
    # Descent Optimization: Better guess for descents
    if F_static < 0:
        # Guess terminal velocity approx
        v_term = np.sqrt(abs(F_static)/K)
        v = v_term

    for _ in range(8): # Increased iterations
        fx = K * v**3 + F_static * v - power
        fpx = 3 * K * v**2 + F_static
        
        if abs(fpx) < 1e-5: 
            # Flat slope (inflection)? Nudge v
            v += 1.0 
            continue
            
        v_new = v - fx / fpx
        
        if abs(v_new - v) < 0.01:
            return v_new
            
        v = v_new
        
        if v < 0.1: v = 0.1 # Clamp min
        
    return v

def create_smart_segments(resampled_df, cp, rider_mass=85.0):
    """
    Groups the 100m chunks into logical segments (4-20 mins).
    """
    
    # 1. Initial Classification
    def classify(grad):
        if grad > 3.0: return 'Climb'
        elif grad < -2.0: return 'Descent'
        else: return 'Flat'
        
    resampled_df['type'] = resampled_df['gradient'].apply(classify)
    
    # ... (Run length encoding is same) ...
    # 2. Run-Length Encoding (Merge adjacent same types)
    segments = []
    current_seg = None
    
    for idx, row in resampled_df.iterrows():
        # Handle missing surface if not present (defensive)
        r_surf = row['surface'] if 'surface' in row else 'Paved'
        
        if current_seg is None:
            current_seg = {
                'type': row['type'],
                'surface': r_surf,
                'start_idx': idx,
                'end_idx': idx,
                'dist': 100,
                'avg_grad': row['gradient']
            }
        else:
            # Extend if Type AND Surface match
            if row['type'] == current_seg['type'] and r_surf == current_seg['surface']:
                # Extend
                current_seg['end_idx'] = idx
                current_seg['dist'] += 100
                # Update moving average gradient
                n = (current_seg['dist'] / 100)
                current_seg['avg_grad'] = current_seg['avg_grad'] * ((n-1)/n) + row['gradient'] * (1/n)
            else:
                # Finish current
                segments.append(current_seg)
                # Start new
                current_seg = {
                    'type': row['type'],
                    'surface': r_surf,
                    'start_idx': idx,
                    'end_idx': idx,
                    'dist': 100,
                    'avg_grad': row['gradient']
                }
    if current_seg: segments.append(current_seg)
    
    # 3. Time Estimation (Pre-Merge)
    # Estimate duration based on CP (steady state proxy)
    def est_speed(p, g):
        return get_speed(p, g, rider_mass=rider_mass)

    for seg in segments:
        # heuristic power
        if seg['type'] == 'Climb': p = cp * 1.05
        elif seg['type'] == 'Descent': p = cp * 0.1
        else: p = cp * 0.85
        
        speed = est_speed(p, seg['avg_grad'])
        seg['duration_s'] = seg['dist'] / speed

    # 4. Merge Small Segments (< 4 mins / 240s)
    MIN_DURATION = 240 # 4 mins
    
    changed = True
    while changed:
        changed = False
        new_segments = []
        skip = False
        
        for i in range(len(segments)):
            if skip:
                skip = False
                continue
                
            seg = segments[i]
            
            # Check if too small
            if seg['duration_s'] < MIN_DURATION:
                # Try to merge with Next
                if i < len(segments) - 1:
                    next_seg = segments[i+1]
                    
                    # Merge Logic (Require same surface!)
                    if seg['surface'] == next_seg['surface']:
                        total_dist = seg['dist'] + next_seg['dist']
                        w1 = seg['dist'] / total_dist
                        w2 = next_seg['dist'] / total_dist
                        avg_grad = seg['avg_grad'] * w1 + next_seg['avg_grad'] * w2
                        
                        # New Type? Favor the larger one
                        if next_seg['dist'] > seg['dist']:
                            new_type = next_seg['type']
                        else:
                            new_type = seg['type']
                            
                        merged = {
                            'type': new_type,
                            'surface': seg['surface'],
                            'start_idx': seg['start_idx'],
                            'end_idx': next_seg['end_idx'],
                            'dist': total_dist,
                            'avg_grad': avg_grad,
                            'duration_s': seg['duration_s'] + next_seg['duration_s']
                        }
                        new_segments.append(merged)
                        skip = True # Skip next since we consumed it
                        changed = True
                    else:
                        # Cannot merge different surfaces
                        new_segments.append(seg)
                else:
                    # Last segment is short, merge with previous if exists AND surface matches
                    if new_segments:
                        prev = new_segments[-1] # Peek
                        if prev['surface'] == seg['surface']:
                             prev = new_segments.pop()
                             total_dist = prev['dist'] + seg['dist']
                             w1 = prev['dist'] / total_dist
                             w2 = seg['dist'] / total_dist
                             avg_grad = prev['avg_grad'] * w1 + seg['avg_grad'] * w2
                             merged = {
                                 'type': prev['type'],
                                 'surface': prev['surface'],
                                 'start_idx': prev['start_idx'],
                                 'end_idx': seg['end_idx'],
                                 'dist': total_dist,
                                 'avg_grad': avg_grad,
                                 'duration_s': prev['duration_s'] + seg['duration_s']
                             }
                             new_segments.append(merged)
                        else:
                             new_segments.append(seg)
                    else:
                        new_segments.append(seg)
            else:
                new_segments.append(seg)
        
        segments = new_segments

    # 5. Split Long Segments (> 20 mins / 1200s)
    MAX_DURATION = 1200
    final_segments = []
    
    for seg in segments:
        if seg['duration_s'] > MAX_DURATION:
            num_splits = int(np.ceil(seg['duration_s'] / MAX_DURATION))
            chunk_dist = seg['dist'] / num_splits
            chunk_idx_span = (seg['end_idx'] - seg['start_idx']) / num_splits
            
            for k in range(num_splits):
                s_idx = int(seg['start_idx'] + k * chunk_idx_span)
                e_idx = int(seg['start_idx'] + (k+1) * chunk_idx_span)
                if k == num_splits - 1: e_idx = seg['end_idx']
                
                sub_grad = seg['avg_grad'] 
                sub_dur = seg['duration_s'] / num_splits
                
                final_segments.append({
                    'type': f"{seg['type']} {k+1}",
                    'start_idx': s_idx,
                    'end_idx': e_idx,
                    'dist': chunk_dist,
                    'avg_grad': sub_grad,
                    'duration_s': sub_dur
                })
        else:
            final_segments.append(seg)
            
    return final_segments

def optimize_pacing(course_df, cp, w_prime, lstm_model=None, gear_id=None, rider_mass=85.0):
    """
    Generates a pacing plan with smart segmentation, specific bike physics, and LSTM constraints.
    """
    
    # Select Pysics
    crr = 0.005
    cda = 0.32
    if gear_id and gear_id in BIKE_PROFILES:
        crr = BIKE_PROFILES[gear_id]['crr']
        cda = BIKE_PROFILES[gear_id]['cda']
        # Sanity bounds (model fitting can be wild)
        if crr < 0.001: crr = 0.001
        if cda < 0.1: cda = 0.1
    
    # 1. Resample (100m)
    max_dist = course_df['distance'].max()
    new_dist = np.arange(0, max_dist, 100)
    
    resampled = pd.DataFrame({'distance': new_dist})
    resampled['gradient'] = np.interp(new_dist, course_df['distance'], course_df['gradient'])
    resampled['elevation'] = np.interp(new_dist, course_df['distance'], course_df['elevation'])
    resampled['lat'] = np.interp(new_dist, course_df['distance'], course_df['lat'])
    resampled['lon'] = np.interp(new_dist, course_df['distance'], course_df['lon'])
    
    # Map Categorical Surface
    if 'surface' in course_df.columns:
        # Find nearest index in course_df for each new_dist
        idx = np.searchsorted(course_df['distance'], new_dist, side='right') - 1
        idx[idx < 0] = 0
        resampled['surface'] = course_df['surface'].iloc[idx].values
    else:
        resampled['surface'] = 'Paved'
    
    # 2. Segment
    segments = create_smart_segments(resampled, cp, rider_mass=rider_mass)
    
    # 3. Assign Power & Simulate
    current_w_bal = w_prime
    
    resampled['segment_id'] = -1
    resampled['target_power'] = 0.0
    resampled['cues'] = "" 
    
    # LSTM Prep
    # We need to construct a sequence: [watts, hr, cad, w_bal]
    # For simulation, we don't have true HR/Cadence. We must approximate.
    # HR ~ Power (linear lag). Cadence ~ 90.
    recent_history = [] 
    
    cumulative_time = 0
    
    # Surface Multipliers
    SURFACE_CRR_MAP = {
        'Paved': 1.0,
        'Gravel': 1.6,
        'Dirt': 1.8
    }
    
    # ... (Segments loop) ...
    for seg_id, seg in enumerate(segments):
        # Determine segment surface (majority wins or start point)
        # Using the start idx to check surface in original resampled df
        seg_start = seg['start_idx']
        s_type = resampled.loc[seg_start, 'surface'] if 'surface' in resampled.columns else 'Paved'
        crr_mult = SURFACE_CRR_MAP.get(s_type, 1.0)
        
        # Effective params for this segment
        eff_crr = crr * crr_mult
        
        # Base Power Strategy
        base_type = seg['type'].split(" ")[0] 
        grad = seg['avg_grad']
        
        if base_type == 'Climb':
            p_target = cp * 1.05 
            if grad > 6: p_target = cp * 1.15
        elif base_type == 'Descent':
            # "Coasting Logic"
            # When does gravity overcome resistance?
            # F_gravity > F_roll + F_aero (at some V)
            # F_g = m*g*sin(theta) ~ m*g*grad
            # F_r = m*g*cos(theta)*Crr ~ m*g*Crr
            # COAST THRESHOLD: when grad < -Crr (approx, in decimal)
            # Paved Crr ~0.005 -> -0.5% grade
            # Gravel Crr ~0.008 -> -0.8% grade
            
            coast_threshold_grade = -(eff_crr * 100)
            
            # If steeper than threshold (more negative), we can coast
            if grad < coast_threshold_grade - 0.5: # buffer
                p_target = 0 # Coast
            elif grad < coast_threshold_grade: 
                 p_target = cp * 0.1 # Soft pedal
            else:
                 # Shallow descent where friction dominates (esp gravel)
                 # Treat closer to Flat
                 p_target = cp * 0.70
                 
        else: # Flat
            p_target = cp * 0.85
            if s_type != 'Paved':
               # Increase power slightly on flat gravel to maintain momentum/speed? 
               # Or decrease to save energy? 
               # Optimal control usually suggests smoothing effort. 
               # Let's keep constant power -> resulting speed drops.
               pass

        # ... (LSTM Logic Placeholder) ...

        # Refine Power based on W' availability
        speed = get_speed(p_target, grad, eff_crr, cda, rider_mass)
        duration = seg['dist'] / speed
        w_cost = (p_target - cp) * duration
        
        if p_target > cp:
            if current_w_bal - w_cost < 0:
                p_target = cp * 0.98
                speed = get_speed(p_target, grad, eff_crr, cda, rider_mass)
                duration = seg['dist'] / speed
                w_cost = (p_target - cp) * duration
        
        # Update W'
        current_w_bal -= w_cost
        if current_w_bal > w_prime: current_w_bal = w_prime
        
        # Update History (Approximated)
        # Add N seconds worth of data
        # For efficiency, just add 1 point representing the Segment?
        # Or detailed. Detailed is better for LSTM.
        # Let's add 1 point per 100m for history approx
        sim_hr = 60 + (p_target / cp) * 110 # Crude HR model
        sim_cad = 90 if p_target > 0 else 0
        
        # We process this segment length in chunks for history? 
        # Just appending end-state for now to keep loop fast.
        recent_history.append([p_target, sim_hr, sim_cad, current_w_bal])
        
        # Assign to 100m chunks
        s = seg['start_idx']
        e = seg['end_idx']
        
        resampled.loc[s:e, 'segment_id'] = seg_id
        resampled.loc[s:e, 'target_power'] = p_target
        
        seg_dur_min = duration / 60.0
        resampled.loc[s, 'cues'] = f"Seg {seg_id+1}: {seg['type']} ({seg_dur_min:.0f}m) @ {int(p_target)}W"
        
        cumulative_time += duration

    # 4. Final Data Assembly
    resampled['watts'] = resampled['target_power']
    
    def calc_row_speed(r):
        s_type = r['surface'] if 'surface' in r else 'Paved'
        # Check against the local SURFACE_CRR_MAP if possible, or define again. 
        # Since it's inside function scope, we can access it if defined above loop? 
        # Yes, defined at line ~250.
        mult = SURFACE_CRR_MAP.get(s_type, 1.0)
        return get_speed(r['watts'], r['gradient'], crr * mult, cda, rider_mass)

    resampled['speed_mps'] = resampled.apply(calc_row_speed, axis=1)
    resampled['duration_s'] = 100 / resampled['speed_mps']
    resampled['time_seconds'] = resampled['duration_s'].cumsum()
    
    # Nutrition
    resampled['kcal_hr'] = resampled['watts'].apply(calculate_metabolic_cost)
    resampled['carbs_hr'] = resampled['watts'].apply(carb_needs_per_hour)
    
    return resampled

