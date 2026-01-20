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
        elif grad < -0.5: return 'Descent'
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

    # 4. Merge Small Segments
    # Climbs: Min 60s. Others: Min 240s.
    MIN_DURATION_CLIMB = 60
    MIN_DURATION_DEFAULT = 240
    
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
            
            # Determine threshold
            min_dur = MIN_DURATION_CLIMB if seg['type'] == 'Climb' else MIN_DURATION_DEFAULT
            
            # Check if too small
            if seg['duration_s'] < min_dur:
                # Try to merge with Next
                if i < len(segments) - 1:
                    next_seg = segments[i+1]
                    
                    # Merge Logic (Require same surface!)
                    if seg['surface'] == next_seg['surface']:
                        total_dist = seg['dist'] + next_seg['dist']
                        w1 = seg['dist'] / total_dist
                        w2 = next_seg['dist'] / total_dist
                        avg_grad = seg['avg_grad'] * w1 + next_seg['avg_grad'] * w2
                        
                        # New Type? Favor the larger one OR if one is a "Significant Climb", maybe keep it?
                        # Standard logic: favor larger duration
                        if next_seg['duration_s'] > seg['duration_s']:
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
                             
                             # Type logic? Favor previous (larger) usually
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
        
    # --- Path Analysis: Sinuosity / Technicality ---
    def calculate_bearing(lat1, lon1, lat2, lon2):
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        initial_bearing = np.arctan2(x, y)
        # Convert to degrees and normalize to 0-360
        deg = np.degrees(initial_bearing)
        return (deg + 360) % 360

    # Vectorized bearing calculation
    # Shift arrays to get next point
    lat1 = resampled['lat'].values[:-1]
    lon1 = resampled['lon'].values[:-1]
    lat2 = resampled['lat'].values[1:]
    lon2 = resampled['lon'].values[1:]
    
    bearings = calculate_bearing(lat1, lon1, lat2, lon2)
    # Pad last value
    bearings = np.append(bearings, bearings[-1] if len(bearings) > 0 else 0)
    resampled['bearing'] = bearings
    
    # Calculate Bearing *Change* (absolute, accounting for wrap)
    # diff: 350 -> 10 = 20 deg turn (not -340)
    # diff = abs(b2 - b1)
    # if diff > 180: diff = 360 - diff
    b_diff = np.abs(np.diff(bearings, prepend=bearings[0]))
    b_diff = np.where(b_diff > 180, 360 - b_diff, b_diff)
    resampled['bearing_diff'] = b_diff
    
    # Sinuosity Metric: Rolling sum of bearing changes over e.g. 300m (3 points)
    # "Technical" if lots of turning.
    resampled['sinuosity'] = pd.Series(b_diff).rolling(window=3, center=True).mean().fillna(0)
    
    # Define Technical Threshold: Average > 15 degrees change per 100m?
    # A 90 deg turn in 100m is 90.
    # A gentler curve might be 5-10.
    TECHNICAL_THRESHOLD = 15.0 

    # 2. Segment
    segments = create_smart_segments(resampled, cp, rider_mass=rider_mass)
    
    # 3. Assign Power & Simulate
    current_w_bal = w_prime
    
    resampled['segment_id'] = -1
    resampled['target_power'] = 0.0
    resampled['cues'] = "" 
    
    # LSTM Prep
    recent_history = [] 
    
    cumulative_time = 0
    
    # Surface Multipliers
    SURFACE_CRR_MAP = {
        'Paved': 1.0,
        'Gravel': 1.6,
        'Dirt': 1.8
    }

    # Helper to calc target power for a segment
    def get_segment_power(seg):
        # Determine segment surface (majority wins or start point)
        seg_start = seg['start_idx']
        s_type = resampled.loc[seg_start, 'surface'] if 'surface' in resampled.columns else 'Paved'
        crr_mult = SURFACE_CRR_MAP.get(s_type, 1.0)
        eff_crr = crr * crr_mult
        
        # Determine Technicality
        # Average sinuosity of the segment
        seg_sinuosity = resampled.loc[seg['start_idx']:seg['end_idx'], 'sinuosity'].mean()
        is_technical = seg_sinuosity > TECHNICAL_THRESHOLD

        base_type = seg['type'].split(" ")[0] 
        grad = seg['avg_grad']
        
        p_t = cp * 0.85 # Default
        
        if base_type == 'Climb':
            p_t = cp * 1.05 
            if grad > 6: p_t = cp * 1.15
        elif base_type == 'Descent':
            coast_threshold_grade = -(eff_crr * 100)
            
            # Technical Descent Override
            if is_technical:
                # Winding descent: Safety first, cornering means inconsistent power.
                # Often coasting into turns, sprinting out. Avg power lower.
                # Let's be conservative: Coast (0W).
                p_t = 0
            elif grad < coast_threshold_grade - 0.5: 
                p_t = 0 
            elif grad < coast_threshold_grade: 
                 p_t = cp * 0.1 
            else:
                 p_t = cp * 0.70
        else: # Flat
            if is_technical:
                 # Technical flat? (Criterium?)
                 # Reduced power for cornering?
                 p_t = cp * 0.75
            else:
                 p_t = cp * 0.85
            
        return p_t

    # --- Pre-Calculation & Merging Pass ---
    # 1. Calc initial power
    for seg in segments:
        seg['p_target'] = get_segment_power(seg)

    # 2. Merge by Power (diff <= 10W)
    #    AND Duration Constraint (< 20 mins / 1200s)
    MAX_MERGE_DURATION = 1200
    
    if len(segments) > 0:
        merged_segments = []
        curr = segments[0]
        
        for next_seg in segments[1:]:
            diff = abs(curr['p_target'] - next_seg['p_target'])
            
            # Predict new duration
            new_dur = curr['duration_s'] + next_seg['duration_s']
            
            if diff <= 10 and new_dur <= MAX_MERGE_DURATION:
                # Merge next into curr
                total_dist = curr['dist'] + next_seg['dist']
                # Weighted grad
                w1 = curr['dist'] / total_dist
                w2 = next_seg['dist'] / total_dist
                new_grad = curr['avg_grad'] * w1 + next_seg['avg_grad'] * w2
                
                # Update curr
                curr['end_idx'] = next_seg['end_idx']
                curr['dist'] = total_dist
                curr['avg_grad'] = new_grad
                curr['duration_s'] = new_dur
                
                # Re-calc Power for the new merged block (Type might have changed? No, we kept start type)
                curr['p_target'] = get_segment_power(curr)
                
            else:
                merged_segments.append(curr)
                curr = next_seg
        
        merged_segments.append(curr)
        segments = merged_segments

    # 3. Add Variety to Flat Segments
    #    If we have consecutive flats, alternate power (+10W)
    variety_toggle = False
    for seg in segments:
        base_type = seg['type'].split(" ")[0]
        if base_type == 'Flat':
            if variety_toggle:
                seg['p_target'] += 10.0
            variety_toggle = not variety_toggle
        else:
            variety_toggle = False # Reset sequence on non-flat

    # --- Main Simulation Loop ---
    for seg_id, seg in enumerate(segments):
        # Determine segment surface (majority wins or start point)
        seg_start = seg['start_idx']
        s_type = resampled.loc[seg_start, 'surface'] if 'surface' in resampled.columns else 'Paved'
        crr_mult = SURFACE_CRR_MAP.get(s_type, 1.0)
        eff_crr = crr * crr_mult
        
        p_target = seg['p_target']
        grad = seg['avg_grad']
        
        # Refine Power based on W' availability
        speed = get_speed(p_target, grad, eff_crr, cda, rider_mass)
        # Avoid div by zero
        if speed < 0.1: speed = 0.1
        duration = seg['dist'] / speed
        w_cost = (p_target - cp) * duration
        
        if p_target > cp:
            if current_w_bal - w_cost < 0:
                # Bonk mitigation
                p_target = cp * 0.98
                speed = get_speed(p_target, grad, eff_crr, cda, rider_mass)
                if speed < 0.1: speed = 0.1
                duration = seg['dist'] / speed
                w_cost = (p_target - cp) * duration
        
        # Update W'
        current_w_bal -= w_cost
        if current_w_bal > w_prime: current_w_bal = w_prime
        
        # Update History (Approximated)
        sim_hr = 60 + (p_target / cp) * 110 # Crude HR model
        sim_cad = 90 if p_target > 0 else 0
        
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

