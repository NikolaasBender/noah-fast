import pandas as pd
import numpy as np
from modeling.nutrition import calculate_metabolic_cost, carb_needs_per_hour
from modeling.physiology import calculate_w_prime_balance


# Physics Constants
RIDER_MASS = 85 # kg
GRAVITY = 9.81
CRR = 0.005
CDA = 0.32
RHO = 1.225

def get_speed(power, gradient_percent):
    """
    Estimates speed (m/s) for a given power and gradient using a simplified physics model.
    """
    grad_decimal = gradient_percent / 100.0
    sin_theta = np.sin(np.arctan(grad_decimal))
    cos_theta = np.cos(np.arctan(grad_decimal))
    
    # F_resist = F_gravity + F_rolling + F_drag
    # F_const = m*g*sin + m*g*cos*crr
    # P = (F_const + 0.5*rho*cda*v^2) * v
    
    F_const = RIDER_MASS * GRAVITY * (sin_theta + cos_theta * CRR)
    
    # Newton-Raphson to solve for v
    v = 10.0 # Initial guess (10 m/s = 36 km/h)
    for _ in range(5):
        # f(v) = 0.5*rho*cda*v^3 + F_const*v - P
        # f'(v) = 1.5*rho*cda*v^2 + F_const
        
        fx = 0.5 * RHO * CDA * v**3 + F_const * v - power
        fpx = 1.5 * RHO * CDA * v**2 + F_const
        
        if abs(fpx) < 1e-5: break # Avoid div/0
        v = v - fx / fpx
        
        if v < 0.1: v = 0.1 # Min speed
        
    return v

def create_smart_segments(resampled_df, cp):
    """
    Groups the 100m chunks into logical segments (4-20 mins).
    """
    
    # 1. Initial Classification
    def classify(grad):
        if grad > 3.0: return 'Climb'
        elif grad < -2.0: return 'Descent'
        else: return 'Flat'
        
    resampled_df['type'] = resampled_df['gradient'].apply(classify)
    
    # 2. Run-Length Encoding (Merge adjacent same types)
    segments = []
    current_seg = None
    
    for idx, row in resampled_df.iterrows():
        if current_seg is None:
            current_seg = {
                'type': row['type'],
                'start_idx': idx,
                'end_idx': idx,
                'dist': 100,
                'avg_grad': row['gradient']
            }
        else:
            if row['type'] == current_seg['type']:
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
                    'start_idx': idx,
                    'end_idx': idx,
                    'dist': 100,
                    'avg_grad': row['gradient']
                }
    if current_seg: segments.append(current_seg)
    
    # 3. Time Estimation (Pre-Merge)
    # Estimate duration based on CP (steady state proxy)
    for seg in segments:
        # heuristic power
        if seg['type'] == 'Climb': p = cp * 1.05
        elif seg['type'] == 'Descent': p = cp * 0.1
        else: p = cp * 0.85
        
        speed = get_speed(p, seg['avg_grad'])
        seg['duration_s'] = seg['dist'] / speed

    # 4. Merge Small Segments (< 4 mins / 240s)
    # We loop until no segments are too small (or can't be merged)
    # Strategy: Merge small segment into its similar neighbor or just the previous one?
    # Simple strategy: Merge into the PREVIOUS segment, changing type to the dominator (or keep prev)
    # Exception: Don't merge a steep Climb into a Descent.
    
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
                    # Merge Logic
                    # Combined properties
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
                        'start_idx': seg['start_idx'],
                        'end_idx': next_seg['end_idx'],
                        'dist': total_dist,
                        'avg_grad': avg_grad,
                        'duration_s': seg['duration_s'] + next_seg['duration_s'] # rough sum
                    }
                    new_segments.append(merged)
                    skip = True # Skip next since we consumed it
                    changed = True
                else:
                    # Last segment is short, merge with previous if exists
                    if new_segments:
                        prev = new_segments.pop()
                        # Merge logic same as above
                        total_dist = prev['dist'] + seg['dist']
                        w1 = prev['dist'] / total_dist
                        w2 = seg['dist'] / total_dist
                        avg_grad = prev['avg_grad'] * w1 + seg['avg_grad'] * w2
                         # Keep prev type usually
                        merged = {
                            'type': prev['type'],
                            'start_idx': prev['start_idx'],
                            'end_idx': seg['end_idx'],
                            'dist': total_dist,
                            'avg_grad': avg_grad,
                            'duration_s': prev['duration_s'] + seg['duration_s']
                        }
                        new_segments.append(merged)
                    else:
                        # Only one segment and it's short, keep it
                        new_segments.append(seg)
            else:
                new_segments.append(seg)
        
        segments = new_segments

    # 5. Split Long Segments (> 20 mins / 1200s)
    MAX_DURATION = 1200
    final_segments = []
    
    for seg in segments:
        if seg['duration_s'] > MAX_DURATION:
            # How many splits?
            num_splits = int(np.ceil(seg['duration_s'] / MAX_DURATION))
            # Split distance equally
            chunk_dist = seg['dist'] / num_splits
            chunk_idx_span = (seg['end_idx'] - seg['start_idx']) / num_splits
            
            for k in range(num_splits):
                s_idx = int(seg['start_idx'] + k * chunk_idx_span)
                e_idx = int(seg['start_idx'] + (k+1) * chunk_idx_span)
                if k == num_splits - 1: e_idx = seg['end_idx'] # Ensure closure
                
                # Get sub-gradient
                # Approximate from avg (or re-query df, but approximation is okay for V1)
                sub_grad = seg['avg_grad'] 
                
                # Recalc duration
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

def optimize_pacing(course_df, cp, w_prime, lstm_model=None):
    """
    Generates a pacing plan with smart segmentation.
    """
    
    # 1. Resample (100m)
    max_dist = course_df['distance'].max()
    new_dist = np.arange(0, max_dist, 100)
    
    resampled = pd.DataFrame({'distance': new_dist})
    resampled['gradient'] = np.interp(new_dist, course_df['distance'], course_df['gradient'])
    resampled['elevation'] = np.interp(new_dist, course_df['distance'], course_df['elevation'])
    resampled['lat'] = np.interp(new_dist, course_df['distance'], course_df['lat'])
    resampled['lon'] = np.interp(new_dist, course_df['distance'], course_df['lon'])
    
    # 2. Segment
    segments = create_smart_segments(resampled, cp)
    
    # 3. Assign Power & Simulate
    # Global state
    current_w_bal = w_prime
    
    # We will write 'segment_id' and 'target_power' back to resampled df
    resampled['segment_id'] = -1
    resampled['target_power'] = 0.0
    resampled['cues'] = "" # For cues later
    
    final_powers = []
    
    cumulative_time = 0
    
    for seg_id, seg in enumerate(segments):
        # Base Power Strategy
        base_type = seg['type'].split(" ")[0] # Handle "Climb 1"
        grad = seg['avg_grad']
        
        if base_type == 'Climb':
            p_target = cp * 1.05 # Start optimistic
            if grad > 6: p_target = cp * 1.15
        elif base_type == 'Descent':
            p_target = cp * 0.1 # Recovery
            if grad < -5: p_target = 0
        else: # Flat
            p_target = cp * 0.85 # Sweetspot/Tempo
            
        # Refine Power based on W' availability
        # Simulate this segment
        # dist is seg['dist']
        speed = get_speed(p_target, grad)
        duration = seg['dist'] / speed
        
        w_cost = (p_target - cp) * duration
        
        if p_target > cp:
            if current_w_bal - w_cost < 0:
                # We will bonk!
                # Reduce power to CP (or slightly under)
                p_target = cp * 0.98
                # Re-calc duration
                speed = get_speed(p_target, grad)
                duration = seg['dist'] / speed
                w_cost = (p_target - cp) * duration # should be negative (recharge) or tiny cost
        
        # Update W'
        current_w_bal -= w_cost
        if current_w_bal > w_prime: current_w_bal = w_prime
        
        # Assign to 100m chunks
        # Range in resampled df is [start_idx, end_idx) roughly?
        # Actually start_idx to end_idx logic in create_smart_segments needs careful indexing
        # Let's trust indices from create_smart_segments
        
        # Apply to DataFrame
        # Note: end_idx in 'segments' logic (run length) was inclusive? 
        # In my logic: current_seg['end_idx'] = idx. So yes inclusive.
        
        s = seg['start_idx']
        e = seg['end_idx']
        
        # Fill rows
        resampled.loc[s:e, 'segment_id'] = seg_id
        resampled.loc[s:e, 'target_power'] = p_target
        
        # Set Cue at Start
        seg_dur_min = duration / 60.0
        resampled.loc[s, 'cues'] = f"Seg {seg_id+1}: {seg['type']} ({seg_dur_min:.0f}m) @ {int(p_target)}W"
        
        cumulative_time += duration

    # 4. Final Data Assembly
    # We essentially have target_power filled.
    # Recalculate accurate speeds/times row-by-row for the export
    
    resampled['watts'] = resampled['target_power']
    resampled['speed_mps'] = resampled.apply(lambda r: get_speed(r['watts'], r['gradient']), axis=1)
    resampled['duration_s'] = 100 / resampled['speed_mps']
    resampled['time_seconds'] = resampled['duration_s'].cumsum()
    
    # Nutrition
    resampled['kcal_hr'] = resampled['watts'].apply(calculate_metabolic_cost)
    resampled['carbs_hr'] = resampled['watts'].apply(carb_needs_per_hour)
    
    return resampled

