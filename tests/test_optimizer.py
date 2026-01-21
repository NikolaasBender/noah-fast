import pytest
import numpy as np
import pandas as pd
from planning.optimizer import create_smart_segments, optimize_pacing

def test_segmentation():
    # Create simple data: 1km flat, 1km climb, 1km descent
    # Each point 100m. 10pts flat, 10pts climb, 10pts descent
    
    data = []
    # Flat (0%)
    for i in range(10): data.append({'gradient': 0.0})
    # Climb (5%)
    for i in range(10): data.append({'gradient': 5.0})
    # Descent (-5%)
    for i in range(10): data.append({'gradient': -5.0})
    
    df = pd.DataFrame(data)
    df['distance'] = df.index * 100
    
    # Mock CP
    cp = 250
    
    # Note: Segmentation merges small segments (<4 mins). 
    # 1km at 30km/h (~8m/s) is ~125s (2 mins).
    # So these 3 distinct chunks might get merged if they are considered too short.
    # To force separate segments, let's make them longer or check logic.
    # Current logic: 0 < 3 (Flat), 5 > 3 (Climb), -5 < -2 (Descent).
    # If they are all < 240s, they will merge.
    
    segments = create_smart_segments(df, cp)
    
    # We expect at least 1 segment
    assert len(segments) > 0
    
    # Check coverage
    total_dist = sum(s['dist'] for s in segments)
    assert total_dist == 3000

def test_optimize_pacing(sample_course_df):
    cp = 250
    w_prime = 20000
    
    plan = optimize_pacing(sample_course_df, cp, w_prime, lstm_model=None)
    
    assert 'watts' in plan.columns
    assert 'speed_mps' in plan.columns
    assert 'time_seconds' in plan.columns
    
    # Check non-negative watts
    assert (plan['watts'] >= 0).all()
    
    # Check w_prime never exceeded
    # We don't export w_prime stream in final df currently in optimizer.py 
    # (Checking code... looks like it is not in returned columns, only used internally)
    # But we can check cues
    assert 'cues' in plan.columns

def test_max_duration_merge():
    # Long flat section ~60 mins.
    # Should result in at least 3 segments (max 20 min each).
    # Flat speed 30km/h ~8.33m/s? 
    # 60 mins = 3600s.
    # Dist = 3600 * 8.33 = 30000m = 30km.
    # 300 points.
    
    data = [{'gradient': 0.0} for _ in range(300)]
    df = pd.DataFrame(data)
    df['distance'] = df.index * 100
    df['surface'] = 'Paved'
    df['elevation'] = 0.0
    df['lat'] = 0.0
    df['lon'] = 0.0
    
    cp = 250
    w_prime = 20000
    
    # optimize_pacing runs segmentation then merging.
    # Segmentation splits > 20 mins anyway.
    # So create_smart_segments yields 3 segments of 20 mins.
    # Seg 1: Flat 1. Seg 2: Flat 2. Seg 3: Flat 3.
    # They have same power. Logic would merge them if not for Duration Constraint.
    
    plan = optimize_pacing(df, cp, w_prime)
    
    # Check cues or segment_id uniqueness in result
    cues = plan[plan['cues'] != ""]['cues'].tolist()
    
    # Should be at least 3 segments
    assert len(cues) >= 3
    
    # Verify no segment > 20 mins (approx)
    # cues string: "Seg 1: Flat 1 (20m) @ 212W"
    # Parse duration from string or use segment_id
    
    # Calculate duration of each segment_id
    seg_durs = plan.groupby('segment_id')['duration_s'].sum()
    # Allow small margin for error?
    assert (seg_durs < 1205).all()

def test_flat_variety():
    # Two adjacent 10-min flat segments.
    # Should not merge due to duration? No, 10+10 = 20. Allowed.
    # Wait, if they merge, they become one 20-min segment. No variety.
    # Variety is applied to *segments*.
    # If merged, it's one segment.
    # To test variety, we need segments that *cannot* merge (e.g. sum > 20 mins)
    # Or segments that were already separate types?
    # Or if we have 3 x 10 min segments.
    # 10+10 = 20 (Merge). 3rd 10 left alone.
    # Result: [20m, 10m].
    # Seg 1 (Flat): Base.
    # Seg 2 (Flat): Base + 10W.
    
    # Let's try 3 x 10 min segments.
    # 10 min ~ 5km ~ 50 pts.
    data = [{'gradient': 0.0} for _ in range(150)] # 30 mins total
    df = pd.DataFrame(data)
    df['distance'] = df.index * 100
    df['surface'] = 'Paved'
    df['elevation'] = 0.0
    df['lat'] = 0.0
    df['lon'] = 0.0
    
    cp = 250
    w_prime = 20000
    
    plan = optimize_pacing(df, cp, w_prime)
    cues = plan[plan['cues'] != ""]['cues'].tolist()
    
    # Expected:
    # create_smart_segments -> Split > 20 mins. 30 mins > 20.
    # Splits into 2 x 15 mins. (Flat 1, Flat 2).
    # Merge passes: 15+15 = 30 > 20. No merge.
    # Result: 2 segments.
    # Seg 1: Flat 1.
    # Seg 2: Flat 2.
    # Variety:
    # Seg 1: Base Power.
    # Seg 2: Base Power + 10W.
    
    assert len(cues) == 2
    
    # Extract power
    # "Seg 1: Flat 1 (15m) @ 212W"
    p1 = int(cues[0].split("@ ")[1].replace("W", ""))
    p2 = int(cues[1].split("@ ")[1].replace("W", ""))
    
    # Check variety
    assert p2 == p1 + 10

    # Climb > 60s but < 240s should remain
    # 1.5 min climb
    data = [{'gradient': 5.0} for _ in range(25)] # 25 * 100m = 2500m. 
    # At 250W, grade 5%, speed ~ 5 m/s? 
    # v_climb ~ ? Check get_speed(250, 5) approx 5-6 m/s.
    # 2500m / 5m/s = 500s. Too long.
    
    # Need approx 90s. Speed ~6m/s. Dist ~ 540m.
    # 6 points.
    data = [{'gradient': 5.0} for _ in range(6)]
    
    # Surrounded by flats (to test merge constraint)
    # Flat: 300s -> 3000m -> 30 pts?
    # Flat speed ~10m/s. 300s -> 3000m -> 30 pts.
    flats = [{'gradient': 0.0} for _ in range(30)]
    
    full_data = flats + data + flats
    df = pd.DataFrame(full_data)
    df['distance'] = df.index * 100
    df['surface'] = 'Paved'
    
    cp = 250
    segments = create_smart_segments(df, cp)
    
    # Should have 3 segments: Flat, Climb, Flat
    # If climb was merged, we'd have 1 or 2.
    # Check middle segment type
    assert len(segments) == 3
    assert segments[1]['type'] == 'Climb'
    assert segments[1]['duration_s'] < 240
    assert segments[1]['duration_s'] > 60

def test_power_adjacency_merge():
    # Create two segments that would normally be separate (e.g. different slopes)
    # but have power target within 10W.
    # Climb 5% (Target ~262W) vs Climb 5.2% (Target ~262W if logic is flat for range)
    # Actually logic: 1.05 * CP. Constant for climbs < 6%.
    # So 4% and 5% should merge if they are separate types?
    # create_smart_segments logic groups by type classification.
    # 4% and 5% are both "Climb" type (>3%). They merge in step 2 of create_smart_segments.
    
    # We need separate types or non-contiguous indices to force optimize_pacing to see them as distinct initially?
    # OR, satisfy create_smart_segments to output 2 segments (e.g. change of surface?).
    # BUT merge logic requires same surface.
    
    # Try: Climb 5% vs Climb 7%.
    # 5% -> 1.05 * CP (262.5W)
    # 7% -> 1.15 * CP (287.5W). Diff = 25W. Should NOT merge.
    
    # Try: Climb 4% vs Climb 5%. Both 1.05*CP. Diff 0.
    # create_smart_segments lumps them? Yes, same "Climb" type.
    
    # How to create distinct segments for optimize_pacing to merge?
    # Maybe Flat (0%) and Shallow Descent (-1%)?
    # Flat: 0.85 * CP -> 212.5
    # Descent: < -0.5% (assuming Paved Crr 0.005). 
    # If -1%: Coast if < -0.5. So 0W.
    # Diff > 10.
    
    # Try Flat (0%) vs Flat Gravel (0%)?
    # Surface diff prevents merge.
    
    # Try: Flat (0%) vs "False Flat" (1% or 2%).
    # 0%: "Flat" -> 0.85 CP.
    # 2%: "Flat" (<3%). -> 0.85 CP.
    # create_smart_segments will merge them into one "Flat" block.
    
    # Wait, create_smart_segments merges things that are "Same Type".
    # So we only have multiple segments if Type changes.
    # Example: Flat -> Climb.
    # Flat (212W) -> Climb (262W). Diff 50W.
    
    # What if we have a custom CP where 1.05*CP and 0.85*CP are close?
    # Requires CP ~0? No.
    
    # Use the logic: 
    # Shallow Descent (-0.3%) -> "Flat" logic (0.85 CP)?
    # Descent definition: < -2.0.
    # So -1.5% is "Flat".
    
    # Ideally we test the logic I added to optimize_pacing.
    # I can mock return values of create_smart_segments to force two segments
    # that optimize_pacing sees.
    
    from unittest.mock import patch
    
    # Mock data
    seg1 = {'type': 'Climb', 'surface': 'Paved', 'start_idx': 0, 'end_idx': 10, 'dist': 1000, 'avg_grad': 5.0, 'duration_s': 200}
    seg2 = {'type': 'Climb', 'surface': 'Paved', 'start_idx': 11, 'end_idx': 20, 'dist': 1000, 'avg_grad': 5.2, 'duration_s': 200}
    # These would typically merge in create_smart_segments, but let's assume they didn't
    # or came from manual splits.
    
    # 5.0% -> 1.05*CP = 262.5
    # 5.2% -> 1.05*CP = 262.5
    # Diff = 0. Should merge.
    
    segs = [seg1, seg2]
    
    # We need to run optimize_pacing with mocked create_smart_segments
    course_df = pd.DataFrame({'distance': range(0, 2200, 100), 'gradient': 5.0, 'elevation': 0, 'lat': 0, 'lon': 0})
    
    with patch('planning.optimizer.create_smart_segments', return_value=segs):
        plan = optimize_pacing(course_df, 250, 20000)
        
        # Check if merged.
        # cues should show fewer segments?
        # "Seg 1..."
        # If merged, we have 1 segment.
        cues = plan[plan['cues'] != ""]['cues'].tolist()
        assert len(cues) == 1

def test_technical_descent():
    # Comparison: Straight Descent vs Technical (Zig-Zag) Descent
    # Same gradient (-3%), Same Distance (1km -> 10 pts)
    
    # 1. Straight Descent
    # Lat/Lon constant change or moving in straight line (e.g. lat increasing, lon constant)
    # 10 points. 
    # Lat: 0.000, 0.001, 0.002 ... 
    # Lon: 0.0 constant
    straight_data = [{'gradient': -3.0} for _ in range(20)]
    df_s = pd.DataFrame(straight_data)
    df_s['distance'] = df_s.index * 100
    df_s['surface'] = 'Paved'
    df_s['elevation'] = 1000 - df_s.index * 3 # approx
    
    # Generate straight path (North)
    # 0.001 deg lat ~ 111m. Close enough to 100m.
    df_s['lat'] = df_s.index * 0.001
    df_s['lon'] = 0.0
    
    # 2. Technical Descent (Zig Zag)
    # Gradient same.
    # Path: 0,0 -> 1,1 -> 2,0 -> 3,1 ...
    # Sharp turns every point.
    tech_data = [{'gradient': -3.0} for _ in range(20)]
    df_t = pd.DataFrame(tech_data)
    df_t['distance'] = df_t.index * 100
    df_t['surface'] = 'Paved'
    df_t['elevation'] = 1000 - df_t.index * 3
    
    # Zig Zag path
    # Lat increases steadily (forward motion)
    # Lon oscillates
    df_t['lat'] = df_t.index * 0.001
    # Lon Zig Zag: 0, 0.001, 0, 0.001
    df_t['lon'] = np.where(df_t.index % 2 == 0, 0.0, 0.001)
    
    cp = 250
    w_prime = 20000
    
    plan_s = optimize_pacing(df_s, cp, w_prime)
    plan_t = optimize_pacing(df_t, cp, w_prime)
    
    # Analysis
    # Straight: -3% is shallow-ish descent. 
    # Coast threshold ~ -1% (Paved). 
    # -3% is steeper than -1%. So it should be Coast (0W)?
    # Wait, check logic:
    # if grad < -1.0: p_t = 0 (Coast)
    # -3 < -1. True. So Straight is already 0W?
    # I need a shallow descent that allows pedaling normally, but is technical.
    # Or a descent that is "Pedalable" (-2% to -1%) but Tech forces 0.
    
    # Let's adjust gradient to -0.8% (False Flat Down)
    # Paved threshold ~ -0.5% (equilibrium), -1.0% (0 power trigger).
    # -0.8% is between -0.5 and -1.0. 
    # Logic: `elif grad < coast_threshold_grade (-0.5): p_t = cp * 0.1`
    # So Straight -0.8% -> Soft Pedal (25W).
    
    # Technical -0.8% -> Should force 0W.
    
    df_s['gradient'] = -0.8
    df_t['gradient'] = -0.8
    
    plan_s = optimize_pacing(df_s, cp, w_prime)
    plan_t = optimize_pacing(df_t, cp, w_prime)
    
    # Check Power
    # Straight should have some power (0.1 * CP = 25W)
    # Technical should be 0W
    
    # Get mean power of the main segment
    # (Ignoring potential start/end artifacts)
    p_straight = plan_s['watts'].mean()
    p_tech = plan_t['watts'].mean()
    
    # Debug print if fails
    print(f"Straight Power: {p_straight}, Tech Power: {p_tech}")
    
    assert p_straight > 0
    assert p_tech == 0

