
import pytest
import numpy as np
import pandas as pd
from planning.optimizer import optimize_pacing

def test_match_burning_count():
    """
    Verify that the number of matches corresponds to distance / interval.
    60 miles / 20 miles = 3 matches.
    """
    # 60 miles approx 96.5 km ~ 96,500m.
    # Create a course with multiple climbs.
    
    dist_m = 97000
    points = int(dist_m / 100)
    
    data = []
    # Create oscillating course: Flat -> Climb -> Flat -> Climb...
    # We need enough climbs to be candidates.
    # 3 matches needed. Let's make 5 climbs.
    
    climb_grad = 5.0
    flat_grad = 0.0
    
    # 5 climbs.
    # Length of each climb: 500m (5 pts).
    # Rest flat.
    
    # Simple construction
    for i in range(points):
        # Every 2000m (20 pts), put a climb
        if (i % 200) < 10: # Climb for 1000m
             grad = 5.0
             # vary gradient slightly to make them sortable?
             # Climb 1: 5.0
             # Climb 2: 6.0
             # Climb 3: 7.0
             # Climb 4: 8.0
             # Climb 5: 4.0
             # Matches should pick 4, 3, 2 (Steepest).
             
             climb_idx = int(i / 200)
             if climb_idx == 0: grad = 5.0
             elif climb_idx == 1: grad = 6.0
             elif climb_idx == 2: grad = 7.0
             elif climb_idx == 3: grad = 8.0
             elif climb_idx == 4: grad = 4.0
             else: grad = 3.0 # Extra climbs
             
        else:
             grad = 0.0
             
        data.append({'gradient': grad})
        
    df = pd.DataFrame(data)
    df['distance'] = df.index * 100
    df['surface'] = 'Paved'
    df['elevation'] = 0 # Ignored by logic mostly
    df['lat'] = 0.0
    df['lon'] = 0.0
    
    cp = 250
    w_prime = 20000
    
    # Optimize
    plan = optimize_pacing(df, cp, w_prime, match_interval_miles=20.0)
    
    # Check cues for "(MATCH)"
    cues = plan[plan['cues'].str.contains("MATCH")].cues.unique()
    
    # Should have 3 matches
    assert len(cues) == 3
    
    # Verify selection (Steepest)
    # The climbs were 8%, 7%, 6%, 5%, 4%.
    # Top 3: 8, 7, 6.
    
    # Extract power from cues to verify elevated power?
    # Or just check which segments got it.
    # Cues string: "Seg X: Climb (Ym) @ ZW" -> "Seg X: Climb (MATCH) ..."
    
    # Let's inspect the target_power of the match segments
    matches = plan[plan['cues'].str.contains("MATCH")]
    # Get the max gradient of these segments
    # Note: 'gradient' in plan is per 100m. 
    # Match segments cover a range.
    
    # We can check the power target.
    # Normal climb power: CP * 1.05 or 1.15.
    # Match power: CP * 1.2 = 300W.
    
    # Check that we have ~300W segments
    high_power_mask = plan['watts'] >= (cp * 1.2 * 0.99) # Approx
    assert high_power_mask.any()
    
    
def test_no_matches_short_course():
    """
    Verify 0 matches on short course.
    """
    # 10 miles ~ 16000m.
    data = [{'gradient': 5.0} for _ in range(160)]
    df = pd.DataFrame(data)
    df['distance'] = df.index * 100
    df['surface'] = 'Paved'
    df['elevation'] = 0
    df['lat'] = 0.0
    df['lon'] = 0.0

    cp = 250
    
    plan = optimize_pacing(df, cp, 20000, match_interval_miles=20.0)
    
    matches = plan[plan['cues'].str.contains("MATCH")]
    assert len(matches) == 0

