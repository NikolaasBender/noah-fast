import pytest
import numpy as np
import pandas as pd
from modeling.physiology import cp_model, calculate_cp_and_w_prime, calculate_w_prime_balance, extract_power_profile

def test_cp_model():
    cp = 200
    w_prime = 20000
    t = 100
    # P = CP + W'/t
    expected = 200 + 20000/100 # 400
    assert cp_model(t, cp, w_prime) == expected

def test_extract_power_profile():
    sample_activity_df = pd.DataFrame({
        'time': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-01 12:00:01']),
        'watts': [400, 300],
        'heart_rate': [150, 155],
        'cadence': [90, 92],
        'speed': [10, 11],
        'lat': [34.0, 34.001],
        'lon': [-118.0, -118.001],
        'elevation': [100, 101],
        'distance': [0, 10],
        'cues': ["Start", ""]
    })
    durations, powers = extract_power_profile(sample_activity_df)
    
    # In our sample, max 1s power is 400
    # There are 1s, 5s, etc. in default durations
    
    assert 1 in durations
    # Find power for duration 1
    idx = np.where(durations == 1)[0][0]
    assert powers[idx] == 400

def test_calculate_w_prime_balance():
    cp = 200
    w_prime = 20000
    
    # 1. Steady low power -> Should stay full
    power_stream = [100] * 10
    bal = calculate_w_prime_balance(power_stream, cp, w_prime)
    assert np.allclose(bal, 20000)
    
    # 2. Burst -> Should drain
    # 300W for 10s -> (300-200)*10 = 1000J drain
    power_stream = [300] * 10
    bal = calculate_w_prime_balance(power_stream, cp, w_prime)
    expected_end = 20000 - 1000
    assert np.isclose(bal[-1], expected_end)
