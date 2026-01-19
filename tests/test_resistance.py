import pytest
import pandas as pd
import numpy as np
from modeling.resistance import learn_bike_physics

def test_learn_bike_physics_empty():
    assert learn_bike_physics(pd.DataFrame()) == {}

def test_learn_bike_physics_success():
    # Create synthetic data following P = m*g*grad*v + Crr*m*g*v + 0.5*rho*v^3*CdA
    # Mass 85, Crr 0.005, CdA 0.32
    mass = 85.0
    g = 9.81
    rho = 1.225
    
    v = np.linspace(5, 15, 1000) # 5 to 15 m/s
    grad = np.zeros(1000) # Flat
    
    # Expected power
    p_roll = mass * g * v * 0.005
    p_aero = 0.5 * rho * (v**3) * 0.32
    p_grav = mass * g * grad * v
    
    watts = p_roll + p_aero + p_grav
    
    df = pd.DataFrame({
        'velocity_smooth': v,
        'grade_smooth': grad * 100, # percent
        'watts': watts,
        'moving': [True]*1000,
        'gear_id': ['b1']*1000
    })
    
    profiles = learn_bike_physics(df)
    
    assert 'b1' in profiles
    # Check within reasonable bounds (floating point fit)
    assert 0.004 < profiles['b1']['crr'] < 0.006
    assert 0.30 < profiles['b1']['cda'] < 0.34
