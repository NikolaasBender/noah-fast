import pytest
import os
import sys
import pandas as pd
import numpy as np
from flask import Flask

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def sample_activity_df():
    """Returns a dummy dataframe mimicking a Strava activity stream."""
    return pd.DataFrame({
        'watts': [100, 200, 300, 400, 0] * 12, # 60 seconds
        'heartrate': [100, 110, 120, 130, 90] * 12,
        'cadence': [80, 85, 90, 95, 0] * 12,
        'time': range(60)
    })

@pytest.fixture
def sample_course_df():
    """Returns a dummy dataframe mimicking a fetched RWGPS route."""
    return pd.DataFrame({
        'distance': np.linspace(0, 1000, 11), # 0 to 1000m
        'elevation': [10] * 11, # Flat
        'lat': [34.0] * 11,
        'lon': [-118.0] * 11,
        'gradient': [0.0] * 11,
        'ele_smooth': [10.0] * 11
    })
