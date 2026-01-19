import pytest

def test_index(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"Race Simulator" in rv.data

def test_generate_no_data(client):
    rv = client.post('/generate', data={})
    assert rv.status_code == 400
    assert b"No URL provided" in rv.data

# Note: Testing valid generate requires mocking inside the app context or mocking the planning module.
# Since we tested the planning module separately, an integration test here would mostly test Flask wiring.
# We can mock fetch_route to verify the happy path.

from unittest.mock import patch
import pandas as pd

def test_generate_success(client):
    # Mock data
    mock_df = pd.DataFrame({
        'distance': [0, 100],
        'gradient': [0, 0],
        'elevation': [10, 10],
        'lat': [0, 0],
        'lon': [0, 0]
    })
    mock_df.attrs['name'] = "Mock Route"
    
    with patch('app.fetch_route', return_value=mock_df):
        # Also need to ensure optimizer doesn't crash on this small df
        # The optimizer is imported in app.py. We rely on it working or mock it too.
        # Let's mock optimizer to return a ready dataframe for export
        
        mock_plan = mock_df.copy()
        mock_plan['watts'] = [200, 200]
        mock_plan['time_seconds'] = [0, 10]
        mock_plan['speed_mps'] = [10, 10]
        mock_plan['cues'] = ["", ""]
        mock_plan['kcal_hr'] = [500, 500]
        mock_plan['carbs_hr'] = [60, 60]
        
        with patch('app.optimize_pacing', return_value=mock_plan):
            rv = client.post('/generate', data={'route_url': 'http://example.com'})
            
            # Should be 200 and return a file
            assert rv.status_code == 200
            assert rv.headers['Content-Disposition'].startswith("attachment")
