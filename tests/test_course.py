import pytest
import requests_mock
import pandas as pd
from planning.course import fetch_route

def test_fetch_route_success():
    route_id = "12345"
    url = f"https://ridewithgps.com/routes/{route_id}.json"
    
    mock_response = {
        "name": "Test Route",
        "track_points": [
            {'e': 10, 'x': -118, 'y': 34, 'd': 0},
            {'e': 12, 'x': -118, 'y': 34, 'd': 100},
            {'e': 15, 'x': -118, 'y': 34, 'd': 200}
        ]
    }
    
    with requests_mock.Mocker() as m:
        m.get(url, json=mock_response)
        
        df = fetch_route(route_id)
        
        assert len(df) == 3
        assert df.attrs['name'] == "Test Route"
        assert 'gradient' in df.columns

def test_fetch_route_failure():
    route_id = "bad_route"
    url = f"https://ridewithgps.com/routes/{route_id}.json"
    
    with requests_mock.Mocker() as m:
        m.get(url, status_code=404)
        
        with pytest.raises(Exception) as excinfo:
            fetch_route(route_id)
        
        assert "Failed to fetch route" in str(excinfo.value)
