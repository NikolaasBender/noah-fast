import pytest
from unittest.mock import patch, MagicMock
from fetch_data import fetch_recent_activities

def test_fetch_recent_activities():
    # Mock client passed to function
    with patch('stravalib.client.Client') as MockClient:
        client_instance = MockClient.return_value
        # Mock activities iterator
        mock_act = MagicMock()
        mock_act.id = 1
        mock_act.start_date = "2023-01-01"
        mock_act.type = "Ride"
        client_instance.get_activities.return_value = [mock_act]
        
        with patch('pandas.DataFrame') as MockDF:
            with patch('builtins.print'):
                 # Mock streams
                 client_instance.get_activity_streams.return_value = {'watts': MagicMock(data=[100, 100]), 'time': MagicMock(data=[1, 2])}
                 
                 # Need to allow real DF creation or mock effectively to avoid concat errors
                 # Better to use real pandas/logic for simple data manipulation
                 pass
                 
def test_fetch_recent_activities_real_pandas():
    # Pass a MagicMock client but let pandas doing its thing
    mock_client = MagicMock()
    mock_act = MagicMock()
    mock_act.id = 1
    mock_act.name = "Ride 1"
    mock_act.type = "Ride"
    mock_act.gear_id = "g1"
    mock_act.start_date = "2023-01-01"
    
    mock_client.get_activities.return_value = [mock_act]
    
    # Streams
    mock_streams = {}
    for k in ['time', 'distance', 'watts', 'heartrate', 'cadence', 'altitude', 'grade_smooth', 'velocity_smooth', 'moving']:
        mock_streams[k] = MagicMock(data=[10, 20])
    
    mock_client.get_activity_streams.return_value = mock_streams
    
    with patch('time.sleep'): # optimize speed
        df = fetch_recent_activities(mock_client, limit=1)
    
    assert len(df) == 2
    assert 'watts' in df.columns
