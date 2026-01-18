import requests
import pandas as pd
import numpy as np

def fetch_route(url_or_id):
    """
    Fetches route data from RideWithGPS.
    Supports full URLs or just IDs.
    """
    # Extract ID if URL is passed
    if "ridewithgps.com" in url_or_id:
        # Expected format: https://ridewithgps.com/routes/50057635
        # or .json or .gpx
        route_id = url_or_id.split("/routes/")[1].split(".")[0].split("?")[0]
    else:
        route_id = url_or_id
        
    json_url = f"https://ridewithgps.com/routes/{route_id}.json"
    print(f"Fetching route from {json_url}...")
    
    resp = requests.get(json_url)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch route: {resp.status_code} {resp.text}")
        
    data = resp.json()
    
    # Extract track points
    # structure: data['track_points'] is a list of dicts: {'e': elevation, 'x': lon, 'y': lat, 'd': dist_from_start, 't': timestamp(optional)}
    # Note: 'd' might not be in all responses, sometimes it is 'D'. RWGPS is usually 'd' (meters).
    
    track_points = data.get('track_points', [])
    if not track_points:
        raise Exception("No track points found in route data.")
        
    df = pd.DataFrame(track_points)
    
    # Normalize columns
    # e -> elevation, x -> lon, y -> lat, d -> distance
    rename_map = {'e': 'elevation', 'x': 'lon', 'y': 'lat', 'd': 'distance'}
    df = df.rename(columns=rename_map)
    
    # Ensure distance is monotonic
    if 'distance' not in df.columns:
        # Calculate Haversine distance if missing (unlikely for RWGPS json)
        # For V1 assuming 'd' exists as it's standard RWGPS
        raise Exception("Distance data missing from Route.")
        
    # Calculate Gradient
    # Gradient = d_elevation / d_distance
    # Smoothing is essential as raw points are noisy
    
    window = 5 # 5 points smoothing
    df['ele_smooth'] = df['elevation'].rolling(window=window, center=True).mean()
    
    df['dist_diff'] = df['distance'].diff()
    df['ele_diff'] = df['ele_smooth'].diff()
    
    # avoid div by zero
    df['gradient'] = (df['ele_diff'] / df['dist_diff']) * 100
    df['gradient'] = df['gradient'].fillna(0)
    
    # Clamp extreme gradients (GPS errors)
    df['gradient'] = df['gradient'].clip(-25, 25)
    
    # Add metadata
    df.attrs['name'] = data.get('name', 'Unknown Route')
    df.attrs['id'] = route_id
    
    return df
