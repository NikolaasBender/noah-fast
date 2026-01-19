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
    
    # --- Surface Parsing ---
    # Default to Paved
    df['surface'] = 'Paved'
    
    course_points = data.get('course_points', [])
    # Filter for surface markers if any (RWGPS often uses type='Surface' or notes)
    # Heuristic: simple parsing. RWPGS sometimes puts surface in "n" (note) or "t" (type)
    # We will assume a format where we can find "Gravel", "Dirt", "Paved" in the notes or type.
    
    surface_changes = []
    if course_points:
        for cp in course_points:
            # d is distance in meters
            dist = cp.get('d')
            note = str(cp.get('n', '')).lower()
            kind = str(cp.get('t', '')).lower()
            
            # Check keywords
            s_type = None
            if 'gravel' in note or 'gravel' in kind: s_type = 'Gravel'
            elif 'dirt' in note or 'dirt' in kind: s_type = 'Dirt'
            elif 'paved' in note or 'paved' in kind: s_type = 'Paved'
            elif 'road' in note and 'unpaved' not in note: s_type = 'Paved'
            
            if s_type and dist is not None:
                surface_changes.append((dist, s_type))
                
        # Sort by distance
        surface_changes.sort(key=lambda x: x[0])
        
        # Apply changes
        # Vectorized way is hard with non-uniform grid, simple iter logic:
        # Paved (0) -> Gravel (10km) -> Paved (15km)...
        if surface_changes:
            current_surf = 'Paved'
            # We can use pd.cut or searchsorted, but let's just iter rows for clarity or use a function
            # Optimized: use searchsorted
            dists = [x[0] for x in surface_changes]
            surfs = [x[1] for x in surface_changes]
            
            # For each row, find index of last change
            # np.searchsorted(dists, df['distance'], side='right') - 1
            idx_surf = np.searchsorted(dists, df['distance'], side='right') - 1
            
            # Map index to surface. -1 means before first change (default Paved)
            # Create a lookup array. We need to handle the -1 case.
            # Add a dummy start if not present
            if dists[0] > 0:
                dists.insert(0, 0)
                surfs.insert(0, 'Paved')
                idx_surf = np.searchsorted(dists, df['distance'], side='right') - 1
            
            surf_array = np.array(surfs)
            df['surface'] = surf_array[idx_surf]

    # Add metadata
    df.attrs['name'] = data.get('name', 'Unknown Route')
    df.attrs['id'] = route_id
    
    return df
