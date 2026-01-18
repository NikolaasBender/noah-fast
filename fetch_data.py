import os
import pandas as pd
from auth import authenticate
import time

DATA_DIR = 'data/raw'

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def fetch_activities(limit=10):
    client = authenticate()
    if not client:
        return

    print(f"Fetching last {limit} videos... just kidding, activities.")
    
    # Get activities
    activities = client.get_activities(limit=limit)
    
    ensure_data_dir()

    for activity in activities:
        activity_id = activity.id
        activity_name = activity.name
        activity_type = activity.type
        
        # Only interested in rides/virtual rides
        if activity_type not in ['Ride', 'VirtualRide']:
            continue
            
        file_path = os.path.join(DATA_DIR, f"{activity_id}.parquet")
        
        if os.path.exists(file_path):
            print(f"Skipping {activity_id} ({activity_name}) - already exists")
            continue

        print(f"Downloading {activity_id} ({activity_name})...")
        
        try:
            # Request streams
            streams = client.get_activity_streams(
                activity_id,
                types=['time', 'distance', 'watts', 'heartrate', 'cadence', 'altitude', 'grade_smooth', 'velocity_smooth', 'moving'],
                resolution='high'
            )
            
            data = {}
            if 'time' in streams:
                data['time'] = streams['time'].data
            else:
                print(f"No time stream for {activity_id}, skipping.")
                continue
                
            for key in ['distance', 'watts', 'heartrate', 'cadence', 'altitude', 'grade_smooth', 'velocity_smooth', 'moving']:
                if key in streams:
                    data[key] = streams[key].data
                else:
                    # Fill missing streams with None/NaN if necessary, or just omit
                    # For a simple dataframe construction, lengths must match.
                    # Strava streams usually match length of 'time' if returned.
                     pass

            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add metadata
            df.attrs['activity_id'] = activity_id
            df.attrs['name'] = activity_name
            df.attrs['start_date'] = str(activity.start_date)
            # Capture Gear ID (Bike)
            df.attrs['gear_id'] = activity.gear_id if activity.gear_id else "Unknown"
            
            # Save to parquet
            df.to_parquet(file_path, index=False)
            print(f"Saved {file_path}")
            
            # Rate limiting
            time.sleep(1) 

        except Exception as e:
            print(f"Failed to download {activity_id}: {e}")

if __name__ == "__main__":
    fetch_activities(limit=50) # Start with 50 recent rides
