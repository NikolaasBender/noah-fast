import pandas as pd
import time

def fetch_recent_activities(client, limit=50):
    """
    Fetches recent activities and their streams into a single DataFrame in memory.
    """
    if not client:
        return pd.DataFrame()

    print(f"Fetching last {limit} videos... just kidding, activities.")
    
    # Get activities
    activities = client.get_activities(limit=limit)
    
    frames = []

    for activity in activities:
        activity_id = activity.id
        activity_name = activity.name
        activity_type = activity.type
        
        # Only interested in rides/virtual rides
        if activity_type not in ['Ride', 'VirtualRide']:
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

            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add metadata as columns (safer for concatenation)
            df['activity_id'] = activity_id
            df['gear_id'] = activity.gear_id if activity.gear_id else "Unknown"
            df['start_date'] = activity.start_date
            
            frames.append(df)
            
            # Rate limiting
            time.sleep(0.5) 

        except Exception as e:
            print(f"Failed to download {activity_id}: {e}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Test stub (requires manual client injection if run directly)
    print("This module is now a library function.")

