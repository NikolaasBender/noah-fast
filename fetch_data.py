import pandas as pd
import time

def fetch_recent_activities(client, limit=20, yield_progress=False):
    """
    Fetches the most recent 'limit' cycling activities.
    
    Args:
        client: stravalib.client.Client object (authenticated)
        limit: max number of activities to fetch
        yield_progress: If True, yields dicts of {'type': 'progress', 'msg': '...'} 
                        and finally {'type': 'result', 'data': df}.
                        If False, returns df directly.
    
    Returns:
        pd.DataFrame (if yield_progress=False)
        Generator (if yield_progress=True)
    """
    try:
        activities = client.get_activities(limit=limit)
        
        # We need to exhaust the generator to list to know count, 
        # but typically we just iterate.
        # Let's iterate and fetch streams one by one.
        
        data_frames = []
        
        count = 0
        for summary_activity in activities:
            if summary_activity.type != 'Ride':
                continue
                
            activity_id = summary_activity.id
            name = summary_activity.name
            start_date = summary_activity.start_date
            gear_id = summary_activity.gear_id
            
            msg = f"Downloading {activity_id} ({name})..."
            if yield_progress:
                yield {'type': 'progress', 'msg': msg}
            else:
                print(msg)
            
            # Fetch Streams
            try:
                streams = client.get_activity_streams(
                    activity_id,
                    types=['time', 'watts', 'heartrate', 'velocity_smooth', 'cadence', 'grade_smooth', 'moving', 'latlng', 'altitude'],
                    series_type='time'  # Important!
                )
                
                if not streams:
                    continue
                    
                # Convert to DataFrame
                df_dict = {}
                for key, stream in streams.items():
                    df_dict[key] = stream.data
                
                activity_df = pd.DataFrame(df_dict)
                
                # Add Metadata columns (repeated for all rows, handled efficiently by pandas)
                activity_df['activity_id'] = activity_id
                activity_df['start_date'] = start_date
                activity_df['gear_id'] = gear_id
                
                data_frames.append(activity_df)
                count += 1
                
                # Respect limit (get_activities limit is for summary fetch, but we might skip non-rides)
                if count >= limit:
                    break
                    
                # Polite delay
                time.sleep(0.5)
                
            except Exception as e:
                err_msg = f"Error fetching streams for {activity_id}: {e}"
                if yield_progress:
                    yield {'type': 'error', 'msg': err_msg}
                else:
                    print(err_msg)
                continue

        if not data_frames:
            empty_df = pd.DataFrame()
            if yield_progress:
                yield {'type': 'result', 'data': empty_df}
            else:
                return empty_df
                
        # Concatenate all
        final_df = pd.concat(data_frames, ignore_index=True)
        
        if yield_progress:
            yield {'type': 'result', 'data': final_df}
        else:
            return final_df
            
    except Exception as e:
        if yield_progress:
             yield {'type': 'error', 'msg': str(e)}
             # Return empty df as result to avoid breaking consumer
             yield {'type': 'result', 'data': pd.DataFrame()}
        else:
             raise e

if __name__ == "__main__":
    # Test stub (requires manual client injection if run directly)
    print("This module is now a library function.")
