import argparse
import os
import pandas as pd
import numpy as np
import time
from stravalib.client import Client
from app import app, db, User
from planning.optimizer import optimize_pacing

def get_latest_user():
    with app.app_context():
        # Get the most recently updated user or just first
        return User.query.first()

def fetch_activity_streams(client, activity_id):
    print(f"Fetching streams for Activity {activity_id}...")
    types = ['time', 'latlng', 'distance', 'altitude', 'velocity_smooth', 'watts', 'grade_smooth', 'moving']
    streams = client.get_activity_streams(activity_id, types=types)
    
    data = {}
    for k, v in streams.items():
        data[k] = v.data
    
    df = pd.DataFrame(data)
    
    # Normalize columns
    # latlng is list of [lat, lon]
    df['lat'] = df['latlng'].apply(lambda x: x[0])
    df['lon'] = df['latlng'].apply(lambda x: x[1])
    # altitude -> elevation
    df['elevation'] = df['altitude']
    
    return df

def run_backtest(activity_id):
    user = get_latest_user()
    if not user:
        print("No user found in DB. Please login via App first.")
        return

    client = Client(access_token=user.access_token)
    
    # Token Refresh Logic (Simplified for script)
    if time.time() > user.expires_at:
        print("Refreshing token...")
        refresh_resp = client.refresh_access_token(
            client_id=os.getenv('STRAVA_CLIENT_ID'),
            client_secret=os.getenv('STRAVA_CLIENT_SECRET'),
            refresh_token=user.refresh_token
        )
        client.access_token = refresh_resp['access_token']
    
    # 1. Get Actual Data
    try:
        df_actual = fetch_activity_streams(client, activity_id)
    except Exception as e:
        print(f"Error fetching activity: {e}")
        return

    # Filter invalid
    # df_actual = df_actual.dropna()
    
    # 2. Extract Course from Activity
    # We need to format it exactly like fetch_route output
    course_df = pd.DataFrame({
        'distance': df_actual['distance'],
        'elevation': df_actual['elevation'],
        'gradient': df_actual['grade_smooth'],
        'lat': df_actual['lat'],
        'lon': df_actual['lon']
    })
    
    # Mock surface as Paved for now (Strava streams don't have surface easily)
    course_df['surface'] = 'Paved' 
    course_df.attrs['name'] = f"Backtest {activity_id}"
    
    # 3. Determine Inputs
    avg_power = df_actual['watts'].mean()
    # Simple CP assumption: Avg Power (assuming max effort)
    # Or use user's CP? User wants to know "Does model predict similar times"
    # So we should use the actual power put out to simulate speed.
    # If we use 300W and go 20mph, model should also go 20mph at 300W.
    
    # We will FORCE the pacing to be the actual power profile? 
    # No, optimize_pacing generates its own power.
    # To test PHYSICS, we should check: Given Power X, do we get Speed Y?
    
    print(f"--- Backtest Configuration ---")
    print(f"Activity: {activity_id}")
    print(f"Distance: {course_df['distance'].max()/1000:.2f} km")
    print(f"Actual Avg Power: {avg_power:.0f} W")
    
    # Simulation
    # We use a trick: We want to test pure physics. 
    # But optimize_pacing changes power based on slope.
    # To strictly test "39h vs 5h", running the optimizer with AvgPower as CP is a good start.
    
    rider_mass = 85.0 # Default
    
    print("Running Simulation...")
    plan_df = optimize_pacing(course_df, cp=avg_power, w_prime=20000, rider_mass=rider_mass)
    
    # DEBUG: Inspect the plan
    print("\n--- Plan Sample (First 20 rows) ---")
    print(plan_df[['distance', 'gradient', 'watts', 'speed_mps', 'surface']].head(20).to_string())
    print("\n--- Plan Stats ---")
    print(plan_df[['watts', 'speed_mps', 'gradient']].describe().to_string())
    
    # 4. Compare
    actual_time_s = df_actual['time'].max() - df_actual['time'].min() # Elapsed? Or Moving?
    # Use moving time typically
    # But Strava stream time is elapsed from start.
    
    # Calculate simulated stats
    sim_time_s = plan_df['duration_s'].sum()
    sim_dist_m = plan_df['distance'].max() # Approx
    
    print("\n--- Results ---")
    print(f"Actual Time:    {actual_time_s/3600:.2f} hours")
    print(f"Simulated Time: {sim_time_s/3600:.2f} hours")
    
    diff_s = sim_time_s - actual_time_s
    diff_pct = (diff_s / actual_time_s) * 100
    
    print(f"Difference:     {diff_s/60:.0f} min ({diff_pct:+.1f}%)")
    
    if abs(diff_pct) > 10:
        print("\n[FAIL] Significant Discrepancy detected!")
        # Diagnosis
        act_speed = df_actual['distance'].max() / actual_time_s
        sim_speed = sim_dist_m / sim_time_s
        print(f"Actual Avg Speed:    {act_speed:.2f} m/s ({(act_speed*2.237):.1f} mph)")
        print(f"Simulated Avg Speed: {sim_speed:.2f} m/s ({(sim_speed*2.237):.1f} mph)")
    else:
        print("\n[PASS] Model is within 10% of reality.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('activity_id', type=str, help='Strava Activity ID')
    args = parser.parse_args()
    
    run_backtest(args.activity_id)
