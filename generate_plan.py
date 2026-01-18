import argparse
import os
import joblib
import pandas as pd
from planning.course import fetch_route
from planning.optimizer import optimize_pacing

# Hardcode paths for V1
MODEL_DIR = 'data/models'

def main():
    parser = argparse.ArgumentParser(description="Generate a cycling race plan.")
    parser.add_argument("--route", required=True, help="RideWithGPS URL or ID")
    parser.add_argument("--output", default="race_plan.csv", help="Output file")
    parser.add_argument("--format", default="csv", choices=["csv", "tcx"], help="Output format")
    
    args = parser.parse_args()
    
    print(f"1. Loading Rider Model from {MODEL_DIR}...")
    try:
        phys = joblib.load(os.path.join(MODEL_DIR, 'physiology.pkl'))
        cp = phys['cp']
        w_prime = phys['w_prime']
        print(f"   Rider CP: {cp:.0f} W, W': {w_prime:.0f} J")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using default values (250W, 20kJ)")
        cp = 250
        w_prime = 20000
    
    print(f"2. Fetching Route: {args.route}...")
    try:
        course_df = fetch_route(args.route)
        print(f"   Route: {course_df.attrs['name']}")
        print(f"   Distance: {course_df['distance'].max()/1000:.1f} km")
        print(f"   Climbing: {course_df['ele_diff'].clip(lower=0).sum():.0f} m")
    except Exception as e:
        print(f"Error fetching route: {e}")
        return

    print("3. Optimizing Pacing Strategy...")
    plan_df = optimize_pacing(course_df, cp, w_prime)
    
    if args.format == 'csv':
        plan_df['cues'] = ""
        print(f"4. Saving Plan to {args.output}...")
        plan_df.to_csv(args.output, index=False)
    elif args.format == 'tcx':
        from export.garmin import export_tcx
        # If output is csv default, change to tcx
        out = args.output
        if out == "race_plan.csv": out = "race_plan.tcx"
        print(f"4. Exporting Plan to {out}...")
        export_tcx(plan_df, out)
        
    print("Done! Go crush it.")

if __name__ == "__main__":
    main()
