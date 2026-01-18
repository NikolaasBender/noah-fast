from flask import Flask, render_template, request, send_file
import io
import joblib
import os
import traceback
from planning.course import fetch_route
from planning.optimizer import optimize_pacing
from export.garmin import export_tcx

app = Flask(__name__)
MODEL_DIR = 'data/models'

# Load Bike Profiles
BIKE_PROFILES = {}
try:
    with open(os.path.join(MODEL_DIR, 'bike_profiles.json'), 'r') as f:
        BIKE_PROFILES = json.load(f)
except Exception as e:
    print(f"Warning: Could not load bike profiles: {e}")

# Load Model once at startup
print("Loading Rider Model...")
try:
    phys = joblib.load(os.path.join(MODEL_DIR, 'physiology.pkl'))
    cp = phys['cp']
    w_prime = phys['w_prime']
    print(f"Loaded: CP={cp:.0f}W, W'={w_prime:.0f}J")
except Exception as e:
    print(f"Warning: Could not load model ({e}). Using defaults.")
    cp = 250
    w_prime = 20000

@app.route('/')
def index():
    return render_template('index.html', cp=int(cp), w_prime=int(w_prime), bikes=BIKE_PROFILES)

@app.route('/generate', methods=['POST'])
def generate():
    route_url = request.form.get('route_url')
    gear_id = request.form.get('gear_id')
    
    # Rider Settings (with defaults)
    try:
        user_mass = float(request.form.get('rider_mass', 85.0))
        user_cp = float(request.form.get('cp', cp))
        user_w_prime = float(request.form.get('w_prime', w_prime))
    except ValueError:
        return "Error: Invalid number format for rider settings", 400
    
    if not route_url:
        return "Error: No URL provided", 400
        
    try:
        # 1. Fetch
        course_df = fetch_route(route_url)
        
        # 2. Optimize
        plan_df = optimize_pacing(course_df, user_cp, user_w_prime, gear_id=gear_id, rider_mass=user_mass)
        
        # 3. Export to Memory
        mem = io.BytesIO()
        export_tcx(plan_df, mem)
        mem.seek(0)
        
        # 4. Return
        filename = "race_plan.tcx"
        if 'name' in course_df.attrs:
            # sanitize
            safe_name = "".join([c for c in course_df.attrs['name'] if c.isalnum() or c in (' ','-','_')]).strip()
            filename = f"{safe_name}.tcx"
            
        return send_file(
            mem, 
            as_attachment=True, 
            download_name=filename, 
            mimetype='application/xml'
        )
        
    except Exception as e:
        traceback.print_exc()
        return f"Error processing route: {str(e)}", 500

if __name__ == '__main__':
    # Local dev
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
