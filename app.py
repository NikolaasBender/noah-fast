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

# Load Model once at startup
print("Loading Rider Model...")
try:
    # Check if we are in Docker or local? Path should be relative to where we run.
    # Docker WORKDIR is /app (which maps to /home/nick/projects/race_simulator)
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
    return render_template('index.html', cp=int(cp), w_prime=int(w_prime))

@app.route('/generate', methods=['POST'])
def generate():
    route_url = request.form.get('route_url')
    if not route_url:
        return "Error: No URL provided", 400
        
    try:
        # 1. Fetch
        course_df = fetch_route(route_url)
        
        # 2. Optimize
        plan_df = optimize_pacing(course_df, cp, w_prime)
        
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
