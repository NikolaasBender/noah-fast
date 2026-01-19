import os
import io
import json
import traceback
from flask import Flask, render_template, request, send_file, redirect, url_for, session, flash
from stravalib.client import Client
from planning.optimizer import optimize_pacing
from planning.course import fetch_route
from export.garmin import export_tcx
import joblib
import pickle
import time
from dotenv import load_dotenv
from models import db, User, MinerModel

# Pipeline Imports
from fetch_data import fetch_recent_activities
from modeling.physiology import extract_power_profile, calculate_cp_and_w_prime
from modeling.resistance import learn_bike_physics

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Security Headers
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# In production, set this to True. For localhost Docker, False is fine.
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'

# Database Config
db_path = os.path.join(os.path.abspath(os.getcwd()), 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{db_path}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Create Tables
with app.app_context():
    db.create_all()

MODEL_DIR = 'data/models'
CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')

# Default Physiology
cp = 250
w_prime = 20000

# Helper: Get Current User
def get_current_user():
    user_id = session.get('user_id')
    if user_id:
        return User.query.get(user_id)
    return None

@app.route('/')
def index():
    user = get_current_user()
    
    # Default values (Guest)
    current_cp = 250
    current_w_prime = 20000
    current_bikes = {}
    
    # Logic to load User models if logged in
    # This comes from the DB in stateless mode
    if user:
        # Try fetch physiology
        phys_model = MinerModel.query.filter_by(user_id=user.id, model_type='physiology').first()
        if phys_model:
            phys = pickle.loads(phys_model.data)
            current_cp = phys['cp']
            current_w_prime = phys['w_prime']

        # Try fetch bikes
        bike_model = MinerModel.query.filter_by(user_id=user.id, model_type='bike_profiles').first()
        if bike_model:
            current_bikes = pickle.loads(bike_model.data)
    
    # Fallback to local file defaults if Guest (or if user has no model yet)
    if not user and not current_bikes:
        try:
             with open(os.path.join(MODEL_DIR, 'bike_profiles.json'), 'r') as f:
                current_bikes = json.load(f)
             # Phys defaults are 250/20000
        except: pass

    return render_template('index.html', 
                          cp=int(current_cp), 
                          w_prime=int(current_w_prime), 
                          bikes=current_bikes,
                          user=user)

@app.route('/login')
def login():
    client = Client()
    redirect_uri = url_for('authorized', _external=True)
    authorize_url = client.authorization_url(
        client_id=CLIENT_ID,
        redirect_uri=redirect_uri,
        scope=['read_all','profile:read_all','activity:read_all']
    )
    return redirect(authorize_url)

@app.route('/authorized')
def authorized():
    code = request.args.get('code')
    if not code:
        return "Error: No code received", 400
        
    client = Client()
    try:
        token_response = client.exchange_code_for_token(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            code=code
        )
        client.access_token = token_response['access_token']
        athlete = client.get_athlete()
        
        # Check if user exists
        user = User.query.filter_by(strava_id=athlete.id).first()
        if not user:
            user = User(strava_id=athlete.id)
            db.session.add(user)
            
        user.firstname = athlete.firstname
        user.lastname = athlete.lastname
        user.access_token = token_response['access_token']
        user.refresh_token = token_response['refresh_token']
        user.expires_at = token_response['expires_at']
        
        db.session.commit()
        
        session['user_id'] = user.id
        
        # Trigger Sync? For V1, maybe redirect to a sync page or just index?
        return redirect(url_for('index'))
        
    except Exception as e:
        traceback.print_exc()
        return f"Authentication failed: {e}", 500

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

def get_valid_client(user):
    client = Client(access_token=user.access_token)
    if time.time() > user.expires_at:
        print(f"Refreshing token for {user.strava_id}")
        refresh_response = client.refresh_access_token(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            refresh_token=user.refresh_token
        )
        user.access_token = refresh_response['access_token']
        user.refresh_token = refresh_response['refresh_token']
        user.expires_at = refresh_response['expires_at']
        db.session.commit()
        client.access_token = user.access_token
    return client

from flask import Flask, render_template, request, send_file, redirect, url_for, session, flash, Response, stream_with_context

# ... (imports remain same)

@app.route('/sync')
def sync_page():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    return render_template('sync.html', user=user)

@app.route('/sync_stream')
def sync_stream():
    user = get_current_user()
    if not user:
        return "Unauthorized", 401
        
    def generate():
        try:
            # 1. Get Client
            client = get_valid_client(user)
            
            yield f"data: Connected to Strava. Fetching recent rides...\n\n"
            
            # 1b. Fetch Bike Names (for UI)
            try:
                athlete = client.get_athlete()
                gear_map = {g.id: g.name for g in athlete.bikes}
                yield f"data: Found {len(gear_map)} bikes in your garage ({', '.join(gear_map.values())}).\n\n"
            except Exception as e:
                gear_map = {}
                yield f"data: Warning: Could not fetch bike names ({str(e)})\n\n"

            # 2. Fetch Data (Streaming)
            fetch_gen = fetch_recent_activities(client, limit=50, yield_progress=True)
            
            df = None
            for item in fetch_gen:
                if item['type'] == 'progress':
                    yield f"data: {item['msg']}\n\n"
                elif item['type'] == 'error':
                    yield f"data: Error: {item['msg']}\n\n"
                elif item['type'] == 'result':
                    df = item['data']
            
            if df is None or df.empty:
                yield f"data: No ride data found to analyze.\n\n"
                yield f"data: DONE\n\n"
                return
                
            yield f"data: Analyze {len(df)} data points...\n\n"
            
            # 3. Train Physiology
            yield f"data: calculating critical power...\n\n"
            durations, powers = extract_power_profile(df)
            cp, w_prime = calculate_cp_and_w_prime(durations, powers)
            
            if cp:
                phys_data = {'cp': cp, 'w_prime': w_prime}
                # Save or Update
                # Note: We must manage DB session carefully in generator context
                with app.app_context():
                     model = MinerModel.query.filter_by(user_id=user.id, model_type='physiology').first()
                     if not model:
                         model = MinerModel(user_id=user.id, model_type='physiology')
                         db.session.add(model)
                     model.data = pickle.dumps(phys_data)
                     model.updated_at = db.func.now()
                     db.session.commit()
                yield f"data: CP: {cp:.0f}W, W': {w_prime:.0f}J saved.\n\n"

            # 4. Train Bike Physics
            yield f"data: Learning bike aerodynamics (Crr/CdA)...\n\n"
            bikes = learn_bike_physics(df, gear_map=gear_map)
            if bikes:
                # Save or Update
                with app.app_context():
                    model = MinerModel.query.filter_by(user_id=user.id, model_type='bike_profiles').first()
                    if not model:
                        model = MinerModel(user_id=user.id, model_type='bike_profiles')
                        db.session.add(model)
                    model.data = pickle.dumps(bikes)
                    model.updated_at = db.func.now()
                    db.session.commit()
                yield f"data: Learned profiles for {len(bikes)} bikes.\n\n"
            
            yield f"data: Sync Complete!\n\n"
            yield f"data: DONE\n\n"
            
        except Exception as e:
            traceback.print_exc()
            yield f"data: Error: {str(e)}\n\n"
            yield f"data: DONE\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# Helper: Core Planning Logic
def get_plan_dataframe(form):
    route_url = form.get('route_url')
    gear_id = form.get('gear_id')
    
    # Rider Settings (with defaults)
    try:
        user_mass = float(form.get('rider_mass', 85.0))
        user_cp = float(form.get('cp', cp))
        user_w_prime = float(form.get('w_prime', w_prime))
        
        # Delusion Logic (0-10 scale, each step adds 1%)
        delusion_lvl = int(form.get('delusion', 0))
        delusion_factor = 1.0 + (delusion_lvl * 0.01)
        effective_cp = user_cp * delusion_factor
        
    except ValueError:
        raise ValueError("Invalid number format for rider settings")
        
    if not route_url:
        raise ValueError("No URL provided")
        
    # 1. Fetch
    course_df = fetch_route(route_url)
    
    # 2. Optimize (Use Effective CP)
    plan_df = optimize_pacing(course_df, effective_cp, user_w_prime, gear_id=gear_id, rider_mass=user_mass)
    
    # Attach Delusion Metadata for UI
    if delusion_lvl > 0:
        plan_df.attrs['delusion_level'] = delusion_lvl
        plan_df.attrs['effective_cp'] = effective_cp
    
    return plan_df, course_df

@app.route('/generate', methods=['POST'])
def generate():
    try:
        plan_df, course_df = get_plan_dataframe(request.form)
        
        # Prepare Data for Preview Template
        # Need to aggregate segments into a list of dicts for Jinja
        # The optimizer returns resampled 100m chunks.
        # We need to reconstruct the "Directives" / Segments from segment_id
        
        preview_segments = []
        grouped = plan_df.groupby('segment_id')
        
        for seg_id, group in grouped:
            if seg_id == -1: continue # Skip initialization rows if any
            
            first = group.iloc[0]
            
            # Extract Cues (stored in the first row of segment usually, or we reconstruct)
            # The 'cues' column already has formatted text like "Seg 1: Flat (4m) @ 200W"
            # But let's verify if cues are populated on first row of chunk
            
            # Reconstruct cleaner object for UI
            avg_power = group['watts'].mean()
            duration_s = group['duration_s'].sum()
            dist_km = group['distance'].min() / 1000.0
            
            # Determine type from cues or infer
            s_type = first['cues'].split(':')[1].split('(')[0].strip() if ':' in first['cues'] else "Segment"
            surface = first['surface'] if 'surface' in first else 'Paved'
            
            avg_grad = group['gradient'].mean()
            
            notes = ""
            if s_type == 'Descent' and avg_grad < -5:
                notes = "High Speed Descent - Tuck!"
            if surface == 'Gravel':
                notes = "Loose Surface - Maintain Momentum"
                
            seg_obj = {
                'id': seg_id + 1,
                'start_dist_km': f"{dist_km:.1f}",
                'type': s_type,
                'type_class': s_type.split(' ')[0], # Climb, Descent, Flat
                'surface': surface,
                'power': int(avg_power),
                'duration_str': f"{int(duration_s // 60)}m {int(duration_s % 60)}s",
                'avg_grad': f"{avg_grad:.1f}",
                'notes': notes
            }
            preview_segments.append(seg_obj)
            
        # Overall Stats
        total_time_s = plan_df['duration_s'].sum()
        avg_p = plan_df['watts'].mean()
        total_carbs = plan_df['carbs_hr'].mean() * (total_time_s/3600.0)
        
        return render_template('plan_preview.html', 
                             segments=preview_segments,
                             route_name=course_df.attrs.get('name', 'Route'),
                             total_time_str=f"{int(total_time_s//3600)}h {int((total_time_s%3600)//60)}m",
                             avg_power=int(avg_p),
                             total_carbs=int(total_carbs),
                             delusion_level=plan_df.attrs.get('delusion_level', 0),
                             effective_cp=int(plan_df.attrs.get('effective_cp', 0))
                             )
        
    except Exception as e:
        traceback.print_exc()
        return f"Error creating preview: {str(e)}", 500

@app.route('/generate_file', methods=['POST'])
def generate_file():
    try:
        plan_df, course_df = get_plan_dataframe(request.form)
        
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
