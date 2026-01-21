import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gpxpy
import gpxpy.gpx

class CoastingPredictor:
    def __init__(self, mass_kg=75, cda=0.3, crr=0.005, rho=1.225):
        """
        :param mass_kg: Rider + Bike mass
        :param cda: Aerodynamic drag coefficient times area
        :param crr: Coefficient of rolling resistance
        :param rho: Air density (kg/m^3)
        """
        self.mass = mass_kg
        self.cda = cda
        self.crr = crr
        self.rho = rho
        self.g = 9.81

    def calculate_forces(self, grade_percent, speed_ms):
        """
        Calculates forces acting on the rider.
        positive grade = climbing
        positive speed = moving forward
        """
        grade_rad = np.arctan(grade_percent / 100.0)
        
        # Gravity Force (Positive helps descent)
        f_gravity = -self.mass * self.g * np.sin(grade_rad)
        
        # Air Drag: 0.5 * rho * CdA * v^2
        f_drag = 0.5 * self.rho * self.cda * (speed_ms ** 2)
        
        # Rolling Resistance: Crr * mg * cos(theta)
        f_rolling = self.crr * self.mass * self.g * np.cos(grade_rad)
        
        return f_gravity, f_drag, f_rolling

    def predict_terminal_velocity(self, grade_percent):
        """
        Finds speed where Gravity = Drag + Rolling (approx).
        Returns speed in m/s.
        If climbing or too shallow to overcome friction, returns 0.
        """
        grade_rad = np.arctan(grade_percent / 100.0)
        
        f_grav_forward = -self.mass * self.g * np.sin(grade_rad) 
        f_resist_base = self.crr * self.mass * self.g * np.cos(grade_rad)
        
        net_drive = f_grav_forward - f_resist_base
        
        if net_drive <= 0:
            return 0.0 # Cannot coast (decelerates)
            
        v_squared = net_drive / (0.5 * self.rho * self.cda)
        return np.sqrt(v_squared)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000 # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        # Calculate bearing between two points
        y = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
        x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
            np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))
        return np.degrees(np.arctan2(y, x))

    def calculate_curvature(self, lat, lon):
        """
        Estimates curvature (1/Radius) from 3 consecutive points.
        Simplified placeholder.
        """
        return np.zeros(len(lat))

    def parse_gpx(self, file_path):
        with open(file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    # Extract Cadence from extensions
                    cad = 0
                    for ext in point.extensions:
                        # Depends on namespace, often simple tag search works
                        if 'cad' in ext.tag:
                            try:
                                cad = int(ext.text)
                            except:
                                pass
                        # Recursive search for children if needed (Garmin often nests in TrackPointExtension)
                        for child in ext:
                            if 'cad' in child.tag:
                                try:
                                    cad = int(child.text)
                                except:
                                    pass
                                    
                    points.append({
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'ele': point.elevation,
                        'time': point.time,
                        'cadence': cad
                    })
        
        if not points:
            return pd.DataFrame()
            
        df = pd.DataFrame(points)
        
        # 1. Distance
        result_points = []
        # We need to reconstruct the loop logic here since we are replacing the whole method block
        
        lats = df['lat'].values
        lons = df['lon'].values
        eles = df['ele'].values
        
        dists = [0.0]
        grades = [0.0]
        bearings = [0.0]
        
        # Calculate time deltas for speed
        times = df['time'].values
        speeds = [0.0] # m/s
        
        for i in range(1, len(lats)):
            d = self.haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
            ele_diff = eles[i] - eles[i-1]
            
            # Time delta in seconds
            # numpy datetime64 handling
            dt = (times[i] - times[i-1]) / np.timedelta64(1, 's')
            
            if d > 0:
                grade = (ele_diff / d) * 100
                b = self.calculate_bearing(lats[i-1], lons[i-1], lats[i], lons[i])
                v = d / dt if dt > 0 else 0
            else:
                grade = 0
                b = bearings[-1]
                v = 0
                
            dists.append(d)
            grades.append(grade)
            bearings.append(b)
            speeds.append(v)
            
        df['step_dist'] = dists
        df['distance'] = df['step_dist'].cumsum()
        df['raw_grade'] = grades
        df['bearing'] = bearings
        df['speed_ms'] = speeds
        
        # SMOOTHING
        df['grade'] = df['raw_grade'].rolling(window=10, center=True).mean().fillna(0)
        
        # Curvature
        df['delta_bearing'] = df['bearing'].diff().fillna(0)
        df['delta_bearing'] = (df['delta_bearing'] + 180) % 360 - 180
        df['smooth_delta_bearing'] = df['delta_bearing'].rolling(window=5, center=True).mean().fillna(0)
        df['curvature'] = np.radians(df['smooth_delta_bearing']) / df['step_dist'].replace(0, 1.0)
        
        return df

    def solve_speed_for_power(self, grade_percent, target_power_watts):
        """
        Solves for speed (m/s) given a power input and grade.
        Simple iterative solver for P = F*v.
        """
        # P = (F_drag + F_roll + F_grav) * v
        # P = (0.5*rho*CdA*v^2 + Crr*mg*cos + mg*sin) * v
        # P = A*v^3 + B*v (where B includes rolling and gravity)
        
        grade_rad = np.arctan(grade_percent / 100.0)
        A = 0.5 * self.rho * self.cda
        F_resist_constant = self.crr * self.mass * self.g * np.cos(grade_rad) + self.mass * self.g * np.sin(grade_rad)
        
        # Iterative guess (Newton-Raphson or just simple search)
        # Function: f(v) = A*v^3 + F*v - P = 0
        # Derivative: f'(v) = 3*A*v^2 + F
        
        v = 10.0 # Start guess (10 m/s)
        for _ in range(10):
            f_v = A * v**3 + F_resist_constant * v - target_power_watts
            df_v = 3 * A * v**2 + F_resist_constant
            if abs(df_v) < 0.001: break
            v_new = v - f_v / df_v
            if abs(v_new - v) < 0.01:
                v = v_new
                break
            v = v_new
            
        return max(0, v)

    def analyze_segment(self, segment_df):
        results = []
        mu_friction = 0.8
        
        PEDALING_POWER_WATTS = 200.0
        
        for idx, row in segment_df.iterrows():
            grade = row['grade']
            curvature = row.get('curvature', 0)
            cadence = row.get('cadence', 0)
            actual_speed = row.get('speed_ms', 0)
            step_dist = row.get('step_dist', 0)
            
            # 1. Physics Predictions
            v_terminal_ms = self.predict_terminal_velocity(grade)
            
            if abs(curvature) > 0.002:
                radius = 1.0 / abs(curvature)
                v_corner_limit_ms = np.sqrt(mu_friction * self.g * radius)
            else:
                v_corner_limit_ms = 999.0
            
            is_fast_descent = v_terminal_ms > 14.0 
            forcing_brake = (v_terminal_ms > 0) and (v_terminal_ms > v_corner_limit_ms) # If unchecked speed > corner speed
            # Note: Comparing terminal velocity to corner limit is safer than current speed
            
            required_coasting = is_fast_descent or forcing_brake
            
            # 2. Time Loss Analysis
            # "Unnecessary Coasting": Cadence=0 AND required_coasting=False
            time_lost_sec = 0
            potential_speed = actual_speed
            
            is_unnecessary = False
            if cadence < 5 and not required_coasting and step_dist > 0:
                is_unnecessary = True
                # How fast COULD we go?
                potential_speed = self.solve_speed_for_power(grade, PEDALING_POWER_WATTS)
                
                # Cap potential speed at corner limit!
                potential_speed = min(potential_speed, v_corner_limit_ms)
                
                if potential_speed > actual_speed and actual_speed > 0.5:
                    time_taken = step_dist / actual_speed
                    time_optimal = step_dist / potential_speed
                    time_lost_sec = max(0, time_taken - time_optimal)

            results.append({
                'distance': row['distance'],
                'grade': grade, # Include grade for context
                'speed': actual_speed,
                'predicted_coasting': required_coasting,
                'reason': 'Gravity' if is_fast_descent else ('Corner' if forcing_brake else 'Pedal'),
                'is_unnecessary_coasting': is_unnecessary,
                'time_lost_sec': time_lost_sec
            })
            
        return pd.DataFrame(results)

    def predict_ride(self, gpx_path):
        """
        One-shot method to process a GPX file and return analysis.
        Returns empty DataFrame on failure.
        """
        import os
        if not os.path.exists(gpx_path):
            print(f"File not found: {gpx_path}")
            return pd.DataFrame()
            
        df = self.parse_gpx(gpx_path)
        if df.empty:
            return pd.DataFrame()
            
        results = self.analyze_segment(df)
        
        # Merge key columns for the user
        df['predicted_coasting'] = results['predicted_coasting'].values
        df['coasting_reason'] = results['reason'].values
        df['is_unnecessary_coasting'] = results['is_unnecessary_coasting'].values
        df['time_lost_sec'] = results['time_lost_sec'].values
        
        return df

def generate_synthetic_segment_data():
    x = np.linspace(0, 2000, 200) 
    grade = np.zeros_like(x)
    grade[0:50] = -8.0 
    grade[50:100] = -2.0 
    grade[100:150] = 6.0 
    grade[150:200] = -10.0 
    
    curvature = np.zeros_like(x)
    curvature[60:90] = 0.05 
    
    return pd.DataFrame({'distance': x, 'grade': grade, 'curvature': curvature})

def main():
    import os
    import glob
    
    print("--- Physics Engine Demo ---")
    predictor = CoastingPredictor(cda=0.32, mass_kg=80) 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpx_dir = os.path.join(script_dir, '..', 'gpx')
    gpx_files = glob.glob(os.path.join(gpx_dir, '*.gpx'))
    
    if gpx_files:
        print(f"Found {len(gpx_files)} GPX files. Analyzing...")
        for f in gpx_files:
            df = predictor.predict_ride(f)
            if not df.empty:
               coast_dist = df[df['predicted_coasting']]['step_dist'].sum()
               print(f"{os.path.basename(f)}: Predicted Coasting {coast_dist/1000:.2f}km")
    else:
        print("No GPX files found for demo.")

if __name__ == "__main__":
    main()

