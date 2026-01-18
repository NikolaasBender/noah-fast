import pandas as pd
import datetime
import xml.etree.ElementTree as ET

def export_tcx(plan_df, output_path="race_plan.tcx"):
    """
    Generates a TCX file from the race plan.
    """
    NS_TCX = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
    NS_AE = "http://www.garmin.com/xmlschemas/ActivityExtension/v2"
    
    ET.register_namespace("", NS_TCX)
    ET.register_namespace("ax", NS_AE)
    
    root = ET.Element(f"{{{NS_TCX}}}TrainingCenterDatabase")
    courses = ET.SubElement(root, f"{{{NS_TCX}}}Courses")
    course = ET.SubElement(courses, f"{{{NS_TCX}}}Course")
    
    # Name
    ET.SubElement(course, f"{{{NS_TCX}}}Name").text = "AI Race Plan"
    
    # LAP
    lap = ET.SubElement(course, f"{{{NS_TCX}}}Lap")
    ET.SubElement(lap, f"{{{NS_TCX}}}TotalTimeSeconds").text = f"{plan_df['time_seconds'].max():.1f}"
    ET.SubElement(lap, f"{{{NS_TCX}}}DistanceMeters").text = f"{plan_df['distance'].max():.1f}"
    
    begin_lat = plan_df.iloc[0]['lat']
    begin_lon = plan_df.iloc[0]['lon']
    end_lat = plan_df.iloc[-1]['lat']
    end_lon = plan_df.iloc[-1]['lon']
    
    bp = ET.SubElement(lap, f"{{{NS_TCX}}}BeginPosition")
    ET.SubElement(bp, f"{{{NS_TCX}}}LatitudeDegrees").text = f"{begin_lat:.6f}"
    ET.SubElement(bp, f"{{{NS_TCX}}}LongitudeDegrees").text = f"{begin_lon:.6f}"
    
    ep = ET.SubElement(lap, f"{{{NS_TCX}}}EndPosition")
    ET.SubElement(ep, f"{{{NS_TCX}}}LatitudeDegrees").text = f"{end_lat:.6f}"
    ET.SubElement(ep, f"{{{NS_TCX}}}LongitudeDegrees").text = f"{end_lon:.6f}"
    
    ET.SubElement(lap, f"{{{NS_TCX}}}Intensity").text = "Active"
    
    # TRACK
    track = ET.SubElement(course, f"{{{NS_TCX}}}Track")
    
    # Start time
    start_time = datetime.datetime.now() + datetime.timedelta(days=1)
    start_time = start_time.replace(hour=8, minute=0, second=0, microsecond=0)
    
    # Nutrition Tracking
    last_fed_time = 0
    course_points = [] # List of tuples (time, lat, lon, name)

    for i, row in plan_df.iterrows():
        tp = ET.SubElement(track, f"{{{NS_TCX}}}Trackpoint")
        
        # Time
        ts = start_time + datetime.timedelta(seconds=row['time_seconds'])
        ET.SubElement(tp, f"{{{NS_TCX}}}Time").text = ts.isoformat() + "Z"
        
        # Position
        pos = ET.SubElement(tp, f"{{{NS_TCX}}}Position")
        ET.SubElement(pos, f"{{{NS_TCX}}}LatitudeDegrees").text = f"{row['lat']:.6f}"
        ET.SubElement(pos, f"{{{NS_TCX}}}LongitudeDegrees").text = f"{row['lon']:.6f}"
        
        ET.SubElement(tp, f"{{{NS_TCX}}}AltitudeMeters").text = f"{row['elevation']:.1f}"
        ET.SubElement(tp, f"{{{NS_TCX}}}DistanceMeters").text = f"{row['distance']:.1f}"
        
        # Power Extension
        exts = ET.SubElement(tp, f"{{{NS_TCX}}}Extensions")
        tpx = ET.SubElement(exts, f"{{{NS_AE}}}TPX")
        ET.SubElement(tpx, f"{{{NS_AE}}}Watts").text = f"{int(row['watts'])}"
        
        if 'cues' in row and row['cues']:
            # Parse the detailed cue to make a short label
            # Format in df: "Seg X: Type (Ym) @ ZW"
            # We want: "Type Ym ZW" e.g. "Climb 12m 300W"
            try:
                parts = str(row['cues']).split(':')
                # parts[0] = "Seg X"
                # parts[1] = " Type (Ym) @ ZW"
                details = parts[1].strip()
                # Remove parens and @
                clean = details.replace('(', '').replace(')', '').replace('@', '').replace('  ', ' ')
                # "Type Ym ZW"
                # Shorten types
                clean = clean.replace('Climb', 'Clmb').replace('Descent', 'Dsc').replace('Flat', 'Flt')
                short_label = clean[:15]
            except:
                short_label = "Seg"

            course_points.append({
                 'Time': ts.isoformat() + "Z",
                 'Lat': row['lat'],
                 'Lon': row['lon'],
                 'Name': short_label, 
                 'Type': 'Generic'
            })

        # Check Nutrition Cue
        # Logic: If > 20 mins since last fed AND needed carbs > 30g/hr
        # Add a CoursePoint
        t_sec = row['time_seconds']
        if (t_sec - last_fed_time) > 1200: # 20 mins
             if row['carbs_hr'] > 50:
                 course_points.append({
                     'Time': ts.isoformat() + "Z",
                     'Lat': row['lat'],
                     'Lon': row['lon'],
                     'Name': f"EAT {int(row['carbs_hr']/3)}g", # rough estimate
                     'Type': 'Food'
                 })
                 last_fed_time = t_sec
                 
    # Add CoursePoints to XML
    for cp in course_points:
        pt = ET.SubElement(course, f"{{{NS_TCX}}}CoursePoint")
        ET.SubElement(pt, f"{{{NS_TCX}}}Name").text = cp['Name']
        ET.SubElement(pt, f"{{{NS_TCX}}}Time").text = cp['Time']
        pos = ET.SubElement(pt, f"{{{NS_TCX}}}Position")
        ET.SubElement(pos, f"{{{NS_TCX}}}LatitudeDegrees").text = f"{cp['Lat']:.6f}"
        ET.SubElement(pos, f"{{{NS_TCX}}}LongitudeDegrees").text = f"{cp['Lon']:.6f}"
        ET.SubElement(pt, f"{{{NS_TCX}}}PointType").text = cp['Type']

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)
    print(f"Exported TCX to {output_path}")
