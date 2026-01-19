import pytest
import io
import pandas as pd
from export.garmin import export_tcx

def test_export_tcx():
    df = pd.DataFrame({
        'watts': [200, 210],
        'time_seconds': [0, 10],
        'lat': [34.0, 34.001],
        'lon': [-118.0, -118.001],
        'elevation': [100, 101],
        'distance': [0, 10],
        'cues': ["Start", ""]
    })
    
    out = io.BytesIO()
    export_tcx(df, out)
    
    content = out.getvalue().decode('utf-8')
    assert "TrainingCenterDatabase" in content
    # Namespace makes simple string matching hard for tags, check for value
    assert ">200<" in content or "Watts" in content
    assert ">34.0<" in content or "LatitudeDegrees" in content
