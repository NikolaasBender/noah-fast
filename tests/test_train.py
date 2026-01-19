import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from modeling import train

@pytest.fixture
def mock_training_data():
    df = pd.DataFrame({
        'watts': np.random.randint(100, 400, 1000),
        'heartrate': np.random.randint(100, 180, 1000),
        'cadence': np.random.randint(80, 100, 1000),
        'time': range(1000)
    })
    return df

def test_train_main(mock_training_data):
    # Mock glob to return 1 file
    with patch('glob.glob', return_value=['dummy.parquet']):
        # Mock pandas read_parquet
        with patch('pandas.read_parquet', return_value=mock_training_data):
            # Mock os.path.exists/makedirs
            with patch('os.path.exists', return_value=False):
                with patch('os.makedirs'):
                    # Mock joblib dump
                    with patch('joblib.dump'):
                        # Mock model save
                        with patch('tensorflow.keras.models.Sequential.save'):
                            # Mock fit to save time
                            with patch('tensorflow.keras.models.Sequential.fit'):
                                train.main()

def test_train_no_data():
    with patch('glob.glob', return_value=[]):
        with patch('os.path.exists'):
             # Should print "No data found" and return
             train.main()

def test_load_data_error():
    with patch('glob.glob', return_value=['bad.parquet']):
        with patch('pandas.read_parquet', side_effect=Exception("Bad file")):
            dfs = train.load_all_data()
            assert len(dfs) == 0
