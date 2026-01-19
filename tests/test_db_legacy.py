import pytest
from unittest.mock import patch, MagicMock, mock_open
import db

def test_init_db():
    with patch('sqlite3.connect') as mock_conn:
        db.init_db()
        mock_conn.return_value.cursor.return_value.execute.assert_called()
        mock_conn.return_value.commit.assert_called()

def test_save_user_new():
    with patch('sqlite3.connect') as mock_conn:
        mock_cursor = mock_conn.return_value.cursor.return_value
        # First execute (select) returns None (not exists)
        mock_cursor.fetchone.return_value = None
        
        user_data = {
            'strava_id': 1, 'firstname': 'A', 'lastname': 'B',
            'access_token': 'x', 'refresh_token': 'y', 'expires_at': 100
        }
        db.save_user(user_data)
        
        # Check insert called
        assert mock_cursor.execute.call_count == 2
        ins_call = mock_cursor.execute.call_args_list[1]
        assert "INSERT INTO users" in ins_call[0][0]

def test_get_user_found():
    with patch('sqlite3.connect') as mock_conn:
        # mock execute return
        mock_row = MagicMock()
        mock_row.__iter__.return_value = [('strava_id', 1)]
        # This is tricky with sqlite3.Row simulation, but standard fetchone return tuple/row
        # The code does: dict(user)
        # So user must be iterable as k,v or compatible
        
        # simplified: mock execute().fetchone() -> result
        # result cast to dict.
        
        mock_conn.return_value.execute.return_value.fetchone.return_value = {'strava_id': 1, 'firstname': 'Test'}
        
        u = db.get_user(1)
        assert u['firstname'] == 'Test'

def test_update_sync_time():
    with patch('sqlite3.connect') as mock_conn:
        db.update_sync_time(1)
        mock_conn.return_value.execute.assert_called()
        mock_conn.return_value.commit.assert_called()
