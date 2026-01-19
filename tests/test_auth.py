import pytest
import os
import json
from unittest.mock import patch, mock_open, MagicMock
from auth import save_tokens, load_tokens, authenticate

TOKEN_DATA = {
    "access_token": "fake_access",
    "refresh_token": "fake_refresh",
    "expires_at": 1234567890
}

def test_save_tokens():
    with patch("builtins.open", mock_open()) as mock_file:
        save_tokens(TOKEN_DATA)
        mock_file.assert_called_with('strava_tokens.json', 'w')
        handle = mock_file()
        # Verify write was called
        handle.write.assert_called()

def test_load_tokens_exists():
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=json.dumps(TOKEN_DATA))):
            tokens = load_tokens()
            assert tokens['access_token'] == "fake_access"

def test_load_tokens_missing():
    with patch("os.path.exists", return_value=False):
        tokens = load_tokens()
        assert tokens is None

@patch('auth.Client')
def test_authenticate_saved_valid(MockClient):
    # Setup mock client
    client_instance = MockClient.return_value
    
    # Mock loaded tokens
    with patch('auth.load_tokens', return_value=TOKEN_DATA):
        # Mock time to be BEFORE expiry
        with patch('time.time', return_value=TOKEN_DATA['expires_at'] - 100):
            client = authenticate()
            
    assert client.access_token == "fake_access"
    client_instance.refresh_access_token.assert_not_called()

@patch('auth.Client')
def test_authenticate_cli_flow(MockClient):
    # Case: No env vars -> returns None (prints error)
    # Must also mock load_tokens to ensure we don't accidentally find tokens
    with patch('auth.load_tokens', return_value=None):
        # We need to patch the MODULE level variables, not just os.getenv
        # changing os.getenv won't affect already imported CLIENT_ID global
        with patch('auth.CLIENT_ID', None):
             with patch('auth.CLIENT_SECRET', None):
                 assert authenticate() is None
