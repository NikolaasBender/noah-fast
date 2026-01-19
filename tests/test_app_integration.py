import pytest
from unittest.mock import patch, MagicMock

def test_login_route(client):
    # Should redirect to Strava
    rv = client.get('/login')
    assert rv.status_code == 302
    assert "strava.com/oauth/authorize" in rv.headers['Location']

def test_authorized_route_success(client):
    # Mock Client exchange_code_for_token
    with patch('app.Client') as MockClient:
        instance = MockClient.return_value
        instance.exchange_code_for_token.return_value = {
            'access_token': 'new_token',
            'refresh_token': 'new_refresh',
            'expires_at': 1234567890
        }
        
        mock_athlete = MagicMock()
        mock_athlete.id = 100
        mock_athlete.firstname = "Auth"
        mock_athlete.lastname = "User"
        instance.get_athlete.return_value = mock_athlete
        
        # Mock User search/creation
        # The app looks up user by strava_id likely
        with patch('app.User') as MockUser:
            mock_user_db = MagicMock()
            mock_user_db.id = 500
            MockUser.query.filter_by.return_value.first.return_value = mock_user_db
            
            rv = client.get('/authorized?code=some_code')
            # Should redirect to index and set session
            assert rv.status_code == 302
            with client.session_transaction() as sess:
                assert sess['user_id'] == 500

def test_sync_no_user(client):
    # Not logged in
    rv = client.get('/sync')
    assert rv.status_code == 302 # Redirect to login

def test_dashboard_access(client):
    # Logged in
    with client.session_transaction() as sess:
        sess['user_id'] = 100
        
    # Mock DB user query
    with patch('app.User') as MockUser:
        mock_user = MagicMock(firstname="Dash")
        mock_user.id = 100 # Real int for SQL
        MockUser.query.get.return_value = mock_user
        rv = client.get('/')
        assert rv.status_code == 200
        assert b"Dash" in rv.data
