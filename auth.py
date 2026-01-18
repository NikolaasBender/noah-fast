import os
import json
import time
from stravalib.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')
TOKEN_FILE = 'strava_tokens.json'

def save_tokens(token_response):
    with open(TOKEN_FILE, 'w') as f:
        json.dump(token_response, f)
    print(f"Tokens saved to {TOKEN_FILE}")

def load_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f)
    return None

def authenticate():
    client = Client()

    if not CLIENT_ID or not CLIENT_SECRET:
        print("Error: STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set in .env file.")
        return None

    # check if we have saved tokens
    saved_tokens = load_tokens()
    
    if saved_tokens:
        print("Found saved tokens.")
        client.access_token = saved_tokens['access_token']
        client.refresh_token = saved_tokens['refresh_token']
        client.token_expires_at = saved_tokens['expires_at']
        
        if time.time() > client.token_expires_at:
            print("Token expired, refreshing...")
            refresh_response = client.refresh_access_token(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                refresh_token=client.refresh_token
            )
            save_tokens(refresh_response)
            client.access_token = refresh_response['access_token']
            client.refresh_token = refresh_response['refresh_token']
            client.token_expires_at = refresh_response['expires_at']
            print("Token refreshed.")
        else:
            print("Token is valid.")
            
        return client

    # Steps to get new token
    authorize_url = client.authorization_url(
        client_id=CLIENT_ID,
        redirect_uri='http://localhost:8000/authorized',
        scope=['read_all','profile:read_all','activity:read_all']
    )

    print("\nPlease go to the following URL to authorize access:")
    print(authorize_url)
    print("\nAfter authorizing, you will be redirected to a URL like 'http://localhost:8000/authorized?state=&code=...'")
    code = input("Please paste the full Redirect URL here: ")

    # Extract code from URL if full URL is pasted
    if "code=" in code:
        code = code.split("code=")[1].split("&")[0]

    token_response = client.exchange_code_for_token(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        code=code
    )

    save_tokens(token_response)
    client.access_token = token_response['access_token']
    client.refresh_token = token_response['refresh_token']
    client.token_expires_at = token_response['expires_at']
    
    return client

if __name__ == "__main__":
    client = authenticate()
    if client:
        athlete = client.get_athlete()
        print(f"\nSuccessfully authenticated as: {athlete.firstname} {athlete.lastname}")
