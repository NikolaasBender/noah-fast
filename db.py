import sqlite3
import time
import os

DB_FILE = 'users.db'

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            strava_id INTEGER PRIMARY KEY,
            firstname TEXT,
            lastname TEXT,
            access_token TEXT,
            refresh_token TEXT,
            expires_at INTEGER,
            last_sync REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_FILE}")

def save_user(user_data):
    """
    Saves or updates a user in the DB.
    user_data must include: strava_id, firstname, lastname, access_token, refresh_token, expires_at
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check if exists
    c.execute('SELECT * FROM users WHERE strava_id = ?', (user_data['strava_id'],))
    exists = c.fetchone()
    
    if exists:
        c.execute('''
            UPDATE users SET 
                firstname = ?, lastname = ?, access_token = ?, refresh_token = ?, expires_at = ?
            WHERE strava_id = ?
        ''', (user_data['firstname'], user_data['lastname'], 
              user_data['access_token'], user_data['refresh_token'], user_data['expires_at'], 
              user_data['strava_id']))
    else:
        c.execute('''
            INSERT INTO users (strava_id, firstname, lastname, access_token, refresh_token, expires_at, last_sync)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_data['strava_id'], user_data['firstname'], user_data['lastname'], 
              user_data['access_token'], user_data['refresh_token'], user_data['expires_at'], 0.0))
    
    conn.commit()
    conn.close()

def get_user(strava_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE strava_id = ?', (strava_id,)).fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

def update_sync_time(strava_id):
    conn = get_db_connection()
    conn.execute('UPDATE users SET last_sync = ? WHERE strava_id = ?', (time.time(), strava_id))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
