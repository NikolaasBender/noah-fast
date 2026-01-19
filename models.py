from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    strava_id = db.Column(db.Integer, unique=True, nullable=False)
    firstname = db.Column(db.String(50))
    lastname = db.Column(db.String(50))
    access_token = db.Column(db.String(100))
    refresh_token = db.Column(db.String(100))
    expires_at = db.Column(db.Integer)
    last_sync = db.Column(db.DateTime)
    
    # Relationship
    models = db.relationship('MinerModel', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.firstname} {self.lastname}>'

class MinerModel(db.Model):
    __tablename__ = 'miner_models'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False) # 'physiology', 'bike_profiles'
    data = db.Column(db.LargeBinary, nullable=False) # Pickled data
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<MinerModel {self.model_type} for User {self.user_id}>'
