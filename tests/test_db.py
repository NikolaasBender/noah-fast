import pytest
from models import User, MinerModel, db

@pytest.fixture(autouse=True)
def clean_db(app):
    with app.app_context():
        db.create_all()
        # Clean tables
        db.session.query(MinerModel).delete()
        db.session.query(User).delete()
        db.session.commit()
        yield
        db.session.remove()
        db.drop_all()

def test_user_creation(app):
    with app.app_context():
        user = User(
            strava_id=123,
            firstname="Test",
            lastname="User"
        )
        db.session.add(user)
        db.session.commit()
        
        retrieved = User.query.filter_by(strava_id=123).first()
        assert retrieved.firstname == "Test"
        assert retrieved.lastname == "User"

def test_miner_model_creation(app):
    with app.app_context():
        # Ensure user exists (depend on test order or fixture? Better create fresh)
        user = User(strava_id=999, firstname="Miner", lastname="Test")
        db.session.add(user)
        db.session.commit()
        
        m = MinerModel(user_id=user.id, model_type="test_model", data=b"some binary data")
        db.session.add(m)
        db.session.commit()
        
        retrieved = MinerModel.query.filter_by(user_id=user.id).first()
        assert retrieved.model_type == "test_model"
        assert retrieved.data == b"some binary data"
