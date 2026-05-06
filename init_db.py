from app1 import app, db

with app.app_context():
    db.create_all()
    print("Database tables initialized successfully!")
