from app1 import app, db, User, bcrypt
from datetime import datetime

def seed_admin():
    with app.app_context():
        # Check if admin already exists
        admin_username = 'danish'
        admin_email = 'admin@medvision.ai'
        
        existing_admin = User.query.filter_by(username=admin_username).first()
        
        if existing_admin:
            print(f"Admin user '{admin_username}' already exists.")
            return

        # Create new admin user
        hashed_password = bcrypt.generate_password_hash('danish123').decode('utf-8')
        
        new_admin = User(
            username=admin_username,
            email=admin_email,
            password_hash=hashed_password,
            role='admin',
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        try:
            db.session.add(new_admin)
            db.session.commit()
            print(f"Admin user '{admin_username}' created successfully!")
        except Exception as e:
            db.session.rollback()
            print(f"Error creating admin user: {str(e)}")

if __name__ == '__main__':
    seed_admin()
