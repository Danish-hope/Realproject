from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, text
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images with CORS headers"""
    response = send_from_directory(os.path.join(os.path.dirname(__file__), 'uploads'), filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    return response

# Enable CORS with explicit settings for development
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Database Configuration (PostgreSQL)
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'noschoolokayG1')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'postgres')

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'medvision-secret-key-123')

# Initialize DB and Bcrypt
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Run schema maintenance on app startup
with app.app_context():
    # Schema Maintenance: Cleanup old enum type
    try:
        from sqlalchemy import text
        # First, alter the role column to VARCHAR to be safe
        db.session.execute(text("""
            DO $$ 
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.columns 
                           WHERE table_name = 'users' AND column_name = 'role' 
                           AND data_type = 'USER-DEFINED') THEN
                    ALTER TABLE users ALTER COLUMN role TYPE VARCHAR(20) USING role::VARCHAR(20);
                END IF;
            END $$;
        """))
        # Drop the old enum type if it exists
        db.session.execute(text("DROP TYPE IF EXISTS user_roles;"))
        db.session.commit()
        print("Old enum type cleaned up successfully!")
    except Exception as e:
        print(f"Schema maintenance notice (enum cleanup): {str(e)}")
        db.session.rollback()
    
    # Now create all tables
    db.create_all()
    print("Database tables initialized!")

# User Model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    specialty = db.Column(db.String(100), nullable=True)
    clinic_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    feedback = db.Column(db.Text, nullable=True)
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<User {self.username}>'

# Prediction Model
class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    image_path = db.Column(db.Text, nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    ai_description = db.Column(db.Text, nullable=True)
    prevention_tips = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.disease} - {self.user_id}>'

# Patient Model
class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_age = db.Column(db.Integer, nullable=False)
    patient_gender = db.Column(db.String(10), nullable=False)
    contact = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to predictions
    patient_predictions = db.relationship('PatientPrediction', backref='patient', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Patient {self.patient_name}>'

# Patient Prediction Model
class PatientPrediction(db.Model):
    __tablename__ = 'patient_predictions'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    image_path = db.Column(db.Text, nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PatientPrediction {self.disease} for patient {self.patient_id}>'

# Enable CORS for all routes (allows frontend to communicate with backend)
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit

# Global model variable
model = None

# Disease labels 
labels = {
    0: 'Chickenpox',
    1: 'Cowpox',
    2: 'HFMD',
    3: 'Healthy',
    4: 'Measles',
    5: 'MPOX'
}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_ml_model():
    """Load the TensorFlow model"""
    global model
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best_model (1).h5')

        print(f"Looking for model at: {model_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")

        if not os.path.exists(model_path):
            # Try alternative filename without parentheses
            alt_model_path = os.path.join(script_dir, 'best_model.h5')
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
                print(f"Using alternative model path: {model_path}")
            else:
                # List all .h5 files in the directory
                h5_files = [f for f in os.listdir(script_dir) if f.endswith('.h5')]
                print(f"Available .h5 files: {h5_files}")
                raise FileNotFoundError(f"Model file '{model_path}' not found. Available .h5 files: {h5_files}")

        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Resize image to 224x224 (matching Streamlit app)
        image = image.resize((224, 224))

        # Convert to numpy array
        image_array = np.array(image)

        # Add batch dimension to match model's expected input shape
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def predict_disease(image):
    """Make prediction using the loaded model"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)

        # Get the predicted class index
        predicted_class = np.argmax(predictions[0])

        # Get confidence score (highest probability)
        confidence = float(predictions[0][predicted_class] * 100)

        # Get disease name
        disease_name = labels.get(predicted_class, 'Unknown')

        return {
            'disease': disease_name,
            'confidence': round(confidence, 2),
            'class_index': int(predicted_class)
        }

    except Exception as e:
        raise ValueError(f"Error making prediction: {str(e)}")

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'MedVision AI API is running',
        'endpoints': {
            'health': '/health',
            'register': '/register',
            'login': '/login'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    db_status = False
    try:
        db.session.execute(text('SELECT 1'))
        db_status = True
    except Exception as e:
        print(f"Health check DB error: {str(e)}")
            
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'db_initialized': db_status,
        'message': 'MedVision AI API is running'
    })

@app.route('/debug/predict-modes', methods=['GET'])
def debug_predict_modes():
    """Debug endpoint to describe predict() behavior for different clients"""
    return jsonify({
        'predict_endpoint': '/predict',
        'user_flow': {
            'description': 'Used by user dashboard without patient/doctor IDs',
            'requires_patient_id': False,
            'requires_doctor_id': False,
            'form_fields': ['image', 'is_user_request=true'],
            'automatic_save': False,
            'notes': 'Predictions are saved via /api/save-prediction using user_id'
        },
        'doctor_flow': {
            'description': 'Used by doctor dashboard with patient and doctor IDs',
            'requires_patient_id': True,
            'requires_doctor_id': True,
            'form_fields': ['image', 'patient_id', 'doctor_id'],
            'automatic_save': True,
            'notes': 'Predictions are saved automatically to patient_predictions table'
        }
    })

@app.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    data = request.json
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password are required'}), 400

    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if data.get('email') and User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    
    # Create new user
    new_user = User(
        username=data.get('username'),
        email=data.get('email'),
        password_hash=hashed_password,
        role=data.get('role', 'user')
    )
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User registered successfully'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.json
    role = data.get('role', 'user')
    
    if not data or not data.get('password'):
        return jsonify({'error': 'Credentials required'}), 400

    # Find user by email (if provided) or username
    user = None
    credential = data.get('email') or data.get('username')
    
    if role == 'user':
        # For users, allow login via email or username
        user = User.query.filter((User.email == credential) | (User.username == credential)).first()
    else:
        # For doctors/admins, usually username-based
        user = User.query.filter_by(username=data.get('username')).first()
        
    if user and bcrypt.check_password_hash(user.password_hash, data['password']):
        # Check if user is active
        if user.is_active is False:
            return jsonify({'error': 'Account is deactivated. Please contact support.'}), 403

        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'email': user.email,
                'username': user.username,
                'role': user.role
            }
        })
    else:
        return jsonify({'error': 'Invalid username/email or password'}), 401

@app.route('/verify-email', methods=['POST'])
def verify_email():
    """Verify email exists for password reset"""
    data = request.json
    if not data or not data.get('email'):
        return jsonify({'error': 'Email is required'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    if user:
        return jsonify({'success': True, 'message': 'Email verified'}), 200
    else:
        return jsonify({'error': 'Email not found'}), 404

@app.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset user password"""
    data = request.json
    if not data or not data.get('email') or not data.get('new_password'):
        return jsonify({'error': 'Email and new password are required'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    try:
        # Update password
        hashed_password = bcrypt.generate_password_hash(data['new_password']).decode('utf-8')
        user.password_hash = hashed_password
        db.session.commit()
        return jsonify({'success': True, 'message': 'Password updated successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

# --- Profile Management Endpoints ---

@app.route('/api/profile/get/<int:user_id>', methods=['GET'])
def get_profile(user_id):
    """Get user profile information"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        response_data = {
            'success': True,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'phone': user.phone,
                'role': user.role,
                'created_at': user.created_at.isoformat() if user.created_at else None
            }
        }

        # Add doctor-specific fields if applicable
        if user.role == 'doctor':
            response_data['user']['specialty'] = user.specialty
            response_data['user']['clinic_name'] = user.clinic_name

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

@app.route('/api/profile/update', methods=['PUT'])
def update_profile():
    """Update user profile information"""
    data = request.json
    
    if not data or not data.get('user_id'):
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        user = User.query.get(data['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Validate and update username
        if 'username' in data:
            new_username = data['username'].strip()
            if not new_username:
                return jsonify({'error': 'Username cannot be empty'}), 400
            
            # Check if username is taken by another user
            existing_user = User.query.filter_by(username=new_username).first()
            if existing_user and existing_user.id != user.id:
                return jsonify({'error': 'Username already taken'}), 400
            
            user.username = new_username
        
        # Validate and update email
        if 'email' in data:
            new_email = data['email'].strip()
            if new_email:
                # Basic email validation
                import re
                email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
                if not re.match(email_regex, new_email):
                    return jsonify({'error': 'Invalid email format'}), 400
                
                # Check if email is taken by another user
                existing_user = User.query.filter_by(email=new_email).first()
                if existing_user and existing_user.id != user.id:
                    return jsonify({'error': 'Email already taken'}), 400
                
                user.email = new_email
        
        # Update phone (optional field, no validation needed)
        if 'phone' in data:
            user.phone = data['phone'].strip() if data['phone'] else None
        
        # Update doctor-specific fields
        if user.role == 'doctor':
            if 'specialty' in data:
                user.specialty = data['specialty'].strip() if data['specialty'] else None
            if 'clinic_name' in data:
                user.clinic_name = data['clinic_name'].strip() if data['clinic_name'] else None
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'phone': user.phone,
                'specialty': user.specialty,
                'clinic_name': user.clinic_name
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

@app.route('/api/profile/delete', methods=['DELETE'])
def delete_profile():
    """Soft delete user profile"""
    data = request.json
    
    if not data or not data.get('user_id'):
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        user = User.query.get(data['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Soft delete: mark as inactive
        user.is_active = False
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Account deactivated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

@app.route('/api/profile/update-password', methods=['PUT'])
def update_password():
    """Update user password"""
    data = request.json
    
    if not data or not data.get('user_id'):
        return jsonify({'error': 'User ID is required'}), 400
    
    if not data.get('current_password') or not data.get('new_password'):
        return jsonify({'error': 'Current password and new password are required'}), 400
    
    try:
        user = User.query.get(data['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Verify current password
        if not bcrypt.check_password_hash(user.password_hash, data['current_password']):
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Validate new password strength
        new_password = data['new_password']
        if len(new_password) < 8:
            return jsonify({'error': 'New password must be at least 8 characters long'}), 400
        
        # Hash and update password
        user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Password updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Database error', 'message': str(e)}), 500


# --- Admin Doctor Management Endpoints ---


@app.route('/admin/analytics', methods=['GET'])
def get_system_analytics():
    """Get system-wide analytics data with optional filtering"""
    try:
        period = request.args.get('period', 'month')
        now = datetime.utcnow()
        
        # Calculate start date based on period
        if period == 'today':
            start_date = datetime(now.year, now.month, now.day)
        elif period == 'week':
            start_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'month':
            start_date = datetime(now.year, now.month, 1)
        elif period == 'all':
            start_date = datetime.min
        else:
            start_date = datetime(now.year, now.month, 1) # Default to month

        # 1. User Metrics (Filtered by date for creation, though typically users are total count)
        # For analytics, we might want NEW users in this period vs TOTAL users.
        # Let's keep Total/Active as global stats, but maybe filter "New Users" if we had that metric.
        # For now, we'll keep Total/Active as global snapshots as requested by most dashboards.
        total_users = User.query.filter_by(role='user').count()
        active_users = User.query.filter_by(role='user', is_active=True).count()
        
        # 2. Doctor Metrics
        total_doctors = User.query.filter_by(role='doctor').count()
        active_doctors = User.query.filter_by(role='doctor', is_active=True).count()
        
        # 3. Patient Metrics
        total_patients = Patient.query.count()
        
        # 4. Prediction Metrics (Filtered by period)
        user_predictions_query = Prediction.query
        doctor_predictions_query = PatientPrediction.query
        
        if period != 'all':
            user_predictions_query = user_predictions_query.filter(Prediction.created_at >= start_date)
            doctor_predictions_query = doctor_predictions_query.filter(PatientPrediction.created_at >= start_date)
            
        user_predictions_count = user_predictions_query.count()
        doctor_predictions_count = doctor_predictions_query.count()
        total_predictions = user_predictions_count + doctor_predictions_count
        
        # 5. Disease Distribution (Filtered by period)
        user_diseases = db.session.query(Prediction.disease, func.count(Prediction.disease))
        doctor_diseases = db.session.query(PatientPrediction.disease, func.count(PatientPrediction.disease))
        
        if period != 'all':
            user_diseases = user_diseases.filter(Prediction.created_at >= start_date)
            doctor_diseases = doctor_diseases.filter(PatientPrediction.created_at >= start_date)
            
        user_diseases = user_diseases.group_by(Prediction.disease).all()
        doctor_diseases = doctor_diseases.group_by(PatientPrediction.disease).all()
        
        disease_counts = {}
        for disease, count in user_diseases:
            disease_counts[disease] = disease_counts.get(disease, 0) + count
        for disease, count in doctor_diseases:
            disease_counts[disease] = disease_counts.get(disease, 0) + count
            
        disease_distribution = [{'name': k, 'value': v} for k, v in disease_counts.items()]
        disease_distribution.sort(key=lambda x: x['value'], reverse=True)
        
        # 6. Monthly Stats (Keep as is for the "Monthly Growth" card context, or adjust if needed)
        # The prompt asks for "This Month" filter functionality.
        # If the user selects "Today", the main metrics update.
        # The "Monthly Stats" section in the UI (Growth, Last Month) specifically refers to Monthly performance.
        # We should probably keep those as fixed monthly comparators unless we want to change the UI labels dynamically.
        # For stability, let's keep the monthly comparison logic fixed to actual months, 
        # but the TOP cards will reflect the filtered period (e.g. Total Predictions in this period).
        
        current_month_start = datetime(now.year, now.month, 1)
        last_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
        
        # Last Month (Fixed for comparison)
        last_month_end = current_month_start
        last_month_user = Prediction.query.filter(Prediction.created_at >= last_month_start, Prediction.created_at < last_month_end).count()
        last_month_doctor = PatientPrediction.query.filter(PatientPrediction.created_at >= last_month_start, PatientPrediction.created_at < last_month_end).count()
        last_month_total = last_month_user + last_month_doctor
        
        # This Month (Fixed for comparison)
        this_month_user = Prediction.query.filter(Prediction.created_at >= current_month_start).count()
        this_month_doctor = PatientPrediction.query.filter(PatientPrediction.created_at >= current_month_start).count()
        this_month_total = this_month_user + this_month_doctor

        # Growth
        if last_month_total > 0:
            growth = round(((this_month_total - last_month_total) / last_month_total) * 100, 1)
        else:
            growth = 100 if this_month_total > 0 else 0
            
        # Daily Usage (Today) - Fixed for specific card
        today_start = datetime(now.year, now.month, now.day)
        today_user = Prediction.query.filter(Prediction.created_at >= today_start).count()
        today_doctor = PatientPrediction.query.filter(PatientPrediction.created_at >= today_start).count()
        daily_usage = today_user + today_doctor
            
        return jsonify({
            'success': True,
            'metrics': {
                'total_users': total_users,
                'active_users': active_users,
                'total_doctors': total_doctors,
                'active_doctors': active_doctors,
                'total_patients': total_patients,
                'total_predictions': total_predictions, # This now respects the filter
                'daily_usage': daily_usage
            },
            'monthly_stats': {
                'this_month': this_month_total,
                'last_month': last_month_total,
                'growth': growth
            },
            'disease_distribution': disease_distribution # This now respects the filter
        })
        
    except Exception as e:
        print(f"Analytics Error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        user_id = data.get('user_id')
        message = data.get('message')
        
        if not user_id or not message:
            return jsonify({'error': 'Missing required fields'}), 400
            
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        user.feedback = message
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Feedback submitted successfully'}), 200
        
    except Exception as e:
        print(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/admin/feedbacks', methods=['GET'])
def get_all_feedbacks():
    try:
        # Get users who have submitted feedback (feedback is not null and not empty)
        users_with_feedback = User.query.filter(User.feedback.isnot(None), User.feedback != '').all()
        
        feedback_list = []
        for u in users_with_feedback:
            feedback_list.append({
                'user_id': u.id,
                'username': u.username,
                'role': u.role,
                'email': u.email,
                'feedback': u.feedback,
                'submitted_at': u.last_login.isoformat() if u.last_login else datetime.utcnow().isoformat()
            })
            
        return jsonify(feedback_list)
        
    except Exception as e:
        print(f"Error getting feedbacks: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


@app.route('/admin/users', methods=['GET'])
def get_all_users():
    """Get all registered users"""
    users = User.query.filter_by(role='user').all()
    return jsonify([{
        'id': u.id,
        'username': u.username,
        'email': u.email,
        'is_active': u.is_active,
        'created_at': u.created_at.isoformat() if u.created_at else None,
        'predictions_count': len(u.predictions)
    } for u in users])

@app.route('/admin/toggle-user/<int:user_id>', methods=['POST'])
def toggle_user(user_id):
    """Toggle user active status"""
    user = User.query.get_or_404(user_id)
    if user.role != 'user':
        return jsonify({'error': 'Not a user account'}), 400
        
    user.is_active = not user.is_active
    db.session.commit()
    
    status = 'activated' if user.is_active else 'deactivated'
    return jsonify({'success': True, 'message': f'User {status} successfully', 'is_active': user.is_active})

@app.route('/admin/doctors', methods=['GET'])
def get_all_doctors():
    """Get all registered doctors"""
    doctors = User.query.filter_by(role='doctor').all()
    return jsonify([{
        'id': d.id,
        'username': d.username,
        'email': d.email,
        'specialty': d.specialty,
        'clinic_name': d.clinic_name,
        'is_active': d.is_active,
        'created_at': d.created_at.isoformat() if d.created_at else None
    } for d in doctors])

@app.route('/admin/add-doctor', methods=['POST'])
def add_doctor():
    """Add a new doctor account"""
    data = request.json
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password are required'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if data.get('email') and User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    
    new_doctor = User(
        username=data['username'],
        email=data.get('email'),
        password_hash=hashed_password,
        role='doctor',
        specialty=data.get('specialty'),
        clinic_name=data.get('clinic_name'),
        is_active=True
    )
    
    try:
        db.session.add(new_doctor)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Doctor added successfully'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

@app.route('/admin/toggle-doctor/<int:doctor_id>', methods=['POST'])
def toggle_doctor(doctor_id):
    """Toggle doctor active status"""
    doctor = User.query.get_or_404(doctor_id)
    if doctor.role != 'doctor':
        return jsonify({'error': 'Not a doctor account'}), 400
        
    doctor.is_active = not doctor.is_active
    db.session.commit()
    
    status = 'activated' if doctor.is_active else 'deactivated'
    return jsonify({'success': True, 'message': f'Doctor {status} successfully', 'is_active': doctor.is_active})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with automatic saving"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Machine learning model failed to load'
            }), 500

        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please upload an image file'
            }), 400

        file = request.files['image']
        patient_id = request.form.get('patient_id')
        doctor_id = request.form.get('doctor_id')
        is_user_request = request.form.get('is_user_request') == 'true'

        if not is_user_request and (not patient_id or not doctor_id):
            return jsonify({
                'error': 'Missing identification',
                'message': 'Patient ID and Doctor ID are required for automatic saving'
            }), 400

        # Check if file is empty
        if file.filename == '':
            return jsonify({
                'error': 'Empty file',
                'message': 'No file selected'
            }), 400

        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Only PNG, JPG, JPEG, and BMP files are allowed'
            }), 400

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer

        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': 'File too large',
                'message': f'File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit'
            }), 400

        try:
            # Save image to uploads folder
            UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            
            filename = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save the file
            file.save(image_path)
            
            # Relative path for frontend access
            relative_image_path = f"uploads/{filename}"

            # Open image for prediction
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Make prediction
            result = predict_disease(image)

            # Automatic Save to Database for doctor workflow
            prediction_saved = False
            if not is_user_request:
                try:
                    new_prediction = PatientPrediction(
                        patient_id=int(patient_id),
                        doctor_id=int(doctor_id),
                        image_path=relative_image_path,
                        disease=result['disease'],
                        confidence=result['confidence']
                    )
                    db.session.add(new_prediction)
                    db.session.commit()
                    prediction_saved = True
                    print(f"DEBUG: Automatic prediction save successful for patient {patient_id}")
                except Exception as db_err:
                    db.session.rollback()
                    prediction_saved = False
                    print(f"DEBUG: Automatic prediction save failed: {str(db_err)}")
            else:
                print("DEBUG: User prediction request - skipping automatic patient/doctor save")

            return jsonify({
                'success': True,
                'prediction': result,
                'image_path': relative_image_path,
                'prediction_saved': prediction_saved,
                'message': 'Prediction completed and saved successfully' if prediction_saved else 'Prediction completed but failed to save to database'
            })

        except Exception as e:
            return jsonify({
                'error': 'Processing error',
                'message': f'Error processing image: {str(e)}'
            }), 400

    except Exception as e:
        print(f"Unexpected error in /predict: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method used is not allowed for this endpoint'
    }), 405

# --- Prediction History Endpoints ---

@app.route('/api/save-prediction', methods=['POST'])
def save_prediction():
    """Save a prediction result to history"""
    data = request.json
    print(f"DEBUG: save_prediction called with data: {data}")
    if not data or not data.get('user_id') or not data.get('disease'):
        print("DEBUG: save_prediction failed - missing fields")
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        new_prediction = Prediction(
            user_id=data['user_id'],
            image_path=data.get('image_path', 'N/A'),
            disease=data['disease'],
            confidence=data.get('confidence', 0),
            ai_description=data.get('ai_description'),
            prevention_tips=data.get('prevention_tips')
        )
        db.session.add(new_prediction)
        db.session.commit()
        print(f"DEBUG: Prediction saved successfully for user {data['user_id']}")
        return jsonify({'success': True, 'message': 'Prediction saved successfully'}), 201
    except Exception as e:
        db.session.rollback()
        print(f"DEBUG: save_prediction database error: {str(e)}")
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

@app.route('/api/prediction-history/<int:user_id>', methods=['GET'])
def get_prediction_history(user_id):
    """Get prediction history for a specific user"""
    print(f"DEBUG: get_prediction_history called for user: {user_id}")
    try:
        predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
        print(f"DEBUG: Found {len(predictions)} predictions for user {user_id}")
        return jsonify([{
            'id': p.id,
            'disease': p.disease,
            'confidence': p.confidence,
            'image_path': p.image_path,
            'ai_description': p.ai_description,
            'prevention_tips': p.prevention_tips,
            'created_at': p.created_at.isoformat()
        } for p in predictions])
    except Exception as e:
        print(f"DEBUG: get_prediction_history error: {str(e)}")
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

# --- Doctor-Patient Management APIs ---

@app.route('/api/patients/add', methods=['POST'])
def add_patient():
    """Add a new patient record"""
    data = request.json
    print(f"DEBUG: add_patient called with data: {data}")
    
    required_fields = ['doctor_id', 'patient_name', 'patient_age', 'patient_gender']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        new_patient = Patient(
            doctor_id=data['doctor_id'],
            patient_name=data['patient_name'],
            patient_age=data['patient_age'],
            patient_gender=data['patient_gender'],
            contact=data.get('contact')
        )
        db.session.add(new_patient)
        db.session.commit()
        return jsonify({
            'success': True, 
            'message': 'Patient added successfully',
            'patient_id': new_patient.id
        }), 201
    except Exception as e:
        db.session.rollback()
        print(f"DEBUG: add_patient error: {str(e)}")
        return jsonify({'error': 'Database error', 'message': str(e)}), 500


@app.route('/api/patients/delete/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Delete a patient record"""
    try:
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
            
        db.session.delete(patient)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Patient deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        print(f"DEBUG: delete_patient error: {str(e)}")
        return jsonify({'error': 'Database error', 'message': str(e)}), 500


@app.route('/api/patients/doctor/<int:doctor_id>', methods=['GET'])
def get_doctor_patients(doctor_id):
    """Fetch all patients for a specific doctor"""
    try:
        patients = Patient.query.filter_by(doctor_id=doctor_id).order_by(Patient.patient_name).all()
        return jsonify([{
            'id': p.id,
            'name': p.patient_name,
            'age': p.patient_age,
            'gender': p.patient_gender,
            'contact': p.contact,
            'createdAt': p.created_at.isoformat()
        } for p in patients]), 200
    except Exception as e:
        print(f"DEBUG: get_doctor_patients error: {str(e)}")
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

@app.route('/api/history/doctor/<int:doctor_id>', methods=['GET'])
def get_doctor_history(doctor_id):
    """Fetch all predictions for a doctor with patient names"""
    try:
        results = db.session.query(PatientPrediction, Patient.patient_name).\
            join(Patient, PatientPrediction.patient_id == Patient.id).\
            filter(PatientPrediction.doctor_id == doctor_id).\
            order_by(PatientPrediction.created_at.desc()).all()
        
        history = []
        for pred, patient_name in results:
            history.append({
                'id': pred.id,
                'patient_id': pred.patient_id,
                'patient_name': patient_name,
                'disease': pred.disease,
                'confidence': pred.confidence,
                'image_path': pred.image_path,
                'timestamp': pred.created_at.isoformat()
            })
        return jsonify(history), 200
    except Exception as e:
        print(f"DEBUG: get_doctor_history error: {str(e)}")
        return jsonify({'error': 'Database error', 'message': str(e)}), 500

@app.route('/api/patients/report/<int:patient_id>', methods=['GET'])
def get_patient_report(patient_id):
    """Fetch patient details and prediction history"""
    try:
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
            
        predictions = PatientPrediction.query.filter_by(patient_id=patient_id).\
            order_by(PatientPrediction.created_at.desc()).all()
            
        return jsonify({
            'patient_info': {
                'id': patient.id,
                'name': patient.patient_name,
                'age': patient.patient_age,
                'gender': patient.patient_gender,
                'contact': patient.contact,
                'created_at': patient.created_at.isoformat()
            },
            'predictions': [{
                'id': p.id,
                'disease': p.disease,
                'confidence': p.confidence,
                'image_path': p.image_path,
                'timestamp': p.created_at.isoformat()
            } for p in predictions]
        }), 200
    except Exception as e:
        print(f"DEBUG: get_patient_report error: {str(e)}")
        return jsonify({'error': 'Database error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Load the model when starting the server
    if load_ml_model():
        print("Starting Flask server...")

        print("API Endpoints:")
        print("  GET  /health     - Health check")
        print("  POST /predict    - Image prediction")
        print("Server will run on http://localhost:5000")

        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True
        )
    else:
        print("Failed to load ML model. Exiting...")
        exit(1)
