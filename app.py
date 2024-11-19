from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Will be replaced with environment variable
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///referrals.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Import models after db initialization
from models import Referral

# AI Model configuration
model_id = "microsoft/resnet-50"  # Using a smaller model for the prototype
model = None
processor = None

def initialize_model():
    global model, processor
    try:
        model = AutoModelForImageClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image(image_path):
    """Prototype function to analyze images"""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # For prototype, we'll use a simple logic
        # In production, this would be replaced with proper medical image analysis
        score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][0].item()
        
        return {
            'urgency': 'urgent' if score > 0.7 else 'routine',
            'appointment_type': 'comprehensive' if score > 0.7 else 'standard',
            'field_test_required': score > 0.5
        }
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return {
            'urgency': 'urgent',  # Default to urgent in case of errors
            'appointment_type': 'comprehensive',
            'field_test_required': True
        }

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    referrals = Referral.query.order_by(Referral.created_at.desc()).all()
    return render_template('dashboard.html', referrals=referrals)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'referral' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['referral']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the uploaded file
            result = analyze_image(filepath)
            
            # Save to database
            referral = Referral()
            referral.filename = filename
            referral.urgency = result['urgency']
            referral.appointment_type = result['appointment_type']
            referral.field_test_required = result['field_test_required']
            
            db.session.add(referral)
            db.session.commit()
            
            flash('Referral processed successfully')
            return redirect(url_for('dashboard'))
            
        flash('Invalid file type')
        return redirect(request.url)
    
    return render_template('upload.html')

# Initialize database and model
with app.app_context():
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Initialize model
    initialize_model()
    # Create database tables
    db.create_all()
