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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Import models after db initialization
from models import Referral

# AI Model configuration
model_id = "microsoft/resnet-50"  # Using a smaller model for the prototype
model = None
processor = None

def initialize_model():
    """Initialize the AI model for image analysis"""
    global model, processor
    try:
        print("Loading AI model...")
        model = AutoModelForImageClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        
        # Verify model initialization
        if model is None or processor is None:
            raise RuntimeError("Failed to initialize AI model components")
            
        print("AI model loaded successfully")
        return True
    except Exception as e:
        print(f"Error initializing AI model: {str(e)}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image(image_path):
    """Prototype function to analyze images for glaucoma screening"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        if not processor or not model:
            raise Exception("Model not initialized")
            
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # For prototype, we'll use confidence scores to determine urgency
        # In production, this would be replaced with proper medical image analysis
        confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0][0].item()
        
        # Analyze image features for potential indicators
        is_urgent = confidence > 0.7
        needs_comprehensive = confidence > 0.6
        needs_field_test = confidence > 0.5
        
        return {
            'urgency': 'urgent' if is_urgent else 'routine',
            'appointment_type': 'comprehensive' if needs_comprehensive else 'standard',
            'field_test_required': needs_field_test,
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        # Default to urgent in case of errors for patient safety
        return {
            'urgency': 'urgent',
            'appointment_type': 'comprehensive',
            'field_test_required': True,
            'confidence': None
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
        # Check if the post request has the file part
        if 'referral' not in request.files:
            flash('No file part in the request', 'error')
            return redirect(request.url)
        
        file = request.files['referral']
        
        # Check if a file was actually selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Validate file type
        if not allowed_file(file.filename):
            flash('Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS), 'error')
            return redirect(request.url)
        
        try:
            if not file.filename or not isinstance(file.filename, str):
                raise ValueError("Invalid filename")
                
            # Secure the filename and create save path
            filename = secure_filename(file.filename)
            if not filename:
                raise ValueError("Invalid filename after sanitization")
                
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Analyze the uploaded file
            result = analyze_image(filepath)
            
            # Save to database
            referral = Referral(
                filename=filename,
                urgency=result['urgency'],
                appointment_type=result['appointment_type'],
                field_test_required=result['field_test_required']
            )
            
            db.session.add(referral)
            db.session.commit()
            
            flash(f'Referral processed successfully. Urgency: {result["urgency"].title()}', 'success')
            return redirect(url_for('dashboard'))
            
        except ValueError as e:
            flash(f'Invalid file: {str(e)}', 'error')
            return redirect(request.url)
        except Exception as e:
            print(f"Error processing referral: {str(e)}")  # Log the error
            flash('An error occurred while processing the referral', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

# Initialize application
with app.app_context():
    try:
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print("Upload directory created/verified")
        
        # Create database tables
        db.create_all()
        print("Database tables created/verified")
        
        # Initialize AI model
        if not initialize_model():
            print("Warning: AI model initialization failed, system will use fallback analysis")
        
        print("Application initialization completed")
    except Exception as e:
        print(f"Error during application initialization: {str(e)}")
        # The application will still run, but with limited functionality
        print("Application will run with limited functionality")
