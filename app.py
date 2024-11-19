import os
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.utils import secure_filename
import torch
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from qwen_utils import analyze_referral
from PIL import Image
import numpy as np

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)

# Configuration
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "glaucoma_triage_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///glaucoma_triage.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize database
db.init_app(app)

# Initialize ResNet model for image analysis
from transformers import AutoFeatureExtractor, ResNetForImageClassification
model_id = "microsoft/resnet-50"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id)
model.eval()  # Set to evaluation mode

# Routes
@app.route('/')
def index():
    return render_template('dashboard.html')

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
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process referral with AI model
            result = analyze_referral(filepath, model, feature_extractor)
            
            # Save to database
            from models import Referral
            referral = Referral(
                filename=filename,
                urgency=result['urgency'],
                appointment_type=result['appointment_type'],
                field_test_required=result['field_test_required']
            )
            db.session.add(referral)
            db.session.commit()
            
            flash('Referral processed successfully')
            return redirect(url_for('dashboard'))
    
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    from models import Referral
    referrals = Referral.query.order_by(Referral.created_at.desc()).all()
    return render_template('dashboard.html', referrals=referrals)

with app.app_context():
    import models
    db.create_all()
