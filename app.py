from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image, ImageFile, TiffImagePlugin
import logging
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images gracefully

# Flask app configuration
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'development-key-only'),
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///referrals.db'),
    UPLOAD_FOLDER=os.path.join(os.getcwd(), 'uploads'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
)
# Configure logging
logging.basicConfig(level=logging.INFO)
# Request logging middleware
class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        method = environ.get('REQUEST_METHOD', '')
        request_id = os.urandom(8).hex()
        
        logger.info(f"Request started - ID: {request_id} - Method: {method} - Path: {path}")
        
        def custom_start_response(status, headers, exc_info=None):
            logger.info(f"Request completed - ID: {request_id} - Status: {status}")
            return start_response(status, headers, exc_info)
        
        try:
            return self.app(environ, custom_start_response)
        except Exception as e:
            logger.error(f"Request failed - ID: {request_id} - Error: {str(e)}", exc_info=True)
            raise

# Application readiness flag
app_ready = False
logger = logging.getLogger(__name__)
# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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
    """Initialize the model for medical image analysis"""
    global model, processor
    try:
        print("Loading AI model...")
        from transformers import AutoModelForImageClassification, AutoFeatureExtractor
        
        model = AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=2,  # binary classification: urgent vs routine
            ignore_mismatched_sizes=True
        )
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
    """Analyze medical images with enhanced TIFF support for the prototype"""
    try:
        if not processor or not model:
            raise Exception("Model not initialized")
            
        # Log start of image processing
        print(f"Starting analysis of: {image_path}")
        
        # Open image with enhanced TIFF handling
        try:
            img = Image.open(image_path)
            
            # Get detailed image information
            img_info = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'bits': getattr(img, 'bits', None),
            }
            
            if img.format == 'TIFF':
                print("TIFF file detected, gathering TIFF-specific information...")
                # Get TIFF-specific tags
                tags = {
                    tag: img.tag[tag] for tag in img.tag.keys()
                    if tag in [256, 257, 258, 259]  # Width, Height, BitsPerSample, Compression
                }
                print(f"TIFF Tags: {tags}")
                
                # Check compression
                compression = tags.get(259, [1])[0]  # Default to uncompressed
                if compression not in [1, 5, 7]:  # Uncompressed, LZW, JPEG
                    raise OSError(f"Unsupported TIFF compression method: {compression}")
                    
                print(f"TIFF Compression: {compression}")
                
            print(f"Image Details: {img_info}")
            
        except OSError as e:
            print(f"Error opening image file: {str(e)}")
            if "truncated" in str(e).lower():
                raise OSError("Truncated or corrupted TIFF file. Please ensure the file is complete.")
            elif "compression" in str(e).lower():
                raise OSError("Unsupported TIFF compression. Please provide an uncompressed or LZW/JPEG compressed TIFF.")
            else:
                raise OSError(f"Error opening image: {str(e)}")

        # Log image details
        print(f"Image Format: {img.format}")
        print(f"Image Mode: {img.mode}")
        print(f"Image Size: {img.size}")
        print(f"Bits per pixel: {img.bits if hasattr(img, 'bits') else 'Unknown'}")
        
        # Handle multi-page TIFF
        n_frames = getattr(img, 'n_frames', 1)
        if n_frames > 1:
            print(f"Multi-page TIFF detected with {n_frames} frames")
            print("Processing first frame for prototype")
            img.seek(0)
        
        # Enhanced image conversion with proper bit depth handling
        try:
            # Handle multi-page TIFF
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f"Multi-page TIFF detected with {img.n_frames} frames")
                print("Processing first frame for prototype")
                img.seek(0)
            
            # Convert based on image mode
            if img.mode in ['I;16', 'I']:
                print(f"Converting {img.mode} (16-bit) image to 8-bit")
                # Scale 16-bit to 8-bit while preserving relative values
                img = img.point(lambda i: i * (255/65535)).convert('L')
            elif img.mode in ['LA', 'PA']:
                print(f"Converting {img.mode} to RGB with alpha")
                img = img.convert('RGBA')
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode == 'RGBA':
                print("Converting RGBA to RGB with white background")
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode not in ['RGB', 'L']:
                print(f"Converting {img.mode} to RGB")
                img = img.convert('RGB')
            
            # Final conversion to RGB if not already
            image = img if img.mode == 'RGB' else img.convert('RGB')
            print("Image successfully converted to RGB mode")
            
            # Preprocess image
            inputs = processor(images=image, return_tensors="pt")
            
            # Get model prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence = probs[0][0].item()  # Confidence score for urgent class
            
            # For prototype: Use confidence thresholds to determine urgency
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
            print(f"Error during image processing: {str(e)}")
            raise  # Re-raise the exception to be caught by the outer try-except block
    except OSError as e:
        print(f"TIFF/Image file error: {str(e)}")
        if "truncated" in str(e).lower():
            print("Error: Truncated or corrupted TIFF file")
        elif "unknown" in str(e).lower():
            print("Error: Unknown or unsupported TIFF compression")
        else:
            print("Error: General image file error")
        # Default to urgent in case of errors for patient safety
        return {
            'urgency': 'urgent',
            'appointment_type': 'comprehensive',
            'field_test_required': True,
            'confidence': None,
            'error': str(e)
        }
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        # Default to urgent in case of errors for patient safety
        return {
            'urgency': 'urgent',
            'appointment_type': 'comprehensive',
            'field_test_required': True,
            'confidence': None,
            'error': str(e)
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
            error_msg = str(e)
            if isinstance(e, OSError):
                if "compression" in error_msg.lower():
                    flash('Unsupported TIFF compression. Please provide an uncompressed or LZW/JPEG compressed TIFF.', 'error')
                elif "truncated" in error_msg.lower():
                    flash('The TIFF file appears to be incomplete or corrupted. Please check the file and try again.', 'error')
                elif "bits" in error_msg.lower():
                    flash('Unsupported bit depth in TIFF file. Please provide an 8-bit or 16-bit TIFF.', 'error')
                else:
                    flash(f'Error processing the TIFF file: {error_msg}', 'error')
            else:
                flash('An error occurred while processing the referral', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

# Initialize application components
def init_app():
    global app_ready
    with app.app_context():
        try:
            # Configure app for production
            app.config.update(
                ENV='production',
                DEBUG=False,
                PREFERRED_URL_SCHEME='https'
            )
            
            # Register middleware
            app.wsgi_app = RequestLoggingMiddleware(app.wsgi_app)
            logger.info("Request logging middleware registered")
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            logger.info("Upload directory created/verified")
            
            # Create database tables
            db.create_all()
            logger.info("Database tables created/verified")
            
            # Initialize AI model
            if not initialize_model():
                logger.warning("AI model initialization failed, system will use fallback analysis")
            
            # Mark application as ready
            app_ready = True
            logger.info("Application initialization completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during application initialization: {str(e)}")
            # The application will still run, but with limited functionality
            logger.warning("Application will run with limited functionality")
            return False

# Initialize app when imported
init_app()
