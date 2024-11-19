import os
from PIL import Image
import torch
import numpy as np
from datetime import datetime

def analyze_referral(filepath, model, feature_extractor):
    """Analyze referral document using ResNet model"""
    image = Image.open(filepath).convert('RGB')
    
    # Prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
    # Simple logic for demo purposes
    # Using confidence scores to determine urgency and requirements
    confidence = probs.max().item()
    
    # Analyze the image features for decision making
    result = {
        'urgency': 'urgent' if confidence > 0.8 else 'routine',
        'appointment_type': 'comprehensive assessment' if confidence > 0.7 else 'standard assessment',
        'field_test_required': confidence > 0.75
    }
    
    return result
    
    # Parse model output
    result = {
        'urgency': 'urgent' if 'urgent' in output_text.lower() else 'routine',
        'appointment_type': 'standard assessment',  # Default, should be parsed from output
        'field_test_required': 'field test' in output_text.lower()
    }
    
    return result
