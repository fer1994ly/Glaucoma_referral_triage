import os
from PIL import Image
import torch
from datetime import datetime
from pdf2image import convert_from_path
import re

def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for message in messages:
        if message["role"] != "user":
            continue
        for content in message["content"]:
            if content["type"] == "image":
                image_inputs.append(content["image"])
            elif content["type"] == "video":
                video_inputs.append(content["video"])
def convert_pdf_to_images(pdf_path):
    """Convert PDF to list of PIL Images"""
    try:
        return convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")
        return None

def extract_numerical_values(text):
    """Extract numerical values from text"""
    pressure_pattern = r'(\d{2,3}(?:\.\d+)?)\s*(?:mm ?Hg|mmHg)'
    cup_disc_pattern = r'(?:cup[\s-]*to[\s-]*disc|c/d|cd)\s*(?:ratio)?\s*(?:of)?\s*(\d+\.?\d*)'
    
    pressure_matches = re.findall(pressure_pattern, text.lower())
    cup_disc_matches = re.findall(cup_disc_pattern, text.lower())
    
    pressures = [float(p) for p in pressure_matches]
    cup_disc = [float(cd) for cd in cup_disc_matches]
    
    return {
        'max_pressure': max(pressures) if pressures else None,
        'cup_disc_ratio': max(cup_disc) if cup_disc else None
    }
    return image_inputs, video_inputs

def analyze_referral(filepath, model, feature_extractor, tokenizer):
    """Analyze referral document using Qwen2-VL model for medical document analysis"""
    try:
        # Handle PDF files
        if filepath.lower().endswith('.pdf'):
            images = convert_pdf_to_images(filepath)
            if not images:
                raise Exception("Failed to process PDF document")
            image = images[0]  # Process first page for prototype
        else:
            image = Image.open(filepath).convert("RGB")

        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        
        # Generate detailed analysis using Qwen2-VL model
        prompt = """Please analyze this medical document and describe:
        1. Eye pressure measurements
        2. Cup-to-disc ratio
        3. Visual field test results
        4. Any signs of glaucoma damage"""
        
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_new_tokens=200,
                num_beams=4,
                no_repeat_ngram_size=3,
                return_dict_in_generate=True
            ).sequences
            
        description = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Extract numerical values and indicators
        values = extract_numerical_values(description)
        max_pressure = values.get('max_pressure')
        cup_disc_ratio = values.get('cup_disc_ratio')
        
        # Check for field test damage indicators
        field_damage_keywords = ['field defect', 'visual field loss', 'scotoma', 'field test abnormal']
        field_damage_present = any(keyword in description.lower() for keyword in field_damage_keywords)
        
        # Updated urgency determination logic
        is_urgent = (max_pressure is not None and max_pressure > 25 and 
                    (cup_disc_ratio is not None and cup_disc_ratio > 0.7 or field_damage_present))
        
        # Determine if field test is required
        field_test_required = (
            field_damage_present or 
            (cup_disc_ratio is not None and cup_disc_ratio > 0.6) or 
            (max_pressure is not None and max_pressure > 21)
        )
        
        result = {
            'urgency': 'urgent' if is_urgent else 'routine',
            'appointment_type': 'comprehensive assessment' if is_urgent else 'standard assessment',
            'field_test_required': field_test_required,
            'metrics': {
                'max_pressure': max_pressure,
                'cup_disc_ratio': cup_disc_ratio,
                'field_damage_indicated': field_damage_present
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        # Default to urgent in case of errors
        return {
            'urgency': 'urgent',
            'appointment_type': 'comprehensive assessment',
            'field_test_required': True,
            'metrics': {
                'max_pressure': None,
                'cup_disc_ratio': None,
                'field_damage_indicated': None
            }
        }
