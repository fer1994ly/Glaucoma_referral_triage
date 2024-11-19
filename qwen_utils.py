import os
from PIL import Image
import torch
from datetime import datetime

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
    return image_inputs, video_inputs

def analyze_referral(filepath, model, feature_extractor, tokenizer):
    """Analyze referral document using Vision-Language model"""
    try:
        # Open and preprocess image
        image = Image.open(filepath).convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        
        # Generate description
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                return_dict_in_generate=True
            ).sequences
            
        description = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Simple rule-based analysis for prototype
        description = description.lower()
        urgent_keywords = ['urgent', 'emergency', 'immediate', 'severe']
        comprehensive_keywords = ['detailed', 'comprehensive', 'complete', 'thorough']
        field_test_keywords = ['field', 'vision test', 'peripheral']
        
        result = {
            'urgency': 'urgent' if any(keyword in description for keyword in urgent_keywords) else 'routine',
            'appointment_type': 'comprehensive assessment' if any(keyword in description for keyword in comprehensive_keywords) else 'standard assessment',
            'field_test_required': any(keyword in description for keyword in field_test_keywords)
        }
        
        return result
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        # Default to urgent in case of errors
        return {
            'urgency': 'urgent',
            'appointment_type': 'comprehensive assessment',
            'field_test_required': True
        }
