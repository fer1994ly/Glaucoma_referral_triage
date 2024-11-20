import os
from PIL import Image
import torch
from datetime import datetime

import re
import logging
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_vision_info(messages):
    """Process vision information from messages"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image":
                image = Image.open(content["image"]).convert("RGB")
                image_inputs.append(image)
            elif content["type"] == "video":
                video_inputs.append(content["video"])
    
    return image_inputs, video_inputs



def extract_numerical_values(analysis):
    """Extract numerical values from analysis text"""
    values = {
        'max_pressure': None,
        'cup_disc_ratio': None,
        'field_damage_indicated': False
    }
    
    # Extract IOP (pressure)
    pressure_match = re.search(r'(\d+(?:\.\d+)?)\s*mm?Hg', analysis, re.IGNORECASE)
    if pressure_match:
        values['max_pressure'] = float(pressure_match.group(1))
    
    # Extract cup-to-disc ratio
    ratio_match = re.search(r'(?:cup.?to.?disc|c/?d).?ratio.*?(\d+(?:\.\d+)?)', analysis, re.IGNORECASE)
    if ratio_match:
        values['cup_disc_ratio'] = float(ratio_match.group(1))
    
    # Check for field damage indicators
    field_damage_terms = ['field defect', 'visual field loss', 'scotoma']
    values['field_damage_indicated'] = any(term in analysis.lower() for term in field_damage_terms)
    
    return values

def evaluate_urgency(values, analysis):
    """Evaluate if the case is urgent based on values and analysis"""
    # Check numerical values
    max_pressure = values.get('max_pressure')
    cup_disc_ratio = values.get('cup_disc_ratio')
    
    # Critical value thresholds
    if max_pressure and max_pressure > 25:
        return True
    if cup_disc_ratio and cup_disc_ratio > 0.7:
        return True
    
    # Check for concerning keywords in analysis
    urgent_indicators = [
        'severe', 'advanced', 'immediate', 'urgent',
        'significant loss', 'marked damage', 'hemorrhage'
    ]
    return any(indicator in analysis.lower() for indicator in urgent_indicators)

def evaluate_field_test_requirement(values, analysis):
    """Evaluate if visual field testing is required"""
    max_pressure = values.get('max_pressure')
    cup_disc_ratio = values.get('cup_disc_ratio')
    
    # Criteria for requiring field test
    if max_pressure and max_pressure > 21:
        return True
    if cup_disc_ratio and cup_disc_ratio > 0.6:
        return True
    
    # Check for indicators in analysis
    field_test_indicators = [
        'visual field', 'field test', 'perimetry',
        'scotoma', 'field loss', 'field defect'
    ]
    return any(indicator in analysis.lower() for indicator in field_test_indicators)

def analyze_referral(filepath):
    """Analyze referral document using Qwen2-VL model for medical document analysis"""
    try:
        # Initialize Qwen2-VL model
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        adapter_path = "sergiopaniego/qwen2-7b-instruct-trl-sft-ChartQA"
        model.load_adapter(adapter_path)
        processor = Qwen2VLProcessor.from_pretrained(model_id)
        
        # Open and process image
        image = Image.open(filepath)
        if image.format == 'TIFF':
            # Handle TIFF-specific processing
            if getattr(image, 'n_frames', 1) > 1:
                logger.info(f"Multi-page TIFF detected with {image.n_frames} frames, using first frame")
                image.seek(0)
            
            # Convert high bit-depth images
            if image.mode in ['I;16', 'I']:
                image = image.point(lambda i: i * (255/65535)).convert('L')
        
        image = image.convert("RGB")
        
        # Enhanced medical analysis prompt
        prompt = """Analyze this ophthalmological image in detail and provide:
        1. Precise intraocular pressure (IOP) measurements in mmHg if visible
        2. Detailed cup-to-disc ratio assessment with specific values
        3. Any visual field test abnormalities or defects
        4. Specific signs of glaucomatous damage (disc hemorrhage, RNFL defects)
        5. Evaluation of optic nerve head appearance
        6. Assessment of retinal nerve fiber layer
        7. Overall glaucoma risk assessment
        Please provide numerical values where applicable and note any concerning findings."""
        
        # Process with Qwen2-VL
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": filepath},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate analysis
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            analysis = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        # Extract values and determine urgency
        values = extract_numerical_values(analysis)
        is_urgent = evaluate_urgency(values, analysis)
        
        return {
            'urgency': 'urgent' if is_urgent else 'routine',
            'appointment_type': 'comprehensive assessment' if is_urgent else 'standard assessment',
            'field_test_required': evaluate_field_test_requirement(values, analysis),
            'analysis': analysis,
            'metrics': values
        }
    
    except Exception as e:
        logger.error(f"Error during Qwen2-VL analysis: {str(e)}")
        if isinstance(e, OSError):
            if "compression" in str(e).lower():
                raise OSError("Unsupported TIFF compression. Please use uncompressed or standard compression.")
            elif "truncated" in str(e).lower():
                raise OSError("Truncated or corrupted TIFF file detected.")
        
        # Default to urgent for safety
        return {
            'urgency': 'urgent',
            'appointment_type': 'comprehensive assessment',
            'field_test_required': True,
            'error': str(e),
            'metrics': {
                'max_pressure': None,
                'cup_disc_ratio': None,
                'field_damage_indicated': None
            }
        }