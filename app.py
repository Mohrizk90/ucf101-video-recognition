"""
Simple Flask web UI for UCF101 video classification.
"""

import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_rnn import CNNRNN
from datasets.ucf101 import UCF101Dataset
from utils.common import load_config, set_seed
from utils.engine import Evaluator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables
model = None
class_names = None
config = None
device = None

def load_model():
    """Load the trained model and class names."""
    global model, class_names, config, device
    
    print("Loading model...")
    
    # Load config
    config = load_config('config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class names by creating a temporary dataset
    temp_dataset = UCF101Dataset(
        root=config['data']['root'],
        split='val',  # Doesn't matter for getting class names
        clip_len=config['data']['clip_len'],
        frame_stride=config['data']['frame_stride']
    )
    class_names = temp_dataset.get_class_names()
    
    # Create model
    model = CNNRNN(
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['backbone'],
        finetune_layers=config['model']['finetune_layers'],
        lstm_hidden=config['model']['lstm_hidden'],
        lstm_layers=config['model']['lstm_layers'],
        bidirectional=config['model']['bidirectional'],
        dropout=config['model']['dropout']
    )
    
    # Load trained weights
    checkpoint_path = 'runs/ucf101_cnn_rnn_20250817_145949/best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Check which key contains the model state dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Warning: No model state dict found in checkpoint keys: {list(checkpoint.keys())}")
            print("Using untrained model weights")
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using untrained model weights")
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {len(class_names)} classes")

def allowed_file(filename):
    """Check if file extension is allowed."""
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_video(video_path):
    """Process video and return classification results."""
    try:
        print(f"Processing video: {video_path}")
        
        # Create temporary dataset for single video
        temp_dataset = UCF101Dataset(
            root=config['data']['root'],
            split='val',  # Doesn't matter for inference
            clip_len=config['data']['clip_len'],
            frame_stride=config['data']['frame_stride'],
            img_size=config['data']['img_size']
        )
        
        print(f"Dataset created with {len(temp_dataset.class_names)} classes")
        
        # Load video frames using the same method as dataset
        print("Loading video frames...")
        frames = temp_dataset._load_video_frames(video_path)
        print(f"Frames loaded: {frames.shape}")
        
        # Apply transforms
        print("Applying transforms...")
        frames = temp_dataset._apply_transforms(frames)
        print(f"After transforms: {frames.shape}")
        
        # Add batch dimension
        frames = frames.unsqueeze(0)  # [1, C, T, H, W]
        frames = frames.to(device)
        print(f"Final input shape: {frames.shape}")
        
        # Inference
        print("Running inference...")
        with torch.no_grad():
            outputs = model(frames)
            probabilities = torch.softmax(outputs, dim=1)
        print(f"Model output shape: {outputs.shape}")
        
        # Get top-5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
        
        results = []
        for i in range(5):
            class_idx = top5_indices[0][i].item()
            probability = top5_prob[0][i].item()
            class_name = temp_dataset.class_names[class_idx]
            results.append({
                'class': class_name,
                'probability': f"{probability:.3f}",
                'percentage': f"{probability * 100:.1f}%"
            })
        
        print(f"Classification completed successfully: {results[0]['class']} ({results[0]['percentage']})")
        return results, None
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and classification."""
    print("=== UPLOAD REQUEST RECEIVED ===")
    
    if 'video' not in request.files:
        print("No video file in request")
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        print(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, WMV, or FLV'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Process video
        print("Calling process_video...")
        results, error = process_video(filepath)
        print(f"process_video returned: results={results}, error={error}")
        
        if error:
            print(f"Error from process_video: {error}")
            return jsonify({'error': f'Error processing video: {error}'}), 500
        
        # Clean up uploaded file
        os.remove(filepath)
        print(f"Cleaned up file: {filepath}")
        
        response_data = {
            'success': True,
            'filename': filename,
            'results': results
        }
        print(f"Sending response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Exception in upload_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': len(class_names) if class_names else 0,
        'device': str(device) if device else 'unknown'
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 