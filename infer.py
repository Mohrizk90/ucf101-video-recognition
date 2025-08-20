#!/usr/bin/env python3
"""
Inference script for UCF101 CNN-RNN video action recognition.

This script loads a trained model and performs inference on single videos
or directories of videos, with optional visualization output.
"""

import os
import argparse
import torch
import torch.nn as nn
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional

from utils.common import set_seed, load_config, get_device
from utils.logger import PrettyPrinter
from models.cnn_rnn import create_model
from datasets.ucf101 import UCF101Dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference with UCF101 CNN-RNN model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file or directory')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                       help='Output directory for results')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video with predictions')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device):
    """Load model checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_acc' in checkpoint:
            print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("Model state dict loaded directly")
    
    return checkpoint


def create_inference_dataset(config, video_path: str):
    """Create dataset for inference."""
    # Create a temporary dataset to get transforms and class names
    temp_dataset = UCF101Dataset(
        root=config['data']['root'],
        split='val',  # Use validation transforms
        clip_len=config['data']['clip_len'],
        frame_stride=config['data']['frame_stride'],
        img_size=config['data']['img_size'],
        cache_decoded=False
    )
    
    return temp_dataset


def load_video_frames(video_path: str, config: dict) -> torch.Tensor:
    """
    Load video frames for inference.
    
    Args:
        video_path: Path to video file
        config: Configuration dictionary
        
    Returns:
        Tensor of frames [C, T, H, W]
    """
    # Try to import decord for faster video decoding
    try:
        import decord
        DECORD_AVAILABLE = True
    except ImportError:
        DECORD_AVAILABLE = False
    
    try:
        if DECORD_AVAILABLE:
            # Use decord
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            
            # Center sampling for inference
            frame_indices = []
            if total_frames <= config['data']['clip_len'] * config['data']['frame_stride']:
                # Video too short, repeat frames
                indices = list(range(total_frames)) * (config['data']['clip_len'] // total_frames + 1)
                frame_indices = indices[:config['data']['clip_len']]
            else:
                # Center sampling
                start_idx = (total_frames - config['data']['clip_len'] * config['data']['frame_stride']) // 2
                frame_indices = [start_idx + i * config['data']['frame_stride'] 
                               for i in range(config['data']['clip_len'])]
            
            frames = vr.get_batch(frame_indices)
            frames = torch.from_numpy(frames.asnumpy()).float() / 255.0
            
        else:
            # Use torchvision.io as fallback
            import torchvision.io as tvio
            video, audio, info = tvio.read_video(video_path)
            
            total_frames = video.size(0)
            
            # Center sampling for inference
            if total_frames <= config['data']['clip_len'] * config['data']['frame_stride']:
                indices = list(range(total_frames)) * (config['data']['clip_len'] // total_frames + 1)
                frame_indices = indices[:config['data']['clip_len']]
            else:
                start_idx = (total_frames - config['data']['clip_len'] * config['data']['frame_stride']) // 2
                frame_indices = [start_idx + i * config['data']['frame_stride'] 
                               for i in range(config['data']['clip_len'])]
            
            frames = video[frame_indices]
            frames = frames.float() / 255.0
        
        return frames
        
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        raise


def preprocess_frames(frames: torch.Tensor, dataset) -> torch.Tensor:
    """
    Preprocess frames using dataset transforms.
    
    Args:
        frames: Input frames [T, H, W, C]
        dataset: Dataset instance for transforms
        
    Returns:
        Preprocessed frames [C, T, H, W]
    """
    # Apply transforms to each frame
    transformed_frames = []
    for i in range(frames.size(0)):
        frame = frames[i]  # [H, W, C]
        frame = dataset.transform(frame)  # [C, H, W]
        transformed_frames.append(frame)
    
    # Stack frames and rearrange to [C, T, H, W]
    frames_tensor = torch.stack(transformed_frames, dim=1)
    
    return frames_tensor


def predict_video(model: nn.Module, video_path: str, config: dict, 
                 dataset, device: torch.device, top_k: int = 5) -> Tuple[List[int], List[float]]:
    """
    Predict action class for a single video.
    
    Args:
        model: Trained model
        video_path: Path to video file
        config: Configuration dictionary
        dataset: Dataset instance for transforms
        device: Device to run inference on
        top_k: Number of top predictions to return
        
    Returns:
        Tuple of (top_k_indices, top_k_probabilities)
    """
    model.eval()
    
    # Load and preprocess video
    frames = load_video_frames(video_path, config)
    frames = preprocess_frames(frames, dataset)
    
    # Add batch dimension
    frames = frames.unsqueeze(0).to(device)  # [1, C, T, H, W]
    
    # Inference
    with torch.no_grad():
        outputs = model(frames)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=1)
    
    return top_k_indices[0].cpu().numpy(), top_k_probs[0].cpu().numpy()


def create_output_video(video_path: str, predictions: List[Tuple[str, float]], 
                       output_path: str, config: dict):
    """
    Create output video with overlaid predictions.
    
    Args:
        video_path: Path to input video
        predictions: List of (class_name, probability) tuples
        output_path: Path to save output video
        config: Configuration dictionary
    """
    try:
        import decord
        DECORD_AVAILABLE = True
    except ImportError:
        DECORD_AVAILABLE = False
    
    if DECORD_AVAILABLE:
        # Use decord for reading
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        # Get video dimensions
        sample_frame = vr[0]
        height, width = sample_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        for i in range(min(total_frames, 100)):  # Limit to first 100 frames for demo
            frame = vr[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add prediction text
            y_offset = 30
            for j, (class_name, prob) in enumerate(predictions[:3]):  # Show top 3
                text = f"{j+1}. {class_name}: {prob:.3f}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                y_offset += 30
            
            out.write(frame)
        
        out.release()
        print(f"Output video saved to: {output_path}")
        
    else:
        print("Decord not available, skipping video output creation")


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Print configuration
    PrettyPrinter.print_header("UCF101 CNN-RNN Inference")
    PrettyPrinter.print_data_info(config)
    
    # Setup environment
    set_seed(config['system']['seed'])
    device = get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.ckpt, model, device)
    
    # Create dataset for transforms and class names
    dataset = create_inference_dataset(config, args.video)
    class_names = dataset.get_class_names()
    
    # Process video(s)
    video_path = Path(args.video)
    
    if video_path.is_file():
        # Single video
        print(f"\nProcessing video: {video_path}")
        
        try:
            # Get predictions
            top_k_indices, top_k_probs = predict_video(
                model, str(video_path), config, dataset, device, args.top_k
            )
            
            # Print results
            print(f"\nTop-{args.top_k} predictions:")
            for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
                class_name = class_names[idx]
                print(f"  {i+1}. {class_name}: {prob:.4f}")
            
            # Save results
            results_file = os.path.join(args.output_dir, f"{video_path.stem}_predictions.txt")
            with open(results_file, 'w') as f:
                f.write(f"Predictions for {video_path.name}\n")
                f.write("=" * 50 + "\n\n")
                for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
                    class_name = class_names[idx]
                    f.write(f"{i+1}. {class_name}: {prob:.4f}\n")
            
            # Create output video if requested
            if args.save_video:
                predictions = [(class_names[idx], prob) for idx, prob in zip(top_k_indices, top_k_probs)]
                output_video_path = os.path.join(args.output_dir, f"{video_path.stem}_output.mp4")
                create_output_video(str(video_path), predictions, output_video_path, config)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
    
    elif video_path.is_dir():
        # Directory of videos
        print(f"\nProcessing videos in directory: {video_path}")
        
        video_files = list(video_path.glob("*.avi")) + list(video_path.glob("*.mp4")) + list(video_path.glob("*.mkv"))
        
        if not video_files:
            print("No video files found in directory")
            return
        
        print(f"Found {len(video_files)} video files")
        
        all_results = []
        
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            
            try:
                # Get predictions
                top_k_indices, top_k_probs = predict_video(
                    model, str(video_file), config, dataset, device, args.top_k
                )
                
                # Store results
                result = {
                    'video': video_file.name,
                    'predictions': [(class_names[idx], prob) for idx, prob in zip(top_k_indices, top_k_probs)]
                }
                all_results.append(result)
                
                # Print top prediction
                top_class, top_prob = result['predictions'][0]
                print(f"  Top prediction: {top_class} ({top_prob:.4f})")
                
            except Exception as e:
                print(f"  Error processing {video_file.name}: {e}")
        
        # Save all results
        results_file = os.path.join(args.output_dir, "batch_predictions.txt")
        with open(results_file, 'w') as f:
            f.write(f"Batch predictions for {len(all_results)} videos\n")
            f.write("=" * 50 + "\n\n")
            
            for result in all_results:
                f.write(f"\n{result['video']}:\n")
                for i, (class_name, prob) in enumerate(result['predictions']):
                    f.write(f"  {i+1}. {class_name}: {prob:.4f}\n")
        
        print(f"\nBatch results saved to: {results_file}")
    
    else:
        print(f"Error: {args.video} is not a valid file or directory")
        return
    
    print(f"\nInference completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 