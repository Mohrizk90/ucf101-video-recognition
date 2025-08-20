"""
UCF101 dataset implementation for video action recognition.
"""

import os
import json
import random
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# Try to import decord for faster video decoding
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

# Fallback to torchvision.io
import torchvision.io as tvio


class UCF101Dataset(Dataset):
    """
    UCF101 dataset for video action recognition.
    
    This dataset loads video clips and applies temporal sampling and spatial
    transformations. It supports both decord (faster) and torchvision.io
    video loading backends.
    """
    
    def __init__(self, root: str, split: str = "train", clip_len: int = 16, 
                 frame_stride: int = 2, img_size: int = 144, 
                 cache_decoded: bool = False, transform: Optional[transforms.Compose] = None):
        """
        Initialize UCF101 dataset.
        
        Args:
            root: Root directory containing UCF-101 folder
            split: Dataset split ('train', 'val', 'test', or 'split1')
            clip_len: Number of frames per clip
            frame_stride: Stride between sampled frames
            img_size: Size of output images
            cache_decoded: Whether to cache decoded frames in memory
            transform: Optional transforms to apply
        """
        self.root = root
        self.split = split
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.img_size = img_size
        self.cache_decoded = cache_decoded
        self.transform = transform
        
        # Video cache for decoded frames
        self.video_cache = {}
        
        # Setup paths
        self.video_root = os.path.join(root, "UCF-101")
        self.split_root = os.path.join(root, "UCF101TrainTestSplits-RecognitionTask", "ucfTrainTestlist")
        
        # Load class names and create label mapping
        self.class_names = self._load_class_names()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Load video paths and labels
        self.video_paths, self.labels = self._load_split_data()
        
        # Setup transforms
        if self.transform is None:
            self.transform = self._get_default_transforms()
            
        # Print dataset info
        print(f"UCF101 {split} split: {len(self.video_paths)} videos, {len(self.class_names)} classes")
        
    def _load_class_names(self) -> List[str]:
        """
        Load class names from directory structure or split files.
        
        Returns:
            List of class names
        """
        # Try to load from split files first
        class_ind_path = os.path.join(self.split_root, "classInd.txt")
        if os.path.exists(class_ind_path):
            class_names = []
            with open(class_ind_path, 'r') as f:
                for line in f:
                    idx, class_name = line.strip().split()
                    class_names.append(class_name)
            return class_names
        
        # Fallback: scan directory structure
        if os.path.exists(self.video_root):
            class_names = sorted([d for d in os.listdir(self.video_root) 
                                if os.path.isdir(os.path.join(self.video_root, d))])
            return class_names
        
        raise RuntimeError(f"Could not find UCF101 dataset at {self.root}")
    
    def _load_split_data(self) -> Tuple[List[str], List[int]]:
        """
        Load video paths and labels for the specified split.
        
        Returns:
            Tuple of (video_paths, labels)
        """
        video_paths = []
        labels = []
        
        # Try to load from split files
        if self.split == "split1":
            train_list_path = os.path.join(self.split_root, "trainlist01.txt")
            test_list_path = os.path.join(self.split_root, "testlist01.txt")
            
            if os.path.exists(train_list_path) and os.path.exists(test_list_path):
                # Load training data
                with open(train_list_path, 'r') as f:
                    for line in f:
                        video_path, class_idx = line.strip().split()
                        class_name = video_path.split('/')[0]
                        full_path = os.path.join(self.video_root, video_path)
                        if os.path.exists(full_path):
                            video_paths.append(full_path)
                            labels.append(int(class_idx) - 1)  # Convert to 0-indexed
                
                # Load test data
                with open(test_list_path, 'r') as f:
                    for line in f:
                        video_path = line.strip()
                        class_name = video_path.split('/')[0]
                        class_idx = self.class_to_idx[class_name]
                        full_path = os.path.join(self.video_root, video_path)
                        if os.path.exists(full_path):
                            video_paths.append(full_path)
                            labels.append(class_idx)
                
                return video_paths, labels
        
        # Auto-generate split if split files not found
        print("Split files not found, auto-generating 80/20 train/val split...")
        return self._auto_generate_split()
    
    def _auto_generate_split(self) -> Tuple[List[str], List[int]]:
        """
        Auto-generate train/val split deterministically.
        
        Returns:
            Tuple of (video_paths, labels)
        """
        video_paths = []
        labels = []
        
        # Collect all videos
        for class_name in self.class_names:
            class_dir = os.path.join(self.video_root, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_videos = []
            for video_file in os.listdir(class_dir):
                if video_file.endswith(('.avi', '.mp4', '.mkv')):
                    video_path = os.path.join(class_dir, video_file)
                    class_videos.append(video_path)
            
            # Sort for deterministic split
            class_videos.sort()
            
            # Split videos: 80% train, 20% val
            split_idx = int(len(class_videos) * 0.8)
            
            if self.split == "train":
                selected_videos = class_videos[:split_idx]
            elif self.split == "val":
                selected_videos = class_videos[split_idx:]
            else:  # test uses val split
                selected_videos = class_videos[split_idx:]
            
            for video_path in selected_videos:
                video_paths.append(video_path)
                labels.append(self.class_to_idx[class_name])
        
        return video_paths, labels
    
    def _get_default_transforms(self) -> transforms.Compose:
        """
        Get default transforms for the dataset.
        
        Returns:
            Compose transform
        """
        if self.split == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(self.img_size * 1.14)),  # Resize short side
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        Load video frames using available backend.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tensor of frames [T, H, W, C]
        """
        if self.cache_decoded and video_path in self.video_cache:
            return self.video_cache[video_path]
        
        try:
            if DECORD_AVAILABLE:
                frames = self._load_with_decord(video_path)
            else:
                frames = self._load_with_torchvision(video_path)
            
            # Ensure correct format [T, H, W, C]
            if frames.dim() != 4:
                raise ValueError(f"Expected 4D tensor [T, H, W, C], got {frames.shape}")
            
            T, H, W, C = frames.shape
            if C != 3:
                # If we have wrong number of channels, try to reshape
                if frames.numel() == T * H * W * 3:
                    frames = frames.view(T, H, W, 3)
                else:
                    raise ValueError(f"Cannot reshape {frames.shape} to have 3 channels")
            
            if self.cache_decoded:
                self.video_cache[video_path] = frames
                
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return dummy frames if loading fails
            return torch.zeros(self.clip_len, self.img_size, self.img_size, 3)
    
    def _load_with_decord(self, video_path: str) -> torch.Tensor:
        """
        Load video using decord backend.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tensor of frames [T, H, W, C]
        """
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        # Sample frames
        frame_indices = self._sample_frame_indices(total_frames)
        frames = vr.get_batch(frame_indices)
        
        # Convert to torch tensor and ensure correct format
        frames = torch.from_numpy(frames.asnumpy()).float()
        
        # Ensure correct shape [T, H, W, C]
        if frames.dim() != 4:
            raise ValueError(f"Expected 4D tensor from decord, got {frames.shape}")
        
        # Normalize to [0, 1] if needed
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        return frames
    
    def _load_with_torchvision(self, video_path: str) -> torch.Tensor:
        """
        Load video using torchvision backend.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tensor of frames [T, H, W, C]
        """
        video, audio, info = tvio.read_video(video_path)
        
        # Get total frames
        total_frames = video.size(0)
        
        # Sample frames
        frame_indices = self._sample_frame_indices(total_frames)
        frames = video[frame_indices]
        
        # Ensure correct format [T, H, W, C]
        if frames.dim() != 4:
            raise ValueError(f"Expected 4D tensor from torchvision, got {frames.shape}")
        
        # Convert to float and normalize to [0, 1] if needed
        frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        return frames
    
    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """
        Sample frame indices for the clip.
        
        Args:
            total_frames: Total number of frames in video
            
        Returns:
            List of frame indices
        """
        if total_frames <= self.clip_len * self.frame_stride:
            # Video too short, repeat frames or pad
            indices = list(range(total_frames)) * (self.clip_len // total_frames + 1)
            indices = indices[:self.clip_len]
        else:
            # Uniform sampling with random temporal jitter (train) or center sampling (val/test)
            if self.split == "train":
                max_start = total_frames - self.clip_len * self.frame_stride
                start_idx = random.randint(0, max_start)
            else:
                start_idx = (total_frames - self.clip_len * self.frame_stride) // 2
            
            indices = [start_idx + i * self.frame_stride for i in range(self.clip_len)]
        
        return indices
    
    def _apply_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms to video frames.
        
        Args:
            frames: Input frames [T, H, W, C]
            
        Returns:
            Transformed frames [C, T, H, W]
        """
        # Ensure frames are in the correct format [T, H, W, C]
        if frames.dim() != 4:
            raise ValueError(f"Expected 4D tensor [T, H, W, C], got {frames.shape}")
        
        T, H, W, C = frames.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")
        
        # Apply transforms to each frame
        transformed_frames = []
        for i in range(T):
            frame = frames[i]  # [H, W, C]
            
            # Ensure frame is in the correct range [0, 1] and format
            if frame.max() > 1.0:
                frame = frame / 255.0
            
            # Convert to PIL Image for transforms
            import torchvision.transforms.functional as F
            from PIL import Image
            import numpy as np
            
            # Convert tensor to numpy array, then to PIL Image
            frame_np = frame.numpy()
            frame_pil = Image.fromarray((frame_np * 255).astype(np.uint8))
            
            frame = self.transform(frame_pil)  # [C, H, W]
            transformed_frames.append(frame)
        
        # Stack frames and rearrange to [C, T, H, W]
        frames_tensor = torch.stack(transformed_frames, dim=1)  # [C, T, H, W]
        
        return frames_tensor
    
    def __len__(self) -> int:
        """Return number of videos in dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get video clip and label.
        
        Args:
            idx: Index of video
            
        Returns:
            Tuple of (video_clip, label)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        
        # Apply transforms
        video_clip = self._apply_transforms(frames)
        
        return video_clip, label
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.class_names.copy()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Returns:
            Tensor of class weights
        """
        class_counts = torch.zeros(len(self.class_names))
        for label in self.labels:
            class_counts[label] += 1
        
        # Calculate inverse frequency weights
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum()
        
        return weights


def create_ucf101_dataloaders(config: Dict[str, Any], batch_size: int = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create UCF101 dataloaders for training and validation.
    
    Args:
        config: Configuration dictionary
        batch_size: Override batch size from config
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    from torch.utils.data import DataLoader
    
    # Override batch size if specified
    if batch_size is not None:
        config['train']['batch_size'] = batch_size
    
    # Create datasets
    train_dataset = UCF101Dataset(
        root=config['data']['root'],
        split='train',
        clip_len=config['data']['clip_len'],
        frame_stride=config['data']['frame_stride'],
        img_size=config['data']['img_size'],
        cache_decoded=config['data']['cache_decoded']
    )
    
    val_dataset = UCF101Dataset(
        root=config['data']['root'],
        split='val',
        clip_len=config['data']['clip_len'],
        frame_stride=config['data']['frame_stride'],
        img_size=config['data']['img_size'],
        cache_decoded=config['data']['cache_decoded']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['system']['num_workers'] > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['system']['num_workers'] > 0 else False,
        drop_last=False
    )
    
    return train_loader, val_loader 