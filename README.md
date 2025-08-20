# 🎬 UCF101 Video Action Recognition - Web Application

A modern web interface for video action recognition using a pre-trained CNN-RNN model on the UCF101 dataset. Upload videos and get instant AI-powered action classification results.

## ✨ Features

- 🎥 **Easy Video Upload**: Drag & drop or click to browse
- 🤖 **AI-Powered Analysis**: Pre-trained model with 58.39% accuracy
- 📊 **Real-time Results**: Top-5 predictions with confidence scores
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- ⚡ **Fast Processing**: Optimized inference pipeline
- 🎯 **101 Action Classes**: Covers sports, music, fitness, and daily activities

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ucf101_cnn_rnn
pip install -r requirements.txt
```

### 2. Start the Web Application
```bash
python app.py
```

### 3. Open Your Browser
Navigate to `http://localhost:5000`

### 4. Upload a Video
- Drag & drop a video file or click to browse
- Supported formats: MP4, AVI, MOV, MKV, WMV, FLV
- Maximum file size: 100MB

## 📁 Project Structure

```
ucf101_cnn_rnn/
├── app.py                   # Main Flask web application
├── config.yaml              # Model configuration
├── requirements.txt          # Python dependencies
├── models/
│   └── cnn_rnn.py          # CNN-RNN model architecture
├── datasets/
│   └── ucf101.py           # Dataset handling
├── utils/                   # Utility functions
├── templates/
│   └── index.html          # Web interface
├── runs/                    # Pre-trained model weights
└── data/                    # UCF101 dataset (external)
```

## 🎯 How It Works

### 1. **Video Processing**
- Extracts 16 frames from the uploaded video
- Resizes frames to 144x144 pixels
- Applies normalization and augmentation

### 2. **AI Analysis**
- **Spatial Features**: ResNet-18 extracts features from each frame
- **Temporal Modeling**: Bidirectional LSTM processes temporal sequences
- **Classification**: Softmax output gives top-5 action predictions

### 3. **Results Display**
- Shows predictions ranked by confidence
- Displays probability scores and percentages
- Provides clean, intuitive interface

## 🏆 Model Performance

- **Overall Accuracy**: 58.39%
- **Top-5 Accuracy**: 79.64%
- **Best Classes**: Swimming, Musical Instruments, Diving
- **Training Time**: 6.7 hours on GTX 1660 Ti

### High-Performing Actions (90%+ Accuracy)
- 🏊‍♂️ **BreastStroke**: 100%
- 🥊 **Punch**: 100%
- 🪂 **SkyDiving**: 100%
- 🎸 **PlayingGuitar**: 91%
- 🎻 **PlayingViolin**: 90%

## 💻 System Requirements

- **Python**: 3.8+
- **RAM**: 4GB+ (8GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional (CUDA for faster inference)

## 📦 Dependencies

```
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.8.0
flask>=2.3.0
numpy>=1.24.0
pillow>=9.0.0
```

## 🔧 Configuration

The `config.yaml` file contains model parameters:

```yaml
model:
  backbone: "resnet18"
  lstm_hidden: 256
  lstm_layers: 1
  bidirectional: true
  dropout: 0.3
  num_classes: 101

data:
  clip_len: 16
  frame_stride: 2
  img_size: 144
```

## 📱 Web Interface

### Upload Section
- Drag & drop area with visual feedback
- File type validation
- Size limit enforcement (100MB)

### Processing
- Progress bar with percentage
- Loading spinner during AI analysis
- Real-time status updates

### Results Display
- Top-5 predictions with rankings
- Confidence scores and percentages
- Success message with filename
- Clean, card-based layout

## 🎬 Supported Action Categories

### 🏃‍♂️ Sports & Athletics
- Basketball, Soccer, Tennis, Golf
- Swimming, Diving, Skiing
- Gymnastics, Weightlifting

### 🎵 Music & Arts
- Playing instruments (Guitar, Piano, Violin)
- Dancing, Singing, Painting

### 🏋️‍♂️ Fitness & Exercise
- Push-ups, Pull-ups, Squats
- Yoga, Tai Chi, Martial Arts

### 🏠 Daily Activities
- Cooking, Cleaning, Typing
- Personal care, Work activities

## 🚨 Troubleshooting

### Common Issues

**Model not loading**
- Check if `runs/ucf101_cnn_rnn_20250817_145949/best.pth` exists
- Verify PyTorch installation

**Video upload fails**
- Check file format (MP4, AVI, MOV, MKV, WMV, FLV)
- Ensure file size < 100MB
- Check browser console for errors

**Slow processing**
- Use smaller video files
- Check system resources
- Consider GPU acceleration

### Error Messages

- **"Invalid file type"**: Use supported video formats
- **"File too large"**: Reduce video file size
- **"Processing error"**: Check video file integrity

## 🔮 Future Enhancements

- **Batch Processing**: Upload multiple videos
- **Video Preview**: Show uploaded video before processing
- **Export Results**: Download results as CSV/JSON
- **Model Comparison**: Test different architectures
- **Real-time Processing**: Stream video analysis

## 📚 Technical Details

### Architecture
- **Backbone**: ResNet-18 (ImageNet pre-trained)
- **Temporal**: Bidirectional LSTM (256 hidden units)
- **Input**: 16 frames × 144×144 × 3 channels
- **Output**: 101-class probability distribution

### Training
- **Dataset**: UCF101 (13,320 videos, 101 classes)
- **Optimizer**: Adam with cosine annealing
- **Loss**: CrossEntropy with label smoothing
- **Augmentation**: Random crop, flip, color jitter

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **UCF101 Dataset**: University of Central Florida
- **PyTorch**: Facebook AI Research
- **Flask**: Pallets Project
- **OpenCV**: Intel Corporation

---

**Ready to classify videos?** 🚀

```bash
python app.py
```

Then open `http://localhost:5000` and start uploading videos! 