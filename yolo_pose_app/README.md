# YOLO 11 Pose Detection for Cycling Analysis

This project provides a complete YOLO 11-based pose detection system for cycling biomechanics analysis, similar to the MediaPipe-based `app.py` but using YOLO 11 for more accurate pose estimation.

## 📋 Overview

The system consists of two main components:
1. **Training Script** (`training.py`) - Creates and trains custom YOLO 11 pose models
2. **Flask API** (`yolo_app.py`) - Serves pose detection via REST API

## 🎯 Features

- **YOLO 11 Pose Detection**: State-of-the-art pose estimation
- **Custom Training**: Train on your own cycling datasets
- **Bike Fit Analysis**: Calculate angles and body measurements
- **Video Processing**: Process cycling videos with pose overlay
- **H.264 Video Output**: Browser-compatible video encoding
- **RESTful API**: Easy integration with web applications
- **EC2 Deployment**: Ready-to-deploy on AWS EC2

## 📁 Project Structure

```
yolo_pose_app/
├── training.py              # YOLO 11 training script
├── yolo_app.py             # Flask API server
├── requirements_yolo.txt    # Python dependencies
├── deploy_yolo.sh          # EC2 deployment script
├── README.md               # This file
├── models/                 # Trained models directory
├── yolo_training_data/     # Training data structure
│   ├── images/
│   │   ├── train/          # Training images
│   │   └── val/            # Validation images
│   └── labels/
│       ├── train/          # Training labels (YOLO format)
│       └── val/            # Validation labels
└── output_videos/          # Processed video outputs
```

## 🚀 Quick Start

### Local Development

1. **Clone and setup:**
```bash
cd yolo_pose_app
pip install -r requirements_yolo.txt
```

2. **Download pre-trained model:**
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

3. **Run training (optional):**
```bash
python3 training.py
```

4. **Start Flask API:**
```bash
python3 yolo_app.py
```

5. **Test the API:**
```bash
curl http://localhost:5001/health
```

### EC2 Deployment

1. **Upload files to EC2:**
```bash
# On your local machine
scp -r yolo_pose_app/ ubuntu@your-ec2-ip:~/
```

2. **SSH to EC2 and deploy:**
```bash
ssh ubuntu@your-ec2-ip
cd yolo_pose_app
chmod +x deploy_yolo.sh
./deploy_yolo.sh
```

3. **Access your API:**
- Direct: `http://your-ec2-ip:5001`
- Domain: `http://yolo-pose.grity.co` (after DNS setup)

## 🎓 Training Custom Models

### 1. Using COCO-Pose Dataset (Recommended)

The training script uses the **COCO-Pose dataset** - the gold standard for human pose estimation with 200K+ labeled images:

```bash
python3 training.py
```

When prompted, type 'y' to start training. The script will:
- Download COCO-Pose dataset (~20GB)
- Train YOLO 11 on 57K training images
- Validate on 5K validation images  
- Save best model to `runs/pose/yolo11_coco_pose/weights/best.pt`

### 2. Manual Training

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolo11n-pose.pt')

# Train on COCO-Pose dataset
results = model.train(
    data='coco-pose.yaml',  # Built-in COCO-Pose configuration
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)
```

### 3. Dataset Format for Custom Data

If you have cycling-specific data, format it like this:

```
yolo_training_data/
├── images/
│   ├── train/           # Training images
│   └── val/             # Validation images
└── labels/
    ├── train/           # YOLO format labels
    └── val/             # YOLO format labels
```

**YOLO Label Format** (17 keypoints):
```
# Format: class x_center y_center width height px1 py1 visible1 px2 py2 visible2 ... px17 py17 visible17
0 0.5 0.3 0.8 0.9 0.45 0.1 2 0.48 0.09 2 0.52 0.09 2 ...
```

### 4. Custom Dataset Training

```python
# Create your dataset YAML
custom_config = {
    'path': 'path/to/your/dataset',
    'train': 'images/train',
    'val': 'images/val',
    'names': {0: 'person'},
    'nc': 1,
    'kpt_shape': [17, 3],  # 17 keypoints, 3 values each (x, y, visibility)
    'flip_idx': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
}

# Train on your data
model = YOLO('yolo11n-pose.pt')
results = model.train(data='your_dataset.yaml', epochs=100)
```

## 🔧 API Usage

### Process Video Endpoint

**POST** `/process-video`

**Parameters:**
- `video`: Video file (mp4, avi, mov, webm)
- `user_height_cm`: User height in centimeters (default: 175)
- `quality`: Video quality 1-100 (default: 75)

**Example using curl:**
```bash
curl -X POST http://localhost:5001/process-video \
  -F "video=@cycling_video.mp4" \
  -F "user_height_cm=180" \
  -F "quality=85"
```

**Response:**
```json
{
  "max_angles": {
    "shoulder_hip_knee": 165,
    "hip_knee_ankle": 148,
    "shoulder_elbow_wrist": 155,
    "elbow_shoulder_hip": 85
  },
  "min_angles": {
    "shoulder_hip_knee": 95,
    "hip_knee_ankle": 130,
    "shoulder_elbow_wrist": 90,
    "elbow_shoulder_hip": 65
  },
  "body_lengths_cm": {
    "torso_length": 62.5,
    "femur_length": 48.6,
    "lower_leg_length": 45.0,
    "upper_arm_length": 27.0,
    "forearm_length": 21.6,
    "measurement_method": "calculated from YOLO keypoints"
  },
  "recommendations": {
    "general": [
      {
        "component": "SADDLE HEIGHT",
        "issue": "Good position",
        "action": "No change needed",
        "current": "148° knee extension",
        "target": "140-150°",
        "priority": "low"
      }
    ]
  },
  "video": "base64_encoded_video_data...",
  "model_type": "YOLO11"
}
```

### Health Check

**GET** `/health`

```bash
curl http://localhost:5001/health
```

## 🛠️ Management Commands (EC2)

After deployment, use these commands:

```bash
# Check service status
pm2 status

# View logs
pm2 logs yolo-pose-app

# Restart service
pm2 restart yolo-pose-app

# Stop service
pm2 stop yolo-pose-app

# Monitor resources
pm2 monit

# Train new model
python3 training.py
```

## 🔗 Domain Setup

1. **Point your subdomain to EC2:**
```bash
# DNS A Record
yolo-pose.grity.co → your-ec2-public-ip
```

2. **Setup SSL (optional):**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yolo-pose.grity.co
```

## 📊 Model Comparison

| Feature | MediaPipe (app.py) | YOLO 11 (yolo_app.py) |
|---------|-------------------|----------------------|
| **Accuracy** | Good | Excellent |
| **Speed** | Very Fast | Fast |
| **Training** | Pre-trained only | Custom training |
| **Robustness** | Good | Excellent |
| **Memory Usage** | Low | Medium |
| **Model Size** | Small | Medium |

## 🐛 Troubleshooting

### Common Issues

1. **YOLO model not loading:**
```bash
# Check if model exists
ls -la models/
# Reinstall ultralytics
pip install --upgrade ultralytics
```

2. **GPU not detected:**
```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
# Install CUDA version of PyTorch if needed
```

3. **FFmpeg H.264 issues:**
```bash
# Check FFmpeg codecs
ffmpeg -codecs | grep h264
# Reinstall FFmpeg if needed
sudo apt install --reinstall ffmpeg
```

4. **Memory issues during training:**
```bash
# Reduce batch size in training.py
'batch': 8,  # Instead of 16
```

### Performance Optimization

1. **For GPU acceleration:**
```bash
# Install CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. **For faster inference:**
```bash
# Use smaller model
MODEL_PATH = "yolo11s-pose.pt"  # Instead of yolo11n-pose.pt
```

## 📝 License

This project is part of the cycling pose analysis system. Use responsibly for biomechanics research and bike fitting applications.

## 🤝 Contributing

1. Collect and annotate cycling pose datasets
2. Improve training scripts and model architecture
3. Add more sophisticated bike fit recommendations
4. Optimize performance for real-time analysis

## 📞 Support

For deployment issues or custom training requirements, check the logs:
```bash
# Application logs
pm2 logs yolo-pose-app

# System logs
tail -f /var/log/nginx/error.log
``` 