# YOLO 11 Pose Training with COCO-Pose Dataset

## 🎯 Overview

This guide shows you how to train a YOLO 11 pose estimation model using the **COCO-Pose dataset** - the gold standard for human pose estimation. The COCO-Pose dataset contains over 200,000 images with 17 keypoint annotations per person.

## 📊 COCO-Pose Dataset Details

### Dataset Statistics
- **Total Images**: 200K+ labeled images
- **Training Set**: 57K images (train2017)
- **Validation Set**: 5K images (val2017)  
- **Test Set**: 41K images (test2017)
- **Classes**: 1 (person)
- **Keypoints**: 17 per person
- **Dataset Size**: ~20GB download

### 17 COCO Keypoints
1. **Nose** (0)
2. **Left Eye** (1)
3. **Right Eye** (2)
4. **Left Ear** (3)
5. **Right Ear** (4)
6. **Left Shoulder** (5)
7. **Right Shoulder** (6)
8. **Left Elbow** (7)
9. **Right Elbow** (8)
10. **Left Wrist** (9)
11. **Right Wrist** (10)
12. **Left Hip** (11)
13. **Right Hip** (12)
14. **Left Knee** (13)
15. **Right Knee** (14)
16. **Left Ankle** (15)
17. **Right Ankle** (16)

## 🚀 Quick Start Training

### 1. Basic Training Command

```bash
# Train YOLO11n-pose on COCO-Pose dataset
python3 training.py
```

When prompted, type 'y' to start training with COCO-Pose dataset.

### 2. Manual Training (Advanced)

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolo11n-pose.pt')

# Train on COCO-Pose
results = model.train(
    data='coco-pose.yaml',  # Built-in COCO-Pose config
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'  # or 'cpu'
)
```

### 3. CLI Training

```bash
# Using CLI directly
yolo pose train data=coco-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
```

## ⚙️ Training Configuration

### Model Options
```python
# Different YOLO11 pose model sizes
models = {
    'yolo11n-pose.pt': 'Nano - Fastest (2.9M params)',
    'yolo11s-pose.pt': 'Small - Balanced (9.9M params)', 
    'yolo11m-pose.pt': 'Medium - Good accuracy (20.9M params)',
    'yolo11l-pose.pt': 'Large - High accuracy (26.2M params)',
    'yolo11x-pose.pt': 'Extra Large - Best accuracy (58.8M params)'
}
```

### Training Parameters
```python
training_config = {
    'data': 'coco-pose.yaml',       # COCO-Pose dataset
    'epochs': 100,                  # Training epochs
    'imgsz': 640,                   # Image size
    'batch': 16,                    # Batch size (adjust for GPU memory)
    'lr0': 0.01,                    # Initial learning rate
    'lrf': 0.1,                     # Final learning rate ratio
    'momentum': 0.937,              # SGD momentum
    'weight_decay': 0.0005,         # Weight decay
    'warmup_epochs': 3,             # Warmup epochs
    'box': 7.5,                     # Box loss weight
    'cls': 0.5,                     # Classification loss weight
    'pose': 12.0,                   # Pose loss weight (key for pose estimation)
    'kobj': 2.0,                    # Keypoint objectness loss weight
    'device': 'cuda',               # Training device
    'workers': 8,                   # Data loading workers
    'optimizer': 'auto',            # Optimizer (SGD/Adam/AdamW)
    'amp': True,                    # Automatic Mixed Precision
    'save_period': 10,              # Save checkpoint every N epochs
}
```

## 📈 Performance Metrics

### COCO-Pose Evaluation Metrics
- **mAP50-95 (pose)**: Mean Average Precision over IoU thresholds 0.5-0.95
- **mAP50 (pose)**: Mean Average Precision at IoU threshold 0.5
- **mAP75 (pose)**: Mean Average Precision at IoU threshold 0.75
- **OKS (Object Keypoint Similarity)**: COCO-specific pose evaluation metric

### Expected Results (YOLO11n-pose)
```
Model: YOLO11n-pose
mAP50-95: 50.0
mAP50: 81.0  
Speed (CPU): 52.4ms
Speed (GPU T4): 1.7ms
Parameters: 2.9M
```

## 🛠️ Troubleshooting

### Common Training Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
batch = 8  # instead of 16

# Or use smaller model
model = YOLO('yolo11n-pose.pt')  # instead of yolo11l-pose.pt
```

#### 2. Dataset Download Issues
```bash
# Manually download COCO dataset
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip
```

#### 3. Slow Training
```bash
# Use multiple workers
workers = 8  # Increase if you have more CPU cores

# Enable AMP
amp = True

# Use SSD for faster I/O
cache = True  # If you have enough RAM
```

#### 4. Poor Performance
```bash
# Train longer
epochs = 200

# Use larger model
model = YOLO('yolo11m-pose.pt')

# Increase image size
imgsz = 1024  # instead of 640
```

## 🔄 Data Augmentation

YOLO automatically applies pose-specific augmentations:

```python
augmentation = {
    'hsv_h': 0.015,          # Hue augmentation
    'hsv_s': 0.7,            # Saturation augmentation  
    'hsv_v': 0.4,            # Value augmentation
    'degrees': 0.0,          # Rotation degrees
    'translate': 0.1,        # Translation fraction
    'scale': 0.5,            # Scaling factor
    'shear': 0.0,            # Shear degrees
    'perspective': 0.0,      # Perspective factor
    'flipud': 0.0,           # Vertical flip probability
    'fliplr': 0.5,           # Horizontal flip probability
    'mosaic': 1.0,           # Mosaic augmentation probability
    'mixup': 0.0,            # Mixup augmentation probability
    'copy_paste': 0.0,       # Copy-paste augmentation probability
}
```

## 📁 Output Structure

After training, you'll get:

```
runs/pose/yolo11_coco_pose/
├── weights/
│   ├── best.pt              # Best model checkpoint
│   ├── last.pt              # Last model checkpoint
│   └── epoch_*.pt           # Intermediate checkpoints
├── results.png              # Training curves
├── confusion_matrix.png     # Confusion matrix
├── labels.jpg               # Sample labels
├── train_batch*.jpg         # Training batches
├── val_batch*.jpg           # Validation predictions
└── args.yaml                # Training arguments
```

## 🧪 Testing Your Model

### 1. Validation
```python
from ultralytics import YOLO

model = YOLO('runs/pose/yolo11_coco_pose/weights/best.pt')
metrics = model.val(data='coco-pose.yaml')

print(f"mAP50-95: {metrics.pose.map:.3f}")
print(f"mAP50: {metrics.pose.map50:.3f}")
```

### 2. Inference
```python
# Run inference on image
results = model('path/to/image.jpg')

# Run inference on video
results = model('path/to/video.mp4')

# Run inference on webcam
results = model(0)
```

### 3. Export Model
```python
# Export to different formats
model.export(format='onnx')     # ONNX format
model.export(format='engine')   # TensorRT
model.export(format='coreml')   # CoreML for iOS
```

## 🎯 Using Trained Model in Flask App

Update your `yolo_app.py`:

```python
# Use your trained model
MODEL_PATH = "runs/pose/yolo11_coco_pose/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)
```

## 🌟 Advanced Training Tips

### 1. Transfer Learning from Custom Data
```python
# First train on COCO-Pose
model = YOLO('yolo11n-pose.pt')
model.train(data='coco-pose.yaml', epochs=100)

# Then fine-tune on your cycling data
model.train(data='cycling-pose.yaml', epochs=50, lr0=0.001)
```

### 2. Multi-GPU Training
```python
# Use multiple GPUs
model.train(data='coco-pose.yaml', device=[0,1,2,3])
```

### 3. Resume Training
```python
# Resume from checkpoint
model = YOLO('runs/pose/yolo11_coco_pose/weights/last.pt')
model.train(resume=True)
```

## 📚 Additional Resources

- **COCO Dataset**: https://cocodataset.org/
- **Ultralytics YOLO11**: https://docs.ultralytics.com/
- **COCO-Pose Dataset**: https://docs.ultralytics.com/datasets/pose/coco/
- **Pose Estimation Guide**: https://docs.ultralytics.com/tasks/pose/

## 🤝 Contributing

To improve the cycling pose analysis:

1. **Collect cycling-specific data**: Videos of cyclists in different positions
2. **Annotate keypoints**: Use tools like CVAT or LabelMe
3. **Fine-tune the model**: Use pre-trained COCO model as starting point
4. **Evaluate on cycling metrics**: Custom metrics for bike fit analysis

## 📝 Citation

If you use COCO-Pose dataset:

```bibtex
@misc{lin2015microsoft,
    title={Microsoft COCO: Common Objects in Context},
    author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
    year={2015},
    eprint={1405.0312},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
``` 