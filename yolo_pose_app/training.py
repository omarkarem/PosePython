#!/usr/bin/env python3
"""
YOLO 11 Pose Training Script
This script trains a YOLO 11 pose estimation model using the COCO-Pose dataset
"""

import os
import yaml
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import shutil

def check_environment():
    """Check if the environment is ready for training"""
    print("🔍 Checking environment...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"   Device: {device}")
    
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    return device

def download_coco_pose_dataset():
    """Download and setup COCO-Pose dataset"""
    print("📥 Setting up COCO-Pose dataset...")
    
    # The COCO-Pose dataset will be automatically downloaded by YOLO
    # when we use 'coco-pose.yaml' in training
    print("✅ COCO-Pose dataset will be downloaded automatically during training")
    
    return True

def create_custom_dataset_yaml():
    """Create a custom dataset YAML for cycling-specific training (optional)"""
    print("📝 Creating custom dataset configuration...")
    
    # Create a custom cycling pose dataset configuration
    custom_config = {
        'path': 'datasets/cycling_pose',  # Will be created relative to ultralytics
        'train': 'images/train',
        'val': 'images/val', 
        'test': None,
        
        # Standard COCO pose keypoints
        'names': {0: 'person'},
        'nc': 1,  # number of classes
        'kpt_shape': [17, 3],  # [number_of_keypoints, (x, y, visibility)]
        
        # COCO keypoint flip indices for data augmentation
        'flip_idx': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    }
    
    # Save custom dataset configuration
    config_path = "cycling_pose.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(custom_config, f, default_flow_style=False)
    
    print(f"✅ Custom dataset config saved to {config_path}")
    print("💡 Note: This is for future custom cycling data. We'll use COCO-Pose for now.")
    
    return config_path

def train_yolo_pose_on_coco(model_name="yolo11n-pose.pt", epochs=100, device='cpu'):
    """Train YOLO 11 pose estimation model on COCO-Pose dataset"""
    
    print(f"🚀 Starting YOLO 11 pose training on COCO-Pose dataset...")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Device: {device}")
    
    # Load pre-trained YOLO 11 pose model
    model = YOLO(model_name)
    
    # Training parameters optimized for pose estimation
    training_args = {
        'data': 'coco-pose.yaml',  # Use official COCO-Pose dataset
        'epochs': epochs,
        'imgsz': 640,
        'batch': 16 if device == 'cuda' else 8,  # Adjust batch size based on device
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss function weights for pose estimation
        'box': 7.5,      # Box regression loss weight
        'cls': 0.5,      # Classification loss weight  
        'dfl': 1.5,      # Distribution focal loss weight
        'pose': 12.0,    # Pose keypoint loss weight
        'kobj': 2.0,     # Keypoint objectness loss weight
        
        # Training settings
        'optimizer': 'auto',
        'close_mosaic': 10,
        'amp': True,     # Automatic Mixed Precision
        'fraction': 1.0, # Use full dataset
        'device': device,
        'workers': 8 if device == 'cuda' else 4,
        'seed': 0,
        'deterministic': True,
        'val': True,
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'cache': False,     # Don't cache images (use if you have enough RAM)
        'verbose': True,
        
        # Project organization
        'project': 'cycling_pose_training',
        'name': 'yolo11_coco_pose',
        'exist_ok': True,
        'pretrained': True,
    }
    
    # Start training
    try:
        print("📚 Training will use the COCO-Pose dataset with 200K+ labeled images")
        print("   This includes 17 keypoints per person: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles")
        print("   Training may take several hours depending on your hardware...")
        
        results = model.train(**training_args)
        
        print("✅ Training completed successfully!")
        print(f"📊 Best model saved at: runs/pose/yolo11_coco_pose/weights/best.pt")
        print(f"📊 Last model saved at: runs/pose/yolo11_coco_pose/weights/last.pt")
        print(f"📈 Results saved in: runs/pose/yolo11_coco_pose/")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("💡 Common issues:")
        print("   - Insufficient GPU memory: reduce batch size")
        print("   - Network issues: check internet connection for dataset download")
        print("   - CUDA issues: try CPU training with device='cpu'")
        return None

def validate_trained_model(model_path="runs/pose/yolo11_coco_pose/weights/best.pt"):
    """Validate the trained model"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        print(f"🔍 Validating trained model: {model_path}")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Run validation on COCO-Pose validation set
        metrics = model.val(data='coco-pose.yaml')
        
        print("✅ Model validation successful!")
        print(f"📊 mAP50-95 (pose): {metrics.pose.map:.3f}")
        print(f"📊 mAP50 (pose): {metrics.pose.map50:.3f}")
        print(f"📊 mAP75 (pose): {metrics.pose.map75:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        return False

def create_demo_inference():
    """Create a demo inference script"""
    
    demo_script = '''#!/usr/bin/env python3
"""
Demo inference script for trained YOLO 11 pose model
"""

from ultralytics import YOLO
import cv2

# Load the trained model
model_path = "runs/pose/yolo11_coco_pose/weights/best.pt"
model = YOLO(model_path)

# Test on a sample image
results = model("https://ultralytics.com/images/bus.jpg")

# Save results
for i, result in enumerate(results):
    # Save image with pose annotations
    result.save(f"demo_result_{i}.jpg")
    
    # Print keypoint information
    if result.keypoints is not None:
        print(f"Detected {len(result.keypoints)} persons")
        for j, kpts in enumerate(result.keypoints.xy):
            print(f"Person {j+1}: {len(kpts)} keypoints detected")

print("Demo inference completed! Check demo_result_0.jpg")
'''
    
    with open("demo_inference.py", 'w') as f:
        f.write(demo_script)
    
    print("✅ Demo inference script created: demo_inference.py")

def main():
    """Main training pipeline"""
    
    print("🚴 YOLO 11 Cycling Pose Training Pipeline")
    print("=" * 50)
    print("📈 Using COCO-Pose Dataset (200K+ images, 17 keypoints)")
    print("🎯 Training for general human pose estimation")
    print("💡 This model will work great for cycling pose analysis!")
    print("=" * 50)
    
    # Check environment
    device = check_environment()
    
    # Confirm training parameters
    print("\n⚙️ Training Configuration:")
    print("   Dataset: COCO-Pose (official)")
    print("   Model: YOLO11n-pose (nano - fastest)")
    print("   Epochs: 100 (you can adjust this)")
    print("   Keypoints: 17 (COCO standard)")
    print("   Classes: 1 (person)")
    
    # Ask user for confirmation
    if input("\n🤔 Start training? This will take time and download ~20GB dataset. (y/n): ").lower() != 'y':
        print("👋 Training cancelled. Creating demo model instead...")
        
        # Download pre-trained model for demo
        print("📥 Downloading pre-trained YOLO11n-pose model...")
        model = YOLO('yolo11n-pose.pt')
        
        # Save it to models directory
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "cycling_pose_model.pt"
        model.save(str(model_path))
        
        print(f"✅ Demo model ready at: {model_path}")
        create_demo_inference()
        return True
    
    # Download COCO-Pose dataset info
    download_coco_pose_dataset()
    
    # Create custom dataset config for future use
    create_custom_dataset_yaml()
    
    # Train the model
    print(f"\n🚀 Starting training on {device}...")
    results = train_yolo_pose_on_coco(
        model_name="yolo11n-pose.pt",
        epochs=100,  # You can reduce this for faster training
        device=device
    )
    
    if results:
        # Validate the trained model
        validate_trained_model()
        
        # Create demo script
        create_demo_inference()
        
        print("\n🎉 Training pipeline completed successfully!")
        print("📁 Files created:")
        print("   ├── runs/pose/yolo11_coco_pose/weights/best.pt (trained model)")
        print("   ├── cycling_pose.yaml (custom dataset config)")
        print("   └── demo_inference.py (test script)")
        
        print("\n🚀 Next steps:")
        print("   1. Test the model: python demo_inference.py")
        print("   2. Use in Flask app: Update MODEL_PATH in yolo_app.py")
        print("   3. Deploy to EC2: Use the trained model")
        
        return True
    else:
        print("\n❌ Training failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    main() 