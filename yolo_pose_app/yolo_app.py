#!/usr/bin/env python3
"""
YOLO 11 Pose Detection Flask API
Similar to app.py but using YOLO 11 for pose estimation
"""

from flask import Flask, request, jsonify, send_file, Response
import cv2
import numpy as np
import os
import tempfile
import uuid
import io
import base64
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
import subprocess
from ultralytics import YOLO
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Production configurations
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.mp4', '.avi', '.mov', '.webm']

# Create directories
UPLOAD_FOLDER = 'output_videos'
MODEL_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load YOLO model
MODEL_PATH = "models/cycling_pose_model.pt"
if not os.path.exists(MODEL_PATH):
    # Check for trained model first
    trained_model_path = "runs/pose/yolo11_coco_pose/weights/best.pt"
    if os.path.exists(trained_model_path):
        MODEL_PATH = trained_model_path
    else:
        MODEL_PATH = "yolo11n-pose.pt"  # Fallback to default

try:
    yolo_model = YOLO(MODEL_PATH)
    logger.info(f"✅ YOLO model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load YOLO model: {e}")
    yolo_model = None

@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File is too large. Maximum size is 100MB."), 413

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify(error="Internal server error"), 500

# YOLO Keypoint indices (COCO format)
YOLO_KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b) 
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    
    return int(angle)

def calculate_torso_angle(shoulder, hip):
    """Calculate torso angle relative to horizontal"""
    dy = shoulder[1] - hip[1]
    dx = shoulder[0] - hip[0]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    if angle_deg < 0:
        angle_deg = 180 + angle_deg
    
    lean_angle = 90 - angle_deg if angle_deg <= 90 else angle_deg - 90
    lean_angle = abs(lean_angle)
    lean_angle = min(lean_angle, 90)
    
    return int(lean_angle)

def extract_keypoints_from_yolo(results):
    """Extract keypoints from YOLO results"""
    if not results or len(results) == 0:
        return None
    
    # Get the first detection result
    result = results[0]
    
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None
    
    # Get keypoints data [x, y, confidence] for each keypoint
    keypoints = result.keypoints.data[0]  # First person
    
    # Convert to pixel coordinates
    kpts = {}
    for name, idx in YOLO_KEYPOINTS.items():
        if idx < len(keypoints):
            x, y, conf = keypoints[idx]
            if conf > 0.5:  # Confidence threshold
                kpts[name] = (float(x), float(y), float(conf))
            else:
                kpts[name] = None
    
    return kpts

def determine_visible_side(keypoints):
    """Determine which side of the body is more visible"""
    left_shoulder = keypoints.get('left_shoulder')
    right_shoulder = keypoints.get('right_shoulder')
    
    if left_shoulder and right_shoulder:
        # Use confidence scores to determine visible side
        left_conf = left_shoulder[2]
        right_conf = right_shoulder[2]
        return 'left' if left_conf > right_conf else 'right'
    elif left_shoulder:
        return 'left'
    elif right_shoulder:
        return 'right'
    else:
        return 'left'  # Default

def get_side_keypoints(keypoints, visible_side):
    """Get keypoints for the visible side"""
    if visible_side == 'left':
        return {
            'shoulder': keypoints.get('left_shoulder'),
            'elbow': keypoints.get('left_elbow'),
            'wrist': keypoints.get('left_wrist'),
            'hip': keypoints.get('left_hip'),
            'knee': keypoints.get('left_knee'),
            'ankle': keypoints.get('left_ankle')
        }
    else:
        return {
            'shoulder': keypoints.get('right_shoulder'),
            'elbow': keypoints.get('right_elbow'),
            'wrist': keypoints.get('right_wrist'),
            'hip': keypoints.get('right_hip'),
            'knee': keypoints.get('right_knee'),
            'ankle': keypoints.get('right_ankle')
        }

def calculate_body_measurements(keypoints, visible_side, user_height_cm):
    """Calculate body segment lengths from keypoints"""
    side_kpts = get_side_keypoints(keypoints, visible_side)
    
    # Check if we have the required keypoints
    required = ['shoulder', 'hip', 'knee', 'ankle', 'elbow', 'wrist']
    if not all(side_kpts.get(k) for k in required):
        return None
    
    # Extract coordinates (ignore confidence for measurements)
    shoulder = side_kpts['shoulder'][:2]
    hip = side_kpts['hip'][:2]
    knee = side_kpts['knee'][:2]
    ankle = side_kpts['ankle'][:2]
    elbow = side_kpts['elbow'][:2]
    wrist = side_kpts['wrist'][:2]
    
    # Calculate distances in pixels
    torso_px = np.linalg.norm(np.array(shoulder) - np.array(hip))
    femur_px = np.linalg.norm(np.array(hip) - np.array(knee))
    tibia_px = np.linalg.norm(np.array(knee) - np.array(ankle))
    upper_arm_px = np.linalg.norm(np.array(shoulder) - np.array(elbow))
    forearm_px = np.linalg.norm(np.array(elbow) - np.array(wrist))
    
    # Total height in pixels for scaling
    total_height_px = torso_px + femur_px + tibia_px
    
    # Convert to real measurements
    scale_factor = user_height_cm / total_height_px if total_height_px > 0 else 1
    
    measurements = {
        'torso_length': torso_px * scale_factor,
        'femur_length': femur_px * scale_factor,
        'lower_leg_length': tibia_px * scale_factor,
        'upper_arm_length': upper_arm_px * scale_factor,
        'forearm_length': forearm_px * scale_factor,
        'visible_side': visible_side,
        'measurement_method': 'calculated from YOLO keypoints'
    }
    
    return measurements

def draw_yolo_pose(image, keypoints, visible_side):
    """Draw pose keypoints and connections on image"""
    if not keypoints:
        return image
    
    side_kpts = get_side_keypoints(keypoints, visible_side)
    
    # Define connections for the visible side
    connections = [
        ('shoulder', 'elbow'),
        ('elbow', 'wrist'),
        ('shoulder', 'hip'),
        ('hip', 'knee'),
        ('knee', 'ankle')
    ]
    
    # Draw connections
    for start, end in connections:
        start_pt = side_kpts.get(start)
        end_pt = side_kpts.get(end)
        
        if start_pt and end_pt:
            start_xy = (int(start_pt[0]), int(start_pt[1]))
            end_xy = (int(end_pt[0]), int(end_pt[1]))
            cv2.line(image, start_xy, end_xy, (0, 255, 0), 3)
    
    # Draw keypoints
    for name, point in side_kpts.items():
        if point:
            center = (int(point[0]), int(point[1]))
            cv2.circle(image, center, 6, (0, 255, 0), -1)
            cv2.putText(image, name[:3], (center[0]+10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def generate_yolo_recommendations(max_angles, min_angles, body_measurements, torso_angle):
    """Generate bike fit recommendations (simplified version)"""
    recommendations = {
        'general': [],
        'road_bike': {
            'endurance': [],
            'aggressive': []
        },
        'time_trial': []
    }
    
    # Basic knee extension analysis
    knee_extension = max_angles.get('hip_knee_ankle', 0)
    if knee_extension > 155:
        recommendations['general'].append({
            'component': 'SADDLE HEIGHT',
            'issue': 'Saddle too high',
            'action': 'Lower saddle',
            'current': f'{knee_extension}° knee extension',
            'target': '140-150°',
            'priority': 'high'
        })
    elif knee_extension < 135:
        recommendations['general'].append({
            'component': 'SADDLE HEIGHT',
            'issue': 'Saddle too low',
            'action': 'Raise saddle',
            'current': f'{knee_extension}° knee extension',
            'target': '140-150°',
            'priority': 'high'
        })
    else:
        recommendations['general'].append({
            'component': 'SADDLE HEIGHT',
            'issue': 'Good position',
            'action': 'No change needed',
            'current': f'{knee_extension}° knee extension',
            'target': '140-150°',
            'priority': 'low'
        })
    
    # Torso angle analysis
    if torso_angle < 20:
        recommendations['general'].append({
            'component': 'HANDLEBAR HEIGHT',
            'issue': 'Very aggressive position',
            'action': 'Consider raising handlebars',
            'current': f'{torso_angle}° torso angle',
            'target': '20-45°',
            'priority': 'medium'
        })
    elif torso_angle > 50:
        recommendations['general'].append({
            'component': 'HANDLEBAR HEIGHT',
            'issue': 'Very upright position',
            'action': 'Consider lowering handlebars',
            'current': f'{torso_angle}° torso angle',
            'target': '20-45°',
            'priority': 'medium'
        })
    
    return recommendations

def create_yolo_video(frames, fps, width, height, quality=75):
    """Create video from frames with H.264 encoding"""
    with tempfile.NamedTemporaryFile(suffix='_temp.mp4', delete=False) as temp_intermediate:
        temp_intermediate_path = temp_intermediate.name
    
    with tempfile.NamedTemporaryFile(suffix='_final.mp4', delete=False) as temp_final:
        temp_final_path = temp_final.name

    try:
        # Create video with mp4v codec first
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_intermediate_path, fourcc, fps, (width, height), True)

        if not out.isOpened():
            raise Exception("Failed to open VideoWriter with mp4v codec")

        for frame in frames:
            out.write(frame)
        out.release()

        # Convert to H.264 using FFmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_intermediate_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            temp_final_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            with open(temp_intermediate_path, 'rb') as f:
                video_data = f.read()
        else:
            with open(temp_final_path, 'rb') as f:
                video_data = f.read()

        # Base64 encode
        encoded_video = base64.b64encode(video_data).decode('utf-8')

        # Cleanup
        for temp_path in [temp_intermediate_path, temp_final_path]:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return encoded_video
        
    except Exception as e:
        # Cleanup on error
        for temp_path in [temp_intermediate_path, temp_final_path]:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise e

@app.route('/process-video', methods=['POST'])
def process_video():
    """Process video using YOLO 11 pose detection"""
    try:
        if yolo_model is None:
            return jsonify({"error": "YOLO model not loaded"}), 500
        
        # Get uploaded video and user height
        video_file = request.files['video']
        user_height_cm = float(request.form.get('user_height_cm', 175))
        quality = min(100, max(1, int(request.form.get('quality', '75'))))

        # Save video to temporary file
        video_bytes = video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name

        # Open video
        cap = cv2.VideoCapture(temp_file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Scale output if too large
        if width > 1920 or height > 1080:
            scale_factor = min(1920 / width, 1080 / height)
            output_width = int(width * scale_factor)
            output_height = int(height * scale_factor)
        else:
            output_width = width
            output_height = height

        frames = []
        max_angles = {
            'shoulder_hip_knee': 0,
            'shoulder_elbow_wrist': 0,
            'hip_knee_ankle': 0,
            'elbow_shoulder_hip': 0
        }
        min_angles = {
            'shoulder_hip_knee': float('inf'),
            'shoulder_elbow_wrist': float('inf'),
            'hip_knee_ankle': float('inf'),
            'elbow_shoulder_hip': float('inf')
        }

        body_measurements_list = []
        frame_idx = 0
        torso_angle = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame = cv2.resize(frame, (output_width, output_height))

            # Run YOLO inference
            results = yolo_model(frame, verbose=False)
            
            # Extract keypoints
            keypoints = extract_keypoints_from_yolo(results)
            
            if keypoints:
                # Determine visible side
                visible_side = determine_visible_side(keypoints)
                side_kpts = get_side_keypoints(keypoints, visible_side)
                
                # Calculate measurements
                measurements = calculate_body_measurements(keypoints, visible_side, user_height_cm)
                if measurements:
                    body_measurements_list.append(measurements)
                
                # Calculate angles if we have the required keypoints
                if all(side_kpts.get(k) for k in ['shoulder', 'hip', 'knee', 'ankle', 'elbow', 'wrist']):
                    shoulder = side_kpts['shoulder'][:2]
                    hip = side_kpts['hip'][:2]
                    knee = side_kpts['knee'][:2]
                    ankle = side_kpts['ankle'][:2]
                    elbow = side_kpts['elbow'][:2]
                    wrist = side_kpts['wrist'][:2]

                    # Calculate angles
                    shoulder_hip_knee = calculate_angle(shoulder, hip, knee)
                    shoulder_elbow_wrist = calculate_angle(shoulder, elbow, wrist)
                    hip_knee_ankle = calculate_angle(hip, knee, ankle)
                    elbow_shoulder_hip = calculate_angle(elbow, shoulder, hip)
                    torso_angle = calculate_torso_angle(shoulder, hip)

                    # Update min/max angles
                    max_angles['shoulder_hip_knee'] = max(max_angles['shoulder_hip_knee'], shoulder_hip_knee)
                    max_angles['shoulder_elbow_wrist'] = max(max_angles['shoulder_elbow_wrist'], shoulder_elbow_wrist)
                    max_angles['hip_knee_ankle'] = max(max_angles['hip_knee_ankle'], hip_knee_ankle)
                    max_angles['elbow_shoulder_hip'] = max(max_angles['elbow_shoulder_hip'], elbow_shoulder_hip)
                    
                    min_angles['shoulder_hip_knee'] = min(min_angles['shoulder_hip_knee'], shoulder_hip_knee)
                    min_angles['shoulder_elbow_wrist'] = min(min_angles['shoulder_elbow_wrist'], shoulder_elbow_wrist)
                    min_angles['hip_knee_ankle'] = min(min_angles['hip_knee_ankle'], hip_knee_ankle)
                    min_angles['elbow_shoulder_hip'] = min(min_angles['elbow_shoulder_hip'], elbow_shoulder_hip)

                    # Add angle annotations to frame
                    cv2.putText(frame, f"YOLO Pose - Side: {visible_side}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Torso Angle: {torso_angle}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Hip-Knee-Ankle: {hip_knee_ankle}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Elbow-Shoulder-Hip: {elbow_shoulder_hip}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw pose on frame
                frame = draw_yolo_pose(frame, keypoints, visible_side)

            frames.append(frame)

            if frame_idx % 10 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames")

        cap.release()
        os.remove(temp_file_path)

        # Calculate average body measurements
        if body_measurements_list:
            avg_measurements = {}
            for key in ['torso_length', 'femur_length', 'lower_leg_length', 'upper_arm_length', 'forearm_length']:
                values = [m[key] for m in body_measurements_list if key in m]
                avg_measurements[key] = np.mean(values) if values else 0
            avg_measurements['measurement_method'] = 'calculated from YOLO keypoints'
            avg_measurements['visible_side'] = body_measurements_list[0]['visible_side']
        else:
            # Fallback measurements
            avg_measurements = {
                'torso_length': user_height_cm * 0.35,
                'femur_length': user_height_cm * 0.27,
                'lower_leg_length': user_height_cm * 0.25,
                'upper_arm_length': user_height_cm * 0.15,
                'forearm_length': user_height_cm * 0.12,
                'measurement_method': 'estimated from height'
            }

        # Fix infinite values in min_angles
        for key in min_angles:
            if min_angles[key] == float('inf'):
                min_angles[key] = 0

        # Generate recommendations
        recommendations = generate_yolo_recommendations(max_angles, min_angles, avg_measurements, torso_angle)

        # Prepare response
        response_data = {
            "max_angles": max_angles,
            "min_angles": min_angles,
            "body_lengths_cm": {k: round(v, 2) if isinstance(v, (int, float)) else v 
                               for k, v in avg_measurements.items()},
            "recommendations": recommendations,
            "model_type": "YOLO11"
        }

        # Create video
        if frames:
            encoded_video = create_yolo_video(frames, fps, output_width, output_height, quality)
            response_data["video"] = encoded_video
            logger.info(f"Video created from {len(frames)} frames using YOLO 11")

        return jsonify(response_data)

    except Exception as e:
        import traceback
        logger.error(f"Error processing video: {str(e)}")
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": traceback_str}), 500

@app.route('/')
def index():
    return jsonify({
        "service": "YOLO 11 Pose Detection API",
        "version": "1.0",
        "model": "YOLO 11 Pose",
        "status": "active",
        "endpoints": [
            "/process-video (POST)",
            "/health (GET)"
        ]
    })

@app.route('/health')
def health_check():
    model_status = "loaded" if yolo_model is not None else "not loaded"
    return jsonify({
        "status": "healthy",
        "service": "yolo-pose-detection",
        "model_status": model_status,
        "model_path": MODEL_PATH,
        "timestamp": os.popen('date').read().strip()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Different port from MediaPipe app 