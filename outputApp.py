from flask import Flask, request, jsonify, send_file, Response
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
import uuid
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create a directory for output videos if it doesn't exist
UPLOAD_FOLDER = 'output_videos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
# Replace the existing calculate_angle function with:
def calculate_angle(a, b, c):
    """Calculate the angle between three points with better stability"""
    a = np.array(a)  # First point
    b = np.array(b)  # Apex point
    c = np.array(c)  # Third point
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Clamp value to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    
    return int(angle)

# Add this new function after the calculate_angle function
def calculate_torso_angle(shoulder, hip):
    """Calculate torso angle relative to horizontal properly"""
    # Create a true horizontal reference point
    horizontal_ref = (hip[0] + 100, hip[1])  # 100 pixels to the right
    
    # Calculate angle between shoulder, hip and horizontal reference
    dy = shoulder[1] - hip[1]
    dx = shoulder[0] - hip[0]
    angle_rad = np.arctan2(dy, dx)
    
    # Convert to degrees and get angle from vertical (not horizontal)
    # This gives us the forward lean angle that's more intuitive for bike fitting
    angle_deg = np.degrees(angle_rad)
    
    # Normalize angle to be between 0 and 90 degrees (typical range for bike fit)
    if angle_deg < 0:
        angle_deg = 180 + angle_deg  # Convert negative angles
    
    # Adjust angle to represent forward lean from vertical (90° would be upright)
    lean_angle = 90 - angle_deg if angle_deg <= 90 else angle_deg - 90
    
    # Ensure the angle is positive and in a realistic range (0-90°)
    lean_angle = abs(lean_angle)
    lean_angle = min(lean_angle, 90)
    
    return int(lean_angle)

# Function to calculate distance between two points in pixels
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to convert pixel distances to real-world measurements (cm)
def pixel_to_cm(pixel_distance, user_height_cm, height_in_pixels):
    # Scale factor based on known user height and pixel height
    return (pixel_distance / height_in_pixels) * user_height_cm

# Function to calculate body segment lengths from landmarks
def calculate_body_lengths(landmarks, frame_shape, user_height_cm):
    def to_pixel(landmark):
        return (int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0]))
    
    # Determine which side of the body is more visible
    left_visibility = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
    right_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
    visible_side = 'left' if left_visibility > right_visibility else 'right'
    
    if visible_side == 'left':
        shoulder = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        elbow = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        wrist = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        hip = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        knee = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        ankle = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        heel = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value])
    else:
        shoulder = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        elbow = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        wrist = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        hip = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        knee = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        ankle = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        heel = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value])
    
    # Calculate pixel distances between key landmarks
    torso_pixels = calculate_distance(shoulder, hip)
    femur_pixels = calculate_distance(hip, knee)
    lower_leg_pixels = calculate_distance(knee, ankle)
    upper_arm_pixels = calculate_distance(shoulder, elbow)
    forearm_pixels = calculate_distance(elbow, wrist)
    
    # Calculate total body height in pixels for scaling
    # Using the sum of segments that make up the body height
    total_height_pixels = torso_pixels + femur_pixels + lower_leg_pixels + calculate_distance(ankle, heel)
    
    # Convert pixel measurements to cm based on user's known height
    torso_length = pixel_to_cm(torso_pixels, user_height_cm, total_height_pixels)
    femur_length = pixel_to_cm(femur_pixels, user_height_cm, total_height_pixels)
    lower_leg_length = pixel_to_cm(lower_leg_pixels, user_height_cm, total_height_pixels)
    upper_arm_length = pixel_to_cm(upper_arm_pixels, user_height_cm, total_height_pixels)
    forearm_length = pixel_to_cm(forearm_pixels, user_height_cm, total_height_pixels)
    
    return {
        'torso_length': torso_length,
        'femur_length': femur_length,
        'lower_leg_length': lower_leg_length,
        'upper_arm_length': upper_arm_length,
        'forearm_length': forearm_length,
        'visible_side': visible_side,
        'landmarks': {
            'shoulder': shoulder,
            'elbow': elbow,
            'wrist': wrist,
            'hip': hip,
            'knee': knee,
            'ankle': ankle
        }
    }

# Function to generate recommendations
def generate_recommendations(max_angles, min_angles, body_lengths, torso_angle):
    """
    Generate evidence-based bike fit recommendations using research-backed angle ranges,
    personalized to the user's specific body measurements.
    
    Parameters:
    - max_angles: Dictionary containing maximum joint angles observed
    - min_angles: Dictionary containing minimum joint angles observed
    - body_lengths: Dictionary containing user's body segment measurements in cm
    - torso_angle: Measured torso angle relative to horizontal
    
    Returns:
    - Dictionary with three categories of recommendations
    """
    recommendations = {
        'general': [],
        'endurance': [],
        'aggressive': []
    }

    # Extract body measurements
    femur_length = body_lengths['femur_length']
    torso_length = body_lengths['torso_length']
    lower_leg_length = body_lengths['lower_leg_length']
    
    # Calculate user-specific proportions and adjustments
    torso_femur_ratio = torso_length / femur_length if femur_length > 0 else 1.0
    leg_length = femur_length + lower_leg_length
    
    # Adjust optimal angles based on body proportions
    # Riders with longer femurs relative to torso often need different angles
    
    # -------- SADDLE HEIGHT RECOMMENDATIONS --------
    # Personalize knee extension angle targets based on leg proportions
    femur_lower_leg_ratio = femur_length / lower_leg_length if lower_leg_length > 0 else 1.0
    
    # Adjust optimal knee extension angles based on leg proportions
    # Longer femurs relative to lower legs often benefit from slightly greater extension
    knee_ext_adjustment = 0
    if femur_lower_leg_ratio > 1.1:  # Longer femur
        knee_ext_adjustment = 2  # Increase optimal angle
    elif femur_lower_leg_ratio < 0.9:  # Shorter femur
        knee_ext_adjustment = -2  # Decrease optimal angle
    
    # Personalized optimal ranges
    min_knee_ext = 140 + knee_ext_adjustment
    max_knee_ext = 148 + knee_ext_adjustment
    
    knee_extension = max_angles['hip_knee_ankle']
    
    # Convert angle differences to precise saddle height adjustments
    # Based on the user's leg length (more accurate than fixed mm per degree)
    mm_per_degree = leg_length * 0.01  # About 1% of leg length per degree
    
    if knee_extension > max_knee_ext:
        adj_mm = (knee_extension - max_knee_ext) * mm_per_degree
        recommendations['general'].append(f"Saddle too high for your proportions. Lower by {adj_mm:.1f}mm to achieve optimal knee extension of {min_knee_ext}-{max_knee_ext}°.")
    elif knee_extension < min_knee_ext:
        adj_mm = (min_knee_ext - knee_extension) * mm_per_degree
        recommendations['general'].append(f"Saddle too low for your proportions. Raise by {adj_mm:.1f}mm to achieve optimal knee extension of {min_knee_ext}-{max_knee_ext}°.")
    elif knee_extension > (min_knee_ext + max_knee_ext)/2 and knee_extension <= max_knee_ext:
        recommendations['general'].append(f"Saddle height good for endurance riding (current knee extension: {knee_extension}°).")
        recommendations['aggressive'].append(f"For more power, consider lowering saddle by {5 * mm_per_degree:.1f}mm to reduce knee extension angle slightly.")
    elif knee_extension >= min_knee_ext and knee_extension < (min_knee_ext + max_knee_ext)/2:
        recommendations['general'].append(f"Saddle height acceptable but on lower end (current knee extension: {knee_extension}°).")
        recommendations['endurance'].append(f"For more comfort on longer rides, consider raising saddle by {5 * mm_per_degree:.1f}mm.")
    else:
        recommendations['general'].append(f"Saddle height optimal for your proportions (current knee extension: {knee_extension}°).")
    
    # -------- KNEE OVER PEDAL SPINDLE (KOPS) / FORE-AFT POSITIONING --------
    # Personalize based on femur length and riding style
    knee_flexion_tdc = min_angles['hip_knee_ankle']
    
    # Adjust optimal range based on femur length
    # Riders with longer femurs often benefit from different fore/aft position
    tdc_adjustment = 0
    if femur_length > 45:  # Longer femur
        tdc_adjustment = 2  # Allow slightly more flexion
    elif femur_length < 38:  # Shorter femur
        tdc_adjustment = -2  # Recommend slightly less flexion
    
    min_tdc_angle = 65 + tdc_adjustment
    max_tdc_angle = 75 + tdc_adjustment
    
    # Calculate saddle adjustment based on femur length
    mm_per_tdc_degree = femur_length * 0.05  # 5% of femur length per degree change
    
    if knee_flexion_tdc > max_tdc_angle:
        adj_mm = (knee_flexion_tdc - max_tdc_angle) * mm_per_tdc_degree
        recommendations['general'].append(f"Excessive knee flexion at top of pedal stroke ({knee_flexion_tdc}°). Based on your femur length of {femur_length:.1f}cm, move saddle forward by {adj_mm:.1f}mm.")
    elif knee_flexion_tdc < min_tdc_angle:
        adj_mm = (min_tdc_angle - knee_flexion_tdc) * mm_per_tdc_degree
        recommendations['general'].append(f"Insufficient knee flexion at top of pedal stroke ({knee_flexion_tdc}°). Based on your femur length of {femur_length:.1f}cm, move saddle backward by {adj_mm:.1f}mm.")
    else:
        recommendations['general'].append(f"Saddle fore/aft position good for your proportions (knee flexion at top: {knee_flexion_tdc}°).")
        
        if knee_flexion_tdc > (min_tdc_angle + max_tdc_angle)/2:
            recommendations['aggressive'].append(f"For more aggressive position, consider moving saddle back {5 * mm_per_tdc_degree:.1f}mm for better power transfer.")
        else:
            recommendations['endurance'].append(f"For more comfort on long rides, consider moving saddle forward {5 * mm_per_tdc_degree:.1f}mm.")
    
    # -------- TORSO ANGLE RECOMMENDATIONS --------
    # Adjust recommendation based on torso-femur ratio - longer torsos can go lower
    # and shorter riders may need more upright positions
    
    # Calculate more personalized adjustment based on proportions
    adjustment = 0
    if torso_femur_ratio > 1.1:  # Long torso
        adjustment = 5  # Can go lower
    elif torso_femur_ratio < 0.9:  # Short torso
        adjustment = -5  # Should be more upright
    
    # Further adjust based on overall height
    height_adj = 0
    total_height = body_lengths.get('measurement_method') == 'calculated from landmarks' \
                    and (torso_length + femur_length + lower_leg_length) \
                    or body_lengths.get('user_height_cm', 175)
                    
    if total_height > 185:  # Taller rider
        height_adj = 2  # Can go slightly lower
    elif total_height < 165:  # Shorter rider
        height_adj = -2  # Should be slightly more upright
    
    total_adjustment = adjustment + height_adj
    
    # Apply the personalized adjustment to the angle ranges
    min_general = 25 - total_adjustment
    max_general = 40 - total_adjustment
    min_aggressive = 15 - total_adjustment
    max_aggressive = 25 - total_adjustment
    min_endurance = 40 - total_adjustment
    max_endurance = 55 - total_adjustment
    
    # Ensure all angles are in reasonable ranges
    min_general = max(10, min_general)
    max_general = max(min_general + 10, max_general)
    min_aggressive = max(5, min_aggressive)
    max_aggressive = max(min_aggressive + 5, max_aggressive)
    min_endurance = max(max_general, min_endurance)
    max_endurance = max(min_endurance + 10, max_endurance)
    
    # Generate personalized recommendations based on current torso angle
    mm_per_angle_degree = (torso_length * 0.07)  # Approx 7% of torso length per degree
    
    if torso_angle < min_aggressive:
        recommendations['general'].append(f"Torso angle too low ({torso_angle}°) for your body proportions. Raise handlebars by {(min_aggressive - torso_angle) * mm_per_angle_degree:.1f}mm for better comfort.")
    elif torso_angle > max_endurance:
        recommendations['general'].append(f"Torso angle too upright ({torso_angle}°) for your proportions. Lower handlebars by {(torso_angle - max_endurance) * mm_per_angle_degree:.1f}mm for better aerodynamics.")
    elif min_aggressive <= torso_angle < min_general:
        recommendations['general'].append(f"Current torso angle ({torso_angle}°) is very aggressive for your body proportions.")
        recommendations['endurance'].append(f"For endurance riding with your proportions, raise handlebars by {(min_endurance - torso_angle) * mm_per_angle_degree:.1f}mm to achieve a more comfortable torso angle.")
    elif max_general < torso_angle <= max_endurance:
        recommendations['general'].append(f"Current torso angle ({torso_angle}°) is upright, good for endurance with your proportions.")
        recommendations['aggressive'].append(f"For more aerodynamic position, lower handlebars by {(torso_angle - min_aggressive) * mm_per_angle_degree:.1f}mm to achieve a more aggressive torso angle.")
    else:
        recommendations['general'].append(f"Torso angle ({torso_angle}°) is optimal for general road riding with your body proportions.")
        
        if torso_angle > (min_general + max_general)/2:
            recommendations['aggressive'].append(f"For more aggressive position with your proportions, consider lowering handlebars by {(torso_angle - min_aggressive) * mm_per_angle_degree:.1f}mm.")
        else:
            recommendations['endurance'].append(f"For more comfort on long rides with your proportions, consider raising handlebars by {(min_endurance - torso_angle) * mm_per_angle_degree:.1f}mm.")
    
    # -------- ELBOW ANGLE / REACH RECOMMENDATIONS --------
    # Personalize based on arm length and torso length
    elbow_angle = max_angles['shoulder_elbow_wrist']
    
    # Get arm measurements
    arm_length = body_lengths.get('upper_arm_length', 0) + body_lengths.get('forearm_length', 0)
    
    # Adjust optimal elbow angle range based on arm-to-torso ratio
    arm_torso_ratio = arm_length / torso_length if torso_length > 0 else 1.0
    elbow_adjustment = 0
    if arm_torso_ratio > 0.8:  # Longer arms relative to torso
        elbow_adjustment = 5  # Can have more extension
    elif arm_torso_ratio < 0.6:  # Shorter arms relative to torso
        elbow_adjustment = -5  # Need more bend
    
    min_elbow = 60 + elbow_adjustment
    max_elbow = 80 + elbow_adjustment
    
    # Calculate personalized stem adjustment
    stem_change_factor = arm_length * 0.02  # 2% of arm length per degree of angle change
    
    if elbow_angle < min_elbow:
        stem_change = (min_elbow - elbow_angle) * stem_change_factor
        recommendations['general'].append(f"Arms too bent ({elbow_angle}°) for your proportions. Increase reach with longer stem (+{stem_change:.1f}mm).")
    elif elbow_angle > max_elbow:
        stem_change = (elbow_angle - max_elbow) * stem_change_factor
        recommendations['general'].append(f"Arms too extended ({elbow_angle}°) for your proportions. Decrease reach with shorter stem (-{stem_change:.1f}mm).")
    else:
        recommendations['general'].append(f"Elbow angle optimal ({elbow_angle}°) for your arm length and proportions.")
    
    # -------- HIP ANGLE RECOMMENDATIONS --------
    hip_angle = min_angles['shoulder_hip_knee']
    
    # Adjust hip angle recommendations based on leg-torso proportions
    hip_adjustment = 0
    if torso_femur_ratio > 1.1:  # Long torso relative to femur
        hip_adjustment = -3  # Can handle more closed hip angle
    elif torso_femur_ratio < 0.9:  # Short torso relative to femur
        hip_adjustment = 3  # Needs more open hip angle
        
    min_hip = 45 + hip_adjustment
    max_hip = 55 + hip_adjustment
    
    if hip_angle < min_hip - 5:
        recommendations['general'].append(f"Hip angle too closed ({hip_angle}°) for your proportions. Based on your torso-to-femur ratio, adjust either:")
        recommendations['general'].append(f"- Move saddle backward by {(min_hip - hip_angle) * femur_length * 0.03:.1f}mm, or")
        recommendations['general'].append(f"- Raise handlebars by {(min_hip - hip_angle) * torso_length * 0.05:.1f}mm")
    elif hip_angle > max_hip + 5:
        recommendations['general'].append(f"Hip angle too open ({hip_angle}°) for your proportions. Based on your torso-to-femur ratio, adjust either:")
        recommendations['general'].append(f"- Move saddle forward by {(hip_angle - max_hip) * femur_length * 0.03:.1f}mm, or")
        recommendations['general'].append(f"- Lower handlebars by {(hip_angle - max_hip) * torso_length * 0.05:.1f}mm")
    elif hip_angle >= min_hip - 5 and hip_angle < min_hip:
        recommendations['general'].append(f"Hip angle ({hip_angle}°) on aggressive side but acceptable for your proportions.")
        recommendations['endurance'].append(f"For more comfort with your body proportions, consider small adjustments to open hip angle by 2-3°.")
    elif hip_angle > max_hip and hip_angle <= max_hip + 5:
        recommendations['general'].append(f"Hip angle ({hip_angle}°) on relaxed side but acceptable for your proportions.")
        recommendations['aggressive'].append(f"For more power transfer with your proportions, consider adjustments to close hip angle by 2-3°.")
    else:
        recommendations['general'].append(f"Hip angle optimal ({hip_angle}°) for balanced position with your body proportions.")
    
    return recommendations

# Custom drawing function that only shows the visible side
def draw_visible_side_landmarks(image, landmarks, connections, visible_side='left'):
    # Get image dimensions
    height, width, _ = image.shape
    
    # Create a list to store the connections we want to draw
    connections_to_draw = []
    
    # Define landmarks for each side
    left_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.LEFT_HEEL.value,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    ]
    
    right_landmarks = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_HEEL.value,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    ]
    
    # Define the unwanted landmarks (face and hands except wrist)
    face_landmarks = [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE.value,
        mp_pose.PoseLandmark.RIGHT_EYE.value,
        mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.MOUTH_LEFT.value,
        mp_pose.PoseLandmark.MOUTH_RIGHT.value
    ]
    
    hand_landmarks = [
        mp_pose.PoseLandmark.LEFT_PINKY.value,
        mp_pose.PoseLandmark.RIGHT_PINKY.value,
        mp_pose.PoseLandmark.LEFT_INDEX.value,
        mp_pose.PoseLandmark.RIGHT_INDEX.value,
        mp_pose.PoseLandmark.LEFT_THUMB.value,
        mp_pose.PoseLandmark.RIGHT_THUMB.value
    ]
    
    unwanted_landmarks = face_landmarks + hand_landmarks
    
    # Choose which side landmarks to draw based on visibility
    side_landmarks = left_landmarks if visible_side == 'left' else right_landmarks
    
    # Filter connections to only include the visible side and exclude unwanted connections
    for connection in connections:
        start_idx, end_idx = connection
        
        # Skip if either landmark is unwanted
        if start_idx in unwanted_landmarks or end_idx in unwanted_landmarks:
            continue
        
        # Skip if connection involves the wrong side
        if visible_side == 'left' and (start_idx in right_landmarks or end_idx in right_landmarks):
            continue
        if visible_side == 'right' and (start_idx in left_landmarks or end_idx in left_landmarks):
            continue
        
        # Add connections that join left/right sides, but only if they involve the torso
        if visible_side == 'left' and (start_idx in [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value] or 
                                     end_idx in [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value]):
            connections_to_draw.append(connection)
        elif visible_side == 'right' and (start_idx in [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value] or 
                                       end_idx in [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value]):
            connections_to_draw.append(connection)
        # Add connections that are fully on the visible side
        elif (visible_side == 'left' and start_idx in left_landmarks and end_idx in left_landmarks) or \
             (visible_side == 'right' and start_idx in right_landmarks and end_idx in right_landmarks):
            connections_to_draw.append(connection)
    
    # Draw landmark points for visible side only
    for idx, landmark in enumerate(landmarks.landmark):
        # Skip if landmark is not visible enough
        if landmark.visibility < 0.5:
            continue
        
        # Skip if landmark is unwanted
        if idx in unwanted_landmarks:
            continue
        
        # Skip if landmark is not on the visible side (and not central)
        if (visible_side == 'left' and idx in right_landmarks) or \
           (visible_side == 'right' and idx in left_landmarks):
            continue
        
        # Convert normalized coordinates to pixel coordinates
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        
        # Draw landmark point
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
    
    # Draw connections
    for connection in connections_to_draw:
        start_idx, end_idx = connection
        
        # Skip if any of the landmarks is not visible enough
        if landmarks.landmark[start_idx].visibility < 0.5 or landmarks.landmark[end_idx].visibility < 0.5:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        start_point = (int(landmarks.landmark[start_idx].x * width), int(landmarks.landmark[start_idx].y * height))
        end_point = (int(landmarks.landmark[end_idx].x * width), int(landmarks.landmark[end_idx].y * height))
        
        # Draw connection line
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    return image

# Flask route to process video
@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        # Get the uploaded video file and user height
        video_file = request.files['video']
        user_height_cm = float(request.form.get('user_height_cm'))
        
        # Get optional quality parameter (0-100, default 40)
        quality = min(100, max(1, int(request.form.get('quality', '40'))))

        # Save uploaded video to a memory buffer
        video_bytes = video_file.read()
        video_buffer = np.frombuffer(video_bytes, dtype=np.uint8)
        
        # Open the video from memory buffer
        cap = cv2.imdecode(video_buffer, cv2.IMREAD_UNCHANGED)
        if cap is None:
            # Fall back to temporary file if memory buffer doesn't work
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(video_bytes)
                temp_file_path = temp_file.name
            
            cap = cv2.VideoCapture(temp_file_path)
            temp_file_created = True
        else:
            cap = cv2.VideoCapture(io.BytesIO(video_bytes))
            temp_file_created = False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate scaled dimensions for output video to reduce size
        if width > 1280 or height > 720:
            scale_factor = min(640 / width, 360 / height)
            output_width = int(width * scale_factor)
            output_height = int(height * scale_factor)
        else:
            scale_factor = min(0.75, min(640 / width, 360 / height))
            output_width = int(width * scale_factor)
            output_height = int(height * scale_factor)
        
        # Create a list to store frames for video
        frames = []
        
        # Store up to 10 key frames for summary (but still process full video)
        key_frames = []
        
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
        
        # Will store body lengths from frames with good landmark visibility
        body_length_measurements = []
        torso_angle = 0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                
                # Resize to reduce memory usage and output size
                frame = cv2.resize(frame, (output_width, output_height))

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Flag for key frames (for analytics, still process all frames)
                is_key_frame = False

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Determine visible side
                    left_visibility = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
                    right_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
                    visible_side = 'left' if left_visibility > right_visibility else 'right'
                    
                    # Draw only the visible side landmarks
                    image = draw_visible_side_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, visible_side)
                    
                    # Check if landmarks have good visibility for measurements
                    if max(left_visibility, right_visibility) > 0.7:
                        # Calculate body segment lengths for this frame
                        body_lengths = calculate_body_lengths(landmarks, frame.shape, user_height_cm)
                        body_length_measurements.append(body_lengths)
                        is_key_frame = True  # Good visibility makes this a key frame

                    # Get pixel coordinates for various landmarks
                    def to_pixel(landmark):
                        return (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

                    if visible_side == 'left':
                        shoulder = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                        elbow = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
                        wrist = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                        hip = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
                        knee = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
                        ankle = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
                    else:
                        shoulder = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                        elbow = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                        wrist = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                        hip = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                        knee = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
                        ankle = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

                    # Calculate angles
                    shoulder_hip_knee = calculate_angle(shoulder, hip, knee)
                    shoulder_elbow_wrist = calculate_angle(shoulder, elbow, wrist)
                    hip_knee_ankle = calculate_angle(hip, knee, ankle)
                    elbow_shoulder_hip = calculate_angle(elbow, shoulder, hip)
                    torso_angle = calculate_torso_angle(shoulder, hip)  # Torso angle relative to horizontal

                    # Update max and min angles
                    prev_max_hip_knee_ankle = max_angles['hip_knee_ankle']
                    prev_min_hip_knee_ankle = min_angles['hip_knee_ankle']
                    
                    max_angles['shoulder_hip_knee'] = max(max_angles['shoulder_hip_knee'], shoulder_hip_knee)
                    max_angles['shoulder_elbow_wrist'] = max(max_angles['shoulder_elbow_wrist'], shoulder_elbow_wrist)
                    max_angles['hip_knee_ankle'] = max(max_angles['hip_knee_ankle'], hip_knee_ankle)
                    max_angles['elbow_shoulder_hip'] = max(max_angles['elbow_shoulder_hip'], elbow_shoulder_hip)
                    min_angles['shoulder_hip_knee'] = min(min_angles['shoulder_hip_knee'], shoulder_hip_knee)
                    min_angles['shoulder_elbow_wrist'] = min(min_angles['shoulder_elbow_wrist'], shoulder_elbow_wrist)
                    min_angles['hip_knee_ankle'] = min(min_angles['hip_knee_ankle'], hip_knee_ankle)
                    min_angles['elbow_shoulder_hip'] = min(min_angles['elbow_shoulder_hip'], elbow_shoulder_hip)
                    
                    # If a new max or min angle was set, this is a key frame
                    if (max_angles['hip_knee_ankle'] != prev_max_hip_knee_ankle or 
                        min_angles['hip_knee_ankle'] != prev_min_hip_knee_ankle):
                        is_key_frame = True
                    
                    # Add text with angles to the frame
                    cv2.putText(image, f"Visible Side: {visible_side}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Torso Angle: {torso_angle}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Hip-Knee-Ankle: {hip_knee_ankle}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Elbow-Shoulder-Hip: {elbow_shoulder_hip}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Store ALL frames for the video
                frames.append(image)
                
                # Store a few key frames for summary
                if is_key_frame and len(key_frames) < 10:
                    key_frames.append(image)

                # Add progress info to debug long-running processes
                if frame_idx % 10 == 0:
                    print(f"Processed {frame_idx} frames out of {frame_count}")

        cap.release()
        
        # Clean up temporary file if created
        if temp_file_created and 'temp_file_path' in locals():
            import os
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
        # If we couldn't get good measurements, fall back to estimation
        if not body_length_measurements:
            femur_length = user_height_cm * 0.27
            torso_length = user_height_cm * 0.35
            upper_arm_length = user_height_cm * 0.15
            lower_leg_length = user_height_cm * 0.25
            
            avg_body_lengths = {
                'femur_length': femur_length,
                'torso_length': torso_length,
                'upper_arm_length': upper_arm_length,
                'lower_leg_length': lower_leg_length,
                'measurement_method': 'estimated from height'
            }
        else:
            # Calculate average body segment lengths from all frames with good visibility
            avg_femur = np.mean([bl['femur_length'] for bl in body_length_measurements])
            avg_torso = np.mean([bl['torso_length'] for bl in body_length_measurements])
            avg_lower_leg = np.mean([bl['lower_leg_length'] for bl in body_length_measurements])
            avg_upper_arm = np.mean([bl['upper_arm_length'] for bl in body_length_measurements])
            avg_forearm = np.mean([bl['forearm_length'] for bl in body_length_measurements])
            
            avg_body_lengths = {
                'femur_length': avg_femur,
                'torso_length': avg_torso,
                'lower_leg_length': avg_lower_leg,
                'upper_arm_length': avg_upper_arm,
                'forearm_length': avg_forearm,
                'measurement_method': 'calculated from landmarks'
            }

        # Generate recommendations using the average body lengths
        recommendations = generate_recommendations(max_angles, min_angles, avg_body_lengths, torso_angle)

        # Prepare response
        response_data = {
            "max_angles": max_angles,
            "min_angles": min_angles,
            "body_lengths_cm": {k: round(v, 2) if isinstance(v, float) else v for k, v in avg_body_lengths.items()},
            "recommendations": recommendations,
        }
        
        # Now create the full video and add to response
        if len(frames) > 0:
            # Create MP4 video from all processed frames
            encoded_video = create_full_video(frames, fps, output_width, output_height, quality)
            response_data["video"] = encoded_video
            print(f"Video created from {len(frames)} frames")
            
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": traceback_str}), 500

# Add this improved function to create a video from ALL frames
def create_full_video(frames, fps, width, height, quality=40):
    """Create an MP4 video from all frames and return as base64 string"""
    # Create temporary file for video output
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output_file:
        temp_output_path = temp_output_file.name
    
    try:
        # Try different codecs in order of compatibility
        codecs = ['avc1', 'mp4v', 'xvid']
        success = False
        
        for codec in codecs:
            try:
                print(f"Trying codec: {codec}")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height), True)
                
                if not out.isOpened():
                    print(f"Failed to open VideoWriter with codec {codec}")
                    continue
                
                # Write ALL frames to video
                for frame in frames:
                    out.write(frame)
                
                out.release()
                success = True
                print(f"Successfully created video with codec {codec}")
                break
            except Exception as e:
                print(f"Error with codec {codec}: {str(e)}")
                if out:
                    out.release()
        
        if not success:
            raise Exception("Failed to create video with any codec")
        
        # Read the video file into memory
        with open(temp_output_path, 'rb') as f:
            video_data = f.read()
            
        # Base64 encode the video
        encoded_video = base64.b64encode(video_data).decode('utf-8')
        
        # Delete the temp file
        import os
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            
        return encoded_video
    except Exception as e:
        import os
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise e

# New endpoint to get keyframes - can be used if full video is too large
@app.route('/get-keyframes', methods=['POST'])
def get_keyframes():
    try:
        # Same processing as /process-video, but only return key frames
        # Implementation would be similar but only returns the keyframes
        # This is a placeholder for a potential implementation
        return jsonify({"error": "Not implemented yet"}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)