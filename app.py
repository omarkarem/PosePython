from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Apex point
    c = np.array(c)  # Third point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

# Function to estimate body segment lengths from user height
def estimate_body_lengths(user_height_cm):
    femur_length = user_height_cm * 0.27  # ~27% of height
    torso_length = user_height_cm * 0.35  # ~35% of height
    upper_arm_length = user_height_cm * 0.15  # ~15% of height
    lower_leg_length = user_height_cm * 0.25  # ~25% of height
    return femur_length, torso_length, upper_arm_length, lower_leg_length

# Function to generate recommendations
def generate_recommendations(max_angles, min_angles, femur_length, torso_length, torso_angle):
    recommendations = {
        'general': [],
        'endurance': [],
        'aggressive': []
    }

    # Analyze saddle height (hip_knee_ankle at BDC)
    bdc_angle = max_angles['hip_knee_ankle']
    if bdc_angle > 155:
        delta_theta = bdc_angle - 155
        delta_h = femur_length * np.sin(np.radians(delta_theta))
        recommendations['general'].append(f"Saddle height is too high. Lower saddle height by {delta_h:.2f} cm.")
        recommendations['endurance'].append("Lower saddle height slightly for endurance comfort.")
        recommendations['aggressive'].append("No change needed for aggressive posture.")
    elif bdc_angle < 150:
        delta_theta = 150 - bdc_angle
        delta_h = femur_length * np.sin(np.radians(delta_theta))
        recommendations['general'].append(f"Saddle height is too low. Raise saddle height by {delta_h:.2f} cm.")
        recommendations['endurance'].append("Raise saddle height slightly for endurance posture.")
        recommendations['aggressive'].append("Raise saddle height slightly for aggressive posture.")
    else:
        recommendations['general'].append("Saddle height is within optimal range. No changes needed.")

    # Analyze saddle fore/aft position (hip_knee_ankle at TDC)
    tdc_angle = min_angles['hip_knee_ankle']
    if tdc_angle > 110:
        delta_theta = tdc_angle - 110
        delta_x = femur_length * np.cos(np.radians(delta_theta))  # Horizontal adjustment
        recommendations['general'].append(f"Saddle is too far back. Move saddle forward by {delta_x:.2f} mm.")
        recommendations['endurance'].append("Move saddle forward slightly for endurance posture.")
        recommendations['aggressive'].append("Move saddle forward slightly for aggressive posture.")
    elif tdc_angle < 90:
        delta_theta = 90 - tdc_angle
        delta_x = femur_length * np.cos(np.radians(delta_theta))  # Horizontal adjustment
        recommendations['general'].append(f"Saddle is too far forward. Move saddle backward by {delta_x:.2f} mm.")
        recommendations['endurance'].append("Move saddle backward slightly for endurance posture.")
        recommendations['aggressive'].append("Move saddle backward slightly for aggressive posture.")
    else:
        recommendations['general'].append("Saddle fore/aft position is within optimal range. No changes needed.")

    # Analyze handlebar height (torso angle)
    if torso_angle > 45:
        delta_h = torso_length * np.sin(np.radians(torso_angle - 45))
        recommendations['general'].append(f"Handlebars are too high. Lower handlebars by {delta_h:.2f} cm.")
        recommendations['endurance'].append("Lower handlebars slightly for endurance posture.")
        recommendations['aggressive'].append("Lower handlebars further for aggressive posture.")
    elif torso_angle < 30:
        delta_h = torso_length * np.sin(np.radians(30 - torso_angle))
        recommendations['general'].append(f"Handlebars are too low. Raise handlebars by {delta_h:.2f} cm.")
        recommendations['endurance'].append("Raise handlebars for endurance posture.")
        recommendations['aggressive'].append("Raise handlebars slightly for aggressive posture.")
    else:
        recommendations['general'].append("Handlebar height is within optimal range. No changes needed.")

    # Analyze handlebar reach (elbow_shoulder_hip angle)
    elbow_shoulder_hip = max_angles['elbow_shoulder_hip']
    if elbow_shoulder_hip > 100:
        delta_reach = (elbow_shoulder_hip - 100) * 0.1  # Convert angle difference to approximate stem length adjustment
        recommendations['general'].append(f"Handlebars are too far away. Shorten stem length by {delta_reach:.2f} cm.")
        recommendations['endurance'].append("Shorten stem length for endurance posture.")
        recommendations['aggressive'].append("Shorten stem length slightly for aggressive posture.")
    elif elbow_shoulder_hip < 80:
        delta_reach = (80 - elbow_shoulder_hip) * 0.1  # Convert angle difference to approximate stem length adjustment
        recommendations['general'].append(f"Handlebars are too close. Lengthen stem length by {delta_reach:.2f} cm.")
        recommendations['endurance'].append("Lengthen stem length for endurance posture.")
        recommendations['aggressive'].append("Lengthen stem length slightly for aggressive posture.")
    else:
        recommendations['general'].append("Handlebar reach is within optimal range. No changes needed.")

    return recommendations

# Flask route to process video
@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        # Get the uploaded video file and user height
        video_file = request.files['video']
        user_height_cm = float(request.form.get('user_height_cm'))

        # Save the video to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = temp_file.name
            video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)

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

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    left_visibility = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
                    right_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
                    visible_side = 'left' if left_visibility > right_visibility else 'right'

                    if visible_side == 'left':
                        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    else:
                        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                    def to_pixel(landmark):
                        return (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

                    shoulder_px = to_pixel(shoulder)
                    elbow_px = to_pixel(elbow)
                    wrist_px = to_pixel(wrist)
                    hip_px = to_pixel(hip)
                    knee_px = to_pixel(knee)
                    ankle_px = to_pixel(ankle)

                    # Calculate angles
                    shoulder_hip_knee = calculate_angle(shoulder_px, hip_px, knee_px)
                    shoulder_elbow_wrist = calculate_angle(shoulder_px, elbow_px, wrist_px)
                    hip_knee_ankle = calculate_angle(hip_px, knee_px, ankle_px)
                    elbow_shoulder_hip = calculate_angle(elbow_px, shoulder_px, hip_px)
                    torso_angle = calculate_angle(shoulder_px, hip_px, (hip_px[0] + 1, hip_px[1]))  # Torso angle relative to horizontal

                    # Update max and min angles
                    max_angles['shoulder_hip_knee'] = max(max_angles['shoulder_hip_knee'], shoulder_hip_knee)
                    max_angles['shoulder_elbow_wrist'] = max(max_angles['shoulder_elbow_wrist'], shoulder_elbow_wrist)
                    max_angles['hip_knee_ankle'] = max(max_angles['hip_knee_ankle'], hip_knee_ankle)
                    max_angles['elbow_shoulder_hip'] = max(max_angles['elbow_shoulder_hip'], elbow_shoulder_hip)
                    min_angles['shoulder_hip_knee'] = min(min_angles['shoulder_hip_knee'], shoulder_hip_knee)
                    min_angles['shoulder_elbow_wrist'] = min(min_angles['shoulder_elbow_wrist'], shoulder_elbow_wrist)
                    min_angles['hip_knee_ankle'] = min(min_angles['hip_knee_ankle'], hip_knee_ankle)
                    min_angles['elbow_shoulder_hip'] = min(min_angles['elbow_shoulder_hip'], elbow_shoulder_hip)

        cap.release()

        # Estimate body lengths
        femur_length, torso_length, _, _ = estimate_body_lengths(user_height_cm)

        # Generate recommendations
        recommendations = generate_recommendations(max_angles, min_angles, femur_length, torso_length, torso_angle)

        # Return the response with all required fields
        return jsonify({
            "max_angles": max_angles,
            "min_angles": min_angles,
            "femur_length_cm": round(femur_length, 2),
            "torso_length_cm": round(torso_length, 2),
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)