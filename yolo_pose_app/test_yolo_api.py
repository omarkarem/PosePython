#!/usr/bin/env python3
"""
Test script for YOLO 11 Pose Detection API
This script demonstrates how to use the YOLO pose detection API
"""

import requests
import json
import base64
import cv2
import numpy as np
import tempfile
import os

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🏥 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:5001/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Model Status: {data['model_status']}")
            print(f"   Service: {data['service']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def create_test_video():
    """Create a simple test video with a person-like figure"""
    print("🎬 Creating test video...")
    
    # Create a temporary video file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_file.close()
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 2  # seconds
    frame_count = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
    
    for frame_idx in range(frame_count):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a simple stick figure that moves slightly
        offset = int(10 * np.sin(frame_idx * 0.2))
        center_x = width // 2 + offset
        center_y = height // 2
        
        # Head
        cv2.circle(frame, (center_x, center_y - 80), 20, (255, 255, 255), -1)
        
        # Body
        cv2.line(frame, (center_x, center_y - 60), (center_x, center_y + 40), (255, 255, 255), 3)
        
        # Arms
        cv2.line(frame, (center_x, center_y - 20), (center_x - 40, center_y), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y - 20), (center_x + 40, center_y), (255, 255, 255), 3)
        
        # Legs
        cv2.line(frame, (center_x, center_y + 40), (center_x - 30, center_y + 100), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y + 40), (center_x + 30, center_y + 100), (255, 255, 255), 3)
        
        # Add some text
        cv2.putText(frame, f"Test Frame {frame_idx+1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created: {temp_file.name}")
    return temp_file.name

def test_video_processing():
    """Test the video processing endpoint"""
    print("🎯 Testing video processing...")
    
    # Create test video
    video_path = create_test_video()
    
    try:
        # Prepare the request
        files = {
            'video': ('test_video.mp4', open(video_path, 'rb'), 'video/mp4')
        }
        data = {
            'user_height_cm': '175',
            'quality': '75'
        }
        
        print("📤 Sending video to API...")
        response = requests.post(
            "http://localhost:5001/process-video",
            files=files,
            data=data,
            timeout=60  # 60 second timeout
        )
        
        files['video'][1].close()  # Close the file
        
        if response.status_code == 200:
            print("✅ Video processing successful!")
            
            result = response.json()
            
            # Print results
            print("\n📊 Results:")
            print(f"   Model Type: {result.get('model_type', 'Unknown')}")
            
            print("\n🔄 Max Angles:")
            for angle, value in result.get('max_angles', {}).items():
                print(f"   {angle}: {value}°")
            
            print("\n🔄 Min Angles:")
            for angle, value in result.get('min_angles', {}).items():
                print(f"   {angle}: {value}°")
            
            print("\n📏 Body Measurements (cm):")
            body_lengths = result.get('body_lengths_cm', {})
            for measurement, value in body_lengths.items():
                if isinstance(value, (int, float)):
                    print(f"   {measurement}: {value:.1f}")
                else:
                    print(f"   {measurement}: {value}")
            
            print("\n💡 Recommendations:")
            recommendations = result.get('recommendations', {}).get('general', [])
            for i, rec in enumerate(recommendations[:3]):  # Show first 3
                print(f"   {i+1}. {rec.get('component', 'Unknown')}: {rec.get('action', 'No action')}")
                print(f"      Current: {rec.get('current', 'N/A')}, Target: {rec.get('target', 'N/A')}")
            
            # Save processed video if available
            if 'video' in result:
                print("\n💾 Saving processed video...")
                video_data = base64.b64decode(result['video'])
                output_path = 'test_output_video.mp4'
                with open(output_path, 'wb') as f:
                    f.write(video_data)
                print(f"✅ Processed video saved: {output_path}")
            
            return True
            
        else:
            print(f"❌ Video processing failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout - video processing took too long")
        return False
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)

def test_with_real_video():
    """Test with a real video file if available"""
    print("\n🎥 Looking for real video files...")
    
    # Look for common video files in current directory
    video_extensions = ['.mp4', '.avi', '.mov', '.webm']
    video_files = []
    
    for ext in video_extensions:
        files = [f for f in os.listdir('.') if f.lower().endswith(ext)]
        video_files.extend(files)
    
    if not video_files:
        print("ℹ️  No real video files found in current directory")
        print("   You can place a cycling video here and run the test again")
        return False
    
    # Use the first video file found
    video_file = video_files[0]
    print(f"🎬 Testing with real video: {video_file}")
    
    try:
        files = {
            'video': (video_file, open(video_file, 'rb'), 'video/mp4')
        }
        data = {
            'user_height_cm': '175',
            'quality': '50'  # Lower quality for faster processing
        }
        
        print("📤 Processing real video (this may take a while)...")
        response = requests.post(
            "http://localhost:5001/process-video",
            files=files,
            data=data,
            timeout=300  # 5 minute timeout for real videos
        )
        
        files['video'][1].close()
        
        if response.status_code == 200:
            print("✅ Real video processing successful!")
            
            result = response.json()
            
            # Save the result
            output_filename = f"real_video_result_{video_file.split('.')[0]}.json"
            with open(output_filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"📄 Results saved to: {output_filename}")
            
            # Save processed video
            if 'video' in result:
                video_data = base64.b64decode(result['video'])
                output_video = f"processed_{video_file}"
                with open(output_video, 'wb') as f:
                    f.write(video_data)
                print(f"🎬 Processed video saved: {output_video}")
            
            return True
        else:
            print(f"❌ Real video processing failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Real video processing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 YOLO 11 Pose Detection API Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    health_ok = test_health_endpoint()
    
    if not health_ok:
        print("\n❌ Health check failed. Make sure the API is running:")
        print("   python3 yolo_app.py")
        return
    
    print("\n" + "=" * 50)
    
    # Test 2: Video processing with synthetic video
    processing_ok = test_video_processing()
    
    if processing_ok:
        print("\n✅ Basic video processing test passed!")
    else:
        print("\n❌ Basic video processing test failed!")
    
    print("\n" + "=" * 50)
    
    # Test 3: Real video processing (optional)
    real_video_ok = test_with_real_video()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Video Processing: {'✅ PASS' if processing_ok else '❌ FAIL'}")
    print(f"   Real Video Test: {'✅ PASS' if real_video_ok else 'ℹ️  SKIPPED'}")
    
    if health_ok and processing_ok:
        print("\n🎉 YOLO 11 API is working correctly!")
        print("\n🚀 Next steps:")
        print("   1. Deploy to EC2 using deploy_yolo.sh")
        print("   2. Train custom model with your cycling data")
        print("   3. Integrate with your web application")
    else:
        print("\n⚠️  Some tests failed. Check the API logs for details.")

if __name__ == "__main__":
    main() 