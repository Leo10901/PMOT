import mediapipe as mp
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions, RunningMode
from mediapipe.tasks.python.components.containers import Detection
import cv2
import time
import subprocess
import random

# Aliases
BaseOptions = mp.tasks.BaseOptions
DetectionResult = Detection
ObjectDetector = ObjectDetector
ObjectDetectorOptions = ObjectDetectorOptions
VisionRunningMode = RunningMode

last_alert_time = 0
cooldown_seconds = 10
phone_visible = False
FEM = ["RobotWatcher.mp3","HangOn.mp3","Ambition.mp3"]


# Callback function to print results
def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    global last_alert_time, phone_visible
    
    phone_in_frame = False
    
    for detection in result.detections:
        label = detection.categories[0].category_name
        confidence = detection.categories[0].score
        if label.lower() == "cell phone" and confidence > 0.5:
            phone_in_frame = True
            
            break
    
    current_time = time.time()
    time_since_last_alert = current_time - last_alert_time
    
    
    if phone_in_frame and not phone_visible and time_since_last_alert > cooldown_seconds:
        
        alert_audio = random.choice(FEM)
        subprocess.Popen(["afplay", alert_audio])
        last_alert_time = current_time
    
    phone_visible = phone_in_frame


options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result
)


cap = cv2.VideoCapture(0)

with ObjectDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to MediaPipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Get timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        
        # Run detection with timestamp
        detector.detect_async(mp_image, timestamp_ms)
        
        # Show the webcam feed
        cv2.imshow("Ultron", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()