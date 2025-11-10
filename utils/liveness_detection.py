import cv2
import numpy as np
import time
import dlib
import os
from flask import Flask, g
import sqlite3
import threading
from utils.liveness import EnhancedLivenessDetector, LivenessConfig
from utils.drawing import draw_enhanced_landmarks, draw_challenge_indicator, draw_metrics_panel, draw_face_guide
from utils.camera_config import *


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Required for session management
app.config['DATABASE'] = 'attendance.db'

# Global shared camera and predictor
camera_config = None
dlib_predictor = None
dlib_face_detector = None


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(base_dir, "..", "models", "shape_predictor_68_face_landmarks.dat")
    print(f"Attempting to load dlib model from: {predictor_path}")
    dlib_predictor = dlib.shape_predictor(predictor_path)
    dlib_face_detector = dlib.get_frontal_face_detector()
    print("✅ Dlib models loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load dlib models. Reason: {e}")
    
    
# Database connection management
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
        # Enable foreign key constraints
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def reset_state(self):
    """Reset detector state but preserve current challenge if in progress"""
    # Keep these values if a challenge is in progress
    current_challenge = self.current_challenge
    challenge_start_time = self.challenge_start_time
    
    # Reset all other state
    self.__init__()
    
    # Restore challenge state if one was in progress
    if current_challenge:
        self.current_challenge = current_challenge
        self.challenge_start_time = challenge_start_time
        

def generate_frames():
    """
    Stream processed video frames to the client
    """
    global LATEST_CLEAN_FRAME, frame_lock
    camera = get_camera()
    if camera is None:
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not available\r\n'
        return

    # Initialize frame event
    frame_ready = threading.Event()
    
    prev_time = time.time()
    fps = 0
    detector = EnhancedLivenessDetector() # create an instance for this stream
    # Instruction stabilization
    current_instruction = ""
    last_stable_instruction = "Position face in frame"
    last_instruction_change = time.time()
    min_instruction_duration = 1.0  # seconds


    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            processed_frame = frame.copy()
            frame = cv2.flip(frame, 1)
            
            # update the clean frame
            with frame_lock:
                LATEST_CLEAN_FRAME = frame.copy()
                frame_ready.set()  # Signal that a new frame is available

            
            # --- Analysis and Drawing ---
            # Face detection (every frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = dlib_face_detector(gray, 0)
            landmarks = None
            
            if len(faces) > 0:
                # Get the first face
                face = faces[0]
                
                # Get landmarks for the face
                shape = dlib_predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            # Analyze frame
            analysis = detector.analyze_frame(frame, landmarks)
                            
            # Calculate fps
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
            prev_time = curr_time
            
            current_instruction = analysis['instruction']
            
            # Stabilize instructions
            if current_instruction != last_stable_instruction:
                if curr_time - last_instruction_change >= min_instruction_duration:
                    last_stable_instruction = current_instruction
                    last_instruction_change = curr_time
            else:
                analysis['instruction'] = last_stable_instruction
            
            # Draw all UI components onto the frame
            if landmarks is not None:
                draw_enhanced_landmarks(frame, landmarks, analysis)
            
            if analysis.get('challenge_result'):
                draw_challenge_indicator(frame, analysis['challenge_result'])
            else:
                cv2.putText(frame, last_stable_instruction, 
                           (frame.shape[1]//2 - 100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, LivenessConfig.GUIDE, 2)
            
            draw_metrics_panel(frame, analysis, fps)
            draw_face_guide(frame)

            # Yield frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except GeneratorExit:
        print("Client disconnected")
    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        # Good practice to release camera and clear frame
        if camera:
            release_camera()
        with frame_lock:
            LATEST_CLEAN_FRAME = None
        print("Video stream and camera released.")
        