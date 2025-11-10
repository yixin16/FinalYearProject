import cv2
import numpy as np
import time
from collections import deque
from scipy.spatial import distance as dist
import random
import math
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class LivenessConfig:
    # Enhanced Color Scheme
    PRIMARY = (0, 150, 255)        # Vibrant Orange
    SECONDARY = (100, 255, 255)    # Cyan (landmarks)
    SUCCESS = (50, 255, 50)        # Bright Green
    WARNING = (0, 255, 255)        # Yellow
    ERROR = (0, 0, 255)            # Red
    CHALLENGE = (255, 100, 255)    # Magenta
    TEXT = (255, 255, 255)         # White
    BACKGROUND = (40, 40, 40)      # Darker Gray
    GUIDE = (200, 200, 200)        # Light Gray
    ACCENT = (255, 165, 0)         # Orange accent

    # Performance Settings
    TARGET_FPS = 30
    DETECTION_INTERVAL = 1  # Process every frame for better responsiveness
    FRAME_BUFFER_SIZE = 10
    
    # Enhanced Detection Thresholds
    EAR_THRESHOLD = 0.23
    EAR_CONSEC_FRAMES = 3
    MAR_THRESHOLD = 0.65  # Mouth Aspect Ratio
    HEAD_POSE_THRESHOLD = 25  # degrees
    TEXTURE_THRESHOLD = 100  # Texture analysis
    MOTION_THRESHOLD = 2.0
    
    # Liveness Challenge Configuration
    CHALLENGE_TIMEOUT = 15  # seconds
    BLINK_PATTERN_TIMEOUT = 8
    SMILE_DURATION = 2
    HEAD_MOVEMENT_RANGE = 15
    
    # UI Constants
    STATUS_BAR_HEIGHT = 150
    FONT_SCALE = 0.7
    LANDMARK_RADIUS = 2
    LANDMARK_THICKNESS = -1
    GUIDE_ALPHA = 0.2
    HIGHLIGHT_GLOW = 6
    CONNECTION_THICKNESS = 2
    
    # Quality Assessment
    MIN_FACE_SIZE = 120
    MAX_FACE_SIZE = 400
    BRIGHTNESS_RANGE = (50, 200)
    CONTRAST_THRESHOLD = 30
    
    
class EnhancedLivenessDetector:
    def __init__(self):
        self.config = LivenessConfig()
        self.challenges = [
            "blink_twice", "smile", "turn_left", "turn_right", 
            "nod", "open_mouth"
        ]
        self.reset_state()
        
    def reset_state(self):
        """Reset detector state"""
        self.ear_history = deque(maxlen=10)
        self.mar_history = deque(maxlen=10)
        self.pose_history = deque(maxlen=15)
        self.motion_history = deque(maxlen=20)
        self.texture_history = deque(maxlen=8)
        
        self.blink_count = 0
        self.smile_frames = 0
        self.head_movements = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
        self.mouth_open_frames = 0
        self.consecutive_blinks = 0
        
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_progress = 0.0
        self.challenge_completed = False
        
        self.liveness_score = 0.0
        self.quality_score = 0.0
        self.anti_spoofing_score = 0.0
        
        self.last_frame = None
        self.frame_count = 0
        
    def select_random_challenge(self) -> str:
        """Select a random liveness challenge"""
        return random.choice(self.challenges)
    
    def calculate_ear(self, eye_landmarks) -> float:
        """Enhanced Eye Aspect Ratio calculation"""
        try:
            # Vertical distances
            A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
            # Horizontal distance
            C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
            return (A + B) / (2.0 * C + 1e-6)
        except:
            return 0.0
    
    def calculate_mar(self, mouth_landmarks) -> float:
        """Calculate Mouth Aspect Ratio"""
        try:
            # Vertical distances
            A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])  # 51-59
            B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])   # 53-57
            # Horizontal distance
            C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])   # 49-55
            return (A + B) / (2.0 * C + 1e-6)
        except:
            return 0.0
    
    def calculate_head_pose(self, landmarks, frame_shape) -> Tuple[float, float, float]:
        """Calculate head pose angles (pitch, yaw, roll)."""
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
            ], dtype="double")
            
            image_points = np.array([
                landmarks[30], landmarks[8], landmarks[36],
                landmarks[45], landmarks[48], landmarks[54]
            ], dtype="double")
            
            focal_length = frame_shape[1]
            center = (frame_shape[1] / 2, frame_shape[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype="double")
            
            dist_coeffs = np.zeros((4, 1))
            
            success, rotation_vector, _ = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
                singular = sy < 1e-6
                
                if not singular:
                    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    y = math.atan2(-rotation_matrix[2, 0], sy)
                    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                    y = math.atan2(-rotation_matrix[2, 0], sy)
                    z = 0
                return tuple(math.degrees(angle) for angle in [x, y, z]) # pitch, yaw, roll
            
        except Exception:
            pass
        
        return (0.0, 0.0, 0.0)
    
    def calculate_texture_score(self, face_roi) -> float:
        """Calculate texture analysis score for anti-spoofing."""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return min(100.0, laplacian_var / 10) # Normalize
        except Exception:
            return 0.0
    
    def calculate_motion_score(self, current_frame, previous_frame) -> float:
        """Calculate motion score between frames."""
        if previous_frame is None: return 0.0
        try:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(current_gray, previous_gray)
            return np.mean(diff)
        except Exception:
            return 0.0
    
    def assess_image_quality(self, face_roi) -> Dict[str, float]:
        """Comprehensive image quality assessment."""
        quality_metrics = {'brightness': 0.0, 'contrast': 0.0, 'sharpness': 0.0, 'overall': 0.0}
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            quality_metrics['brightness'] = max(0, min(100, 100 - abs(brightness - 128)))
            quality_metrics['contrast'] = min(100, contrast)
            quality_metrics['sharpness'] = min(100, sharpness / 20)
            quality_metrics['overall'] = np.mean(list(quality_metrics.values())[:-1])
        except Exception:
            pass
        return quality_metrics
    
    def process_challenge(self, challenge_type: str, landmarks, frame_shape) -> Tuple[bool, float, str]:
        """Handles the logic for the active liveness challenge."""
        if self.challenge_start_time is None:
            self.challenge_start_time = time.time()
            return False, 0.0, f"Perform: {challenge_type.replace('_', ' ')}"
        
        elapsed = time.time() - self.challenge_start_time
        if elapsed > self.config.CHALLENGE_TIMEOUT:
            self.current_challenge = None
            self.challenge_start_time = None
            return False, 0.0, "Challenge timeout"
        
        progress, instruction, completed = 0.0, "", False
        pitch, yaw, _ = self.calculate_head_pose(landmarks, frame_shape)
        
        if challenge_type == "blink_twice":
            progress = min(1.0, self.blink_count / 2)
            instruction = f"Blink twice ({self.blink_count}/2)"
            completed = self.blink_count >= 2
        elif challenge_type == "smile":
            mar = self.calculate_mar(landmarks[48:68])
            if mar < 0.4: # Smilimg typically has low MAR and wide mouth
                self.smile_frames += 1
            else:
                self.smile_frames = max(0, self.smile_frames - 2)
            req_frames = int(self.config.SMILE_DURATION * self.config.TARGET_FPS / 2)
            progress = min(1.0, self.smile_frames / req_frames)
            instruction = f"Smile ({int(progress*100)}%)"
            completed = self.smile_frames >= req_frames
        elif challenge_type == "turn_left":
            if yaw < -self.config.HEAD_MOVEMENT_RANGE: self.head_movements['left'] += 1
            progress = min(1.0, self.head_movements['left'] / 10)
            instruction = "Turn head right"
            completed = self.head_movements['left'] >= 10
        elif challenge_type == "turn_right":
            if yaw > self.config.HEAD_MOVEMENT_RANGE: self.head_movements['right'] += 1
            progress = min(1.0, self.head_movements['right'] / 10)
            instruction = "Turn head left"
            completed = self.head_movements['right'] >= 10
        elif challenge_type == "open_mouth":
            mar = self.calculate_mar(landmarks[48:68])
            if mar > self.config.MAR_THRESHOLD: self.mouth_open_frames += 1
            req_frames = int(1 * self.config.TARGET_FPS) # 1 second
            progress = min(1.0, self.mouth_open_frames / req_frames)
            instruction = "Open your mouth"
            completed = self.mouth_open_frames >= req_frames
        elif challenge_type == "nod":
            if pitch > self.config.HEAD_MOVEMENT_RANGE: self.head_movements['down'] += 1
            progress = min(1.0, self.head_movements['down'] / 10)
            instruction = "Nod your head"
            completed = self.head_movements['down'] >= 10
            
        return completed, progress, instruction

    def analyze_frame(self, frame, landmarks):
        analysis = {
            'is_live': False, 
            'confidence': 0.0, 
            'instruction': "Position face in frame",
            'challenge_result': 
                {'type': self.current_challenge, 
                 'progress': 0.0, 
                 'instruction': 'Position face', 
                 'completed': False
                 },
            'metrics': 
                {'blinks': self.blink_count, 
                 'ear': 0.5, 'mar': 0.1
                },
            'anti_spoofing_score': 0.0
        }
        
        if landmarks is None: 
            self.reset_state()
            return analysis
        
        try:
            # Calculate face ROI for anti-spoofing
            x_coords = [p[0] for p in landmarks]
            y_coords = [p[1] for p in landmarks]
            x, y = min(x_coords), min(y_coords)
            w, h = max(x_coords) - x, max(y_coords) - y
            face_roi = frame[y:y+h, x:x+w]
            
            # Calculate anti-spoofing metrics
            texture_score = self.calculate_texture_score(face_roi) / 100  # Normalize to 0-1
            motion_score = min(1.0, self.calculate_motion_score(frame, self.last_frame) / 50)  # Normalize
            self.last_frame = frame.copy()
            
            # Combine scores (you can adjust weights as needed)
            self.anti_spoofing_score = 0.7 * texture_score + 0.3 * motion_score
            
            # Update analysis dict
            analysis['anti_spoofing_score'] = self.anti_spoofing_score
        
            ear = (self.calculate_ear(landmarks[42:48]) + self.calculate_ear(landmarks[36:42]))/2.0
            if ear<self.config.EAR_THRESHOLD: 
                self.consecutive_blinks+=1
                
            elif self.consecutive_blinks>=self.config.EAR_CONSEC_FRAMES:
                self.blink_count+=1; self.consecutive_blinks=0
                
            if self.current_challenge is None and not self.challenge_completed:
                self.current_challenge = self.select_random_challenge()
                self.reset_state(keep_challenge=True)
                
            challenge_result = analysis['challenge_result']
            
            if self.current_challenge:
                completed, progress, instruction = self.process_challenge(self.current_challenge, landmarks, frame.shape)
                challenge_result.update({'type': self.current_challenge, 'progress': progress, 'instruction': instruction})
                if completed:
                    self.challenge_completed = True; self.current_challenge = None
                    challenge_result['instruction'] = "Challenge Complete!"
                    
            confidence = 1.0 if self.challenge_completed else challenge_result['progress']
            analysis.update({
                'is_live': self.challenge_completed, 'confidence': confidence,
                'instruction': challenge_result['instruction'],
                'metrics': {'blinks': self.blink_count, 
                            'ear': ear, 
                            'mar': self.calculate_mar(landmarks[48:68])},
                            })
        except Exception as e: analysis['instruction'] = f"Error: {e}"
        return analysis
    