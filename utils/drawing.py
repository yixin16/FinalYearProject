import cv2
from utils.liveness import LivenessConfig # Use a relative import


def draw_enhanced_landmarks(frame, landmarks, analysis):
    """Draw landmarks with enhanced visual feedback"""
    if landmarks is None:
        return
    
    # Color based on liveness confidence
    confidence = analysis.get('confidence', 0)
    if confidence > 0.7:
        color = LivenessConfig.SUCCESS
    elif confidence > 0.4:
        color = LivenessConfig.WARNING
    else:
        color = LivenessConfig.ERROR
    
    # MODIFIED: Reduced line thickness from 2 to 1 for a subtler look
    line_thickness = 1

    # Draw face outline
    jaw_line = landmarks[0:17]
    for i in range(len(jaw_line) - 1):
        cv2.line(frame, tuple(jaw_line[i]), tuple(jaw_line[i + 1]), color, line_thickness)
    
    # Draw eyes with blink detection
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    
    eye_color = LivenessConfig.SUCCESS if analysis['metrics']['ear'] < LivenessConfig.EAR_THRESHOLD else color
    cv2.polylines(frame, [left_eye], True, eye_color, line_thickness)
    cv2.polylines(frame, [right_eye], True, eye_color, line_thickness)
    
    # Draw mouth
    mouth = landmarks[48:68]
    mouth_color = LivenessConfig.CHALLENGE if analysis['metrics']['mar'] > LivenessConfig.MAR_THRESHOLD else color
    cv2.polylines(frame, [mouth], True, mouth_color, line_thickness)
    
    # Draw nose
    nose = landmarks[27:36]
    cv2.polylines(frame, [nose], False, color, line_thickness)
    
    # Draw eyebrows
    left_brow = landmarks[17:22]
    right_brow = landmarks[22:27]
    cv2.polylines(frame, [left_brow], False, color, line_thickness)
    cv2.polylines(frame, [right_brow], False, color, line_thickness)
    

def draw_challenge_indicator(frame, challenge_result):
    """Draw challenge progress indicator"""
    if not challenge_result:
        # Show default guidance when no challenge is active
        cv2.putText(frame, "Face the camera", (frame.shape[1]//2 - 100, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, LivenessConfig.GUIDE, 2)
        return
    
    challenge_type = challenge_result['type']
    progress = challenge_result['progress']
    instruction = challenge_result['instruction']
    
    # Challenge panel
    panel_y = 20 # MODIFIED: Moved panel higher
    panel_height = 60 # MODIFIED: Reduced panel height from 80 to 60
    cv2.rectangle(frame, (10, panel_y), (frame.shape[1] - 10, panel_y + panel_height), 
                  LivenessConfig.BACKGROUND, -1)
    cv2.rectangle(frame, (10, panel_y), (frame.shape[1] - 10, panel_y + panel_height), 
                  LivenessConfig.CHALLENGE, 2)
    
    # Challenge icon
    icon_map = {
        'blink_twice': 'BLINK', 'smile': 'SMILE', 'turn_left': 'LEFT',
        'turn_right': 'RIGHT', 'nod': 'NOD', 'open_mouth': 'MOUTH'
    }
    
    icon = icon_map.get(challenge_type, '?')
    # MODIFIED: Adjusted position and font size
    cv2.putText(frame, icon, (25, panel_y + 38), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, LivenessConfig.CHALLENGE, 2)
    
    # Instruction text
    # MODIFIED: Adjusted position and font size
    cv2.putText(frame, instruction, (110, panel_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, LivenessConfig.TEXT, 1)
    
    # Progress bar
    # MODIFIED: Adjusted position and width
    bar_width = frame.shape[1] - 130 
    bar_height = 10 # MODIFIED: Made progress bar thinner
    bar_y = panel_y + 40 # MODIFIED: Adjusted position
    
    cv2.rectangle(frame, (110, bar_y), (110 + bar_width, bar_y + bar_height), 
                  (60, 60, 60), -1)
    
    progress_width = int(bar_width * progress)
    cv2.rectangle(frame, (110, bar_y), (110 + progress_width, bar_y + bar_height), 
                  LivenessConfig.CHALLENGE, -1)
    
    # Progress percentage - removed for a cleaner look


def draw_metrics_panel(frame, analysis, fps):
    """Draw comprehensive metrics panel"""
    metrics = analysis.get('metrics', {})
    
    # Panel background
    panel_height = 80 # MODIFIED: Reduced panel height from 120 to 80
    panel_y = frame.shape[0] - panel_height
    cv2.rectangle(frame, (0, panel_y), (frame.shape[1], frame.shape[0]), 
                  LivenessConfig.BACKGROUND, -1)
    
    # Status indicator
    confidence = analysis.get('confidence', 0)
    status_color = (
        LivenessConfig.SUCCESS if confidence > 0.8 else
        LivenessConfig.WARNING if confidence > 0.5 else
        LivenessConfig.ERROR
    )
    
    # MODIFIED: Made circle smaller and adjusted position
    cv2.circle(frame, (25, panel_y + 30), 10, status_color, -1)
    
    # Main status text
    status_text = (
        "VERIFIED" if analysis.get('is_live', False) else
        "ANALYZING" if confidence > 0.3 else
        "POSITION FACE"
    )
    
    # MODIFIED: Adjusted position and font size
    cv2.putText(frame, status_text, (45, panel_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Confidence bar
    conf_bar_width = 150 # MODIFIED: Reduced bar width
    conf_bar_height = 8 # MODIFIED: Made bar thinner
    conf_bar_x = 45
    conf_bar_y = panel_y + 55 # MODIFIED: Adjusted position
    
    # MODIFIED: Removed "Confidence:" text to save space
    cv2.rectangle(frame, (conf_bar_x, conf_bar_y), 
                  (conf_bar_x + conf_bar_width, conf_bar_y + conf_bar_height), 
                  (60, 60, 60), -1)
    
    conf_width = int(conf_bar_width * confidence)
    cv2.rectangle(frame, (conf_bar_x, conf_bar_y), 
                  (conf_bar_x + conf_width, conf_bar_y + conf_bar_height), 
                  status_color, -1)
    
    # MODIFIED: Display confidence percentage next to the bar
    cv2.putText(frame, f"{int(confidence * 100)}%", 
                (conf_bar_x + conf_bar_width + 10, conf_bar_y + 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, LivenessConfig.TEXT, 1)
    
    # Right side metrics
    right_x = frame.shape[1] - 150 # MODIFIED: Adjusted position
    
    # MODIFIED: Stacked metrics neatly with smaller font
    cv2.putText(frame, f"FPS: {fps:.1f}", (right_x, panel_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, LivenessConfig.TEXT, 1)
    
    blinks = metrics.get('blinks', 0)
    cv2.putText(frame, f"Blinks: {blinks}", (right_x, panel_y + 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, LivenessConfig.TEXT, 1)
    
    anti_spoof = analysis.get('anti_spoofing_score', 0)
    cv2.putText(frame, f"Anti-spoof: {int(anti_spoof * 100)}%", (right_x, panel_y + 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, LivenessConfig.TEXT, 1)



def draw_face_guide(frame):
    """Draw face positioning guide"""
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    
    # Face oval guide
    # MODIFIED: Slightly larger oval to better frame the face
    axes = (int(w * 0.30), int(h * 0.40)) 
    # MODIFIED: Thinner guide line
    cv2.ellipse(frame, center, axes, 0, 0, 360, LivenessConfig.GUIDE, 2) 
    
    # Corner guides and crosshair removed for a cleaner look