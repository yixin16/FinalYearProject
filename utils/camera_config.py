import cv2
import threading

# This will hold the latest frame from the camera without any drawings
LATEST_CLEAN_FRAME = None
# A lock to ensure thread-safe access to the LATEST_CLEAN_FRAME
frame_lock = threading.Lock()
# An event to signal when the first frame is ready, preventing race conditions
first_frame_ready = threading.Event()

# Global shared camera object
camera = None

def get_camera():
    """Initializes and returns a global camera object."""
    global camera
    if camera is None or not camera.isOpened():
        print("Initializing Camera...")
        # Check a few indices, as the primary camera might not be at 0
        for index in range(4):
            cam = cv2.VideoCapture(index)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cam.set(cv2.CAP_PROP_FPS, 30)
                camera = cam
                print(f"✅ Camera found and opened at index {index}.")
                return camera
        print("❌ ERROR: No Camera Found.")
        return None
    return camera

def release_camera():
    """Releases the global camera object."""
    global camera
    if camera:
        print("Releasing camera.")
        camera.release()
        camera = None
        first_frame_ready.clear()