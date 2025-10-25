"""
Real-time Face Detection Web Application
This application uses YOLOv8 and Flask to provide real-time face detection via a web interface.
"""

import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import threading
import logging
import os
import atexit
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
import torch
from ultralytics.nn.modules import Conv  # 正确

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class Config:
    """Application configuration"""
    CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', 0))
    MODEL_PATH = os.environ.get('MODEL_PATH', 'yolov8n.pt')
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'

class Camera:
    """
    Camera management class for handling video capture operations.
    Provides thread-safe access to camera resources.
    """
    
    def __init__(self, camera_index=0):
        """
        Initialize Camera instance.
        
        Args:
            camera_index (int): Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.lock = threading.Lock()
        self.is_initialized = False
        
    def initialize(self):
        """
        Initialize the camera with optimal settings.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        with self.lock:
            if self.is_initialized:
                return True
                
            if self.cap is not None:
                self.release()
                
            try:
                logger.info(f"Attempting to open camera with index {self.camera_index}")
                self.cap = cv2.VideoCapture(self.camera_index)
                
                if not self.cap.isOpened():
                    logger.error(f"Failed to open camera with index {self.camera_index}")
                    return False
                
                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.is_initialized = True
                logger.info(f"Camera {self.camera_index} initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Error initializing camera: {str(e)}")
                self.is_initialized = False
                return False
    
    def read_frame(self):
        """
        Read a frame from the camera.
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the image data
        """
        with self.lock:
            if not self.is_initialized or self.cap is None or not self.cap.isOpened():
                logger.warning("Camera is not properly initialized")
                return False, None
                
            try:
                success, frame = self.cap.read()
                if not success:
                    logger.warning("Failed to read frame from camera")
                return success, frame
            except Exception as e:
                logger.error(f"Error reading frame from camera: {str(e)}")
                return False, None
    
    def release(self):
        """
        Release the camera resources.
        """
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                    logger.info("Camera released successfully")
                except Exception as e:
                    logger.error(f"Error releasing camera: {str(e)}")
                finally:
                    self.cap = None
                    self.is_initialized = False

class FaceDetector:
    """
    Face detection class using YOLOv8 model.
    Provides thread-safe face detection operations.
    """
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize FaceDetector instance.
        
        Args:
            model_path (str): Path to the YOLOv8 model file
        """
        self.model_path = model_path
        self.model = None
        self.lock = threading.Lock()
        self.is_loaded = False
        
    def load_model(self):
        """
        Load the YOLOv8 model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        with self.lock:
            if self.is_loaded:
                return True
                
            if self.model is not None:
                return True
                
            try:
                with torch.serialization.safe_globals([DetectionModel , Sequential , Conv]):
                    logger.info(f"Loading model from {self.model_path}")
                    self.model = YOLO(self.model_path)
                    self.is_loaded = True
                    logger.info(f"Model {self.model_path} loaded successfully")
                    return True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self.is_loaded = False
                return False
    
    def detect_faces(self, frame):
        """
        Run face detection on a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Annotated frame with detection results
        """
        with self.lock:
            if not self.is_loaded or self.model is None:
                logger.warning("Model is not loaded")
                return frame
                
            try:
                results = self.model(frame)
                annotated_frame = results[0].plot()
                return annotated_frame
            except Exception as e:
                logger.error(f"Error during face detection: {str(e)}")
                return frame

# Global instances
camera = Camera(Config.CAMERA_INDEX)
face_detector = FaceDetector(Config.MODEL_PATH)

def generate_frames():
    """
    Generate frames from the camera with face detection.
    
    Yields:
        Encoded frame data for streaming
    """
    # Initialize camera and model when first accessed
    if not camera.initialize():
        logger.error("Failed to initialize camera")
        # Yield an error frame or message
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\nCamera initialization failed\r\n')
        return
    
    if not face_detector.load_model():
        logger.error("Failed to load model")
        # Yield an error frame or message
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\nModel loading failed\r\n')
        return
    
    frame_count = 0
    while True:
        try:
            success, frame = camera.read_frame()
            if not success:
                logger.warning("Failed to read frame from camera")
                # Try to reinitialize camera
                if not camera.initialize():
                    logger.error("Failed to reinitialize camera")
                    break
                continue
            
            # Run face detection
            annotated_frame = face_detector.detect_faces(frame)
            
            # Encode the frame in JPEG format with optimized quality
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                logger.warning("Failed to encode frame")
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
            # Frame rate limiting to prevent overwhelming the client
            logger.setLevel(logging.DEBUG)
            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug(f"【100】Processed {frame_count} frames")
            if frame_count % 10 == 0:
                logger.debug(f"【10】Processed {frame_count} frames")
            
            # Lab2 Part2 Begin #
            # Hint : use logger.info to output the current framenumber, and current frame_bytes per 10 frame # 
            # Lab2 Part2 End #

        except Exception as e:
            logger.error(f"Error in frame generation: {str(e)}")
            break

@app.route('/')
def index():
    """
    Serve the main page.
    
    Returns:
        Rendered HTML template
    """
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Serve the video feed.
    
    Returns:
        Response with multipart stream of video frames
    """
    logger.info("Starting video feed")
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.before_request
def initialize():
    """
    Initialize resources before first request.
    """
    # Use a flag to ensure initialization only happens once
    if not hasattr(app, 'initialized'):
        logger.info("Initializing application resources...")
        # Any initialization code would go here if needed
        # For now, we're just setting the flag
        app.initialized = True

def cleanup_resources():
    """
    Clean up all resources.
    """
    logger.info("Cleaning up resources...")
    camera.release()

# Register cleanup function to be called on exit
atexit.register(cleanup_resources)

if __name__ == '__main__':
    try:
        logger.info(f"Starting Flask application on {Config.HOST}:{Config.PORT}")
        app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
    finally:
        cleanup_resources()