import os
import cv2
import numpy as np
import warnings
import time
import base64
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

# Configuration
MODEL_DIR = "./resources/anti_spoof_models"
# claude
REAL_THRESHOLD = 0.9  # min confidence to call it "real"
# FAKE_THRESHOLD = 0.6  # min confidence to call it "fake"
# 
DEVICE_ID = 0
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Initialize FastAPI app
app = FastAPI(
    title="Face Anti-Spoofing API",
    description="Detect fake faces (spoofing attacks) in images using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (load once at startup)
model_test = None
image_cropper = None


# Pydantic models for request/response
class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string")


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

# #me
# class PredictionResponse(BaseModel):
#     success: bool
#     prediction: Optional[str] = None
#     score: Optional[float] = None
#     confidence: Optional[float] = None
#     label: Optional[int] = None
#     processing_time: Optional[float] = None
#     models_used: Optional[int] = None
#     bbox: Optional[BoundingBox] = None
#     error: Optional[str] = None

# claude
class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[str] = None  # now: "real" | "fake" | "uncertain"
    score: Optional[float] = None
    reject_reason: Optional[str] = None
    confidence: Optional[float] = None
    real_score: Optional[float] = None
    fake_score: Optional[float] = None
    unknown_score: Optional[float] = None
    label: Optional[int] = None
    processing_time: Optional[float] = None
    models_used: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    error: Optional[str] = None

class VisualizationResponse(PredictionResponse):
    result_image: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_dir: str
    models_available: int


class APIInfo(BaseModel):
    name: str
    version: str
    endpoints: dict
    accepted_formats: list


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model_test, image_cropper
    
    print("Initializing models...")
    model_test = AntiSpoofPredict(DEVICE_ID)
    image_cropper = CropImage()
    print("Models initialized successfully!")
    
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Model directory {MODEL_DIR} does not exist!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down...")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def check_image(image: np.ndarray) -> tuple[bool, Optional[str]]:
#     """Check if image has appropriate aspect ratio"""
#     height, width, channel = image.shape
#     if width/height != 3/4:
#         return False, "Image aspect ratio should be 3:4 (width:height)"
#     return True, None
def check_image(image: np.ndarray) -> tuple[bool, Optional[str]]:
    return True, None  # disable the 3:4 restriction entirely

async def decode_image(file: UploadFile) -> np.ndarray:
    """Decode uploaded file to OpenCV image"""
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image. Please ensure image is valid"
        )
    
    return image


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image"""
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    try:
        # Decode base64 to image
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode base64 image: {str(e)}"
        )


def process_image(image: np.ndarray, model_dir: str = MODEL_DIR) -> dict:
    """
    Process image through anti-spoofing models
    
    Args:
        image: numpy array (BGR format from cv2)
        model_dir: directory containing model files
    
    Returns:
        dict: Result containing prediction, score, and timing info
    """
    # Check image aspect ratio
    is_valid, error_msg = check_image(image)
    if not is_valid:
        return {
            'success': False,
            'error': error_msg
        }
    
    # Get bounding box
    image_bbox = model_test.get_bbox(image)
    
    if image_bbox is None:
        return {
            'success': False,
            'error': 'No face detected in the image'
        }
    
    prediction = np.zeros((1, 3))
    test_speed = 0
    
    # Sum predictions from all models (SORTED for consistency)
    model_count = 0
    for model_name in (os.listdir(model_dir)):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
         
        
        if scale is None:
            param["crop"] = False
        
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start
        model_count += 1
    #me
    # #Get final prediction
    # label = np.argmax(prediction)
    # # ME
    # # value = float(prediction[0  ][label] / 2)
    # # cLAUDE
    # value = float(prediction[0][label] / model_count)
    
    # # Debug: print the prediction array to understand the model output
    # print(f"DEBUG - Prediction array: {prediction}")
    # print(f"DEBUG - Argmax label: {label}")
    # print(f"DEBUG - Value at label {label}: {value}")
    
    # # In the model: label 0 = Fake, label 1 = Real, label 2 = Unknown/Other
    # is_real = label == 1
    
    # return {    
    #     'success': True,
    #     'prediction': 'real' if is_real else 'fake',
    #     'score': round(value, 4),
    #     'confidence': round(value * 100, 2),
    #     'label': int(label),
    #     'processing_time': round(test_speed, 3),
    #     'models_used': model_count,
    #     'bbox': {
    #         'x': int(image_bbox[0]),
    #         'y': int(image_bbox[1]),
    #         'width': int(image_bbox[2]),
    #         'height': int(image_bbox[3])
    #     }
    # }
    #claude
    # Normalize scores by number of models averaged
    real_score = float(prediction[0][1] / model_count)    # class 1 = Real
    fake_score = float(prediction[0][0] / model_count)    # class 0 = Fake
    unknown_score = float(prediction[0][2] / model_count)  # class 2 = Unknown/Other

    label = int(np.argmax(prediction))
    value = float(prediction[0][label] / model_count)

    print(f"DEBUG - Prediction array: {prediction}")
    print(f"DEBUG - real={real_score:.3f} fake={fake_score:.3f} unknown={unknown_score:.3f}")


    # # Three-tier decision
    # if real_score >= REAL_THRESHOLD:
    #     prediction_label = 'real'
    # elif fake_score >= FAKE_THRESHOLD:
    #     prediction_label = 'fake'
    # else:
    #     prediction_label = 'uncertain'

    if real_score >= REAL_THRESHOLD:
        prediction_label = 'real'
    elif unknown_score > fake_score:
        prediction_label = 'uncertain'  # یا حتی همان 'fake' برای سادگی
        reject_reason = 'unrecognized_pattern'
    else:
        prediction_label = 'fake'
        reject_reason = 'classic_spoof_pattern'

    return {
        'success': True,
        'prediction': prediction_label,
        'score': round(value, 4),
        'reject_reason': reject_reason,
        'confidence': round(value * 100, 2),
        'real_score': round(real_score, 4),
        'fake_score': round(fake_score, 4),
        'unknown_score': round(unknown_score, 4),
        'label': label,
        'processing_time': round(test_speed, 3),
        'models_used': model_count,
        'bbox': {
            'x': int(image_bbox[0]),
            'y': int(image_bbox[1]),
            'width': int(image_bbox[2]),
            'height': int(image_bbox[3])
        }
    }
 
# def resize_to_3_4_centered(image_path, output_path):
#     # 1. Load the image
#     # img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError("Could not read the image. Check the file path or format.")
        
#     height, width = img.shape[:2]

#     # 2. Load OpenCV's built-in face detector
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # 3. Determine the center point for cropping
#     if len(faces) > 0:
#         # If a face is found, use the center of the first detected face
#         x, y, w, h = faces[0]
#         center_x = x + w // 2
#         center_y = y + h // 2
#     else:
#         # Fallback: Use the exact geometric center of the image if no face is detected
#         center_x = width // 2
#         center_y = height // 2

#     # 4. Calculate the maximum possible 3:4 dimensions around the center
#     # Target ratio: Width / Height = 3 / 4 -> Height = Width * 4 / 3
#     target_w = width
#     target_h = int(target_w * 4 / 3)

#     # If the calculated height exceeds the original image height, scale based on height instead
#     if target_h > height:
#         target_h = height
#         target_w = int(target_h * 3 / 4)

#     # 5. Define crop boundaries around the center point
#     start_x = max(0, center_x - target_w // 2)
#     start_y = max(0, center_y - target_h // 2)
    
#     end_x = start_x + target_w
#     end_y = start_y + target_h

#     # Adjust boundaries if they overshoot the image edges
#     if end_x > width:
#         start_x -= (end_x - width)
#         end_x = width
#     if end_y > height:
#         start_y -= (end_y - height)
#         end_y = height
        
#     # Ensure coordinates are not negative after adjustment
#     start_x = max(0, start_x)
#     start_y = max(0, start_y)

#     # 6. Crop the image
#     cropped_img = img[start_y:end_y, start_x:end_x]

#     # 7. Optional: Resize to a standard API pixel size (e.g., 600x800)
#     # final_img = cv2.resize(cropped_img, (600, 800), interpolation=cv2.INTER_AREA)

#     # 8. Save the output
#     cv2.imwrite(output_path, cropped_img)
#     print(f"Successfully processed and saved: {output_path}")

# # --- Example Usage ---
# # resize_to_3_4_centered("input_user_photo.jpg", "ready_for_api.jpg")



# def crop_to_3_4_ratio(img):
#     """
#     Detects a face and crops the in-memory image to a 3:4 aspect ratio.
#     """
#     height, width = img.shape[:2]

#     # Load OpenCV's built-in face detector
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
#     # Detect face
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     if len(faces) > 0:
#         x, y, w, h = faces[0]
#         center_x = x + w // 2
#         center_y = y + h // 2
#     else:
#         # Fallback to image center if no face is detected
#         center_x = width // 2
#         center_y = height // 2

#     # Calculate target 3:4 dimensions
#     target_w = width
#     target_h = int(target_w * 4 / 3)

#     if target_h > height:
#         target_h = height
#         target_w = int(target_h * 3 / 4)

#     # Calculate cropping coordinates
#     start_x = max(0, center_x - target_w // 2)
#     start_y = max(0, center_y - target_h // 2)
    
#     end_x = start_x + target_w
#     end_y = start_y + target_h

#     # Adjust coordinates if they exceed boundaries
#     if end_x > width:
#         start_x -= (end_x - width)
#         end_x = width
#     if end_y > height:
#         start_y -= (end_y - height)
#         end_y = height
        
#     start_x = max(0, start_x)
#     start_y = max(0, start_y)

#     # Return the cropped 3:4 image
#     return img[start_y:end_y, start_x:end_x]



@app.get("/", response_model=APIInfo)
async def root():
    """API information endpoint"""
    return {
        'name': 'Face Anti-Spoofing API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict if face is real or fake (file upload)',
            '/predict/base64': 'POST - Predict using base64 encoded image',
            '/predict/visualize': 'POST - Predict with annotated image',
            '/docs': 'GET - Interactive API documentation (Swagger UI)',
            '/redoc': 'GET - Alternative API documentation (ReDoc)'
        },
        'accepted_formats': list(ALLOWED_EXTENSIONS)
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_count = 0
    if os.path.exists(MODEL_DIR):
        models_count = len(os.listdir(MODEL_DIR))
    
    return {
        'status': 'healthy',
        'model_dir': MODEL_DIR,
        'models_available': models_count
    }

# ME
# def prepare_3_4_image(image: np.ndarray) -> np.ndarray:
#     """Center-crop image to 3:4 aspect ratio, then resize to target size."""
#     h, w = image.shape[:2]
    
#     target_w = min(w, int(h * 3 / 4))
#     target_h = min(h, int(w * 4 / 3))
    
#     x = (w - target_w) // 2
#     y = (h - target_h) // 2
    
#     cropped = image[y:y+target_h, x:x+target_w]
#     # return cv2.resize(cropped, (300, 400))  # or whatever size your model expects
#     return cropped

# cLAUDE
def prepare_3_4_image(image: np.ndarray) -> np.ndarray:
    """Pad image to 3:4 aspect ratio (width:height) without cropping content."""
    h, w = image.shape[:2]
    target_ratio = 3 / 4  # width / height

    current_ratio = w / h

    if current_ratio > target_ratio:
        # too wide -> increase height (pad top/bottom)
        new_h = int(w / target_ratio)
        pad_total = new_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    else:
        # too narrow -> increase width (pad left/right)
        new_w = int(h * target_ratio)
        pad_total = new_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = cv2.copyMakeBorder(
            image, 0, 0, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    return padded

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Image file (jpg, jpeg, png)")):
    """
    Predict if face image is real or fake using file upload
    
    - **file**: Image file with face (3:4 aspect ratio required)
    
    Returns prediction with confidence score and bounding box
    """
    try:
        # Validate file extension
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Decode image
        image = await decode_image(file)


        # new added for handel 3:4 images
        # image = prepare_3_4_image(image)  # ← add this line
        
        # Process the image
        result = process_image(image)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: Base64ImageRequest):
    """
    Predict if face image is real or fake using base64 encoded image
    
    - **image_base64**: Base64 encoded image string
    
    Returns prediction with confidence score and bounding box
    """
    try:
        # Decode base64 image
        image = decode_base64_image(request.image_base64)
        
        # Process the image
        result = process_image(image)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict/visualize", response_model=VisualizationResponse)
async def predict_with_visualization(
    file: UploadFile = File(..., description="Image file (jpg, jpeg, png)")
):
    """
    Predict if face is real or fake and return annotated image
    
    - **file**: Image file with face (3:4 aspect ratio required)
    
    Returns prediction along with base64 encoded annotated image showing bounding box
    """
    try:
        # Validate file extension
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Decode image
        image = await decode_image(file)
        # new added for handel 3:4 images
        image = prepare_3_4_image(image)  # ← add this line
        
        # Process the image
        result = process_image(image.copy())
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Draw bounding box and text on image
        bbox = result['bbox']
        #me
        # color = (255, 0, 0) if result['prediction'] == 'real' else (0, 0, 255)
        #claude
        if result['prediction'] == 'real':
            color = (255, 0, 0)
        elif result['prediction'] == 'fake':
            color = (0, 0, 255)
        else:  # uncertain
            color = (0, 165, 255)
        #
        cv2.rectangle(
            image,
            (bbox['x'], bbox['y']),
            (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
            color, 2
        )
        
        result_text = f"{result['prediction'].upper()} - Score: {result['score']:.2f}"
        cv2.putText(
            image,
            result_text,
            (bbox['x'], bbox['y'] - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5 * image.shape[0] / 1024,
            color
        )
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result['result_image'] = f"data:image/jpeg;base64,{image_base64}"
        
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == '__main__':
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Model directory {MODEL_DIR} does not exist!")
    
    # Run the FastAPI app with Uvicorn
    uvicorn.run(
        "fastapi_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )

##########################
