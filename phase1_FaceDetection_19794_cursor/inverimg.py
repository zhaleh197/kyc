import cv2
import numpy as np
import mediapipe as mp


def detect_face_orientation_bruteforce(image):
    """
    چرخش تصویر در 4 جهت و پیدا کردن بهترین تطابق
    """
    orientations = [
        (0, "UPRIGHT"),
        (180, "UPSIDE_DOWN"), 
        (90, "ROTATED_LEFT"),
        (270, "ROTATED_RIGHT")
    ]
    
    best_orientation = "UNKNOWN"
    best_confidence = 0
    best_landmarks = None
    
    for angle, orientation_name in orientations:
        # چرخش تصویر
        if angle == 0:
            rotated_image = image.copy()
        else:
            if angle == 90:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # تشخیص صورت در تصویر چرخیده
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        ) as face_mesh:
            
            results = face_mesh.process(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # محاسبه confidence بر اساس کیفیت تشخیص
                confidence = calculate_detection_confidence(landmarks, rotated_image.shape)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_orientation = orientation_name
                    best_landmarks = landmarks
    
    return best_orientation, best_confidence, best_landmarks

def calculate_detection_confidence(landmarks, image_shape):
    """
    محاسبه اطمینان از تشخیص بر اساس معیارهای هندسی
    """
    points = np.array([(lm.x, lm.y) for lm in landmarks.landmark])
    
    height, width = image_shape[:2]
    
    # 1. بررسی اینکه landmarks درون تصویر هستند
    x_coords = points[:, 0] * width
    y_coords = points[:, 1] * height
    
    within_bounds = np.all((x_coords >= 0) & (x_coords <= width) & 
                          (y_coords >= 0) & (y_coords <= height))
    
    # 2. بررسی تناسب انداز صورت
    face_width = np.max(x_coords) - np.min(x_coords)
    face_height = np.max(y_coords) - np.min(y_coords)
    
    aspect_ratio = face_width / face_height if face_height > 0 else 0
    reasonable_size = 0.2 < aspect_ratio < 2.0  # نسبت معقول برای صورت
    
    # 3. بررسی فاصله بین ویژگی‌ها
    left_eye = points[33]
    right_eye = points[263]
    nose = points[1]
    mouth = points[13]
    
    eye_distance = np.linalg.norm(left_eye - right_eye)
    nose_to_mouth = np.linalg.norm(nose - mouth)
    
    reasonable_distances = 0.3 < (nose_to_mouth / eye_distance) < 1.5
    
    confidence = 0
    if within_bounds:
        confidence += 0.4
    if reasonable_size:
        confidence += 0.3
    if reasonable_distances:
        confidence += 0.3
    
    return confidence

# استفاده عملی
def smart_orientation_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "INVALID_IMAGE"
    
    orientation, confidence, landmarks = detect_face_orientation_bruteforce(image)
    
    # اگر confidence پایین بود، روش جایگزین
    if confidence < 0.6:
        orientation = alternative_orientation_detection(image)
    
    return {
        "orientation": orientation,
        "confidence": confidence,
        "landmarks_count": len(landmarks.landmark) if landmarks else 0
    }

def alternative_orientation_detection(image):
    """
    روش جایگزین: استفاده از تشخیص اشیا
    """
    # استفاده از Haar cascade برای تشخیص صورت
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return "UNKNOWN"
    
    # آنالیز موقعیت صورت در تصویر
    x, y, w, h = faces[0]
    center_x = x + w/2
    center_y = y + h/2
    
    img_center_x = image.shape[1] / 2
    img_center_y = image.shape[0] / 2
    
    # اگر مرکز صورت نزدیک به مرکز تصویر باشد، احتمالاً عادی است
    distance_to_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
    max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
    
    if distance_to_center < max_distance * 0.3:
        return "UPRIGHT"
    else:
        return "UPSIDE_DOWN"

# تست
result = smart_orientation_detection("your_upside_down_image.jpg")
print(f"جهت تشخیص داده شده: {result['orientation']}")
print(f"میزان اطمینان: {result['confidence']:.2f}")