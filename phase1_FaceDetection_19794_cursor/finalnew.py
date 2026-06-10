import math
from flask import Flask, render_template, url_for
import aiohttp 
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import base64
import cv2
import numpy as np
from flask import request, jsonify , send_file
import uuid
import re
from datetime import datetime

import asyncio
import torch
import torch.nn as nn
import torchvision.models as models
from multiprocessing import Pool, cpu_count
from PIL import Image 
import base64
# from rembg import remove 
from rembg import new_session, remove
from deepface import DeepFace
import faiss
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp 
import os
from flask_cors import CORS  # اضافه کردن CORS
import requests
from datetime import datetime
import time
import tempfile


from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from werkzeug.utils import secure_filename
import shutil
from moviepy import VideoFileClip
from database import Base,SessionLocal



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
# //////////////////////////////////////////////////////////////////////////////

app = Flask(__name__)
CORS(app)  # فعال‌سازی CORS برای کل برنامه

#  <<<<<<<<<<<<<<<<  Variable  >>>>>>>>>>>>>

executorprocesor = ProcessPoolExecutor()
executorthread = ThreadPoolExecutor(max_workers=4)

savefliag=False
UPLOAD_FOLDER = "./static/images"
image_for_saving = None
KEY_LANDMARKS = {
    "left_eye": [33, 133],  # گوشه‌های چشم چپ
    "right_eye": [362, 263],  # گوشه‌های چشم راست
    "nose": [1],  # نوک بینی
    "mouth": [13, 14]  # وسط لب بالا و پایین
}


landmark_ids = [1, 152, 263, 33, 287, 57]

# نقاط 3D مرجع از مدل سر (می‌تونن با توجه به صورت تنظیم بشن)
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip        -> 1
    (0.0, -63.6, -12.5),       # Chin            -> 152
    (-43.3, 32.7, -26.0),      # Left eye left   -> 263
    (43.3, 32.7, -26.0),       # Right eye right -> 33
    (-28.9, -28.9, -24.1),     # Left mouth      -> 287
    (28.9, -28.9, -24.1)       # Right mouth     -> 57
], dtype=np.float32)


# <<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>

# **************  Face Cover Detection Models  *****************

#     <<<< 0.  MediaPipe >>>>>>
# تنظیمات MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh()


#     <<<< 1.  Mobilenet >>>>>>
# model =mobilenet.load_state_dict(torch.load("obstruction_detector_aug_ful.pt"),weights_only=False)
# mobilenet = torch.load("obstruction_detector_aug_ful.pt", weights_only=False,map_location=torch.device('cpu'))
mobilenet = torch.load("obstruction_detector50.pt", weights_only=False,map_location=torch.device('cpu'))
# Transformer for mobilenet-pythorch
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToPILImage(),               # تبدیل به تصویر PIL
    transforms.Resize((224, 224)),         # سایز ورودی مدل
    transforms.ToTensor(),                 # تبدیل به تنسور
    transforms.Normalize(                  # نرمال‌سازی مثل آموزش
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

#     <<<< 2.  YOLO >>>>>>
from ultralytics import YOLO
mymodel = YOLO("finalbestzh70.pt")

#     <<<< 3.  VIT >>>>>>   
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification
pipe = pipeline("image-classification", model="dima806/face_obstruction_image_detection")


# *******************************************************************




# *************************  OCR ID CARD  ****************************
import pytesseract
from PIL import Image 

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"# Set the tesseract executable path explicitly
# def process_card_image(image_path):
def process_card_image(img):
    """
    پردازش تصویر کارت ملی و استخراج اطلاعات
    """
    try:
        # استخراج متن با OCR
        text = pytesseract.image_to_string(img, lang='fas')
        
        # پیش‌پردازش متن
        text = preprocess_text(text)
        
        # استخراج اطلاعات
        card_info = extract_card_info(text)
        
        # اعتبارسنجی اطلاعات
        if not validate_card_info(card_info):
            raise ValueError("اطلاعات استخراج شده معتبر نیستند")
            
        return card_info
    except Exception as e:
        raise RuntimeError(f"خطا در پردازش کارت: {str(e)}")

def preprocess_text(text):
    """
    پیش‌پردازش متن استخراج شده
    """
    # حذف فاصله‌های اضافی
    text = ' '.join(text.split())
    # اصلاح اعداد فارسی-انگلیسی
    persian_numbers = '۰۱۲۳۴۵۶۷۸۹'
    english_numbers = '0123456789'
    translation_table = str.maketrans(persian_numbers, english_numbers)
    text = text.translate(translation_table)
    return text

def validate_card_info(card_info):
    """
    اعتبارسنجی اطلاعات استخراج شده
    """
    if not card_info.get('national_id') or len(card_info['national_id']) != 10:
        return False
    if not card_info.get('full_name') or len(card_info['full_name'].split()) < 2:
        return False
    if not card_info.get('dob'):
        return False
    return True

def extract_card_info(text):
    # تابعی برای استخراج اطلاعات از متن
    # به عنوان مثال، شماره ملی، نام، تاریخ تولد و...
    card_info = {
        "national_id": extract_national_id(text),
        "full_name": extract_full_name(text),
        "dob": extract_dob(text)
    }
    return card_info

def extract_national_id(text):
    """
    استخراج شماره ملی از متن استخراج شده توسط OCR
    الگو: ۱۰ رقم پیوسته
    """
    matches = re.findall(r'\d{10}', text)
    if matches:
        # انتخاب طولانی‌ترین تطابق (برای جلوگیری از تشخیص نادرست)
        return max(matches, key=len)
    return None

def extract_full_name(text):
    """
    استخراج نام کامل از متن
    الگو: دنباله‌ای از حروف فارسی و فاصله
    """
    # حذف اعداد و کاراکترهای خاص
    cleaned_text = re.sub(r'[0-9۰-۹]', '', text)
    # پیدا کردن دنباله‌های معتبر نام
    name_matches = re.findall(r'[\u0600-\u06FF\s]{3,}', cleaned_text)
    
    if name_matches:
        # انتخاب طولانی‌ترین دنباله به عنوان نام
        full_name = max(name_matches, key=len).strip()
        # حذف کلمات اضافی
        for word in ['کارت', 'ملی', 'شناسایی', 'جمهوری', 'اسلامی', 'ایران']:
            full_name = full_name.replace(word, '')
        return full_name.strip()
    return None

def extract_dob(text):
    """
    استخراج تاریخ تولد از متن
    الگوهای مختلف تاریخ (۱۳۸۰/۰۵/۱۲ یا ۱۲-۰۵-۱۳۸۰ یا ۱۳۸۰۰۵۱۲)
    """
    # الگوی تاریخ با فرمت ۱۳۸۰/۰۵/۱۲
    date_pattern1 = re.compile(r'(\d{4})/(\d{2})/(\d{2})')
    # الگوی تاریخ با فرمت ۱۲-۰۵-۱۳۸۰
    date_pattern2 = re.compile(r'(\d{2})-(\d{2})-(\d{4})')
    # الگوی تاریخ با فرمت ۱۳۸۰۰۵۱۲
    date_pattern3 = re.compile(r'(\d{4})(\d{2})(\d{2})')
    
    for pattern in [date_pattern1, date_pattern2, date_pattern3]:
        match = pattern.search(text)
        if match:
            year, month, day = match.groups()
            try:
                # تبدیل تاریخ شمسی به میلادی (در اینجا نیاز به کتابخانه jalali-date دارید)
                # return JalaliDate(int(year), int(month), int(day)).to_gregorian()
                return f"{year}/{month}/{day}"
            except:
                return f"{year}/{month}/{day}"
    return None



# *******************************************************************
# **************  phase2 Funstios: Get Info From IdCard *****************
@app.route('/process_idcard', methods=['POST']) 
def upload_card_image(): 
    
    if 'id_cardimg' not in request.files:
        return jsonify({"status": "error", "message": "No image file received"}), 400

    file = request.files['id_cardimg']  
    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400
    
    # cv2.imwrite(filename, frame)
    
      # پردازش تصویر برای استخراج اطلاعات کارت ملی
    extracted_info = process_card_image(frame)
    
    # بازگشت نتایج
    return {"status": "success", "data": extracted_info}
    
# *******************************************************************


# **************  phase1 Funstios:  *****************
# 
 
mp_drawing = mp.solutions.drawing_utils

def get_rotation_angle_from_box(bbox):
    # MediaPipe gives rotation in radians
    angle_rad = bbox.rotation
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def correct_face_rotation(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print("هیچ صورتی پیدا نشد!")
            return image

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            rotation = detection.location_data.relative_keypoints
            try:
                # MediaPipe face detection v2 doesn’t provide rotation directly, so we use eye positions
                left_eye = detection.location_data.relative_keypoints[0]
                right_eye = detection.location_data.relative_keypoints[1]

                # تبدیل مختصات نسبی به مختصات مطلق
                h, w = image.shape[:2]
                left_eye = np.array([int(left_eye.x * w), int(left_eye.y * h)])
                right_eye = np.array([int(right_eye.x * w), int(right_eye.y * h)])

                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))

                # اصلاح چرخش
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

                print(f"[INFO] زاویه چرخش اصلاح شد: {angle:.2f} درجه")
                return rotated

            except Exception as e:
                print("خطا در محاسبه زاویه:", e)

        return image







def sync_processing_subtasks(frame, face_landmarks, h, w):
    results = {}
    try:
        left_eye = np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
        right_eye = np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
        nose_x = face_landmarks.landmark[1].x * w
        center_x = (left_eye[0] + right_eye[0]) / 2
        eye_distance = np.linalg.norm(left_eye - right_eye)
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        looking_straight = abs(nose_x - center_x) < 20
        results.update({
            "eye_distance": eye_distance,
            "angle": angle,
            "looking_straight": looking_straight,
        })
    except Exception as e:
        results["error"] = str(e)
    return results
async def process_all_tasks(frame, face_landmarks, h, w, min_x, min_y, max_x, max_y):
    loop = asyncio.get_event_loop()

    face_roi = frame[min_y:max_y, min_x:max_x]

    # موازی‌سازی ۴ تابع
    task1 = loop.run_in_executor(executorthread, is_blurry_fft, frame)
    task2 = loop.run_in_executor(executorthread, extract_background_only, frame)
    task3 = loop.run_in_executor(executorthread, yoloDetect2, face_roi)
    task4 = loop.run_in_executor(executorthread, is_lighting_uniform_with_hist, face_roi)
    blurry, background, yolo_result, face_light_ok = await asyncio.gather(task1, task2, task3, task4)

    # background, yolo_result, face_light_ok = await asyncio.gather(  task2, task3, task4)

    # یک تابع sync که همش رو با هم حساب می‌کنه:
    extra_features = await loop.run_in_executor(executorthread, sync_processing_subtasks, frame, face_landmarks, h, w)

    return {
        # "blurry": blurry,
        "background": background,
        "isyolo": yolo_result,
        "isFacelightOk": face_light_ok,
        **extra_features,
    }
# Check Backgroung
session = new_session(model_name="u2netp")  # مدل کوچک و سریع‌تر از u2net
# Fast
def extract_background_only(image):
     # اطمینان از اینکه ورودی numpy array هست
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # تبدیل OpenCV BGR → PIL RGB
    # else:
    #     image = image_np  # در صورتی که ورودی خودش PIL بود
 
    # removed = remove(image)
    removed = remove(image, session=session)
    removed_np = np.array(removed)

    alpha = removed_np[:, :, 3]
    original_np = np.array(image.convert("RGB"))

    # ماسک پس‌زمینه: فقط جاهایی که آلفا = 0
    mask = (alpha == 0)
    
    # گرفتن پیکسل‌های واقعی زمینه
    bg_pixels = original_np[mask]
    return bg_pixels  # shape: (N, 3)
# Accure but slow
def extract_background_only2(image):
    # image = Image.open(image_path).convert("RGBA")
    removed = remove(image)  # حذف پیش‌زمینه (فقط شیء باقی می‌مونه)
    
    removed_np = np.array(removed)

    # ماسکی درست می‌کنیم از کانال آلفا (شفافیت)
    alpha = removed_np[:, :, 3]

    # هر جا که آلفا صفره یعنی زمینه بوده → اون قسمت رو نگه می‌داریم
    background_mask = (alpha == 0).astype(np.uint8) * 255

    # تبدیل عکس اصلی به RGBA برای ترکیب
    original = np.array(image)
    background = np.zeros_like(original)

    # فقط پیکسل‌های زمینه رو نگه می‌داریم
    background[background_mask == 255] = original[background_mask == 255]
    # cv2.imshow("background", background)
    return background[:, :, :3]  # حذف کانال آلفا
def is_background_uniformSegment(bg_pixels, threshold=25):
    if len(bg_pixels) == 0:
        return False, 999  # زمینه‌ای پیدا نشده!
    
    gray = np.mean(bg_pixels, axis=1)  # میانگین R,G,B → تبدیل به خاکستری
    std_dev = np.std(gray)
    print("std_dev ++++  :  ",std_dev)
    return (std_dev < (threshold)), std_dev
def run_background_check(frame,t=50):
    background = extract_background_only(frame)
    return is_background_uniformSegment(background, t)
# Check face
def is_lighting_uniform_with_hist(face_img, threshold_std=20, threshold_diff=20, show_hist=True):
    """
    بررسی یکنواختی نور در چهره و نمایش هیستوگرام روشنایی.
    """
    if face_img is None or face_img.size == 0:
        print("تصویر صورت یافت نشد یا خالی است.")
        return False
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # تقسیم به دو نیمه
    left = gray[:, :w // 2]
    right = gray[:, w // 2:]

    # محاسبه آمار
    std_total = np.std(gray)
    mean_left = np.mean(left)
    mean_right = np.mean(right)
    diff = abs(mean_left - mean_right)

    is_uniform_li = diff < threshold_std 
    return is_uniform_li
def draw_closest_face_box(frame, face_landmarks_list):
    h, w = frame.shape[:2]
    face_scores = []

    for face_landmarks in face_landmarks_list:
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        left = np.array([left_eye.x * w, left_eye.y * h])
        right = np.array([right_eye.x * w, right_eye.y * h])

        eye_distance = np.linalg.norm(right - left)

        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        box_w, box_h = x2 - x1, y2 - y1

        norm_eye_distance = eye_distance / w
        norm_box_w = box_w / w
        norm_box_h = box_h / h

        score = norm_eye_distance + 0.3 * norm_box_w + 0.3 * norm_box_h

        face_scores.append((score, (x1, y1, x2, y2),face_landmarks))

    if face_scores:
        # مرتب‌سازی از بیشترین نمره (نزدیک‌ترین چهره)
        face_scores.sort(reverse=True, key=lambda x: x[0])

        # کشیدن کادر برای تمام چهره‌ها (آبی)
        for i, (_, box,f) in enumerate(face_scores):
            color = (255, 0, 0)  # آبی
            if i == 0:
                color = (0, 255, 0)
                # mx1, my1, mx2, my2 = box  # سبز برای نزدیک‌ترین چهره
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame, x1,x2,y1,y2,f
# Check Head
def get_head_pose_info(image,face_landmarks):
    img_h, img_w = image.shape[:2]   
    image_points = np.array([
            (face_landmarks.landmark[i].x * img_w,
             face_landmarks.landmark[i].y * img_h)
            for i in landmark_ids
        ], dtype=np.float32)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    if success:
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        # بررسی محدودیت‌ها
        is_head_ok = abs(pitch) <= 15 and abs(yaw) <= 15 and abs(roll) <= 8

       
            # راهنمایی برای اصلاح جهت سر
        tips = ""
        if  15 > pitch :
            headdir=True
            tips="سرت رو کمی پایین بیار"
        elif pitch < -15:
            headdir=True
            tips="سرت رو کمی بالا ببر"

        elif yaw > 15:
            headdir=True
            tips="سرت رو کمی به چپ بچرخون"
        elif yaw < -15:
            headdir=True
            tips="سرت رو کمی به راست بچرخون"
        elif roll > 8:
            headdir=True
            tips="سرت رو کمی به چپ خم کن"
        elif roll <-8:
            headdir=True
            tips="سرت رو کمی به راست خم کن"
        else:
            headdir=False
            tips="زاویه سر مناسب است"
        return headdir,tips
# Check Eyes
def get_eye_direction(inner, outer, upper, lower, iris):
    # نسبت افقی (x): چپ / راست
    eye_width = abs(outer.x - inner.x)
    iris_x_rel = abs(iris.x - inner.x) / eye_width

    # نسبت عمودی (y): بالا / پایین
    eye_height = abs(lower.y - upper.y)
    iris_y_rel = abs(iris.y - upper.y) / eye_height

    # تشخیص چپ/راست
    horizontal = iris_x_rel < 0.35 or iris_x_rel > 0.65
   
    # تشخیص بالا/پایین
    vertical = iris_y_rel < 0.5 or iris_y_rel > 0.65
    print("iris_y_rel   :", iris_y_rel)
    # اگر چشم به جهتی غیر از جلو نگاه می‌کند
    return horizontal or vertical
def is_eye_off_center(inner, outer, upper, lower, iris, tol=0.11):
    eye_width = abs(outer.x - inner.x)
    eye_height = abs(lower.y - upper.y)
    iris_x_rel = abs(iris.x - inner.x) / eye_width
    iris_y_rel = abs(iris.y - upper.y) / eye_height
    print ("iris_x_rel  ",iris_x_rel)
    print("iris_y_rel  ",iris_y_rel)
    # می‌توان از فاصله اقلیدسی استفاده کرد
    dx = (iris_x_rel - 0.5)**2
    dy = (iris_y_rel - 0.5)**2
    th=math.sqrt(dx + dy) 
    print ("th:",th)
    if th> tol:
        return True
    return False
# import math
# def is_eye_off_center(inner, outer, upper, lower, iris, tol=0.1):
#     eye_width = abs(outer.x - inner.x)
#     eye_height = abs(lower.y - upper.y)

#     eye_center_x = (inner.x + outer.x) / 2
#     eye_center_y = (upper.y + lower.y) / 2

#     iris_x_rel = (iris.x - eye_center_x) / eye_width  # [-0.5, +0.5]
#     iris_y_rel = (iris.y - eye_center_y) / eye_height  # [-0.5, +0.5]

#     print("iris_x_rel:", iris_x_rel)
#     print("iris_y_rel:", iris_y_rel)

#     if math.sqrt(iris_x_rel**2 + iris_y_rel**2) > tol:
#         return True
#     return False


#     # if abs(iris_x_rel - 0.5) > tol or abs(iris_y_rel - 0.5) > tol:
#     #     return True
#     # return False
  
def is_eye_looking_forward(eye_inner, eye_outer, iris):
    # موقعیت افقی
    eye_width = eye_outer.x - eye_inner.x
    iris_offset = iris.x - eye_inner.x
    ratio = iris_offset / eye_width
    return ratio  # عددی بین 0 تا 1 — وسط حدود 0.45 تا 0.55
# Blurry
def is_blurryLaplasian(image, threshold=500):
    
    """ بررسی وضوح تصویر (مات نبودن) """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    print("laaaaaaaaaaaaaaaaaaaaplacian ",laplacian)
    return laplacian < threshold
# def is_blurry_fft(image, threshold=0.3):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     f = np.fft.fft2(gray)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20 * np.log(np.abs(fshift))
#     mean_val = np.mean(magnitude_spectrum)
#     print("mean_val ",mean_val)
#     return mean_val < threshold

def is_blurry_fft(image, threshold=0.35):#threshold=0.45
    cv2.setNumThreads(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center = (h // 2, w // 2)

    # حذف مرکز (low frequencies)
    size = 15  # محدوده‌ی مرکز رو حذف کن
    magnitude[center[0]-size:center[0]+size, center[1]-size:center[1]+size] = 0

    high_freq_energy = np.sum(magnitude)
    total_energy = np.sum(np.abs(fshift))
    
    ratio = high_freq_energy / total_energy
    print("Ratio:", ratio)
    # threshold = 0.1 + (image.shape[0] * image.shape[1]) / (1000*1000) * 0.3
    return ratio < threshold  # مثلاً threshold = 0.1


def is_blurry_fftopt(image, threshold=0.35, size_ratio=0.04):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)  # پیش‌پردازش
    
    h, w = gray.shape
    size = max(int(min(h,w) * size_ratio), 5)
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    center = (h//2, w//2)
    magnitude[center[0]-size:center[0]+size, center[1]-size:center[1]+size] = 0
    
    high_freq_energy = np.sum(magnitude)
    total_energy = np.sum(np.abs(fshift))
    
    ratio = high_freq_energy / (total_energy + 1e-6)  # جلوگیری از تقسیم بر صفر
    return ratio < threshold

# ////
# Cover Face
def vit(image):
    pipe = pipeline("image-classification", model="dima806/face_obstruction_image_detection")
    # processor = AutoImageProcessor.from_pretrained("dima806/face_obstruction_image_detection")
    # model = AutoModelForImageClassification.from_pretrained("dima806/face_obstruction_image_detection")
    
    results=pipe(image) 
    
    if results[0]['label']=='none':
        return  False
    else:
        return True
# def vit2(image_pil):
#     # مسیر محلی مدل (اگه دانلودش کردی)
#     # model_path = "./models/face_obstruction_image_detection"
#     model_path = r"C:\Users\Zhala\.cache\huggingface\hub\models--dima806--face_obstruction_image_detection\snapshots\d58c7ee572d1216c61a039b18423d056eff971f5"

#     processor = AutoImageProcessor.from_pretrained(model_path)
#     model = AutoModelForImageClassification.from_pretrained(model_path)

#     # پردازش تصویر
#     inputs = processor(images=image_pil, return_tensors="pt")

#     # پیش‌بینی
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         pred = logits.argmax(-1).item()

#     label = model.config.id2label[pred]

#     if label=='none':
#         return  False
#     else:
#         return True

# /////// 
# متد:
# اگر هر کدام از نقاط لند مارک را نتواند شناسایی کند، پس آن پوشیده شده. 
# نتیجه:
# خوب جواب نداد زیرا جتی اگر پوشیده باشد نقاط لند مارک را نشان میدهد بر اساس احتمالا فاصله ها و نسبت ها
def detect_face_cover(landmarks, image):
    height, width, _ = image.shape

    key_positions = {}

    for key, indices in KEY_LANDMARKS.items():
        key_positions[key] = [
            (int(landmarks.landmark[i].x * width), int(landmarks.landmark[i].y * height)) for i in indices
        ]

    if any(len(points) == 0 for points in key_positions.values()):
        return True 
     
# متد:
# اگر اختلاف شدت پیکسل های نواحی دهان و بینی کمتر از تریشولدی باشد یعنی ان دو ناحیه همرنگ هستند وا حتمال ماسک یا کاور وجود دارد
# نتیجه: 
# خوب کار نکرد، احتمالا برای ماسک خوب باشد ولی برای دست و یا کاورهای دیگر نه احتمالا

def detect_mask(image,mask_threshold=100):
    """ تشخیص ماسک بر روی چهره با استفاده از MediaPipe Face Detection """
     
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (640, 640))
    results = face_detection.process(image_rgb)
    if not results.detections:
        return "چهره یافت نشد"
    
    for detection in results.detections:
        keypoints = detection.location_data.relative_keypoints
        nose = keypoints[2]  # نقطه مربوط به بینی
        mouth = keypoints[3]  # نقطه مربوط به دهان
        
        nose_x, nose_y = int(nose.x * image.shape[1]), int(nose.y * image.shape[0])
        mouth_x, mouth_y = int(mouth.x * image.shape[1]), int(mouth.y * image.shape[0])
        
        # بررسی پوشیدگی ماسک با مقایسه شدت پیکسلی ناحیه بینی و دهان
        nose_intensity = np.mean(image[nose_y-5:nose_y+5, nose_x-5:nose_x+5])
        mouth_intensity = np.mean(image[mouth_y-5:mouth_y+5, mouth_x-5:mouth_x+5])
        
        # mask_threshold = 40  # آستانه تغییر شدت پیکسل
        if abs(nose_intensity - mouth_intensity) < mask_threshold:
            return True
        else:
            return False

# متد: 
# مبایل نت، شبکه ای عصبی یلک و سریع
# نتیجه:
# دقتش خوب نیست به دلیل ایپاک کم ترین شاید
def detect_mobilenet(frame):
    input_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = mobilenet(input_tensor)
        # print("Raw output  :", output.item())
        prob = output.squeeze().item() 
        threshold = 0.25
        prediction = (prob >= threshold)
    # نمایش نتیجه روی تصویر
    label = "Obstructed" if prediction == 0 else "Clear"
    if label=="Obstructed":
        return True
    else:
        return False

# متد: 
# یولو
def yoloDetect(frame):
    results = mymodel(frame)

    for r in results:
        class_ids = r.boxes.cls.tolist()
        class_names = [r.names[int(cls_id)] for cls_id in class_ids]

        # print("✅ کلاس‌های شناسایی شده:", class_names)

        # اگر فقط none شناسایی شده باشه → False
        if all(name.lower() == 'none' for name in class_names):
            return False
        else:
            return True
        
        
    # for r in results:
    #     # گرفتن لیبل‌ها
    #     class_ids = r.boxes.cls.cpu().numpy().astype(int)  # شناسه کلاس‌ها
    #     class_names = [r.names[int(i)] for i in class_ids]
    # for i, name in enumerate(class_names):
    #     print ("name",name)
    #     if name=="none":
    #         return False
    #     else:
    #         return True

def yoloDetect2(frame):
    results = mymodel(frame)  
    # print("resultsnew",results)
    none_detected = False
    for r in results:
        # print("r result",r)
        class_ids = r.boxes.cls.tolist()
        confidences = r.boxes.conf.tolist()
        class_names = [r.names[int(cls_id)] for cls_id in class_ids]
        # print("confidences")

        # ترکیب کلاس‌ها و دقت‌ها
        class_conf_pairs = [
            (class_names[i], confidences[i])
            for i in range(len(class_names))
            if confidences[i] >= 0.1
        ]
        print("class_conf_pairss: ",class_conf_pairs)

        if not class_conf_pairs:
            # print("🟥 هیچ کلاس معتبری با دقت بالا شناسایی نشد → False")
            none_detected= False

        # پیدا کردن کلاسی با بیشترین دقت
        else:
            best_class, best_conf = max(class_conf_pairs, key=lambda x: x[1])
            # print(f"✅ بهترین کلاس شناسایی‌شده: '{best_class}' با دقت {best_conf:.2f}")

            if best_class.lower() == 'none':
                # print("🟥 فقط 'none' با دقت بالا شناسایی شد → False")
                none_detected= False
            else:
                none_detected= True 
    return none_detected
# متد: 
# Canny. لبه های اطراف دهان و بینی اگز از یه حد کمتر باشند شاید یعنی ماسکی چوشیده و همه جیز صاف شده.
# نتیجه:
# قطعا در نور کم و یا ماسک های چر جین ، این کاملا برعکس کار میکند.
def detect_maskcANNY(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (640, 640))
    results = face_detection.process(image_rgb)
    
    if not results.detections:
        return "چهره یافت نشد"
    
    for detection in results.detections:
        keypoints = detection.location_data.relative_keypoints
        nose = keypoints[2]  # بینی
        mouth = keypoints[3]  # دهان
        
        # تبدیل مختصات به پیکسل
        nose_x, nose_y = int(nose.x * image.shape[1]), int(nose.y * image.shape[0])
        mouth_x, mouth_y = int(mouth.x * image.shape[1]), int(mouth.y * image.shape[0])
        
        # استخراج نواحی اطراف بینی و دهان
        nose_area = image[nose_y-20:nose_y+20, nose_x-20:nose_x+20]
        mouth_area = image[mouth_y-20:mouth_y+20, mouth_x-20:mouth_x+20]
        
        # تشخیص لبه‌ها با استفاده از الگوریتم Canny
        edges_nose = cv2.Canny(nose_area, 100, 200)
        edges_mouth = cv2.Canny(mouth_area, 100, 200)
        
        # محاسبه تعداد لبه‌ها (بیشتر بودن لبه‌ها می‌تواند نشانگر وجود ماسک باشد)
        nose_edges_count = np.count_nonzero(edges_nose) 
        mouth_edges_count = np.count_nonzero(edges_mouth) 
        
        # اگر لبه‌های بینی و دهان کمتر باشد، ماسک ممکن است پوشیده باشد
        if nose_edges_count < 100 and mouth_edges_count < 100:
            return True
        else:
            return False
# متد: 
#َشدت نواحی چشم و دهان وو بینی از یک محدودیت کمتر باشد
# نتیجه:
# برای تشخیص ماسک خوب است ان هم ماسک سیاه مثلا
def detect_face_covering(image, threshold=50):
    # تبدیل تصویر به RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (640, 640))
    results = face_detection.process(image_rgb)
    
    if not results.detections:
        return "چهره یافت نشد"
    
    for detection in results.detections:
        # شناسایی نواحی صورت
        keypoints = detection.location_data.relative_keypoints
        nose = keypoints[2]  # نقطه مربوط به بینی
        mouth = keypoints[3]  # نقطه مربوط به دهان
        left_eye = keypoints[0]  # نقطه مربوط به چشم چپ
        right_eye = keypoints[1]  # نقطه مربوط به چشم راست
        
        # تبدیل مختصات به ابعاد تصویر
        nose_x, nose_y = int(nose.x * image.shape[1]), int(nose.y * image.shape[0])
        mouth_x, mouth_y = int(mouth.x * image.shape[1]), int(mouth.y * image.shape[0])
        left_eye_x, left_eye_y = int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0])
        right_eye_x, right_eye_y = int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0])
        
        # استخراج نواحی اطراف بینی، دهان و چشم‌ها
        nose_area = image[nose_y-20:nose_y+20, nose_x-20:nose_x+20]
        mouth_area = image[mouth_y-20:mouth_y+20, mouth_x-20:mouth_x+20]
        left_eye_area = image[left_eye_y-20:left_eye_y+20, left_eye_x-20:left_eye_x+20]
        right_eye_area = image[right_eye_y-20:right_eye_y+20, right_eye_x-20:right_eye_x+20]
        
        # محاسبه میانگین شدت رنگ در این نواحی
        nose_intensity = np.mean(nose_area)
        mouth_intensity = np.mean(mouth_area)
        left_eye_intensity = np.mean(left_eye_area)
        right_eye_intensity = np.mean(right_eye_area)
        
        # بررسی پوشش بر اساس شدت رنگ
        if nose_intensity < threshold or mouth_intensity < threshold or left_eye_intensity < threshold or right_eye_intensity < threshold:
            return True
        else:
            return False
    
    return "چهره یافت نشد"

    # # استخراج رنگ از نواحی چهره
    # try:
    #     nose_x, nose_y = key_positions["nose"][0]
    #     mouth_x, mouth_y = key_positions["mouth"][0]
    #     left_eye_x, left_eye_y = key_positions["left_eye"][0]
    #     right_eye_x, right_eye_y = key_positions["right_eye"][0]

    #     nose_color = np.mean(image[nose_y-5:nose_y+5, nose_x-5:nose_x+5])
    #     mouth_color = np.mean(image[mouth_y-5:mouth_y+5, mouth_x-5:mouth_x+5])
    #     left_eye_color = np.mean(image[left_eye_y-5:left_eye_y+5, left_eye_x-5:left_eye_x+5])
    #     right_eye_color = np.mean(image[right_eye_y-5:right_eye_y+5, right_eye_x-5:right_eye_x+5])

    #     color_threshold = 20  
    #     if abs(nose_color - mouth_color) < color_threshold and abs(left_eye_color - right_eye_color) < color_threshold:
    #         return True  

    # except Exception as e:
    #     print("Error:", e)  
    #     return True  

    return False

# متد: 
# پیش پردازش تصویر . ریسایز . افزایش کنتراست . تبدیل به RGB
def preprocess_frame(frame):
    if frame is None or frame.size == 0:
        print("Invalid frame in preprocess_frame")
        return None,None
    try: 
        frame,msg = correct_rotation_preciseOK(frame) 
        # cv2.imwrite(f"{UPLOAD_FOLDER}/1p.jpg", frame)

        # افزایش کنتراست و روشنایی
        # frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        # تبدیل BGR به RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # تغییر اندازه
        # frame_rgb = cv2.resize(frame_rgb, (640,480))
        return frame_rgb,frame
    except Exception as e:
        print("Error in preprocess_frame:", e)
        return None,None

# *************************************** END  Funstios *****************

# **************  APIs:  *****************

# *********** API for Setting:
# تنظیمات پیش‌فرض
settings = {
    'check_background': True,
    'check_eye': True,
    'check_mouth': False,
    'check_brightness':True,
    'check_brightness_face':True,
    'check_head':True,
    'check_blurry':True,
    'check_anycoverface':True
}

@app.route('/set_settings', methods=['POST'])
def set_settings():
    global settings
    settings.update(request.json)
    return jsonify({'message': 'Settings updated successfully!'})

#***********************

# ************ API Video *******

## Video Frame by frame
# get a video, return frames and warning for all frames
@app.route("/process_video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "ویدیو ارسال نشده"}), 400

    video_file = request.files["video"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video_path = temp.name
        video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_results = []
    i=0
    while i<100:
        i=i+1
        success, frame = cap.read()
        if not success:
            break
        if frame is None:
            print("Skipping invalid preprocessed frame")
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        raw_encoded = base64.b64encode(buffer).decode('utf-8')

        files = {"frame": ("frame.jpg", image_bytes, "image/jpeg")}
         
        
        url = "http://localhost:5000/process_img" 
        response = requests.post(url, files=files)
        if response.status_code == 200:
            data = response.json() 
            warning = data.get("warning", "")
            processed_encoded = data.get("frame", "")
        else:
            print("Error:", response.json())
            processed_encoded = ""
            warning = "خطا در پردازش" 
        frame_results.append({
            # 'raw': raw_encoded,
            'processed': processed_encoded,
            'warning':warning
        })
        if len(frame_results) >= 5000:  # محدودیت برای جلوگیری از ارسال زیاد
            break 
    cap.release()
    os.remove(video_path)

    return jsonify({"frames": frame_results})  # لیست تمام فریم‌ها به صورت base64


##  Video Multi  Thread --- We Use to thise   ;) ----
import concurrent.futures
@app.route("/process_videoMulti", methods=["POST"])
def process_videoMulti():
    if "video" not in request.files:
        return jsonify({"error": "ویدیو ارسال نشده"}), 400

    video_file = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video_path = temp.name
        video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0

    while i < 100:
        i += 1
        success, frame = cap.read()
        if not success or frame is None:
            continue
        frames.append(frame)

    cap.release()
    os.remove(video_path)

    def process_frame(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        raw_encoded = base64.b64encode(buffer).decode('utf-8')

        files = {"frame": ("frame.jpg", image_bytes, "image/jpeg")}
        url = "http://localhost:5000/process_img"

        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                processed_encoded = data.get("frame", "")
                warning = data.get("warning", "")
            else:
                processed_encoded = ""
                warning = "خطا در پردازش"
        except Exception as e:
            processed_encoded = ""
            warning = f"خطای ارسال: {str(e)}"

        return {
            # 'raw': raw_encoded,
            'processed': processed_encoded,
            'warning': warning
        }

    # ✅ اجرای موازی با ThreadPool
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        frame_results = list(executor.map(process_frame, frames))
    

    new_video = process_video_file(video_file)
    return jsonify({"frames": frame_results})


##  Video Multi  Thread --- We Use to thise   ;) ----
 
@app.route("/process_videoMultiJustCheck", methods=["POST"])
def process_videoMultiJustCheck():
    print("hi video API")
    if "video" not in request.files:
        return jsonify({"error": "ویدیو ارسال نشده"}), 400

    video_file = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video_path = temp.name
        video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0

    while i < 100:
        
        if(i%5==0):
            success, frame = cap.read()
            if not success or frame is None:
                continue
            frames.append(frame)
        i+=1    
        
    cap.release()
    os.remove(video_path)

    def process_frame(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        raw_encoded = base64.b64encode(buffer).decode('utf-8')

        files = {"frame": ("frame.jpg", image_bytes, "image/jpeg")}
        url = "http://localhost:5000/process_imgSetting"

        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                processed_encoded = frame
                warning = data.get("warning", "")[0]
                print("data.get('warning', "")[0]  ",data.get("warning", "")[0] )
            else:
                processed_encoded = ""
                warning = "خطا در پردازش"
        except Exception as e:
            processed_encoded = ""
            warning = f"خطای ارسال: {str(e)}"
        print("warning one by one",warning)
        return {
            # 'raw': raw_encoded,
            'frame': processed_encoded,
            'warning': warning
        }

    # ✅ اجرای موازی با ThreadPool
    # with concurrent.futures.ThreadPoolExecutor(max_workers=len(frames)) as executor:
    #     frame_results = list(executor.map(process_frame, frames))
        # map منتظر میشه همه تردها کارشون تموم بشه. //
        # ولی من میخوام اولین ترد مقدارش اوکی بود تموم بشه.

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(frames)) as executor:
        # futures = {executor.submit(process_frame, frame): frame for frame in frames}
        futures = [executor.submit(process_frame, frame) for frame in frames]
    
    # for future in concurrent.futures.as_completed(futures):
    #     result = future.result()
    #     if result.warning == "Face OK!":  # یا if result == "ok" بسته به تعریف تو
    #         # بقیه رو نادیده می‌گیریم
    #         for f in futures:
    #             f.cancel()  # در واقع فقط تردهایی که هنوز شروع نشده‌اند کنسل می‌شوند
    #         print("Found OK frame.")
    #         framok=result.frame
    #         warningok=result.warning
    #         break 
    framok=None
    warningok="NO"
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print("result[warning]------ ", result["warning"])
        if result["warning"]=="Face OK!":
            framok=result["frame"]
            warningok=result["warning"]
            print("Found OK result!*********************************")

            #Save video in DB
            
            # new_videoj = process_video_file(video_file)
            # print("process_video_file    doneEEEEEEEEE",new_videoj)
            # 
            _, buffer = cv2.imencode('.jpg', framok)
            framok = base64.b64encode(buffer).decode('utf-8')
            return jsonify({"frame": framok,"warning":warningok})
            break
    # return jsonify({"frames": frame_results})
    print("warningok",warningok)
        # تبدیل تصویر پردازش‌شده به base64
    
    return jsonify({"frame": framok,"warning":warningok})

##  Video Multi Async
async def process_frame_async(session, frame):
    if frame is None:
        return {"raw": "", "processed": "", "warning": "فریم نامعتبر"}

    _, buffer = cv2.imencode('.jpg', frame)
    raw_encoded = base64.b64encode(buffer).decode('utf-8')
    data = aiohttp.FormData()
    data.add_field('frame', buffer.tobytes(), filename='frame.jpg', content_type='image/jpeg')

    try:
        async with session.post('http://localhost:5000/process_img', data=data) as resp:
            if resp.status == 200:
                json_data = await resp.json()
                return {
                    # 'raw': raw_encoded,
                    'processed': json_data.get("frame", ""),
                    'warning': json_data.get("warning", "")
                }
            else:
                return {'raw': raw_encoded, 'processed': '', 'warning': 'خطا در پردازش'}
    except Exception as e:
        return {'raw': raw_encoded, 'processed': '', 'warning': f'خطا: {str(e)}'}

async def process_all_frames(frames):
    async with aiohttp.ClientSession() as session:
        tasks = [process_frame_async(session, frame) for frame in frames]
        return await asyncio.gather(*tasks)

@app.route("/process_videoasync", methods=["POST"])
def process_videoasync():
    if "video" not in request.files:
        return jsonify({"error": "ویدیو ارسال نشده"}), 400

    video_file = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video_path = temp.name
        video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while i < 100:
        i += 1
        success, frame = cap.read()
        if not success or frame is None:
            break
        frames.append(frame)
    cap.release()
    os.remove(video_path)

    results = asyncio.run(process_all_frames(frames))
    return jsonify({"frames": results})


# MultiProcessor
def process_frame_mp(frame_data):
    """فانکشنی که هر فریم رو جدا پردازش می‌کنه (برای multiprocessing)"""
    idx, frame = frame_data
    if frame is None:
        return {"index": idx, "raw": "", "processed": "", "warning": "فریم نامعتبر"}

    _, buffer = cv2.imencode('.jpg', frame)
    raw_encoded = base64.b64encode(buffer).decode('utf-8')
    image_bytes = buffer.tobytes()
    files = {"frame": ("frame.jpg", image_bytes, "image/jpeg")}

    try:
        response = requests.post("http://localhost:5000/process_img", files=files)
        if response.status_code == 200:
            data = response.json()
            return {
                "index": idx,
                # "raw": raw_encoded,
                "processed": data.get("frame", ""),
                "warning": data.get("warning", "")
            }
        else:
            return {"index": idx, "raw": raw_encoded, "processed": "", "warning": "خطا در پردازش"}
    except Exception as e:
        return {"index": idx, "raw": raw_encoded, "processed": "", "warning": f"خطا: {str(e)}"}

@app.route("/process_videoMultiProcessor", methods=["POST"])
def process_videoMultiProcessor():
    if "video" not in request.files:
        return jsonify({"error": "ویدیو ارسال نشده"}), 400

    video_file = request.files["video"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video_path = temp.name
        video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while i < 100:
        i += 1
        success, frame = cap.read()
        if not success or frame is None:
            break
        frames.append((i, frame))
    cap.release()
    os.remove(video_path)

    # استفاده از Pool برای پردازش موازی
    with Pool(processes=cpu_count()) as pool:  # یا عدد دلخواه مثلاً 4
        results = pool.map(process_frame_mp, frames)

    # مرتب‌سازی براساس index برای جلوگیری از جابجا شدن ترتیب
    results.sort(key=lambda x: x["index"])
    results_clean = [{"raw": r["raw"], "processed": r["processed"], "warning": r["warning"]} for r in results]

    return jsonify({"frames": results_clean})

#  # فیعلا بمانددددددددد
# @app.route('/processBuffer_img', methods=['POST']) 
# def process_frameBuffer():
#     resultframes = []
#     for key in request.files:
#         file = request.files[key]
#         np_img = np.frombuffer(file.read(), np.uint8)
#         frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR) 
    
#         # if 'frame' not in request.files:
#         #     return jsonify({"status": "error", "message": "No image file received"}), 400
 
#     # frame = preprocess_frame(frame)
    
#         valid = False
#         warning = ""
        

#         if frame is None:
#             return jsonify({"status": "error", "message": "Invalid image data"}), 400
        
    
#     # encoded_image=None


#     # if frame_counter % 5 != 0:  # فقط هر 5 فریم یکبار پردازش شود
#         h, w, _ = frame.shape
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_rgb = cv2.resize(frame_rgb, (640, 640))
#         with mp_face_mesh.FaceMesh(
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         ) as face_mesh:
#             results = face_mesh.process(frame)

#         valid = False
#         warning = ""
        
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 face = face_landmarks.landmark
#                 min_x = int(min([point.x for point in face]) * w)
#                 max_x = int(max([point.x for point in face]) * w)
#                 min_y = int(min([point.y for point in face]) * h)
#                 max_y = int(max([point.y for point in face]) * h)
            
#                 # print(f"Bounding Box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
#                 # if 0 <= min_x < w and 0 <= min_y < h and 0 < max_x <= w and 0 < max_y <= h:
#                 #     print("Bounding box is inside the image")
#                 # else:
#                 #      print("Bounding box is out of image bounds")
#             # رسم کادر دور چهره
#                 cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
                
#                 # cv2.imshow("Processed Frame", frame)
#                 # cv2.waitKey(10000)
#                 left_eye = np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
#                 right_eye = np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
#                 nose = np.array([face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h])

#                     # محاسبه فاصله بین دو چشم
#                 eye_distance = np.linalg.norm(left_eye - right_eye)

#                 # محاسبه زاویه چرخش چهره
#                 dx = right_eye[0] - left_eye[0]
#                 dy = right_eye[1] - left_eye[1]
#                 angle = np.degrees(np.arctan2(dy, dx))
#                 # نقاط کلیدی چشم‌ها
#                 left_eyes = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
#                 right_eyes = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
 
#                 # iris_right = face_landmarks.landmark[468]
#                 # iris_left = face_landmarks.landmark[473]
#                 # نقاط چشم چپ (برای تشخیص لبه‌ها)
#                 left_eye_idxs = [33, 133]  # گوشه داخلی و خارجی چشم چپ
#                 left_eye = [face_landmarks.landmark[i] for i in left_eye_idxs]

#                 # نقاط مردمک چشم چپ
#                 iris_left = [face_landmarks.landmark[i] for i in [474, 475, 476, 477]]
#                                 # گرفتن نقاط از mediapipe
#                 landmarks = face_landmarks.landmark

#                 # چشم چپ
#                 left_inner = landmarks[133]
#                 left_outer = landmarks[33]
#                 left_iris = landmarks[468]

#                 # چشم راست
#                 right_inner = landmarks[362]
#                 right_outer = landmarks[263]
#                 right_iris = landmarks[473]

#                 # محاسبه نسبت نگاه
#                 left_ratio = is_eye_looking_forward(left_inner, left_outer, left_iris)
#                 right_ratio = is_eye_looking_forward(right_inner, right_outer, right_iris)

#                 # بررسی اینکه آیا نگاه مستقیم است
#                 is_left_eye_forward = 0.4 < left_ratio < 0.6
#                 is_right_eye_forward = 0.4 < right_ratio < 0.6

#                 # نتیجه‌گیری کلی
#                 if is_left_eye_forward and is_right_eye_forward:
#                     iris=False
#                     print("هر دو چشم به جلو نگاه می‌کنند ✅")
#                 elif is_left_eye_forward:
#                     iris=False
#                     print("فقط چشم چپ به جلوست 👁⬅️")
#                 elif is_right_eye_forward:
#                     iris=False
#                     print("فقط چشم راست به جلوست ⬅️👁")
#                 else:
#                     iris=True
#                     print("هیچ‌کدام از چشم‌ها به جلو نگاه نمی‌کنند ❌")





#                 def eye_aspect_ratio(eye):
#                     """ محاسبه نسبت بازشدگی چشم (EAR) """
#                     A = np.linalg.norm(np.array([eye[1].x * w, eye[1].y * h]) - np.array([eye[5].x * w, eye[5].y * h]))
#                     B = np.linalg.norm(np.array([eye[2].x * w, eye[2].y * h]) - np.array([eye[4].x * w, eye[4].y * h]))
#                     C = np.linalg.norm(np.array([eye[0].x * w, eye[0].y * h]) - np.array([eye[3].x * w, eye[3].y * h]))
#                     return (A + B) / (2.0 * C)

#                 left_EAR = eye_aspect_ratio(left_eyes)
#                 right_EAR = eye_aspect_ratio(right_eyes)
#                 EAR = (left_EAR + right_EAR) / 2.0  # میانگین دو چشم

#                 # نقاط کلیدی لب‌ها
#                 upper_lip = np.array([
#                     [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
#                     for i in [13, 14, 15, 16, 17]
#                 ])
#                 lower_lip = np.array([
#                     [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
#                     for i in [308, 307, 306, 305, 304]
#                 ])
#                 nose_x = face_landmarks.landmark[1].x * w
#                 center_x = (left_eye[0] + right_eye[0]) / 2
#                 looking_straight = abs(nose_x - center_x) < 20
#                 # is_covered = detect_face_cover(face_landmarks, frame)
#                 # is_covered2=detect_maskcANNY(frame)
#                 # is_covered3=detect_mask(frame,mask_threshold=30)
#                 # is_covered4=detect_face_covering(frame, threshold=60)
                

#                 # # مطمئن شدن از اینکه داخل محدوده تصویر می‌مونه
#                 # min_x = max(min_x, 0)
#                 # min_y = max(min_y, 0)
#                 # max_x = min(max_x, w)
#                 # max_y = min(max_y, h)

#                 # حالا بدون خطا می‌تونی برش بزنی
#                 face_roi = frame[min_y:max_y, min_x:max_x]
                
                
                
#                 isyolo=yoloDetect(face_roi)

#                 # pil_image = Image.fromarray(face_roi)
#                 # vit_check=vit(pil_image)
                
#                 # def lip_aspect_ratio(upper, lower):
#                 #     """ محاسبه نسبت بازشدگی لب (LAR) """
#                 #     A = np.linalg.norm(upper[2] - lower[2])  # وسط لب
#                 #     B = np.linalg.norm(upper[0] - lower[0])  # چپ لب
#                 #     C = np.linalg.norm(upper[4] - lower[4])  # راست لب
#                 #     width = np.linalg.norm(upper[0] - upper[4])  # عرض لب
#                 #     return (A + B + C) / (3.0 * width)

#                 # LAR = lip_aspect_ratio(upper_lip, lower_lip)
                

                

#                 # بررسی میزان روشنایی
#                 brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
#                 blurry=is_blurry(frame)

#                 if brightness < 50:
#                     warning = "Too dark! Increase lighting"
#                 elif brightness > 200:
#                     warning = "Too bright! Reduce lighting"
                
#                 elif eye_distance < 90:
#                     warning = "Move closer!"
#                     color = (0, 0, 255)
#                 elif eye_distance > 300:
#                     warning = "Move Back!"
#                     color = (0, 0, 255)    
#                 elif abs(angle) > 10:
#                     warning = "Straighten your head!"
#                     color = (0, 0, 255)
#                 elif blurry:
#                     warning = "Hold. Image is blurry!!"
#                     color = (0, 0, 255)
#                 elif not looking_straight:
#                         warning = "Look straight!"
                
#                 elif EAR < 0.1:
#                     warning = "Open your eyes!"
#                 # elif LAR >= 0.3:
#                 #     warning = "Close your mouth!"  # اولویت دادن به پیام لب‌ها
#                 elif  iris:
#                     warning = " look here!"
#                     color = (255, 255, 0)
#                 elif  isyolo:
#                         warning = " your fACE obstruct by yolo"
#                         color = (0, 0, 255)
#                 # elif  vit_check:
#                 #         warning = " your face obstructed ViT !"
#                 #         color = (0, 0, 255)
#                 else:
#                     warning = "Face OK!"
#                     valid = True  

#                 break  

#             # تبدیل تصویر پردازش‌شده به base64
#         _, buffer = cv2.imencode('.jpg', frame)
#         encoded_image = base64.b64encode(buffer).decode('utf-8')

#         resultframes.append({
#             "valid": valid,
#             "warning": warning,
#             "frame": encoded_image
#         })
#     selected = next((r for r in resultframes if r["valid"]), resultframes[0])  # اگر هیچکدوم valid نبود، اولی

#     return jsonify({
#         "status": "success",
#         "valid": selected["valid"],
#         "warning": selected["warning"],
#         "frame": selected["frame"]
#     }), 200



# //////////////////////////
@app.route('/process_imgSettingwithoutreapeat', methods=['POST'])
def process_frameSettingwithoutreapeat():
    global settings
    print("settings:",settings)

    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "هیچ تصویری ارسال نشده"}), 400

    file = request.files['frame']
    np_img = np.frombuffer(file.read(), np.uint8)
    frame0 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame0 is None:
        return jsonify({"status": "error", "message": "داده تصویر نامعتبر است"}), 400

    frame = preprocess_frame(frame0)
    if frame is None:
        return jsonify({"status": "error", "message": "خطا در پیش‌پردازش تصویر"}), 400

    h, w, _ = frame.shape
    results = face_mesh_detector.process(frame0)
    if not results.multi_face_landmarks:
        return jsonify({"status": "error", "message": "چهره‌ای شناسایی نشد"}), 200

    warning = []
    valid = True  # فرض می‌کنیم همه چیز خوب است، در ادامه بررسی می‌کنیم

    for face_landmarks in results.multi_face_landmarks:
        face = face_landmarks.landmark
        min_x = int(min(p.x for p in face) * w)
        max_x = int(max(p.x for p in face) * w)
        min_y = int(min(p.y for p in face) * h)
        max_y = int(max(p.y for p in face) * h)

        min_x, min_y = max(min_x, 0), max(min_y, 0)
        max_x, max_y = min(max_x, w), min(max_y, h)

        face_roi0 = frame0[min_y:max_y, min_x:max_x]

        left_eye = np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
        right_eye = np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
        nose = np.array([face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h])

        if settings.get('check_head'):
            dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) > 10:
                warning.append("سر خود را صاف نگه دارید!")

        if settings.get('check_eye'):
            def eye_aspect_ratio(eye):
                A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y])) * w
                B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y])) * w
                C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y])) * w
                return (A + B) / (2.0 * C)

            left_EAR = eye_aspect_ratio([face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]])
            right_EAR = eye_aspect_ratio([face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]])
            EAR = (left_EAR + right_EAR) / 2.0

            threshold = 0.22 - (w / 300.0 * 0.03)
            if EAR < threshold:
                warning.append("چشمان خود را بازتر نگه دارید")

            eye_distance = np.linalg.norm(left_eye - right_eye)
            if eye_distance < 75:
                warning.append("لطفا نزدیک‌تر شوید!")
            elif eye_distance > 300:
                warning.append("لطفا کمی عقب‌تر بایستید!")

            nose_x = nose[0]
            center_x = (left_eye[0] + right_eye[0]) / 2
            if abs(nose_x - center_x) > w * 0.05:
                warning.append("مستقیم به دوربین نگاه کنید!")

            # بررسی حرکت مردمک
            iris1 = is_eye_off_center(
                face_landmarks.landmark[362], face_landmarks.landmark[263],
                face_landmarks.landmark[386], face_landmarks.landmark[374],
                face_landmarks.landmark[473]
            )
            iris2 = is_eye_off_center(
                face_landmarks.landmark[133], face_landmarks.landmark[33],
                face_landmarks.landmark[159], face_landmarks.landmark[145],
                face_landmarks.landmark[468]
            )
            if iris1 or iris2:
                warning.append("لطفاً چشم‌ها را به دوربین متمرکز کنید")

        if settings.get('check_mouth'):
            upper_lip = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
                                  for i in [13, 14, 15, 16, 17]])
            lower_lip = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
                                  for i in [308, 307, 306, 305, 304]])

            def lip_aspect_ratio(upper, lower):
                A = np.linalg.norm(upper[2] - lower[2])
                B = np.linalg.norm(upper[0] - lower[0])
                C = np.linalg.norm(upper[4] - lower[4])
                width = np.linalg.norm(upper[0] - upper[4])
                return (A + B + C) / (3.0 * width)

            LAR = lip_aspect_ratio(upper_lip, lower_lip)
            if LAR > 0.3:
                warning.append("دهان خود را ببندید")

        if settings.get('check_anycoverface'):
            if yoloDetect2(face_roi0):
                warning.append("بخشی از چهره پوشیده شده است")

        if settings.get('check_background'):
            background = extract_background_only(frame0)
            check_bg, std_val = is_background_uniformSegment(background, 30)
            if not check_bg:
                warning.append("پس‌زمینه تصویر یکنواخت نیست")

        if settings.get('check_brightness_face'):
            if not is_lighting_uniform_with_hist(face_roi0, show_hist=True):
                warning.append("نور چهره یکنواخت نیست")

        if settings.get('check_brightness'):
            brightness = np.mean(face_roi0)
            if brightness < 50:
                warning.append("خیلی تاریک است، نور را افزایش دهید")
            elif brightness > 200:
                warning.append("خیلی روشن است، نور را کاهش دهید")

        if settings.get('check_blurry'):
            if is_blurry_fft(face_roi0):
                warning.append("تصویر مات است")

    # نهایی‌سازی نتیجه
    if warning:
        return jsonify({"status": "warning", "message": "، ".join(warning)}), 200
    else:
        return jsonify({"status": "ok", "message": "تصویر مناسب است"}), 200


# //////////////////////////////////    
def get_closest_face(img_rgb):
    """تشخیص چهره روی تصویر RGB؛ نزدیک‌ترین چهره (بیشترین فاصله چشم) را برمی‌گرداند."""
    if img_rgb is None or img_rgb.size == 0:
        return None
    h, w = img_rgb.shape[:2] 

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=5,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            print("❌ هیچ صورتی یافت نشد.")
            return None

        # پیدا کردن صورت با بیشترین فاصله بین چشم‌ها
        max_eye_dist = -1
        closest_face_landmarks = None

        for face in results.multi_face_landmarks:
            left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
            right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
            eye_dist = np.linalg.norm(right_eye - left_eye)

            if eye_dist > max_eye_dist:
                max_eye_dist = eye_dist
                closest_face_landmarks = face
                face_landmarks=face.landmark


        if closest_face_landmarks is None:
            return None

        return face_landmarks,closest_face_landmarks

# 
# 
 
# Zhaleh Api, ok after setting 
# @app.route('/process_imgSetting', methods=['POST']) 
def process_frameSettingbackupOK():
    global settings 


    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No image file received"}), 400

    file = request.files['frame']  
    np_img = np.frombuffer(file.read(), np.uint8)
    frame0 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    frame,frame0 = preprocess_frame(frame0)
    # cv2.imwrite(f"{UPLOAD_FOLDER}/1pooooo.jpg", frame0)
    

    

    # valid = False
    # warning = "تصویر شناسایی نشد"
    

    if frame is None:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400
    
    
    # encoded_image=None
    # frame = cv2.resize(frame, (300, 300))
    # h, w, _ = frame.shape

    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_rgb = cv2.resize(frame_rgb, (300, 300))
    h, w, _ = frame.shape
    print("w  " , w)
    results = face_mesh_detector.process(frame0)

    valid = False
    warning =[]
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face = face_landmarks.landmark
            min_x = int(min([point.x for point in face]) * w)
            max_x = int(max([point.x for point in face]) * w)
            min_y = int(min([point.y for point in face]) * h)
            max_y = int(max([point.y for point in face]) * h)
            face_roi = frame[min_y:max_y, min_x:max_x]
            face_roi0 = frame0[min_y:max_y, min_x:max_x]
             # رسم کادر دور چهره
            cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
            
            

            
            # cv2.imshow("Processed Frame", frame)
            # cv2.waitKey(10000)
            left_eye = np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
            right_eye = np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
            nose = np.array([face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h])
            if settings['check_head']:
            # محاسبه زاویه چرخش چهره
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))
                print("angle",angle)
                
            if settings['check_eye']:
                    # محاسبه فاصله بین دو چشم
                eye_distance = np.linalg.norm(left_eye - right_eye)

                
                # نقاط کلیدی چشم‌ها
                left_eyes = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                right_eyes = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

                def eye_aspect_ratio(eye):
                    """ محاسبه نسبت بازشدگی چشم (EAR) """
                    A = np.linalg.norm(np.array([eye[1].x * w, eye[1].y * h]) - np.array([eye[5].x * w, eye[5].y * h]))
                    B = np.linalg.norm(np.array([eye[2].x * w, eye[2].y * h]) - np.array([eye[4].x * w, eye[4].y * h]))
                    C = np.linalg.norm(np.array([eye[0].x * w, eye[0].y * h]) - np.array([eye[3].x * w, eye[3].y * h]))
                    return (A + B) / (2.0 * C)

                left_EAR = eye_aspect_ratio(left_eyes)
                right_EAR = eye_aspect_ratio(right_eyes)
                EAR = (left_EAR + right_EAR) / 2.0  # میانگین دو چشم
                base_threshold = 0.22
                adjustment = w / 300.0 * 0.03
                earth= base_threshold - adjustment
                print ("earth",earth)
                EARTF=EAR<earth
                print("EAR" , EAR)

                nose_x = face_landmarks.landmark[1].x * w
                center_x = (left_eye[0] + right_eye[0]) / 2
                thlf = w * 0.05  # مثلاً 5% عرض تصویر
                looking_straight = abs(nose_x - center_x) < thlf
                # looking_straight = abs(nose_x - center_x) < 20


                # /////
                right_inner = face_landmarks.landmark[362]
                right_outer = face_landmarks.landmark[263]
                right_upper = face_landmarks.landmark[386]
                right_lower = face_landmarks.landmark[374]
                right_iris = face_landmarks.landmark[473]

                    # نگاه چشم راست
                iris1 = is_eye_off_center(right_inner, right_outer, right_upper, right_lower, right_iris)
                # iris1 = get_eye_direction(right_inner, right_outer, right_upper, right_lower, right_iris)
                # print("Right eye:", horiz_r, vert_r)

                # چشم چپ
                left_inner = face_landmarks.landmark[133]
                left_outer = face_landmarks.landmark[33]
                left_upper = face_landmarks.landmark[159]
                left_lower = face_landmarks.landmark[145]
                left_iris = face_landmarks.landmark[468]

                # نگاه چشم چپ
                iris2 = is_eye_off_center(left_inner, left_outer, left_upper, left_lower, left_iris)
           
                # iris2 = get_eye_direction(left_inner, left_outer, left_upper, left_lower, left_iris)
                # print("Left eye:", horiz_l, vert_l)
                if iris1 or iris2:
                    iris=True
                else:
                    iris=False
            print("settings['check_mouth']",settings['check_mouth'])
            if settings['check_mouth']:
                print("shahh")
            # ////
                # نقاط کلیدی لب‌ها
                upper_lip = np.array([
                    [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
                    for i in [13, 14, 15, 16, 17]
                ])
                lower_lip = np.array([
                    [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
                    for i in [308, 307, 306, 305, 304]
                ])
                def lip_aspect_ratio(upper, lower):
                    # """ محاسبه نسبت بازشدگی لب (LAR) """
                    A = np.linalg.norm(upper[2] - lower[2])  # وسط لب
                    B = np.linalg.norm(upper[0] - lower[0])  # چپ لب
                    C = np.linalg.norm(upper[4] - lower[4])  # راست لب
                    width = np.linalg.norm(upper[0] - upper[4])  # عرض لب
                    return (A + B + C) / (3.0 * width)

                LAR = lip_aspect_ratio(upper_lip, lower_lip)

            
            # is_covered = detect_face_cover(face_landmarks, frame)
            # is_covered2=detect_maskcANNY(frame)
            # is_covered3=detect_mask(frame,mask_threshold=30)
            # is_covered4=detect_face_covering(frame, threshold=60)
            

            # # مطمئن شدن از اینکه داخل محدوده تصویر می‌مونه
            # min_x = max(min_x, 0)
            # min_y = max(min_y, 0)
            # max_x = min(max_x, w)
            # max_y = min(max_y, h)
            
            if settings['check_anycoverface']:
                # حالا بدون خطا می‌تونی برش بزنی
                
                
                # ismobilenet=detect_mobilenet(face_roi)
                
                isyolo=yoloDetect2(face_roi0)

            # pil_image = Image.fromarray(face_roi)
            # vit_check=vit(pil_image)
            
           
            
            if settings['check_background']:
                background = extract_background_only(frame0)
                check_backgroundv, std_val = is_background_uniformSegment(background,30)
                print("check_backgroundv", check_backgroundv)

            if settings['check_brightness_face']:
                isFacelightOk = is_lighting_uniform_with_hist(face_roi0, show_hist=True)

            # AngleHead,war = get_head_pose_info(face_roi,face_landmarks)
            if settings['check_brightness']:
            # بررسی میزان روشنایی
                brightness=np.mean(face_roi0)
                print("brightness",brightness)
                # brightness = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
            if settings['check_blurry']:
                blurry=is_blurry_fft(face_roi0)
                # blurry=is_blurry_fftopt(face_roi0)
            if settings['check_brightness']:
                if brightness < 50:
                    warning.append("خیلی تاریک است نور را افزایش دهید. ")
                elif brightness > 200:
                    warning.append("خیلی روشن است نور را کاهش دهید.")
                
             # elif ismobilenet:
            #         warning = "  . mobilenetصورت شما پوشیده شده است. آن را واضح نشان دهید"
            # elif  vit_check:
            #         warning = " your face obstructed ViT !"
            if settings["check_blurry"]:
                if blurry:
                    warning.append( "تصویر مات است")
            if settings['check_head']:
                # AngleHead,war = get_head_pose_info(face_roi,face_landmarks)
                if abs(angle) > 10:
                    warning.append("سر خود را صاف نگه دارید!")
            if settings['check_eye']:
                
                if eye_distance < 75: #90
                    warning.append( "لطفا نزدیکتر شوید!")
                elif eye_distance > 300:
                    warning.append( "لطفا کمی دور بیاستید!")
                 
                if not looking_straight:
                    warning.append("مستقیم را نگاه کنید!")
                # elif AngleHead:
                #     warning=war    
                elif iris:
                    warning.append("مستقیم به دوربین نگاه کنید . "  )
                    
                # elif EAR < 0.25:
                elif EARTF:
                    warning.append( "چشمان خود را باز نگه دراید!")
            # if settings['check_brightness_face']:
            #     if not isFacelightOk:
            #             warning.append("نور دو طرف باید صورت یکنواخت باشد !")

            if settings['check_anycoverface']:
                if isyolo:
                    warning.append(" صورت خود را واضح نشان دهید"  )  
            print("oooooooooooooooStting::: ", settings)
            print("settings['check_mouth']:::: ", settings['check_mouth'])
            if settings['check_mouth']:
                # pass
                
                print("LAR:  ",LAR)
                if LAR >= 0.5:
                    warning.append(" دهان خود را بسته و واضح نشام دهید"  )   
            

            if settings['check_background']:
                if not check_backgroundv:
                    warning.append( " پس زمینه یکدست نیست")
            
            
            
            
              
            # if warning[0]=="":
            # if len(warning)==0:
            #     warning.append("Face OK!")
            #     valid = True  

            # break  

        # تبدیل تصویر پردازش‌شده به base64
    # _, buffer = cv2.imencode('.jpg', frame)
    # encoded_image = base64.b64encode(buffer).decode('utf-8')
    # print("warning[0]",warning[0])
    # return jsonify({
    #         "status": "success",
    #         "valid": valid,
    #         "warning": warning[0],  # پیام نهایی
    #         # "frame": encoded_image  
    #     }), 200
    if len(warning)>0:
        return jsonify({
            # "frame": encoded_image,  
            "status": "warning",
            # "duration_ms": duration,
            "warning": warning
        }), 200
    else:
        if not results.multi_face_landmarks:
            warning.append("No Face Detected!")
        else :
            warning.append("Face OK!")
        return jsonify({
            # "frame": encoded_image,
            "status": "success",
            # "duration_ms": duration,
            "warning": warning
        }), 200

@app.route('/process_imgSetting', methods=['POST']) 
def process_frameSetting():
    global settings 


    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No image file received"}), 400

    file = request.files['frame']  
    np_img = np.frombuffer(file.read(), np.uint8)
    frame0 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    frame,frame0 = preprocess_frame(frame0)
    # cv2.imwrite(f"{UPLOAD_FOLDER}/1pooooo.jpg", frame0)
    

    

    # valid = False
    # warning = "تصویر شناسایی نشد"
    

    if frame is None:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400
    
    
    # encoded_image=None
    # frame = cv2.resize(frame, (300, 300))
    # h, w, _ = frame.shape

    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_rgb = cv2.resize(frame_rgb, (300, 300))
    h, w, _ = frame.shape
    print("w  " , w) 
    valid = False
    warning =[]
    
    # if results.multi_face_landmarks:
        # for face_landmarks in results.multi_face_landmarks:
            #   face = face_landmarks.landmark
    closest = get_closest_face(frame)
    if closest is None:
        return jsonify({
            "status": "warning",
            "warning": ["No Face Detected!"]
        }), 200
    face, face_landmarks = closest

    min_x = int(min([point.x for point in face]) * w)
    max_x = int(max([point.x for point in face]) * w)
    min_y = int(min([point.y for point in face]) * h)
    max_y = int(max([point.y for point in face]) * h)
    face_roi = frame[min_y:max_y, min_x:max_x]
    face_roi0 = frame0[min_y:max_y, min_x:max_x]
        # رسم کادر دور چهره
    cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
    
    

    
    # cv2.imshow("Processed Frame", frame)
    # cv2.waitKey(10000)
    left_eye = np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
    right_eye = np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
    nose = np.array([face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h])
    if settings['check_head']:
    # محاسبه زاویه چرخش چهره
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        print("angle",angle)
        
    if settings['check_eye']:
            # محاسبه فاصله بین دو چشم
        eye_distance = np.linalg.norm(left_eye - right_eye)

        
        # نقاط کلیدی چشم‌ها
        left_eyes = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        right_eyes = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

        def eye_aspect_ratio(eye):
            """ محاسبه نسبت بازشدگی چشم (EAR) """
            A = np.linalg.norm(np.array([eye[1].x * w, eye[1].y * h]) - np.array([eye[5].x * w, eye[5].y * h]))
            B = np.linalg.norm(np.array([eye[2].x * w, eye[2].y * h]) - np.array([eye[4].x * w, eye[4].y * h]))
            C = np.linalg.norm(np.array([eye[0].x * w, eye[0].y * h]) - np.array([eye[3].x * w, eye[3].y * h]))
            return (A + B) / (2.0 * C)

        left_EAR = eye_aspect_ratio(left_eyes)
        right_EAR = eye_aspect_ratio(right_eyes)
        EAR = (left_EAR + right_EAR) / 2.0  # میانگین دو چشم
        base_threshold = 0.22
        adjustment = w / 300.0 * 0.03
        earth= base_threshold - adjustment
        print ("earth",earth)
        EARTF=EAR<earth
        print("EAR" , EAR)

        nose_x = face_landmarks.landmark[1].x * w
        center_x = (left_eye[0] + right_eye[0]) / 2
        thlf = w * 0.05  # مثلاً 5% عرض تصویر
        looking_straight = abs(nose_x - center_x) < thlf
        # looking_straight = abs(nose_x - center_x) < 20


        # /////
        right_inner = face_landmarks.landmark[362]
        right_outer = face_landmarks.landmark[263]
        right_upper = face_landmarks.landmark[386]
        right_lower = face_landmarks.landmark[374]
        right_iris = face_landmarks.landmark[473]

            # نگاه چشم راست
        iris1 = is_eye_off_center(right_inner, right_outer, right_upper, right_lower, right_iris)
        # iris1 = get_eye_direction(right_inner, right_outer, right_upper, right_lower, right_iris)
        # print("Right eye:", horiz_r, vert_r)

        # چشم چپ
        left_inner = face_landmarks.landmark[133]
        left_outer = face_landmarks.landmark[33]
        left_upper = face_landmarks.landmark[159]
        left_lower = face_landmarks.landmark[145]
        left_iris = face_landmarks.landmark[468]

        # نگاه چشم چپ
        iris2 = is_eye_off_center(left_inner, left_outer, left_upper, left_lower, left_iris)
    
        # iris2 = get_eye_direction(left_inner, left_outer, left_upper, left_lower, left_iris)
        # print("Left eye:", horiz_l, vert_l)
        if iris1 or iris2:
            iris=True
        else:
            iris=False
    print("settings['check_mouth']",settings['check_mouth'])
    if settings['check_mouth']:
        print("shahh")
    # ////
        # نقاط کلیدی لب‌ها
        upper_lip = np.array([
            [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
            for i in [13, 14, 15, 16, 17]
        ])
        lower_lip = np.array([
            [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
            for i in [308, 307, 306, 305, 304]
        ])
        def lip_aspect_ratio(upper, lower):
            # """ محاسبه نسبت بازشدگی لب (LAR) """
            A = np.linalg.norm(upper[2] - lower[2])  # وسط لب
            B = np.linalg.norm(upper[0] - lower[0])  # چپ لب
            C = np.linalg.norm(upper[4] - lower[4])  # راست لب
            width = np.linalg.norm(upper[0] - upper[4])  # عرض لب
            return (A + B + C) / (3.0 * width)

        LAR = lip_aspect_ratio(upper_lip, lower_lip)
 
    if settings['check_anycoverface']:
        # حالا بدون خطا می‌تونی برش بزنی
        
        
        # ismobilenet=detect_mobilenet(face_roi)
        
        isyolo=yoloDetect2(face_roi0)

    # pil_image = Image.fromarray(face_roi)
    # vit_check=vit(pil_image)
    
    
    
    if settings['check_background']:
        background = extract_background_only(frame0)
        check_backgroundv, std_val = is_background_uniformSegment(background,30)
        print("check_backgroundv", check_backgroundv)

    if settings['check_brightness_face']:
        isFacelightOk = is_lighting_uniform_with_hist(face_roi0, show_hist=True)

    # AngleHead,war = get_head_pose_info(face_roi,face_landmarks)
    if settings['check_brightness']:
    # بررسی میزان روشنایی
        brightness=np.mean(face_roi0)
        print("brightness",brightness)
        # brightness = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
    if settings['check_blurry']:
        blurry=is_blurry_fft(face_roi0)
        # blurry=is_blurry_fftopt(face_roi0)
    if settings['check_brightness']:
        if brightness < 50:
            warning.append("خیلی تاریک است نور را افزایش دهید. ")
        elif brightness > 200:
            warning.append("خیلی روشن است نور را کاهش دهید.")
        
        # elif ismobilenet:
    #         warning = "  . mobilenetصورت شما پوشیده شده است. آن را واضح نشان دهید"
    # elif  vit_check:
    #         warning = " your face obstructed ViT !"
    if settings["check_blurry"]:
        if blurry:
            warning.append( "تصویر مات است")
    if settings['check_head']:
        # AngleHead,war = get_head_pose_info(face_roi,face_landmarks)
        if abs(angle) > 10:
            warning.append("سر خود را صاف نگه دارید!")
    if settings['check_eye']:
        
        if eye_distance < 75: #90
            warning.append( "لطفا نزدیکتر شوید!")
        elif eye_distance > 300:
            warning.append( "لطفا کمی دور بیاستید!")
            
        if not looking_straight:
            warning.append("مستقیم را نگاه کنید!")
        # elif AngleHead:
        #     warning=war    
        elif iris:
            warning.append("مستقیم به دوربین نگاه کنید . "  )
            
        # elif EAR < 0.25:
        elif EARTF:
            warning.append( "چشمان خود را باز نگه دراید!")
    # if settings['check_brightness_face']:
    #     if not isFacelightOk:
    #             warning.append("نور دو طرف باید صورت یکنواخت باشد !")

    if settings['check_anycoverface']:
        if isyolo:
            warning.append(" صورت خود را واضح نشان دهید"  )  
    # print("oooooooooooooooStting::: ", settings)
    # print("settings['check_mouth']:::: ", settings['check_mouth'])
    if settings['check_mouth']:
        # pass
        
        print("LAR:  ",LAR)
        if LAR >= 0.5:
            warning.append(" دهان خود را بسته و واضح نشام دهید"  )   
    

    if settings['check_background']:
        if not check_backgroundv:
            warning.append( " پس زمینه یکدست نیست")
    
            
            
            
              
            # if warning[0]=="":
            # if len(warning)==0:
            #     warning.append("Face OK!")
            #     valid = True  

            # break  

        # تبدیل تصویر پردازش‌شده به base64
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    # print("warning[0]",warning[0])
    # return jsonify({
    #         "status": "success",
    #         "valid": valid,
    #         "warning": warning[0],  # پیام نهایی
    #         # "frame": encoded_image  
    #     }), 200
    if len(warning)>0:
        return jsonify({
            # "frame": encoded_image,  
            "status": "warning",
            # "msg":msg,
            # "duration_ms": duration,
            "warning": warning
        }), 200
    else:
        # if not results.multi_face_landmarks:
        #     warning.append("No Face Detected!")
        # else :
        warning.append("Face OK!")
        return jsonify({
            # "frame": encoded_image,
            "status": "success",
            # "msg":msg,
            # "duration_ms": duration,
            "warning": warning
        }), 200

# @app.route('/rotateimg', methods=['POST']) 
# def rotateimg():
#     if 'frame' not in request.files:
#         return jsonify({"status": "error", "message": "هیچ تصویری ارسال نشده"}), 400

#     file = request.files['frame']
#     np_img = np.frombuffer(file.read(), np.uint8)
#     frame0 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     if frame0 is None:
#         return jsonify({"status": "error", "message": "داده تصویر نامعتبر است"}), 400
 
#     frame,m = correct_rotation_preciseOK(frame0)
 
#     if frame is None:
#         return jsonify({"status": "error", "message": "خطا در پیش‌پردازش تصویر"}), 400
#     return jsonify({
#         "status": "success",
#         "frame": frame
#     }), 200
    



# # //////////////////////////
@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()

    # فقط کلیدهایی که در کلاس Config تعریف شده‌اند تغییر می‌کنیم
    valid_keys = {
        "CHECK_BRIGHTNESS", "CHECK_BLURRY", "CHECK_HEAD", "CHECK_EYE",
        "CHECK_BACKGROUND", "CHECK_MOUTH", "CHECK_FACE_COVER",
        "MIN_EYE_DISTANCE", "MAX_EYE_DISTANCE","CHECK_BRIGHTNESS_FACE"
    }
    print("Config.CHECK_BRIGHTNESS",data['CHECK_BRIGHTNESS'])
    updates = []
    for key, value in data.items():
        if key in valid_keys:
            setattr(Config, key, value)
            updates.append(f"{key} set to {value}")
        else:
            return jsonify({
                "status": "error",
                "message": f"Invalid config key: {key}"
            }), 400

    return jsonify({
        "status": "success",
        "message": "Settings updated",
        "updated": updates
    }), 200


# پیکربندی تنظیمات
class Config:
    CHECK_BRIGHTNESS = True
    CHECK_BRIGHTNESS_FACE=False
    CHECK_BLURRY = True
    CHECK_HEAD = True
    CHECK_EYE = True
    CHECK_BACKGROUND = True
    CHECK_MOUTH = False
    CHECK_FACE_COVER = True
    CHECK_FACECOVER=True
    MIN_EYE_DISTANCE = 75
    MAX_EYE_DISTANCE = 300

# Initialize Mediapipe FaceMesh بیرون از توابع برای سرعت بهتر
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,  # پشتیبانی از multi-face
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# توابع کمکی
def preprocess_image(file):
    
    np_img = np.frombuffer(file.read(), np.uint8)
    frame0 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame0 is None:
        raise ValueError("Invalid image data")
    # # افزایش کنتراست و روشنایی
    # frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        
    frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (640,480))
    return frame,frame0

def detect_face_landmarks(frame):
    results = face_mesh_detector.process(frame)
    return results.multi_face_landmarks
def getEyes(face, w,h):
    left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
    right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
    return left_eye,right_eye
def compute_eye_distance( left_eye,right_eye):
    # محاسبه فاصله بین دو چشم
    eye_distance = np.linalg.norm(left_eye - right_eye)
    return  eye_distance
    
def compute_head_angle(left_eye,right_eye): 
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    print("angle  s",angle)
    return angle

def is_eyes_closed(face,w,h):  
    def eye_aspect_ratio(eye_points):
        A = np.linalg.norm(np.array([eye_points[1].x*w, eye_points[1].y*h]) - np.array([eye_points[5].x*w, eye_points[5].y*h]))
        B = np.linalg.norm(np.array([eye_points[2].x*w, eye_points[2].y*h]) - np.array([eye_points[4].x*w, eye_points[4].y*h]))
        C = np.linalg.norm(np.array([eye_points[0].x*w, eye_points[0].y*h]) - np.array([eye_points[3].x*w, eye_points[3].y*h]))
        return (A + B) / (2.0 * C)
    left_eyes = [face.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
    right_eyes = [face.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

    left_ear = eye_aspect_ratio(left_eyes)
    right_ear = eye_aspect_ratio(right_eyes)
    ear = (left_ear + right_ear) / 2.0
    return ear < 0.1

def is_looking_straight(face,w,left_eye, right_eye):
    nose = face.landmark[1].x * w
    center_x = (left_eye[0] + right_eye[0]) / 2
    return abs(nose - center_x) < 20
def get_eye_directionTWOEyes(face_landmarks):
    # /////
    right_inner = face_landmarks.landmark[362]
    right_outer = face_landmarks.landmark[263]
    right_upper = face_landmarks.landmark[386]
    right_lower = face_landmarks.landmark[374]
    right_iris = face_landmarks.landmark[473]

        # نگاه چشم راست
    # iris1 = get_eye_direction(right_inner, right_outer, right_upper, right_lower, right_iris)
    iris1 = is_eye_off_center(right_inner, right_outer, right_upper, right_lower, right_iris)
            
    # print("Right eye:", horiz_r, vert_r)

    # چشم چپ
    left_inner = face_landmarks.landmark[133]
    left_outer = face_landmarks.landmark[33]
    left_upper = face_landmarks.landmark[159]
    left_lower = face_landmarks.landmark[145]
    left_iris = face_landmarks.landmark[468]

    # نگاه چشم چپ
    # iris2 = get_eye_direction(left_inner, left_outer, left_upper, left_lower, left_iris)
    iris2 = is_eye_off_center(left_inner, left_outer, left_upper, left_lower, left_iris)
            
    if iris1 or iris2:
        return True
    else:
        return False

# --- مسیر اصلی پردازش ---
@app.route('/process_imgChat', methods=['POST'])
def process_imgChat():
    print("sssssssssssssssssaaaaaaallam")
    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No image file received"}), 400
    try:
        # start_time = time.time()

        file = request.files['frame']
        frame,frame0 = preprocess_image(file)
        h, w, _ = frame.shape
        print("w" , w)
        landmarks = detect_face_landmarks(frame0)
        if not landmarks:
            print("چهره‌ای پیدا نشد")
            return jsonify({"status": "error", "warning": "چهره‌ای پیدا نشد"}), 200

        face = landmarks[0]  # فقط اولین چهره برای سرعت بهتر

        warnings = []

        # ناحیه صورت
        min_x = int(min([point.x for point in face.landmark]) * w)
        max_x = int(max([point.x for point in face.landmark]) * w)
        min_y = int(min([point.y for point in face.landmark]) * h)
        max_y = int(max([point.y for point in face.landmark]) * h)
        face_roi = frame[min_y:max_y, min_x:max_x] 
        face_roi0 = frame0[min_y:max_y, min_x:max_x] 


        left_eye,right_eye=getEyes(face, w,h)


        # بررسی ها
        if Config.CHECK_BRIGHTNESS:
            brightness = np.mean(cv2.cvtColor(face_roi0, cv2.COLOR_RGB2GRAY))
            if brightness < 50:
                warnings.append("نور تصویر کم است.")
                
            elif brightness > 200:
                warnings.append("نور تصویر زیاد است.")

        if Config.CHECK_BLURRY:
            if is_blurry_fft(face_roi0):
                warnings.append("تصویر مات است.")

        if Config.CHECK_HEAD:
            angle = compute_head_angle(left_eye,right_eye)
            if abs(angle) > 10:
                warnings.append("لطفا سر خود را صاف نگه دارید.")

        if Config.CHECK_EYE:
              # فاصله چشم
            eye_distance = compute_eye_distance(left_eye,right_eye)
            if eye_distance < Config.MIN_EYE_DISTANCE:
                warnings.append("بیشتر به دوربین نزدیک شوید.")
            elif eye_distance > Config.MAX_EYE_DISTANCE:
                warnings.append("کمی عقب‌تر بروید.")
                
            if not is_looking_straight(face,w,left_eye,right_eye):
                warnings.append("به دوربین نگاه کنید.")
            elif is_eyes_closed(face,w,h):
                warnings.append("چشمان خود را باز کنید.")
            elif get_eye_directionTWOEyes(face):
                warnings.append("مستقیم به دوربین نگاه کنید . "  )
              
        # if Config. CHECK_BRIGHTNESS_FACE:
        #     if not is_lighting_uniform_with_hist(face_roi):
        #         warnings.append("نور دو طرف باید صورت یکنواخت باشد !")
    
        

        if Config.CHECK_FACECOVER:
            if yoloDetect2(face_roi0):
                warnings.append("صورت پوشیده شده است.")

      

        if Config.CHECK_BACKGROUND:
            background = extract_background_only(frame)
            is_uniform, _ = is_background_uniformSegment(background, 40)
            if not is_uniform:
                warnings.append("پس‌زمینه یکدست نیست.")

        # duration = round((time.time() - start_time) * 1000)  # ms
        
        # _, buffer = cv2.imencode('.jpg', frame)
        # encoded_image = base64.b64encode(buffer).decode('utf-8')
 
        # if len(warnings)>0:
        #     print("warnings : ")
        #     print(warnings[0])
        #     return jsonify({
        #         # "frame": encoded_image,  
        #         "status": "warning",
        #         # "duration_ms": duration,
        #         "warning": warnings
        #     }), 200
        # else:
        #     warnings.append("Face OK!")
        #     print( "تصویر تایید شد.")
        #     return jsonify({
        #         # "frame": encoded_image,
        #         "status": "success",
        #         # "duration_ms": duration,
        #         "warning": warnings
        #     }), 200


        if len(warnings)>0:
            return jsonify({
            # "frame": encoded_image,  
            "status": "warning",
            # "duration_ms": duration,
            "warning": warnings
            }), 200
        else:
            if not landmarks:
                warnings.append("No Face Detected!")
            else :
                warnings.append("Face OK!")
            return jsonify({
            # "frame": encoded_image,
            "status": "success",
            # "duration_ms": duration,
            "warning": warnings
            }), 200




    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# //////////////////////////////

@app.route('/process_imgparallel', methods=['POST']) 
def process_frameParallel():
    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No image file received"}), 400

    file = request.files['frame']
    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (224, 224))

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(frame)

    if not results.multi_face_landmarks:
        # _, buffer = cv2.imencode('.jpg', frame)
        # encoded_image = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            "status": "success",
            "valid": False,
            "warning": "تصویر شناسایی نشد",
            # "frame": encoded_image
        }), 200

    face_landmarks = results.multi_face_landmarks[0].landmark
    min_x = int(min([p.x for p in face_landmarks]) * w)
    max_x = int(max([p.x for p in face_landmarks]) * w)
    min_y = int(min([p.y for p in face_landmarks]) * h)
    max_y = int(max([p.y for p in face_landmarks]) * h)

    face_roi = frame[min_y:max_y, min_x:max_x]

    # محاسبات کلیدی غیرموازی
    left_eye = np.array([face_landmarks[33].x * w, face_landmarks[33].y * h])
    right_eye = np.array([face_landmarks[263].x * w, face_landmarks[263].y * h])
    nose_x = face_landmarks[1].x * w
    center_x = (left_eye[0] + right_eye[0]) / 2
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    eye_distance = np.linalg.norm(left_eye - right_eye)
    looking_straight = abs(nose_x - center_x) < 20

    # /////
    right_inner = face_landmarks.landmark[362]
    right_outer = face_landmarks.landmark[263]
    right_upper = face_landmarks.landmark[386]
    right_lower = face_landmarks.landmark[374]
    right_iris = face_landmarks.landmark[473]

        # نگاه چشم راست
    iris1 = get_eye_direction(right_inner, right_outer, right_upper, right_lower, right_iris)
    # print("Right eye:", horiz_r, vert_r)

    # چشم چپ
    left_inner = face_landmarks.landmark[133]
    left_outer = face_landmarks.landmark[33]
    left_upper = face_landmarks.landmark[159]
    left_lower = face_landmarks.landmark[145]
    left_iris = face_landmarks.landmark[468]

    # نگاه چشم چپ
    iris2 = get_eye_direction(left_inner, left_outer, left_upper, left_lower, left_iris)
    # print("Left eye:", horiz_l, vert_l)
    if iris1 or iris2:
        iris=True
    else:
        iris=False

    # ////

    def run_parallel_tasks():
        futures = {
            "background": executorprocesor.submit(run_background_check, frame),
            "light": executorprocesor.submit(is_lighting_uniform_with_hist, face_roi, False),
            "yolo": executorprocesor.submit(yoloDetect2, face_roi),
            "head_pose": executorprocesor.submit(get_head_pose_info, face_roi, results.multi_face_landmarks[0]),
            "blurry": executorprocesor.submit(is_blurry_fft, frame),
             
        }

        resultss = {}
        for key, future in futures.items():
            resultss[key] = future.result()

        return resultss

    # اجرای موازی
    parallel_results = run_parallel_tasks()

    warning = ""
    valid = False

    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if brightness < 50:
        warning = "خیلی تاریک است. نور را افزایش دهید. "
    elif brightness > 200:
        warning = "خیلی روشن است و نور را کاهش دهید."
    elif parallel_results["yolo"]:
        warning = "صورت شما پوشیده شده است. آن را واضح نشان دهید"
    elif eye_distance < 90:
        warning = "لطفا نزدیکتر شوید!"
    elif eye_distance > 300:
        warning = "لطفا کمی دور بیاستید!"
    elif abs(angle) > 10:
        warning = "سر خود را صاف نگه دارید!"
    elif not looking_straight:
        warning = "مستقیم را نگاه کنید!"
    elif iris:
        warning = "مستقیم دوربین را نگاه کنید!"  
    elif parallel_results["head_pose"][0]:
        warning = parallel_results["head_pose"][1]
    elif parallel_results["blurry"]:
        warning = " تصویر مات است.!!"
    elif not parallel_results["background"][0]:
        warning = "پشت سرتان باید یک رنگ ثابت مانند دیوار بدون سایه باشد !"
    elif not parallel_results["light"]:
        warning = "نور دو طرف باید صورت یکنواخت باشد !"
    
    else:
        warning = "Face OK!" 
        valid = True

    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "status": "success",
        "valid": valid,
        "warning": warning,
        # "frame": encoded_image
    }), 200
 
@app.route('/process_imgthread', methods=['POST'])
async def process_framePthread():
    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No image file received"}), 400

    file = request.files['frame']
    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400

    h, w, _ = frame.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return jsonify({"status": "success", "valid": False, "warning": "چهره‌ای شناسایی نشد!"}), 200

    face_landmarks = results.multi_face_landmarks[0]
    face = face_landmarks.landmark
    min_x = int(min([pt.x for pt in face]) * w)
    max_x = int(max([pt.x for pt in face]) * w)
    min_y = int(min([pt.y for pt in face]) * h)
    max_y = int(max([pt.y for pt in face]) * h)

    async_results = await process_all_tasks(frame, face_landmarks, h, w, min_x, min_y, max_x, max_y)

    # شرط‌ها و پیام نهایی
    warning = ""
    valid = False

    
    if not async_results["looking_straight"]:
        warning = "مستقیم نگاه کنید!"
    # elif async_results["blurry"]:
    #     warning = "تصویر مات است."
    elif async_results["eye_distance"] < 90:
        warning = "لطفا نزدیک‌تر شوید."
    elif abs(async_results["angle"]) > 10:
        warning = "سرتان را صاف نگه دارید."

    elif async_results["isyolo"]:
        warning = "چهره پوشیده شده است."
        
    elif not async_results["isFacelightOk"]:
        warning = "نور صورت یکنواخت نیست."
    else:
        warning = "چهره تأیید شد!"
        valid = True

    # _, buffer = cv2.imencode('.jpg', frame)
    # encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "status": "success",
        "valid": valid,
        "warning": warning,
        # "frame": encoded_image
    }), 200




# ********** Api for Load Home Page **********
@app.route('/')
def index():
    # return render_template('camera.html')
    return render_template('final.html')
    # return render_template('phase1buffer.html')
 
# ***********************************

# ********** Api for DB**********
import psycopg2
def get_db_connection():
    return psycopg2.connect(host="localhost", database="face_db", user="postgres", password="s123456h")

def save_featuremap_to_db(image):
    
    features = DeepFace.represent(image, model_name="Facenet", enforce_detection=False)
    if features:
        face_vector = np.array(features[0]['embedding'], dtype=np.float32)
        index = faiss.IndexFlatL2(128)
        index.add(np.array([face_vector]))
        faiss.write_index(index, "faiss_index.index")

    print("بردار ویژگی تصویر ذخیره شد")
    return True

@app.route('/save_to_db20', methods=['POST'])
def save_to_db20():
    data = request.json
    image_data = data.get("image")
    
    if not image_data:
        return jsonify({"status": "error", "message": "No image data received"}), 400

    # تبدیل تصویر Base64 به فایل
    filename = f"{UPLOAD_FOLDER}/{int(time.time())}.jpg"
    image_bytes = base64.b64decode(image_data)
    with open(filename, "wb") as f:
        f.write(image_bytes)

    # ذخیره مسیر در دیتابیس
    timestamp = datetime.now()
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO images (image_path, timestamp) VALUES (%s, %s)", (filename, timestamp))
        conn.commit()
        cur.close()
        conn.close()

        # Save Feature maps in faiss BDs.
        save_featuremap_to_db(filename)
        print("Save OK... ")
        return jsonify({"status": "success", "message": "Image saved successfully", "image_url": filename})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/save_to_db2', methods=['POST'])
def save_to_db2():
    data = request.json
    image_data = data.get("image")
    
    if not image_data:
        return jsonify({"status": "error", "message": "No image data received"}), 400

    try:
        # مرحله 1: decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # مرحله 2: تبدیل به OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        corrected_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # # مرحله 3: اصلاح چرخش
        # corrected_image = correct_rotation_precise(image)
        # دیگه نیاز نیست اصلاح روتیت. چون قبلش درستش کردیم. بهد میایم سیو میکنیم.

        # مرحله 4: ذخیره تصویر اصلاح‌شده
        filename = f"{UPLOAD_FOLDER}/{int(time.time())}.jpg"
        cv2.imwrite(filename, corrected_image)

        # مرحله 5: ذخیره مسیر تصویر در دیتابیس
        timestamp = datetime.now()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO images (image_path, timestamp) VALUES (%s, %s)", (filename, timestamp))
        conn.commit()
        cur.close()
        conn.close()

        # مرحله 6: ذخیره feature map
        save_featuremap_to_db(filename)
        print("Save OK... ")
        return jsonify({"status": "success", "message": "Image saved successfully", "image_url": filename})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

 

mp_face_mesh = mp.solutions.face_mesh

# def correct_face_rotation_mesh(image):
#     with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as face_mesh:
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_image)

#         if not results.multi_face_landmarks:
#             print("❌ هیچ صورتی پیدا نشد.")
#             return image

#         landmarks = results.multi_face_landmarks[0].landmark
#         h, w = image.shape[:2]

#         # نقاط کلیدی چشم چپ و راست (با دقت بالا)
#         left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])  # نقطه گوشه‌ی بیرونی چشم چپ
#         right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])  # نقطه گوشه‌ی بیرونی چشم راست

#         # زاویه بین چشم‌ها
#         dx = right_eye[0] - left_eye[0]
#         dy = right_eye[1] - left_eye[1]
#         angle = np.degrees(np.arctan2(dy, dx))

#         # اعمال چرخش معکوس به کل تصویر
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

#         print(f"[✅] تصویر با زاویه {angle:.2f} درجه چرخانده شد.")
#         return rotated_image
 

 

# def correct_rotation_full(image):
#     with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as face_mesh:
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb)

#         if not results.multi_face_landmarks:
#             print("❌ هیچ چهره‌ای یافت نشد.")
#             return image

#         face = results.multi_face_landmarks[0].landmark
#         h, w = image.shape[:2]

#         # نقاط کلیدی (بر اساس Mediapipe)
#         left_eye = np.array([face[33].x * w, face[33].y * h])
#         right_eye = np.array([face[263].x * w, face[263].y * h])
#         nose_tip = np.array([face[1].x * w, face[1].y * h])
#         chin = np.array([face[152].x * w, face[152].y * h])

#         # زاویه بین دو چشم
#         dx = right_eye[0] - left_eye[0]
#         dy = right_eye[1] - left_eye[1]
#         raw_angle = np.degrees(np.arctan2(dy, dx))

#         # جهت عمودی چهره (چانه پایین یا بالا)
#         face_upward = chin[1] > nose_tip[1]

#         # تخمین دقیق‌تر زاویه
#         if face_upward:
#             angle = raw_angle
#         else:
#             angle = raw_angle + 180

#         # نرمال‌سازی زاویه به یکی از حالت‌های 0، 90، 180، 270
#         def snap_angle(a):
#             candidates = [0, 90, 180, 270]
#             return min(candidates, key=lambda x: abs(x - a % 360))

#         final_angle = snap_angle(angle)

#         # چرخش
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
#         rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

#         print(f"[✅] چرخش تصویر اصلاح شد. زاویه: {final_angle} درجه")
#         return rotated

# def correct_face_rotation_precise(image):
#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
#         results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if not results.detections:
#             print("❌ هیچ صورتی پیدا نشد.")
#             return image

#         detection = results.detections[0]
#         try:
#             keypoints = detection.location_data.relative_keypoints
#             left_eye = keypoints[0]
#             right_eye = keypoints[1]

#             h, w = image.shape[:2]
#             left_eye = np.array([left_eye.x * w, left_eye.y * h])
#             right_eye = np.array([right_eye.x * w, right_eye.y * h])

#             dx = right_eye[0] - left_eye[0]
#             dy = right_eye[1] - left_eye[1]
#             angle = np.degrees(np.arctan2(dy, dx))

#             # چرخش تصویر به سمت صاف کردن چشم‌ها
#             center = (w // 2, h // 2)
#             M = cv2.getRotationMatrix2D(center, angle, 1.0)
#             rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

#             print(f"[✅] تصویر با زاویه دقیق {angle:.2f}° صاف شد.")
#             return rotated

#         except Exception as e:
#             print("خطا در محاسبه زاویه:", e)
#             return image

 

def correct_rotation_fully(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("❌ صورت پیدا نشد.")
            return image

        face = results.multi_face_landmarks[0]

        # گرفتن نقاط چشم چپ و راست و مرکز بینی
        h, w = image.shape[:2]
        left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])  # left eye outer
        right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])  # right eye outer
        nose_tip = np.array([face.landmark[1].x * w, face.landmark[1].y * h])  # tip of nose

        # خط بین دو چشم
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # اصلاح چرخش کامل
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        print(f"[✅] زاویه دقیق چرخش: {angle:.2f} درجه → تصویر اصلاح شد.")
        return rotated
 

def correct_rotation_auto(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("❌ صورت یافت نشد.")
            return image

        face = results.multi_face_landmarks[0]

        # نقاط کلیدی
        left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
        right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
        nose_tip = np.array([face.landmark[1].x * w, face.landmark[1].y * h])

        # بردار بین چشم‌ها
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # تبدیل به بازه‌ی ۰ تا 360
        if angle_deg < 0:
            angle_deg += 360

        # حالت صورت وارونه: اگر بینی زیر چشم‌ها باشد احتمال زیاد تصویر وارونه است
        if nose_tip[1] > max(left_eye[1], right_eye[1]):
            angle_deg = (angle_deg + 180) % 360

        # چرخش معکوس برای اصلاح تصویر
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        print(f"[✅] تصویر با زاویه {angle_deg:.2f} درجه اصلاح شد.")
        return corrected_image

def correct_rotation_precise(image):
    message=""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("❌ صورت یافت نشد.")
            message="❌ صورت یافت نشد."
            return image, message 

        face = results.multi_face_landmarks[0]

        left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
        right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
        nose_tip = np.array([face.landmark[1].x * w, face.landmark[1].y * h])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))

        print("noise  : ", nose_tip[1])
        center=(left_eye[1]+right_eye[1])/2
        print("center   : ", center)
        print("********************")
        
        if nose_tip[1] < center:
            print("inverse")
            angle_deg = (angle_deg + 180) % 360

          # اگر زاویه خیلی کم بود، تصویر تغییر نکند
        if abs(angle_deg) < 10:
            print(f"✅  ok {angle_deg:.2f}).")
            message="img is correct"
            return image,message 
        
        if angle_deg < 0:
            angle_deg += 360

        # چک کردن وارونگی با بینی
        # if nose_tip[1]< max(left_eye[1], right_eye[1]):
       

        # رند کردن زاویه به نزدیک‌ترین مضرب ۹۰ درجه برای اصلاح دقیق‌تر
        possible_angles = np.array([0, 90, 180, 270])
        diff_angles = np.abs(possible_angles - angle_deg)
        idx = diff_angles.argmin()
        corrected_angle = possible_angles[idx]

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        print(f"[✅] angle {angle_deg:.2f}  ")  
        print(f"[✅]  correct_angel {corrected_angle} ")
        message="img is rotated!"
        return corrected_image,message

def correct_rotation_preciseCH(image):
    message = ""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        h, w = image.shape[:2]
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb=image
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("❌ صورت یافت نشد.")
            message = "❌ صورت یافت نشد."
            return image, message

        face = results.multi_face_landmarks[0]
        left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
        right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
        nose_tip = np.array([face.landmark[1].x * w, face.landmark[1].y * h])
        chin_y = face.landmark[152].y * h  # Chin
        eye_center_y = (left_eye[1] + right_eye[1]) / 2



        # زاویه چشم‌ها
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))

        # تشخیص وارونگی chi
        print("nose   :", chin_y)
        print("eye_cen:", eye_center_y)
        print("********************") 
        is_upside_down = chin_y < eye_center_y
        if is_upside_down:
            print("🔁 وارونگی تشخیص داده شد.")
            angle_deg += 180

        if angle_deg < 0:
            angle_deg += 360

        if abs(angle_deg % 360) < 10 or abs(angle_deg % 360 - 360) < 10:
            print(f"✅ تصویر تقریباً درست است ({angle_deg:.2f})")
            return image, "img is correct"

        # رند کردن زاویه به نزدیک‌ترین مضرب ۹۰
        possible_angles = np.array([0, 90, 180, 270])
        diff_angles = np.abs(possible_angles - angle_deg)
        corrected_angle = possible_angles[diff_angles.argmin()]

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        print(f"[✅] زاویه اولیه: {angle_deg:.2f}")
        print(f"[✅] چرخش اعمال‌شده: {corrected_angle}")
        message = "img is rotated!"
        return corrected_image, message

def correct_rotation_preciseMe(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("❌ صورت یافت نشد.")
            return image

        face = results.multi_face_landmarks[0]

        left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
        right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
        nose_tip = np.array([face.landmark[1].x * w, face.landmark[1].y * h])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))
          # اگر زاویه خیلی کم بود، تصویر تغییر نکند
        if abs(dy) < 5 and nose_tip[1]> right_eye[1]:
            print(f"✅ تصویر در وضعیت درست است (زاویه {angle_deg:.2f}).")
            return image
        
        if angle_deg < 0:
            angle_deg += 360

        center = (w // 2, h // 2)
        # چک کردن وارونگی با بینی
        if nose_tip[1] < max(left_eye[1], right_eye[1]):
            # angle_deg = (angle_deg + 180) % 360
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            # corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        elif abs(dx) < 5 and nose_tip[0]> right_eye[0] :
            M = cv2.getRotationMatrix2D(center, 30, 1.0)
            # corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        else:
            M = cv2.getRotationMatrix2D(center, 270, 1.0)
            # corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        # # رند کردن زاویه به نزدیک‌ترین مضرب ۹۰ درجه برای اصلاح دقیق‌تر
        # possible_angles = np.array([0, 90, 180, 270])
        # diff_angles = np.abs(possible_angles - angle_deg)
        # idx = diff_angles.argmin()
        # corrected_angle = possible_angles[idx]

        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # print(f"[✅] زاویه تشخیص داده شده: {angle_deg:.2f} درجه")
        # print(f"[✅] زاویه روتیت شده به: {corrected_angle} درجه (رند شده به نزدیک‌ترین عمود)")
        return corrected_image
 
def correct_rotation_flexible(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("❌ صورت یافت نشد.")
            return image

        face = results.multi_face_landmarks[0]

        left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
        right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
        nose_tip = np.array([face.landmark[1].x * w, face.landmark[1].y * h])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))
         # اگر زاویه خیلی کم بود، تصویر تغییر نکند
        if abs(angle_deg < 2 ) :
            print(f"✅ تصویر در وضعیت درست است (زاویه {angle_deg:.2f}).")
            return image

        if angle_deg < 0:
            angle_deg += 360

        # بررسی وارونگی تصویر بر اساس بینی
        if nose_tip[1] > max(left_eye[1], right_eye[1]):
            angle_deg = (angle_deg + 180) % 360

        # چک کردن فاصله زاویه به مضرب ۹۰
        possible_angles = np.array([0, 90, 180, 270])
        diff_angles = np.abs(possible_angles - angle_deg)
        min_diff = diff_angles.min()

        # اگر اختلاف کمتر از 5 درجه بود، رند کن به نزدیک‌ترین 90 درجه
        if min_diff < 5:
            corrected_angle = possible_angles[diff_angles.argmin()]
            print(f"[INFO] زاویه {angle_deg:.2f} به {corrected_angle} رند شد.")
        else:
            corrected_angle = angle_deg
            print(f"[INFO] زاویه دقیق استفاده شد: {corrected_angle:.2f}")

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return corrected_image

def crop_and_correct_face(image):
    h, w = image.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.6) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            print("❌ No face detected")
            return image, "no face detected"

        face = results.multi_face_landmarks[0]

        # ⬛ پیدا کردن محدوده صورت
        x_vals = [lm.x for lm in face.landmark]
        y_vals = [lm.y for lm in face.landmark]
        min_x = int(max(min(x_vals) * w - 20, 0))
        max_x = int(min(max(x_vals) * w + 20, w))
        min_y = int(max(min(y_vals) * h - 20, 0))
        max_y = int(min(max(y_vals) * h + 20, h))

        face_crop = image[min_y:max_y, min_x:max_x]
        hf, wf = face_crop.shape[:2]

        # ⭕ گرفتن مختصات دقیق چشم‌ها
        left_eye = np.array([face.landmark[33].x * w - min_x, face.landmark[33].y * h - min_y])
        right_eye = np.array([face.landmark[263].x * w - min_x, face.landmark[263].y * h - min_y])
        mouth_center = np.array([
            (face.landmark[13].x + face.landmark[14].x) / 2 * w - min_x,
            (face.landmark[13].y + face.landmark[14].y) / 2 * h - min_y
        ])
        eye_center_y = (left_eye[1] + right_eye[1]) / 2

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))
        print("🔍 eye angle", angle_deg)
        print("eye_center_y",eye_center_y)
        print("mouth_center",mouth_center)
        print("hf",hf)

        if -10 < angle_deg < 10:
            if eye_center_y < hf/2:
                print("✅ okkkkkkkkkkk")
                return image, "image is correct"

        # چک کردن وارونگی با دهان
        if eye_center_y > hf/2:
            angle_deg += 180
            print("🔄 180000000000000000000000")

        # اصلاح زاویه
        center = (wf // 2, hf // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated_face = cv2.warpAffine(face_crop, M, (wf, hf), flags=cv2.INTER_CUBIC)

        return rotated_face, "face cropped and rotated"


def correct_rotation_preciseOK(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        mes=""
        if not results.multi_face_landmarks:
            print("❌ no face detected")
            mes="no face detected"
            return image,mes

        face = results.multi_face_landmarks[0]

        left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
        right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
        nose_tip = np.array([face.landmark[1].x * w, face.landmark[1].y * h])
        mouth_center = np.array([
            (face.landmark[13].x + face.landmark[14].x) / 2 * w,
            (face.landmark[13].y + face.landmark[14].y) / 2 * h
        ]) 
        eye_center_y = (left_eye[1] + right_eye[1]) / 2


        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))
        print("angle_deg",angle_deg)
        print("mouth_center",mouth_center)
        print("nose_tip",nose_tip[1])
        print("left_eye",left_eye[1])
        print("right_eye",right_eye[1]) 
        print("h",h)
        print("ww",w)
        # if -10<angle_deg<10 and mouth_center[1] > eye_center_y:
        #     mes="img is correct. No rotated"
        #     return image,mes 

         
        # اگر زاویه تقریباً صفر و بینی در نیمه‌ی بالایی تصویر باشد => تصویر درست است
        if -10 < angle_deg < 10:
            # nose_to_eye_dist = mouth_center[1] - eye_center_y
            # if eye_center_y < h * 0.1:
            # if eye_center_y < h / 2:
                
            if np.abs(mouth_center[1] - eye_center_y)>h * 0.1:
                print("✅ تصویر درست است.")
                return image, "image is correct"
            
        if angle_deg < 0:
            angle_deg += 360

        # چک کردن وارونگی با بینی
        # if nose_tip[1]> max(left_eye[1], right_eye[1]):
        #     angle_deg = (angle_deg + 180) % 360

        # if mouth_center[1] < eye_center_y:
        if np.abs(mouth_center[1] - eye_center_y)<h * 0.1:
            angle_deg = (angle_deg + 180) % 360
            print("180 rotated")

        # رند کردن زاویه به نزدیک‌ترین مضرب ۹۰ درجه برای اصلاح دقیق‌تر
        possible_angles = np.array([0, 90, 180, 270])
        diff_angles = np.abs(possible_angles - angle_deg)
        idx = diff_angles.argmin()
        corrected_angle = possible_angles[idx]

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        print(f"[✅] angle {angle_deg:.2f} ")
        print(f"[✅] rotated angle  {corrected_angle}  ")
        mes="img rotated"
        return corrected_image,mes

 
import face_alignment 
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)

def correct_face_rotation_facealignment(image):
    landmarks_list = fa.get_landmarks(image)
    if landmarks_list is None or len(landmarks_list) == 0:
        print("❌ No face detected.")
        return image, "no face detected"

    landmarks = landmarks_list[0]

    # نقاط چشم‌ها و دهان
    left_eye = np.mean(landmarks[36:42], axis=0)   # چشم چپ
    right_eye = np.mean(landmarks[42:48], axis=0)  # چشم راست
    mouth_center = np.mean(landmarks[48:68], axis=0)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    print(f"🔍angle {angle:.2f}  ")

    # اگر دهان بالاتر از چشم‌ها بود، وارونگی داریم → 180 درجه اضافه کن
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    
    print("angle_deg",angle)
    print("mouth_center",mouth_center[1]) 
    print("left_eye",left_eye[1])
    print("right_eye",right_eye[1]) 
    # if -10<angle<10 and mouth_center[1] >= eye_center_y:
    if -10<angle<10 and np.abs(mouth_center[1] - eye_center_y)>30:
        print("img is correct. No rotated")
        return image,"img is correct. No rotated"
    
    if np.abs(mouth_center[1] - eye_center_y)<20:
        print("🔄 180 rotated" )
        angle += 180

    # چرخش تصویر
    (h, w) = image.shape[:2]
    possible_angles = np.array([0, 90, 180, 270])
    diff_angles = np.abs(possible_angles - angle)
    idx = diff_angles.argmin()
    corrected_angle = possible_angles[idx]

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
    corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
 
    mes="img rotated"
    return corrected_image,mes

# def correct_face_rotation_final(image):
#     h, w = image.shape[:2]
#     mp_face_mesh = mp.solutions.face_mesh

#     with mp_face_mesh.FaceMesh(static_image_mode=True,
#                                 max_num_faces=1,
#                                 refine_landmarks=True,
#                                 min_detection_confidence=0.5) as face_mesh:
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_image)

#         if not results.multi_face_landmarks:
#             print("❌ هیچ صورتی پیدا نشد.")
#             return image

#         landmarks = results.multi_face_landmarks[0].landmark

#         left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
#         right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
#         nose = np.array([landmarks[1].x * w, landmarks[1].y * h])

#         # زاویه چرخش
#         dx = right_eye[0] - left_eye[0]
#         dy = right_eye[1] - left_eye[1]
#         angle = np.degrees(np.arctan2(dy, dx))

#         # بررسی وارونگی
#         eye_center_y = (left_eye[1] + right_eye[1]) / 2
#         is_upside_down = nose[1] < eye_center_y

#         if is_upside_down:
#             angle += 180

#         # نرمال‌سازی زاویه به محدوده [-180, 180]
#         while angle > 180:
#             angle -= 360
#         while angle < -180:
#             angle += 360

#         # اگر زاویه خیلی کم بود، تصویر تغییر نکند
#         if abs(angle) < 2:
#             print(f"✅ تصویر در وضعیت درست است (زاویه {angle:.2f}).")
#             return image

#         # چرخش اصلاحی
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

#         print(f"🔁 چرخش اصلاحی انجام شد. زاویه: {angle:.2f} درجه")
#         return rotated

 
def correct_face_rotation_final(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print("❌ هیچ صورتی پیدا نشد.")
            return image

        detection = results.detections[0]  # فقط اولین چهره
        try:
            left_eye = detection.location_data.relative_keypoints[0]
            right_eye = detection.location_data.relative_keypoints[1]

            h, w = image.shape[:2]
            left = np.array([int(left_eye.x * w), int(left_eye.y * h)])
            right = np.array([int(right_eye.x * w), int(right_eye.y * h)])

            dx = right[0] - left[0]
            dy = right[1] - left[1]

            angle = np.degrees(np.arctan2(dy, dx))

            print(f"[INFO] زاویه محاسبه‌شده بین چشم‌ها: {angle:.2f} درجه")

            # اگر زاویه خیلی نزدیک صفر بود (±10)، نیازی به چرخش نیست
            if -10 < angle < 10:
                return image

            # اگر تصویر وارونه بود (چشم‌ها پایین‌تر از خط افقی بودن)
            if abs(angle) > 170:
                print("↩ اصلاح 180 درجه")
                M = cv2.getRotationMatrix2D((w//2, h//2), 180, 1.0)
                return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

            # سایر حالت‌ها (مثلاً 45، 90، 120)
            print(f"↩ اصلاح زاویه: {-angle:.2f}")
            M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
            return rotated

        except Exception as e:
            print("❌ خطا در پردازش چرخش:", e)
            return image
def correct_face_rotation_precise(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print("❌ هیچ صورتی پیدا نشد.")
            return image

        detection = results.detections[0]
        try:
            h, w = image.shape[:2]
            left = detection.location_data.relative_keypoints[0]
            right = detection.location_data.relative_keypoints[1]

            left = np.array([int(left.x * w), int(left.y * h)])
            right = np.array([int(right.x * w), int(right.y * h)])

            dx = right[0] - left[0]
            dy = right[1] - left[1]
            angle = np.degrees(np.arctan2(dy, dx))

            # مکان میانگین چشم‌ها در تصویر
            eye_center_y = (left[1] + right[1]) / 2
            eye_center_x = (left[0] + right[0]) / 2

            print(f"🔍 زاویه: {angle:.2f} | موقعیت Y چشم‌ها: {eye_center_y:.1f} از {h}")

            # تصمیم‌گیری بر اساس موقعیت و زاویه
            if -10 < angle < 10:
                if eye_center_y > h * 0.6:
                    # چشم‌ها پایینن → rotate 180
                    print("🔄 اصلاح 180 درجه")
                    M = cv2.getRotationMatrix2D((w//2, h//2), 180, 1.0)
                    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
                else:
                    print("✅ تصویر درست است.")
                    return image

            elif 80 < angle < 100:
                print("🔄 اصلاح 90 درجه به عقب")
                M = cv2.getRotationMatrix2D((w//2, h//2), -90, 1.0)
                return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

            elif -100 < angle < -80:
                print("🔄 اصلاح 90 درجه به جلو (270)")
                M = cv2.getRotationMatrix2D((w//2, h//2), 90, 1.0)
                return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

            else:
                print(f"🌀 اصلاح دقیق زاویه: {-angle:.2f}")
                M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        except Exception as e:
            print("❌ خطا:", e)
            return image

def correct_face_rotation_precisee(image):
    import cv2
    import numpy as np
    import mediapipe as mp

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print("❌ هیچ صورتی پیدا نشد.")
            return image

        detection = results.detections[0]
        try:
            h, w = image.shape[:2]
            left = detection.location_data.relative_keypoints[0]
            right = detection.location_data.relative_keypoints[1]

            left = np.array([int(left.x * w), int(left.y * h)])
            right = np.array([int(right.x * w), int(right.y * h)])

            dx = right[0] - left[0]
            dy = right[1] - left[1]
            angle = np.degrees(np.arctan2(dy, dx))

            eye_center_y = (left[1] + right[1]) / 2
            eye_center_x = (left[0] + right[0]) / 2

            print(f"🔍 زاویه: {angle:.2f} | Y: {eye_center_y:.1f}/{h} | X: {eye_center_x:.1f}/{w}")

            # حالت 180: زاویه تقریباً صفر اما چشم‌ها پایین هستن
            if -10 < angle < 10:
                if eye_center_y > h * 0.6:
                    print("🔄 اصلاح 180 درجه")
                    angle_to_rotate = 180
                else:
                    print("✅ تصویر درست است.")
                    return image

            # حالت 90
            elif 80 < angle < 100 and eye_center_x < w * 0.4:
                print("🔄 اصلاح 90 درجه")
                angle_to_rotate = -90

            # حالت 270
            elif -100 < angle < -80 and eye_center_x > w * 0.6:
                print("🔄 اصلاح 270 درجه")
                angle_to_rotate = 90

            else:
                print(f"🌀 اصلاح دقیق: {-angle:.2f}")
                angle_to_rotate = -angle

            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_to_rotate, 1.0)
            # rotated = image.rotate(-angle, center=left_eye, resample=Image.BICUBIC, expand=True)
            return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
 

 

        except Exception as e:
            print("❌ خطا:", e)
            return image

def correct_rotation_preciseZH(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("❌ صورت یافت نشد.")
            return image

        face = results.multi_face_landmarks[0]

        left_eye = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
        right_eye = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
        nose_tip = np.array([face.landmark[1].x * w, face.landmark[1].y * h])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))
        print("angle_deg",angle_deg)
          # اگر زاویه خیلی کم بود، تصویر تغییر نکند
        if abs(angle_deg) < 2:
            print(f"✅ ok {angle_deg:.2f}).")
            return image
                 
        # چرخش برای اصلاح کجی سر
        angle_to_rotate = -angle_deg

        # اگر صورت وارونه باشد (بینی پایین‌تر از چشم‌ها)
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        if nose_tip[1] < eye_center_y:
            angle_to_rotate += 180
        

        

        # این‌بار از زاویه دقیق استفاده می‌کنیم
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_to_rotate, 1.0)  # منفی چون چرخش ساعت‌گرد در OpenCV مثبت است
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        print(f"[✅] angle_to_rotate {angle_to_rotate:.2f}  ")
        return corrected_image

# import cv2
# import dlib
# import numpy as np
# from imutils import face_utils

# # مدل پیش‌بینی‌کننده نقاط چهره
# predictor_path = "shape_predictor_68_face_landmarks.dat"  
# detectordlib = dlib.get_frontal_face_detector()
# predictordlib = dlib.shape_predictor(predictor_path)

# def get_rotation_angle(eye_left, eye_right):
#     dx = eye_right[0] - eye_left[0]
#     dy = eye_right[1] - eye_left[1]
#     angle = np.degrees(np.arctan2(dy, dx))
#     return angle

# def correct_face_rotationdible(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = detectordlib(gray)
#     mes=""

#     if len(faces) == 0:
#         print("no face")
#         mes="no face"
#         return image,mes

#     for face in faces:
#         shape = predictordlib(gray, face)
#         shape = face_utils.shape_to_np(shape)

#         left_eye = shape[36]
#         right_eye = shape[45]

#         angle = get_rotation_angle(left_eye, right_eye)

#         # چرخش برعکس زاویه
#         (h, w) = image.shape[:2]
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

#         print(f"[INFO] angle roteted  {angle:.2f}  o is corected.")
#         mes="img roteted"
#         return rotated,mes

#     return image,mes

 



@app.route('/rotateimgg', methods=['POST'])
def rotateimgg():
    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No image file received"}), 400

    file = request.files['frame']
    
    try:
        # خواندن فایل و تبدیل به آرایه NumPy
        file_bytes = file.read()  # ← فقط یک بار read می‌کنیم
        np_img = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"status": "error", "message": "Could not decode image"}), 400

        # اصلاح چرخش چهره
        # corrected_image = correct_rotation_fully(image)
        # corrected_image = correct_rotation_auto(image)
        # corrected_image1,m = correct_rotation_preciseCH(image)#good
        corrected_image1,m = correct_rotation_preciseOK(image) #good
        # corrected_image1,m = crop_and_correct_face(image)
        # corrected_image1,m =correct_face_rotation_facealignment(image)#other model
        # corrected_image1,m =correct_face_rotationdible(image)# BAD
        
        
        # corrected_image2= correct_rotation_flexible(image)
        # corrected_image2 = correct_face_rotation_final(image)
        # corrected_image = correct_face_rotation_final(image)
        # corrected_image2 = correct_rotation_preciseMe(image)
        # corrected_image = correct_face_rotation_precisee(image)
        # corrected_image3= correct_rotation_preciseZH(image)
        
        
        
        
        
        

        # ذخیره تصویر اصلاح‌شده
        filename = f"{UPLOAD_FOLDER}/rotated.jpg"
        cv2.imwrite(filename, corrected_image1)
        # cv2.imwrite(f"{UPLOAD_FOLDER}/2.jpg", corrected_image2)

# # ذخیره تصویر اصلاح‌شده
#         filename1 = f"{UPLOAD_FOLDER}/rotated2.jpg"
#         cv2.imwrite(filename1, corrected_image2)
#         # cv2.imwrite(f"{UPLOAD_FOLDER}/2.jpg", corrected_image2)
#         # ذخیره تصویر اصلاح‌شده
#         filename2 = f"{UPLOAD_FOLDER}/rotated3.jpg"
#         cv2.imwrite(filename2, corrected_image3)
#         # cv2.imwrite(f"{UPLOAD_FOLDER}/2.jpg", corrected_image2)
        return jsonify({
            "status": "success",
            "message": m,
            "file_path": url_for('static', filename='images/rotated.jpg', _external=True),
            # "file_path": filename
            # "file_path2": filename1,
            # "file_path3": filename2
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# UPLOAD_DIR = "./static/videos/"
# TEMP_DIR = os.path.join(UPLOAD_DIR, "temp")
# os.makedirs(TEMP_DIR, exist_ok=True)
# مسیر صحیح با os.path.join

# 
# @app.route("/process_myvideo", methods=["POST"])
# def upload_myvideo():
#     db = SessionLocal()  
#     if 'video' not in request.files:
#         return jsonify({"error": "No video file provided"}), 400

#     video = request.files['video']

#     if video.mimetype not in ALLOWED_MIME:
#         return jsonify({"error": "Only video files are allowed"}), 400

#     temp_filename = f"{uuid.uuid4()}_{secure_filename(video.filename)}"
#     temp_path = os.path.join(TEMP_DIR, temp_filename)
#     video.save(temp_path)

#     final_filename = f"{video.filename}"
#     final_path = os.path.join(UPLOAD_DIR, final_filename)

#     try:
#         clip = VideoFileClip(temp_path)
#         clip.write_videofile(final_path, codec="libx264", audio_codec="aac")
#         clip.close()
#     except Exception as e:
#         return jsonify({"error": f"Video conversion failed: {e}"}), 500
#     finally:
#         os.remove(temp_path)

#     # ذخیره در دیتابیس
#     new_video = Video(filename=final_filename, path=final_path)
#     db.add(new_video)
#     db.commit()

#     return jsonify({"message": "Video uploaded and converted", "video_id": new_video.id})


# def process_video_file(video_file):
#     """تابع مستقل برای پردازش ویدیو"""
#     db = SessionLocal()
    
#     if video_file.mimetype not in ALLOWED_MIME:
#         raise ValueError("Only video files are allowed")

#     temp_filename = f"{uuid.uuid4()}_{secure_filename(video_file.filename)}"
#     temp_path = os.path.join(TEMP_DIR, temp_filename)
#     video_file.save(temp_path)

#     final_filename = f"{video_file.filename}"
#     final_path = os.path.join(UPLOAD_DIR, final_filename)

#     try:
#         clip = VideoFileClip(temp_path)
#         clip.write_videofile(final_path, codec="libx264", audio_codec="aac")
#         clip.close()
#     except Exception as e:
#         raise Exception(f"Video conversion failed: {e}")
#     finally:
#         os.remove(temp_path)

#     new_video = Video(filename=final_filename, path=final_path)
#     db.add(new_video)
#     db.commit()
#     db.close()
    
#     return new_video

# def process_video_file(video_file):
    # """تابع مستقل برای پردازش ویدیو"""
    # if video_file.mimetype not in ALLOWED_MIME:
    #     raise ValueError("Only video files are allowed")

    # # ایجاد مسیرها و نام فایل
    # os.makedirs(TEMP_DIR, exist_ok=True)
    # os.makedirs(UPLOAD_DIR, exist_ok=True)

    # temp_filename = f"{uuid.uuid4()}_{secure_filename(video_file.filename)}"
    # temp_path = os.path.join(TEMP_DIR, temp_filename)
    # video_file.save(temp_path)

    # final_filename = f"{uuid.uuid4()}_{secure_filename(video_file.filename)}"
    # final_path = os.path.join(UPLOAD_DIR, final_filename)

    # clip = None
    # try:
    #     clip = VideoFileClip(temp_path)
    #     if not clip.size or clip.duration == 0:
    #         raise ValueError("Invalid video file: missing metadata")
    #     clip.write_videofile(final_path, codec="libx264", audio_codec="aac",
    #     ffmpeg_params=[
    #         '-analyzeduration', '100M',  # افزایش مقدار تحلیل
    #         '-probesize', '100M'        # افزایش مقدار پروب
    #     ])
    # except Exception as e:
    #     raise Exception(f"Video conversion failed: {e}")
    # finally:
    #     if clip is not None:
    #         clip.close()
    #     if os.path.exists(temp_path):
    #         os.remove(temp_path)

    # # ذخیره در پایگاه داده
    # db = SessionLocal()
    # try:
    #     new_video = Video(filename=final_filename, path=final_path)
    #     db.add(new_video)
    #     db.commit()
    #     db.refresh(new_video)
    #     return new_video
    # except Exception as e:
    #     db.rollback()
    #     raise e
    # finally:
    #     db.close()


def verify_video_file(filepath):
    try:
        import cv2
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        cap.release()
        return True
    except Exception as e:
        raise ValueError(f"Video verification failed: {str(e)}")
    

# def process_video_file(video_file):
#     try:
#         # 1. ذخیره موقت فایل
#         # temp_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4()}.mp4")
#         # video_file.save(temp_path)
#         # تولید مسیر فایل موقت
#         temp_filename='ww.mp4'
#         # temp_filename = f'temp_{uuid.uuid4()}.mp4'
#         temp_path = os.path.join(TEMP_DIR, temp_filename)
        
#         # temp_path='C:\Users\Zhala\Desktop\myid\static\videos\temp'  
#         video_file.save(temp_path)    
#         # 2. بررسی سلامت فایل
#         verify_video_file(temp_path)
        
#         # 3. تنظیم مسیر FFmpeg
#         from moviepy.config import change_settings
#         ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # مسیر جدید FFmpeg
#         change_settings({"FFMPEG_BINARY": ffmpeg_path})
        
#         # 4. پردازش با MoviePy
#         final_path = os.path.join(TEMP_DIR, f"processed_{uuid.uuid4()}.mp4")
#         clip = VideoFileClip(temp_path)
#         clip.write_videofile(
#             final_path,
#             codec="libx264",
#             audio_codec="aac",
#             ffmpeg_params=[
#                 '-analyzeduration', '100M',
#                 '-probesize', '100M'
#             ]
#         )
#         clip.close()
        
#         # 5. ذخیره در دیتابیس
#         db = SessionLocal()
#         new_video = Video(filename=os.path.basename(final_path), path=final_path)
#         db.add(new_video)
#         db.commit()
#         return new_video
        
#     except Exception as e:
#         # تمیزکاری فایل‌های موقت
#         if 'temp_path' in locals() and os.path.exists(temp_path):
#             os.remove(temp_path)
#         if 'final_path' in locals() and os.path.exists(final_path):
#             os.remove(final_path)
#         raise Exception(f"Processing failed: {str(e)}")
#     finally:
#         if 'clip' in locals():
#             clip.close()
#         if 'db' in locals():
#             db.close()


import subprocess

def repair_video(input_path, output_path):
    """ترمیم فایل ویدیویی معیوب"""
    try:
        cmd = [
            'ffmpeg',
            '-y',               # overwrite output
            '-i', input_path,   # input file
            '-c:v', 'copy',     # copy video stream
            '-c:a', 'copy',     # copy audio stream
            '-movflags', '+faststart',  # enable streaming
            '-f', 'mp4',        # force mp4 format
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg Error: {result.stderr}")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Repair failed: {str(e)}")
def is_video_valid(filepath):
    """بررسی وجود moov atom در فایل"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_format', filepath],
            capture_output=True,
            text=True
        )
        return "moov atom not found" not in result.stderr
    except Exception:
        return False
# import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/path/to/ffmpeg"     
# def process_video_file(video_file):
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
#             video_path = temp.name
#             video_file.save(video_path)
        
#         # # 1. ذخیره موقت فایل
#         # temp_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4()}.mp4")
#         # video_file.save(temp_path)
        
#         # # 2. بررسی سلامت فایل
#         # if not is_video_valid(temp_path):
#         #     repaired_path = os.path.join(TEMP_DIR, f"repaired_{uuid.uuid4()}.mp4")
#         #     repair_video(temp_path, repaired_path)
#         #     os.remove(temp_path)
#         #     temp_path = repaired_path
            
#         #     # بررسی مجدد بعد از ترمیم
#         #     if not is_video_valid(temp_path):
#         #         raise ValueError("Failed to repair video file")
        
#         # 3. پردازش ویدیو
#         clip = VideoFileClip(video_path)
#         output_path = os.path.join(TEMP_DIR, f"processed_{uuid.uuid4()}.mp4")
#         clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
#         clip.close()
        
#         return output_path
        
#     except Exception as e:
#         # مدیریت خطاها
#         raise Exception(f"Video processing failed: {str(e)}")
#     finally:
#         # تمیزکاری فایل‌های موقت
#         if 'temp_path' in locals() and os.path.exists(video_path):
#             os.remove(video_path)

# def process_video_file(video_file):
#     """تابع مستقل برای پردازش ویدیو"""
#     db = SessionLocal()
    
#     if video_file.mimetype not in ALLOWED_MIME:
#         raise ValueError("Only video files are allowed")

#     # temp_filename = f"{uuid.uuid4()}_{secure_filename(video_file.filename)}"
#     # temp_path = TEMP_DIR / temp_filename


#     # temp_path_str = temp_path.as_posix()
    
    
    
#     temp_filename = f"{uuid.uuid4()}_{secure_filename(video_file.filename)}"
#     # temp_filename='1.mp4'
#     temp_path = os.path.join(TEMP_DIR, temp_filename)
#     print("tempppath",temp_path,"shooom")
#     video_file.save(temp_path)
    

#     final_filename = f"{video_file.filename}"
#     final_path = os.path.join(TEMP_DIR, final_filename)
#     # final_filename=temp_filename
#     # final_path=temp_filename
#     try:
#         clip = VideoFileClip(temp_path)
#         clip.write_videofile(final_path, codec="libx264", audio_codec="aac")
#         clip.close()
#     except Exception as e:
#         raise Exception(f"Video conversion failed: {e}")
#     finally:
#         os.remove(temp_path)

#     new_video = Video(filename=final_filename, path=final_path,uploaded_at=datetime.utcnow())
#     db.add(new_video)
#     db.commit()
    
#     db.close()
    
#     return jsonify({"message": "ok"})
 

# def process_video_file(video_file):
#     """تابع اصلاح شده برای پردازش ویدیو"""
#     db = SessionLocal()
    
#     try:
#         # بررسی نوع فایل
#         if video_file.mimetype not in ALLOWED_MIME:
#             raise ValueError("Only video files are allowed")

#         # ایجاد پوشه‌های لازم
#         os.makedirs(TEMP_DIR, exist_ok=True)
#         os.makedirs(UPLOAD_DIR, exist_ok=True)

#         # ذخیره فایل موقت
#         temp_filename = f"temp_{uuid.uuid4()}{os.path.splitext(video_file.filename)[1]}"
#         temp_path = os.path.join(TEMP_DIR, temp_filename)
#         video_file.save(temp_path)

#         # نام فایل نهایی
#         final_filename = f"processed_{uuid.uuid4()}{os.path.splitext(video_file.filename)[1]}"
#         final_path = os.path.join(UPLOAD_DIR, final_filename)

#         # پردازش ویدیو
#         try:
#             clip = VideoFileClip(temp_path)
#             clip.write_videofile(
#                 final_path,
#                 codec="libx264",
#                 audio_codec="aac",
#                 threads=4  # استفاده از چند هسته پردازنده
#             )
#         except Exception as e:
#             if os.path.exists(final_path):
#                 os.remove(final_path)
#             raise Exception(f"Video conversion failed: {e}")
#         finally:
#             if 'clip' in locals():
#                 clip.close()
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)

#         # ذخیره در دیتابیس
#         new_video = Video(
#             filename=final_filename,
#             path=final_path,
#             uploaded_at=datetime.utcnow()
#         )
#         db.add(new_video)
#         db.commit()
        
#         return jsonify({
#             "message": "Video processed successfully",
#             "video_id": new_video.id,
#             "path": final_path
#         })
        
#     except Exception as e:
#         db.rollback()
#         raise Exception(f"Video processing failed: {str(e)}")
#     finally:
#         db.close()

UPLOAD_DIR = os.path.join('.', 'static', 'videos')
# UPLOAD_DIR = "C:\\Users\\Zhala\\Desktop\\test-photo"

TEMP_DIR = os.path.join(UPLOAD_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
# UPLOAD_DIR = Path(".\static") / "videos"
# TEMP_DIR = UPLOAD_DIR / "temp"
# TEMP_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_MIME = ["video/mp4", "video/x-msvideo", "video/quicktime", "video/x-ms-wmv"]
class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False)
    path = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

def process_video_file(video_file):
    """تابع مستقل برای پردازش ویدیو"""
    db = SessionLocal()  
   

    if video_file.mimetype not in ALLOWED_MIME:
        return jsonify({"error": "Only video files are allowed"}), 400

    temp_filename = f"{uuid.uuid4()}_{secure_filename(video_file.filename)}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    video_file.save(temp_path)

    final_filename = f"{video_file.filename}"
    final_path = os.path.join(UPLOAD_DIR, final_filename)

    try:
        clip = VideoFileClip(temp_path)
        clip.write_videofile(final_path, codec="libx264", audio_codec="aac")
        clip.close()
    except Exception as e:
        return jsonify({"error": f"Video conversion failed: {e}"}), 500
    finally:
        os.remove(temp_path)

    # ذخیره در دیتابیس
    new_video = Video(filename=final_filename, path=final_path)
    # new_video = Video(filename=final_filename, path=final_path,upload_at=datatime.utc())
    db.add(new_video)
    db.commit()
    return jsonify({"message": "ok"})

@app.route("/process_myvideo", methods=["POST"])
def upload_myvideo():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    try:
        mm = process_video_file(request.files['video'])
        return  mm
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


# @app.route("/process_myvideo", methods=["POST"])
# def upload_myvideo():
    
#     if 'video' not in request.files:
#         return jsonify({"error": "No video file provided"}), 400
    
#     video_file = request.files["video"]
#     try:
#         new_video = process_video_file(video_file)
#         return jsonify({"message": "Video uploaded", "video_id": new_video.id})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
 
from pathlib import Path
def process_videotodb(input_path):
    input_path = Path(input_path)
    
    # بررسی‌های ضروری
    if not input_path.exists():
        raise FileNotFoundError(f"فایل ورودی وجود ندارد: {input_path}")
    
    if input_path.stat().st_size == 0:
        raise ValueError("فایل ورودی خالی است")
    
    try:
        # پردازش با moviepy
        clip = VideoFileClip(str(input_path))
        output_path = input_path.parent / f"processed_{input_path.name}"
        
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            threads=4
        )
        return output_path
        
    except Exception as e:
        if 'output_path' in locals() and output_path.exists():
            output_path.unlink()
        raise
    finally:
        if 'clip' in locals():
            clip.close()


from pathlib import Path

def get_safe_path(file_obj):
    """ایجاد مسیر ایمن برای ذخیره فایل"""
    upload_dir = Path("static/uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # ایجاد نام فایل ایمن
    safe_name = f"{uuid.uuid4()}_{file_obj.filename}"
    return upload_dir / safe_name

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # ذخیره فایل
        temp_path = get_safe_path(video_file)
        video_file.save(str(temp_path))
        
        # پردازش ویدیو
        output_path = process_videotodb(temp_path)
        
        return jsonify({
            "message": "Video processed successfully",
            "path": str(output_path)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # تمیزکاری فایل‌های موقت
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()

@app.route("/get_video/<int:video_id>", methods=["GET"])
def get_video(video_id):
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return jsonify({"error": "Video not found"}), 404

        return send_file(video.path, mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# ***********************************************
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    