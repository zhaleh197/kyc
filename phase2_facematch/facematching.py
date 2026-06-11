
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from fastapi.middleware.cors import CORSMiddleware
# Initialize model
face_app = FaceAnalysis(name="buffalo_l")  # دقیق‌ترین مدل‌ها
face_app.prepare(ctx_id=0)  # -1 برای CPU

app = FastAPI()

def read_image(file: UploadFile):
    img = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


app = FastAPI()

allow_origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,          
    allow_credentials=True,
    allow_methods=["*"],             
    allow_headers=["*"],            
)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)  


 
@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1_bytes = await file1.read()
    img2_bytes = await file2.read()

    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    faces1 = face_app.get(img1)
    faces2 = face_app.get(img2)
    

    if not faces1 or not faces2:
        return {"match": False, "reason": "No face detected"}

    emb1 = faces1[0]['embedding']
    emb2 = faces2[0]['embedding']
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    distance = np.linalg.norm(emb1 - emb2)

    threshold = 1.1
    similarity = np.dot(emb1, emb2)
    
    # similarity_percent = float(np.clip(similarity, 0, 1)) * 100
    # similarity_percent = ((cosine_similarity + 1) / 2) * 100

    # distance = 1 - similarity 
    similarity_percent = ((similarity + 1) / 2) * 100
    print("sim: " , similarity_percent)
    per = 72.5
    return {
        # "match": bool(distance < threshold),
        "tip": "the threshold of similarity is % 72.5, you can change and handle it in your app ",
        "match": bool(similarity_percent > per),
        # "distance": float(distance),
        "similarity_percent": float(similarity_percent)
    }






# @app.post("/compare0")
# async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
#     img1_bytes = await file1.read()
#     img2_bytes = await file2.read()

#     img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
#     img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

#     faces1 = face_app.get(img1)
#     faces2 = face_app.get(img2)
    

#     if not faces1 or not faces2:
#         return {"match": False, "reason": "No face detected"}

#     emb1 = faces1[0]['embedding']
#     emb2 = faces2[0]['embedding']
#     emb1 = emb1 / np.linalg.norm(emb1)
#     emb2 = emb2 / np.linalg.norm(emb2)
#     distance = np.linalg.norm(emb1 - emb2)
    
#     threshold = 1.0  
#     cosine_similarity = np.dot(emb1, emb2)
#     similarity_percent = ((cosine_similarity + 1) / 2) * 100

#     # similarity_percent = float(np.clip(similarity, 0, 1)) * 100

#     return {
#         "match": bool(distance < threshold),
#         "distance": float(distance),
#         "similarity_percent": float(similarity_percent)
#     }



# from fastapi import FastAPI, File, UploadFile
# import numpy as np
# import cv2
# from insightface.app import FaceAnalysis
# from numpy.linalg import norm
# from fastapi.middleware.cors import CORSMiddleware
# # Initialize model
# face_app = FaceAnalysis(name="buffalo_l")  # دقیق‌ترین مدل‌ها
# face_app.prepare(ctx_id=0)  # -1 برای CPU

# app = FastAPI()

# def read_image(file: UploadFile):
#     img = np.frombuffer(file.file.read(), np.uint8)
#     img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# app = FastAPI()

# allow_origins=["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allow_origins,          
#     allow_credentials=True,
#     allow_methods=["*"],             
#     allow_headers=["*"],            
# )
# face_app = FaceAnalysis(name="buffalo_l")
# face_app.prepare(ctx_id=0)  


# from numpy import dot
# @app.post("/compare")
# async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
#     img1_bytes = await file1.read()
#     img2_bytes = await file2.read()

#     img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
#     img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

#     faces1 = face_app.get(img1)
#     faces2 = face_app.get(img2)
    

#     if not faces1 or not faces2:
#         return {"match": False, "reason": "No face detected"}
    
   

#     emb1 = faces1[0]['embedding']
#     emb2 = faces2[0]['embedding']
#     emb1 = emb1 / np.linalg.norm(emb1)
#     emb2 = emb2 / np.linalg.norm(emb2)
#     threshold =  1
#     cosine_similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    
#     cosine_distance = 1 - cosine_similarity
#     similarity_percent = ((cosine_similarity + 1) / 2) * 100


#     return {
#         "match": bool(cosine_distance < threshold),
#         "distance": float(cosine_distance),
#         "similarity_percent": float(similarity_percent) 
#     }






# # @app.post("/compare0")
# # async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
# #     img1_bytes = await file1.read()
# #     img2_bytes = await file2.read()

# #     img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
# #     img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

# #     faces1 = face_app.get(img1)
# #     faces2 = face_app.get(img2)
    

# #     if not faces1 or not faces2:
# #         return {"match": False, "reason": "No face detected"}

# #     emb1 = faces1[0]['embedding']
# #     emb2 = faces2[0]['embedding']
# #     emb1 = emb1 / np.linalg.norm(emb1)
# #     emb2 = emb2 / np.linalg.norm(emb2)
# #     distance = np.linalg.norm(emb1 - emb2)
    
# #     threshold = 1.0  
# #     cosine_similarity = np.dot(emb1, emb2)
# #     similarity_percent = ((cosine_similarity + 1) / 2) * 100

# #     # similarity_percent = float(np.clip(similarity, 0, 1)) * 100

# #     return {
# #         "match": bool(distance < threshold),
# #         "distance": float(distance),
# #         "similarity_percent": float(similarity_percent)
# #     }

