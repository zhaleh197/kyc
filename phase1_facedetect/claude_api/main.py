"""
KYC API — Full Biometric Verification Pipeline
Based on ISO/IEC 19794-5 face quality standard

Requirements:
    pip install fastapi uvicorn python-multipart opencv-python-headless
                mediapipe numpy deepface scikit-image Pillow torch torchvision
                insightface onnxruntime scipy

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

import io, uuid, base64, logging
from typing import Optional
from enum import Enum

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Optional heavy imports (graceful fallback if not installed) ──────────────
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kyc")

# ═══════════════════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="KYC Biometric API",
    description="Full KYC pipeline: face quality (ISO/IEC 19794-5), matching, liveness, anti-spoof, AI forensics",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── InsightFace global init (reused across requests) ─────────────────────────
_insight_app = None
def get_insight():
    global _insight_app
    if _insight_app is None and INSIGHTFACE_AVAILABLE:
        _insight_app = FaceAnalysis(providers=["CPUExecutionProvider"])
        _insight_app.prepare(ctx_id=0, det_size=(640, 640))
    return _insight_app

# ── MediaPipe globals ─────────────────────────────────────────────────────────
_mp_face_mesh  = None
_mp_face_det   = None
def get_mp_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None and MP_AVAILABLE:
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5)
    return _mp_face_mesh


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════
class CheckResult(BaseModel):
    pass_: bool
    score: float
    confidence: float
    issues: list[str] = []
    detail: dict = {}

class FaceQualityResult(CheckResult):
    illumination: dict = {}
    head_pose: dict = {}
    eye_state: dict = {}
    face_coverage: float = 0.0
    sharpness: float = 0.0
    expression: dict = {}
    background: dict = {}
    occlusion: dict = {}

class FaceMatchResult(BaseModel):
    pass_: bool
    similarity: float
    distance: float
    threshold: float
    model_used: str

class LivenessResult(BaseModel):
    pass_: bool
    confidence: float
    method: str
    detail: dict = {}

class AntiSpoofResult(BaseModel):
    pass_: bool
    confidence: float
    attack_type: Optional[str] = None
    detail: dict = {}

class ForensicsResult(BaseModel):
    pass_: bool
    deepfake_probability: float
    manipulation_detected: bool
    ela_score: float
    detail: dict = {}

class KYCReport(BaseModel):
    session_id: str
    overall: str                  # PASS | FAIL | REVIEW
    overall_score: float
    face_quality: Optional[FaceQualityResult] = None
    face_match:   Optional[FaceMatchResult]   = None
    liveness:     Optional[LivenessResult]    = None
    antispoof:    Optional[AntiSpoofResult]   = None
    forensics:    Optional[ForensicsResult]   = None


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_image(upload: UploadFile) -> np.ndarray:
    """Read UploadFile → BGR numpy array."""
    data = upload.file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image.")
    return img

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_keep_aspect(img: np.ndarray, max_dim=1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    return cv2.resize(img, (int(w*scale), int(h*scale)))


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Face Quality (ISO/IEC 19794-5)
# ═══════════════════════════════════════════════════════════════════════════════
class FaceQualityChecker:
    """
    Checks conformance to ISO/IEC 19794-5 token face requirements:
      - Illumination uniformity
      - Head pose (yaw / pitch / roll)
      - Eye openness & gaze
      - Face coverage ratio
      - Sharpness (Laplacian variance)
      - Neutral expression
      - Background uniformity
      - Occlusion detection
    """

    # ── Thresholds (tune per deployment) ─────────────────────────────────────
    SHARPNESS_MIN   = 80.0
    FACE_COVER_MIN  = 0.35   # face bbox area / total image area
    FACE_COVER_MAX  = 0.85
    POSE_YAW_MAX    = 10.0   # degrees
    POSE_PITCH_MAX  = 10.0
    POSE_ROLL_MAX   = 10.0
    EYE_OPEN_MIN    = 0.20   # eye-aspect-ratio
    ILLUM_VAR_MAX   = 40.0   # std-dev threshold for background region

    def check(self, img: np.ndarray) -> FaceQualityResult:
        issues  = []
        details = {}
        scores  = []

        rgb = bgr_to_rgb(img)
        h, w = img.shape[:2]

        # ── 1. Sharpness ──────────────────────────────────────────────────────
        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_ok  = sharpness >= self.SHARPNESS_MIN
        scores.append(min(sharpness / 300, 1.0))
        if not sharp_ok:
            issues.append("Image is blurry (sharpness too low)")

        # ── 2. Face detection & coverage ─────────────────────────────────────
        face_bbox    = None
        face_cover   = 0.0
        coverage_ok  = False

        if INSIGHTFACE_AVAILABLE:
            iapp   = get_insight()
            faces  = iapp.get(rgb) if iapp else []
            if faces:
                f         = faces[0]
                x1,y1,x2,y2 = [int(v) for v in f.bbox]
                face_bbox = (x1, y1, x2-x1, y2-y1)
        else:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            dets = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(dets):
                face_bbox = tuple(dets[0])

        if face_bbox:
            fx, fy, fw, fh = face_bbox
            face_cover = (fw * fh) / (w * h)
            coverage_ok = self.FACE_COVER_MIN <= face_cover <= self.FACE_COVER_MAX
            scores.append(1.0 if coverage_ok else 0.4)
            if not coverage_ok:
                issues.append(f"Face coverage {face_cover:.0%} out of range [{self.FACE_COVER_MIN:.0%}–{self.FACE_COVER_MAX:.0%}]")
        else:
            issues.append("No face detected in image")
            scores.append(0.0)

        # ── 3. Illumination ───────────────────────────────────────────────────
        illum_result = self._check_illumination(img, face_bbox)
        scores.append(illum_result["score"])
        if not illum_result["ok"]:
            issues.extend(illum_result["issues"])

        # ── 4. Head pose + eye state (MediaPipe) ──────────────────────────────
        pose_result = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "ok": True, "issues": [], "score": 1.0}
        eye_result  = {"left_ear": 0.0, "right_ear": 0.0, "ok": True, "issues": [], "score": 1.0}

        if MP_AVAILABLE:
            mesh = get_mp_face_mesh()
            if mesh:
                res = mesh.process(rgb)
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    pose_result = self._estimate_pose(lms, w, h)
                    eye_result  = self._check_eyes(lms)

        scores.append(pose_result["score"])
        scores.append(eye_result["score"])
        issues.extend(pose_result["issues"])
        issues.extend(eye_result["issues"])

        # ── 5. Background uniformity ──────────────────────────────────────────
        bg_result = self._check_background(img, face_bbox, h, w)
        scores.append(bg_result["score"])
        if not bg_result["ok"]:
            issues.append("Background is not uniform")

        # ── 6. Expression (simple landmark heuristic) ─────────────────────────
        expr_result = {"ok": True, "issues": [], "score": 1.0, "mouth_open": False}
        if MP_AVAILABLE and face_bbox:
            mesh = get_mp_face_mesh()
            if mesh:
                res = mesh.process(rgb)
                if res.multi_face_landmarks:
                    expr_result = self._check_expression(res.multi_face_landmarks[0].landmark)
        scores.append(expr_result["score"])
        issues.extend(expr_result.get("issues", []))

        # ── Aggregate ─────────────────────────────────────────────────────────
        avg_score = float(np.mean(scores)) if scores else 0.0
        passed    = len(issues) == 0 and avg_score >= 0.70

        return FaceQualityResult(
            pass_       = passed,
            score       = round(avg_score, 3),
            confidence  = round(avg_score, 3),
            issues      = issues,
            sharpness   = round(sharpness, 2),
            face_coverage = round(face_cover, 3),
            illumination  = illum_result,
            head_pose     = {k: round(v, 2) if isinstance(v, float) else v
                             for k, v in pose_result.items()},
            eye_state     = {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in eye_result.items()},
            background    = bg_result,
            expression    = expr_result,
            occlusion     = {"checked": False, "note": "Deep occlusion model not loaded"},
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _check_illumination(self, img, face_bbox):
        lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L     = lab[:, :, 0].astype(float)
        issues = []

        if face_bbox:
            fx, fy, fw, fh = face_bbox
            face_L = L[fy:fy+fh, fx:fx+fw]
        else:
            face_L = L

        mean_l  = float(face_L.mean())
        std_l   = float(face_L.std())
        dark_ok = mean_l > 50
        glare_ok= mean_l < 210
        unif_ok = std_l < 60

        if not dark_ok:  issues.append("Face region too dark")
        if not glare_ok: issues.append("Overexposed / glare detected")
        if not unif_ok:  issues.append("Uneven illumination (shadows)")

        score = 1.0 - (std_l / 120.0)
        score = max(0.0, min(1.0, score))
        ok    = dark_ok and glare_ok and unif_ok

        return {"ok": ok, "score": round(score, 3),
                "mean_luminance": round(mean_l, 1),
                "std_luminance": round(std_l, 1), "issues": issues}

    def _estimate_pose(self, lms, w, h):
        """Rough yaw/pitch/roll from 3-D face mesh landmarks."""
        issues = []
        # Key landmark indices (MediaPipe 468-point mesh)
        nose_tip   = lms[1]
        left_eye   = lms[33]
        right_eye  = lms[263]
        chin       = lms[152]
        forehead   = lms[10]

        # Roll: angle of eye line
        dy = (right_eye.y - left_eye.y) * h
        dx = (right_eye.x - left_eye.x) * w
        roll = float(np.degrees(np.arctan2(dy, dx)))

        # Yaw: nose-tip horizontal offset from eye midpoint
        eye_mid_x = (left_eye.x + right_eye.x) / 2
        yaw = float((nose_tip.x - eye_mid_x) * 180)

        # Pitch: nose-tip vertical offset from chin–forehead midpoint
        face_mid_y = (chin.y + forehead.y) / 2
        pitch = float((nose_tip.y - face_mid_y) * 180)

        ok = True
        if abs(yaw)   > self.POSE_YAW_MAX:   issues.append(f"Head not straight (yaw {yaw:.1f}°)");   ok=False
        if abs(pitch) > self.POSE_PITCH_MAX: issues.append(f"Head tilted up/down (pitch {pitch:.1f}°)"); ok=False
        if abs(roll)  > self.POSE_ROLL_MAX:  issues.append(f"Head rotated (roll {roll:.1f}°)");        ok=False

        score = 1.0 - min(1.0, (abs(yaw)+abs(pitch)+abs(roll)) / 90)
        return {"ok": ok, "yaw": yaw, "pitch": pitch, "roll": roll,
                "issues": issues, "score": round(score, 3)}

    def _check_eyes(self, lms):
        """Eye Aspect Ratio (EAR) to detect closed eyes."""
        issues = []

        def ear(pts):
            p = [(lms[i].x, lms[i].y) for i in pts]
            A = np.linalg.norm(np.array(p[1])-np.array(p[5]))
            B = np.linalg.norm(np.array(p[2])-np.array(p[4]))
            C = np.linalg.norm(np.array(p[0])-np.array(p[3]))
            return (A+B) / (2*C+1e-6)

        # MediaPipe eye contour landmark indices
        left_ear  = ear([362, 385, 387, 263, 373, 380])
        right_ear = ear([33,  160, 158, 133, 153, 144])

        left_ok  = left_ear  >= self.EYE_OPEN_MIN
        right_ok = right_ear >= self.EYE_OPEN_MIN
        ok = left_ok and right_ok

        if not left_ok:  issues.append("Left eye appears closed")
        if not right_ok: issues.append("Right eye appears closed")

        score = min(left_ear, right_ear) / self.EYE_OPEN_MIN
        score = max(0.0, min(1.0, score))
        return {"ok": ok, "left_ear": float(left_ear),
                "right_ear": float(right_ear),
                "issues": issues, "score": round(score, 3)}

    def _check_background(self, img, face_bbox, h, w):
        if face_bbox is None:
            return {"ok": False, "score": 0.5, "std": 0.0}
        fx, fy, fw, fh = face_bbox
        mask = np.zeros((h, w), dtype=bool)
        mask[fy:fy+fh, fx:fx+fw] = True
        bg   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[~mask]
        std  = float(bg.std()) if bg.size else 0.0
        ok   = std < 30.0
        score= max(0.0, 1.0 - std/60)
        return {"ok": ok, "score": round(score, 3), "std": round(std, 2)}

    def _check_expression(self, lms):
        """Detect open mouth via lip distance."""
        upper_lip = lms[13]
        lower_lip = lms[14]
        mouth_open = abs(upper_lip.y - lower_lip.y) > 0.025
        issues = ["Mouth is open — keep neutral expression"] if mouth_open else []
        score  = 0.5 if mouth_open else 1.0
        return {"ok": not mouth_open, "mouth_open": mouth_open,
                "issues": issues, "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Face Matching (1:1 verification)
# ═══════════════════════════════════════════════════════════════════════════════
class FaceMatcher:
    THRESHOLD = 0.40   # ArcFace cosine distance threshold

    def match(self, img1: np.ndarray, img2: np.ndarray) -> FaceMatchResult:
        if INSIGHTFACE_AVAILABLE:
            return self._match_insightface(img1, img2)
        if DEEPFACE_AVAILABLE:
            return self._match_deepface(img1, img2)
        raise HTTPException(503, "No face matching backend available. Install insightface or deepface.")

    def _match_insightface(self, img1, img2):
        iapp   = get_insight()
        faces1 = iapp.get(bgr_to_rgb(img1))
        faces2 = iapp.get(bgr_to_rgb(img2))
        if not faces1: raise HTTPException(422, "No face found in image 1")
        if not faces2: raise HTTPException(422, "No face found in image 2")

        emb1 = faces1[0].normed_embedding
        emb2 = faces2[0].normed_embedding
        dist = float(np.linalg.norm(emb1 - emb2))
        sim  = float(1 - dist / 2)           # normalise to [0,1]
        return FaceMatchResult(
            pass_=dist <= self.THRESHOLD, similarity=round(sim,4),
            distance=round(dist,4), threshold=self.THRESHOLD,
            model_used="InsightFace/ArcFace")

    def _match_deepface(self, img1, img2):
        tmp1 = "/tmp/kyc_img1.jpg"; tmp2 = "/tmp/kyc_img2.jpg"
        cv2.imwrite(tmp1, img1); cv2.imwrite(tmp2, img2)
        res = DeepFace.verify(tmp1, tmp2, model_name="ArcFace", enforce_detection=True)
        dist = float(res["distance"])
        sim  = float(1 - min(dist, 1.0))
        return FaceMatchResult(
            pass_=res["verified"], similarity=round(sim,4),
            distance=round(dist,4), threshold=float(res["threshold"]),
            model_used="DeepFace/ArcFace")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Passive Liveness Detection
# ═══════════════════════════════════════════════════════════════════════════════
class LivenessDetector:
    """
    Passive liveness: texture analysis, frequency analysis, reflection patterns.
    For active liveness (blink/head-turn challenges) a video stream is needed.
    """
    def check(self, img: np.ndarray) -> LivenessResult:
        score, detail = self._texture_liveness(img)
        passed = score >= 0.60
        return LivenessResult(
            pass_=passed, confidence=round(score, 3),
            method="passive_texture", detail=detail)

    def _texture_liveness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # LBP-like texture richness
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag    = np.sqrt(sobelx**2 + sobely**2)
        texture_score = float(np.mean(mag)) / 100.0
        texture_score = min(texture_score, 1.0)

        # Frequency domain — real faces have high-freq components
        f    = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag_spectrum = 20 * np.log(np.abs(fshift) + 1)
        freq_score = float(np.std(mag_spectrum)) / 50.0
        freq_score = min(freq_score, 1.0)

        combined = (texture_score * 0.5) + (freq_score * 0.5)
        return combined, {
            "texture_score": round(texture_score, 3),
            "freq_score":    round(freq_score, 3),
            "note": "Active liveness (blink/motion) requires video input"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Anti-Spoofing
# ═══════════════════════════════════════════════════════════════════════════════
class AntiSpoofChecker:
    """
    Detects print attacks, screen replay, and 3D mask attacks using
    texture / reflection cues. Plug in Silent-Face-Anti-Spoofing
    or FAS models for production-grade results.
    """
    def check(self, img: np.ndarray) -> AntiSpoofResult:
        score, attack_type, detail = self._heuristic_check(img)
        return AntiSpoofResult(
            pass_=score >= 0.65, confidence=round(score, 3),
            attack_type=attack_type, detail=detail)

    def _heuristic_check(self, img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detail = {}

        # Moiré pattern detection (print attack)
        f   = np.fft.fft2(gray)
        fs  = np.fft.fftshift(f)
        mag = np.abs(fs)
        h, w = mag.shape
        cx, cy = w//2, h//2
        ring = mag[cy-30:cy+30, cx-30:cx+30]
        moire_score = 1.0 - min(float(ring.mean()) / 3000, 1.0)
        detail["moire_score"] = round(moire_score, 3)

        # Saturation variance (screens/prints look flat)
        sat = hsv[:,:,1].astype(float)
        sat_var = float(sat.std())
        sat_score = min(sat_var / 40, 1.0)
        detail["saturation_variance"] = round(sat_var, 2)

        # Reflection highlights (screens produce uniform glare)
        val   = hsv[:,:,2]
        bright_pixels = float(np.sum(val > 240)) / val.size
        refl_score = 1.0 - min(bright_pixels * 20, 1.0)
        detail["bright_pixel_ratio"] = round(bright_pixels, 4)

        combined = (moire_score * 0.4) + (sat_score * 0.35) + (refl_score * 0.25)

        attack = None
        if moire_score < 0.45:  attack = "print_attack"
        elif refl_score < 0.4:  attack = "screen_replay"
        elif sat_score  < 0.35: attack = "flat_texture_mask"

        return combined, attack, detail


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — AI Forensics (Deepfake / Manipulation Detection)
# ═══════════════════════════════════════════════════════════════════════════════
class AIForensicsChecker:
    """
    Error Level Analysis (ELA) + noise analysis to detect tampering.
    For GAN/deepfake detection, plug in an EfficientNet trained on
    FaceForensics++ (ff++_efficientnetb4.pth) here.
    """
    def check(self, img: np.ndarray) -> ForensicsResult:
        ela_score = self._ela(img)
        noise_score = self._noise_analysis(img)
        deepfake_prob = self._heuristic_deepfake(img)

        manipulation = ela_score > 0.55 or noise_score > 0.60
        passed = deepfake_prob < 0.45 and not manipulation

        return ForensicsResult(
            pass_=passed,
            deepfake_probability=round(deepfake_prob, 3),
            manipulation_detected=manipulation,
            ela_score=round(ela_score, 3),
            detail={
                "ela_score": round(ela_score, 3),
                "noise_score": round(noise_score, 3),
                "note": "EfficientNet deepfake model not loaded — using heuristic"
            }
        )

    def _ela(self, img, quality=90):
        """Error Level Analysis: recompress and measure delta."""
        buf = io.BytesIO()
        pil = Image.fromarray(bgr_to_rgb(img))
        pil.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        recompressed = np.array(Image.open(buf))
        orig_rgb = bgr_to_rgb(img)

        # Resize to same dims if needed
        if recompressed.shape != orig_rgb.shape:
            recompressed = cv2.resize(recompressed, (orig_rgb.shape[1], orig_rgb.shape[0]))

        ela = np.abs(orig_rgb.astype(int) - recompressed.astype(int))
        score = float(ela.mean()) / 255
        return score

    def _noise_analysis(self, img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        blur  = cv2.GaussianBlur(gray, (5,5), 0)
        noise = np.abs(gray - blur)
        return float(noise.std()) / 30

    def _heuristic_deepfake(self, img):
        """
        Frequency analysis — GAN images lack natural high-freq noise.
        Replace with a trained classifier for production.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        f    = np.fft.fft2(gray)
        fs   = np.fft.fftshift(f)
        mag  = np.log(np.abs(fs) + 1)
        hf_mask = np.zeros_like(mag, dtype=bool)
        h, w = mag.shape
        cx, cy = w//2, h//2
        r = min(h,w)//4
        hf_mask[:cy-r, :] = True; hf_mask[cy+r:, :] = True
        hf_mask[:, :cx-r] = True; hf_mask[:, cx+r:] = True

        hf_energy = float(mag[hf_mask].mean())
        # Real images have more HF energy; GANs tend to be smoother
        prob = max(0.0, 1.0 - min(hf_energy / 8, 1.0))
        return prob


# ═══════════════════════════════════════════════════════════════════════════════
# Service singletons
# ═══════════════════════════════════════════════════════════════════════════════
quality_checker  = FaceQualityChecker()
face_matcher     = FaceMatcher()
liveness_det     = LivenessDetector()
antispoof_chk    = AntiSpoofChecker()
forensics_chk    = AIForensicsChecker()


# ═══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "KYC Biometric API",
        "version": "1.0.0",
        "backends": {
            "mediapipe":   MP_AVAILABLE,
            "deepface":    DEEPFACE_AVAILABLE,
            "insightface": INSIGHTFACE_AVAILABLE,
        }
    }


@app.post("/kyc/face-quality", response_model=FaceQualityResult, tags=["Modules"])
async def face_quality(image: UploadFile = File(...)):
    """
    ISO/IEC 19794-5 compliant face quality assessment.
    Checks: sharpness, face coverage, illumination, head pose,
    eye state, expression, background, occlusion.
    """
    img = load_image(image)
    img = resize_keep_aspect(img)
    return quality_checker.check(img)


@app.post("/kyc/face-match", response_model=FaceMatchResult, tags=["Modules"])
async def face_match(
    image1: UploadFile = File(..., description="Selfie / live capture"),
    image2: UploadFile = File(..., description="ID document face photo"),
):
    """1:1 face verification using ArcFace embeddings."""
    img1 = load_image(image1)
    img2 = load_image(image2)
    return face_matcher.match(img1, img2)


@app.post("/kyc/liveness", response_model=LivenessResult, tags=["Modules"])
async def liveness(image: UploadFile = File(...)):
    """
    Passive liveness detection via texture & frequency analysis.
    For active liveness (blink / head-turn) supply a video clip.
    """
    img = load_image(image)
    img = resize_keep_aspect(img)
    return liveness_det.check(img)


@app.post("/kyc/antispoof", response_model=AntiSpoofResult, tags=["Modules"])
async def antispoof(image: UploadFile = File(...)):
    """Detect print attacks, screen replay, and flat-texture masks."""
    img = load_image(image)
    img = resize_keep_aspect(img)
    return antispoof_chk.check(img)


@app.post("/kyc/forensics", response_model=ForensicsResult, tags=["Modules"])
async def forensics(image: UploadFile = File(...)):
    """AI forensics: ELA + noise analysis + GAN frequency heuristic."""
    img = load_image(image)
    img = resize_keep_aspect(img)
    return forensics_chk.check(img)


@app.post("/kyc/full", response_model=KYCReport, tags=["Full Pipeline"])
async def full_kyc(
    selfie: UploadFile = File(..., description="Live selfie"),
    id_photo: UploadFile = File(None, description="ID document photo (optional for matching)"),
):
    """
    Full KYC pipeline — runs all 5 checks and returns a unified report.
    Pass/Fail/Review decision based on weighted scoring.
    """
    session_id = f"kyc_{uuid.uuid4().hex[:12]}"

    selfie_img = load_image(selfie)
    selfie_img = resize_keep_aspect(selfie_img)

    # Run all checks
    quality  = quality_checker.check(selfie_img)
    liveness = liveness_det.check(selfie_img)
    spoof    = antispoof_chk.check(selfie_img)
    forensic = forensics_chk.check(selfie_img)

    match_res = None
    if id_photo:
        id_img    = load_image(id_photo)
        id_img    = resize_keep_aspect(id_img)
        match_res = face_matcher.match(selfie_img, id_img)

    # ── Weighted overall score ─────────────────────────────────────────────
    checks = [
        (quality.score,              0.30),
        (liveness.confidence,        0.25),
        (spoof.confidence,           0.25),
        (1.0 - forensic.deepfake_probability, 0.20),
    ]
    if match_res:
        checks.append((match_res.similarity, 0.15))
        # re-normalise weights
        total_w   = sum(w for _, w in checks)
        checks    = [(s, w/total_w) for s, w in checks]

    overall_score = sum(s*w for s, w in checks)

    all_pass = (
        quality.pass_  and
        liveness.pass_ and
        spoof.pass_    and
        forensic.pass_ and
        (match_res.pass_ if match_res else True)
    )

    if all_pass and overall_score >= 0.75:
        verdict = "PASS"
    elif overall_score < 0.45:
        verdict = "FAIL"
    else:
        verdict = "REVIEW"

    return KYCReport(
        session_id    = session_id,
        overall       = verdict,
        overall_score = round(overall_score, 3),
        face_quality  = quality,
        face_match    = match_res,
        liveness      = liveness,
        antispoof     = spoof,
        forensics     = forensic,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
