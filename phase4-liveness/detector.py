"""
Core MediaPipe-based liveness detection logic.

Refactored from a standalone OpenCV demo into a stateful object suitable for
per-session use in an API. All cv2.imshow/print/keyboard-input concerns have
been removed — the object only consumes frames and returns structured results.
"""

import time
import random
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


ALL_TASKS = [
    "CLOSE_EYES",
    "BLINK",
    "SMILE",
    "LOOK_STRAIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "GO_BACK",
    "GO_FORWARD",
]

TASK_INSTRUCTIONS = {
    "CLOSE_EYES": "Close your eyes",
    "BLINK": "Blink",
    "SMILE": "Smile wide",
    "LOOK_STRAIGHT": "Look straight at camera",
    "TURN_LEFT": "Turn head LEFT",
    "TURN_RIGHT": "Turn head RIGHT",
    "GO_BACK": "Move your head back",
    "GO_FORWARD": "Move your head forward",
}


class LivenessSession:
    """
    Stateful liveness check for a single user session.

    Lifecycle:
      1. Construct -> generates a random 4-task challenge (state=IDLE)
      2. process_frame(frame) repeatedly -> transitions IDLE -> RUNNING -> VERIFIED/FAILED
      3. Discard once VERIFIED/FAILED (or call reset() for a new challenge)

    Each instance owns its own MediaPipe FaceMesh — do not share across threads.
    """

    # MediaPipe landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    SMILE_LEFT = 287
    SMILE_RIGHT = 57
    MOUTH_CENTER_TOP = 0
    MOUTH_CENTER_BOTTOM = 17

    NOSE_TIP = 1
    LEFT_EYE = 33
    RIGHT_EYE = 263
    LEFT_EAR = 234
    RIGHT_EAR = 454

    # Thresholds
    YAW_THRESHOLD = 20.0
    EYE_AR_THRESHOLD = 0.21
    SMILE_RATIO_THRESHOLD = 0.75
    SMILE_FRAMES_REQUIRED = 3
    EYES_CLOSED_HOLD_FRAMES = 5
    BLINK_MIN_FRAMES = 2
    BLINK_MAX_FRAMES = 10
    Z_DEPTH_THRESHOLD = 15.0
    TASK_TIME_LIMIT = 7.0  # seconds per task

    def __init__(self, task_time_limit: Optional[float] = None):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        if task_time_limit is not None:
            self.TASK_TIME_LIMIT = task_time_limit

        self.task_list: list[str] = []
        self.current_task_index = 0
        self.task_start_time: Optional[float] = None
        self.completed_tasks = 0

        self.eyes_closed_frames = 0
        self.smile_frames = 0
        self.initial_z_position: Optional[float] = None

        self.state = "IDLE"  # IDLE, RUNNING, VERIFIED, FAILED
        self.face_captured = False
        self.last_message = ""

        self._generate_task_list()

    def close(self):
        """Release MediaPipe resources. Call when the session is discarded."""
        self.face_mesh.close()

    # ------------------------------------------------------------------ #
    # Setup / reset
    # ------------------------------------------------------------------ #

    def _generate_task_list(self):
        self.task_list = random.sample(ALL_TASKS, 4)
        self.current_task_index = 0
        self.completed_tasks = 0

    def reset(self):
        """Reset session state and generate a fresh 4-task challenge."""
        self.state = "IDLE"
        self.face_captured = False
        self.current_task_index = 0
        self.completed_tasks = 0
        self.task_start_time = None
        self.initial_z_position = None
        self.eyes_closed_frames = 0
        self.smile_frames = 0
        self.last_message = ""
        self._generate_task_list()

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    def _eye_aspect_ratio(self, landmarks, eye_indices, image_shape) -> float:
        h, w = image_shape[:2]
        points = np.array(
            [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in eye_indices]
        )
        a = np.linalg.norm(points[1] - points[5])
        b = np.linalg.norm(points[2] - points[4])
        c = np.linalg.norm(points[0] - points[3])
        return (a + b) / (2.0 * c) if c > 0 else 0.0

    def _smile_metrics(self, landmarks, image_shape):
        """Returns (smile_ratio, corners_lifted, is_symmetric)."""
        h, w = image_shape[:2]

        smile_left = landmarks[self.SMILE_LEFT]
        smile_right = landmarks[self.SMILE_RIGHT]
        mouth_center_top = landmarks[self.MOUTH_CENTER_TOP]
        mouth_center_bottom = landmarks[self.MOUTH_CENTER_BOTTOM]

        smile_width = np.linalg.norm(
            np.array([smile_left.x * w, smile_left.y * h])
            - np.array([smile_right.x * w, smile_right.y * h])
        )

        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        face_width = np.linalg.norm(
            np.array([left_eye.x * w, left_eye.y * h])
            - np.array([right_eye.x * w, right_eye.y * h])
        )

        smile_ratio = smile_width / face_width if face_width > 0 else 0.0

        avg_smile_y = (smile_left.y * h + smile_right.y * h) / 2
        mouth_center_y = mouth_center_bottom.y * h
        corners_lifted = avg_smile_y < mouth_center_y

        mouth_center_x = (mouth_center_top.x + mouth_center_bottom.x) / 2 * w
        left_dist = abs(smile_left.x * w - mouth_center_x)
        right_dist = abs(smile_right.x * w - mouth_center_x)
        max_dist = max(left_dist, right_dist)
        symmetry = 1 - abs(left_dist - right_dist) / max_dist if max_dist > 0 else 1.0
        is_symmetric = symmetry > 0.7

        return smile_ratio, corners_lifted, is_symmetric

    def _head_pose(self, landmarks, image_shape):
        """Returns (yaw_degrees, z_depth)."""
        h, w = image_shape[:2]

        nose = np.array(
            [landmarks[self.NOSE_TIP].x * w, landmarks[self.NOSE_TIP].y * h, landmarks[self.NOSE_TIP].z * w]
        )
        left_eye = np.array(
            [landmarks[self.LEFT_EYE].x * w, landmarks[self.LEFT_EYE].y * h, landmarks[self.LEFT_EYE].z * w]
        )
        right_eye = np.array(
            [landmarks[self.RIGHT_EYE].x * w, landmarks[self.RIGHT_EYE].y * h, landmarks[self.RIGHT_EYE].z * w]
        )
        left_ear = np.array(
            [landmarks[self.LEFT_EAR].x * w, landmarks[self.LEFT_EAR].y * h, landmarks[self.LEFT_EAR].z * w]
        )
        right_ear = np.array(
            [landmarks[self.RIGHT_EAR].x * w, landmarks[self.RIGHT_EAR].y * h, landmarks[self.RIGHT_EAR].z * w]
        )

        face_center = (left_eye + right_eye) / 2

        left_dist = np.linalg.norm(left_ear[:2] - face_center[:2])
        right_dist = np.linalg.norm(right_ear[:2] - face_center[:2])

        if (left_dist + right_dist) > 0:
            ear_ratio = (right_dist - left_dist) / (right_dist + left_dist)
            yaw = ear_ratio * 90
        else:
            yaw = 0.0

        return yaw, nose[2]

    # ------------------------------------------------------------------ #
    # Task detection
    # ------------------------------------------------------------------ #

    def _detect_task(self, task: str, landmarks, image_shape, z_depth: float) -> bool:
        if task == "CLOSE_EYES":
            left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES, image_shape)
            right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES, image_shape)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.EYE_AR_THRESHOLD:
                self.eyes_closed_frames += 1
                if self.eyes_closed_frames >= self.EYES_CLOSED_HOLD_FRAMES:
                    return True
            else:
                self.eyes_closed_frames = 0
            return False

        if task == "BLINK":
            left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES, image_shape)
            right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES, image_shape)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.EYE_AR_THRESHOLD:
                self.eyes_closed_frames += 1
            else:
                if self.BLINK_MIN_FRAMES <= self.eyes_closed_frames <= self.BLINK_MAX_FRAMES:
                    self.eyes_closed_frames = 0
                    return True
                self.eyes_closed_frames = 0
            return False

        if task == "SMILE":
            smile_ratio, corners_lifted, is_symmetric = self._smile_metrics(landmarks, image_shape)
            is_valid_smile = (
                smile_ratio > self.SMILE_RATIO_THRESHOLD and corners_lifted and is_symmetric
            )
            if is_valid_smile:
                self.smile_frames += 1
                if self.smile_frames >= self.SMILE_FRAMES_REQUIRED:
                    self.smile_frames = 0
                    return True
            else:
                self.smile_frames = 0
            return False

        if task == "LOOK_STRAIGHT":
            yaw, _ = self._head_pose(landmarks, image_shape)
            return abs(yaw) < 10

        if task == "TURN_LEFT":
            yaw, _ = self._head_pose(landmarks, image_shape)
            return yaw < -self.YAW_THRESHOLD

        if task == "TURN_RIGHT":
            yaw, _ = self._head_pose(landmarks, image_shape)
            return yaw > self.YAW_THRESHOLD

        if task == "GO_BACK":
            if self.initial_z_position is None:
                self.initial_z_position = z_depth
                return False
            return (z_depth - self.initial_z_position) > self.Z_DEPTH_THRESHOLD

        if task == "GO_FORWARD":
            if self.initial_z_position is None:
                self.initial_z_position = z_depth
                return False
            return (z_depth - self.initial_z_position) < -self.Z_DEPTH_THRESHOLD

        return False

    # ------------------------------------------------------------------ #
    # Frame processing
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single BGR frame (as a numpy array, e.g. from cv2.imdecode).

        Returns a JSON-serializable dict describing current state, the active
        task, time remaining, and the overall verification result.
        """
        if self.state in ("VERIFIED", "FAILED"):
            return self._status_payload()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self.last_message = "No face detected"
            if self.state == "RUNNING":
                self.state = "FAILED"
            return self._status_payload()

        face_landmarks = results.multi_face_landmarks[0]

        if not self.face_captured:
            self.face_captured = True
            self.state = "RUNNING"
            self.task_start_time = time.time()
            self.last_message = "Face detected, starting tasks"

        yaw, z_depth = self._head_pose(face_landmarks.landmark, frame.shape)

        if self.state == "RUNNING" and self.current_task_index < len(self.task_list):
            current_task = self.task_list[self.current_task_index]

            elapsed = time.time() - self.task_start_time
            time_remaining = max(0.0, self.TASK_TIME_LIMIT - elapsed)

            task_completed = self._detect_task(
                current_task, face_landmarks.landmark, frame.shape, z_depth
            )

            if task_completed:
                self.completed_tasks += 1
                self.last_message = f"Task completed: {TASK_INSTRUCTIONS[current_task]}"

                self.current_task_index += 1
                self.task_start_time = time.time()
                self.initial_z_position = None
                self.eyes_closed_frames = 0
                self.smile_frames = 0

                if self.current_task_index >= len(self.task_list):
                    self.state = "VERIFIED"
                    self.last_message = f"Verified - all {len(self.task_list)} tasks completed"
                else:
                    next_task = self.task_list[self.current_task_index]
                    self.last_message = f"Next: {TASK_INSTRUCTIONS[next_task]}"

            elif time_remaining <= 0:
                self.state = "FAILED"
                self.last_message = f"Failed: time expired on '{TASK_INSTRUCTIONS[current_task]}'"

        return self._status_payload()

    def _status_payload(self) -> dict:
        time_remaining = None
        current_task = None

        if self.state == "RUNNING" and self.current_task_index < len(self.task_list):
            current_task = self.task_list[self.current_task_index]
            if self.task_start_time is not None:
                elapsed = time.time() - self.task_start_time
                time_remaining = max(0.0, self.TASK_TIME_LIMIT - elapsed)

        return {
            "state": self.state,
            "message": self.last_message,
            "current_task": current_task,
            "current_task_instruction": TASK_INSTRUCTIONS.get(current_task) if current_task else None,
            "task_index": self.current_task_index,
            "total_tasks": len(self.task_list),
            "completed_tasks": self.completed_tasks,
            "time_remaining": round(time_remaining, 2) if time_remaining is not None else None,
            "task_list": self.task_list,
            "is_live": self.state == "VERIFIED",
        }