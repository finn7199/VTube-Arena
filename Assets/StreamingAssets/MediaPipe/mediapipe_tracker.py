#!/usr/bin/env python3
"""
MediaPipe Full-Body Tracker for VLive-Arena.
Captures webcam, runs MediaPipe pose + face detection,
computes bone rotations, and sends binary UDP packets to Unity.

Uses the modern MediaPipe Tasks API (0.10.9+).

Usage:
    python mediapipe_tracker.py --camera 0 --port 11574
"""

import argparse
import struct
import socket
import time
import sys
import os

# Add local libs folder to path so --target installs work
_libs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
if os.path.isdir(_libs_dir):
    sys.path.insert(0, _libs_dir)

_script_dir = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Constants ---
MAGIC = 0x4D504654  # "MPFT"
NUM_BLENDSHAPES = 52
NUM_BONES = 17
PACKET_SIZE = 512  # 12 + 208 + 272 + 12 + 8

# ARKit blendshape names in packet order (52 total)
ARKIT_BLENDSHAPE_NAMES = [
    "browDownLeft",         # 0
    "browDownRight",        # 1
    "browInnerUp",          # 2
    "browOuterUpLeft",      # 3
    "browOuterUpRight",     # 4
    "cheekPuff",            # 5
    "cheekSquintLeft",      # 6
    "cheekSquintRight",     # 7
    "eyeBlinkLeft",         # 8
    "eyeBlinkRight",        # 9
    "eyeLookDownLeft",      # 10
    "eyeLookDownRight",     # 11
    "eyeLookInLeft",        # 12
    "eyeLookInRight",       # 13
    "eyeLookOutLeft",       # 14
    "eyeLookOutRight",      # 15
    "eyeLookUpLeft",        # 16
    "eyeLookUpRight",       # 17
    "eyeSquintLeft",        # 18
    "eyeSquintRight",       # 19
    "eyeWideLeft",          # 20
    "eyeWideRight",         # 21
    "jawForward",           # 22
    "jawLeft",              # 23
    "jawOpen",              # 24
    "jawRight",             # 25
    "mouthClose",           # 26
    "mouthDimpleLeft",      # 27
    "mouthDimpleRight",     # 28
    "mouthFrownLeft",       # 29
    "mouthFrownRight",      # 30
    "mouthFunnel",          # 31
    "mouthLeft",            # 32
    "mouthLowerDownLeft",   # 33
    "mouthLowerDownRight",  # 34
    "mouthPressLeft",       # 35
    "mouthPressRight",      # 36
    "mouthPucker",          # 37
    "mouthRight",           # 38
    "mouthRollLower",       # 39
    "mouthRollUpper",       # 40
    "mouthShrugLower",      # 41
    "mouthShrugUpper",      # 42
    "mouthSmileLeft",       # 43
    "mouthSmileRight",      # 44
    "mouthStretchLeft",     # 45
    "mouthStretchRight",    # 46
    "mouthUpperUpLeft",     # 47
    "mouthUpperUpRight",    # 48
    "noseSneerLeft",        # 49
    "noseSneerRight",       # 50
    "tongueOut",            # 51
]

# Build a lookup: blendshape name -> packet index
_ARKIT_NAME_TO_IDX = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}

# Bone indices
BONE_HIPS = 0
BONE_SPINE = 1
BONE_CHEST = 2
BONE_NECK = 3
BONE_HEAD = 4
BONE_LEFT_SHOULDER = 5
BONE_LEFT_UPPER_ARM = 6
BONE_LEFT_LOWER_ARM = 7
BONE_LEFT_HAND = 8
BONE_RIGHT_SHOULDER = 9
BONE_RIGHT_UPPER_ARM = 10
BONE_RIGHT_LOWER_ARM = 11
BONE_RIGHT_HAND = 12
BONE_LEFT_UPPER_LEG = 13
BONE_LEFT_LOWER_LEG = 14
BONE_RIGHT_UPPER_LEG = 15
BONE_RIGHT_LOWER_LEG = 16

# MediaPipe pose landmark indices
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

# Rest-pose directions for each bone (in Unity left-hand coords: Y-up, Z-forward)
REST_DIRECTIONS = {
    BONE_HIPS:            np.array([0.0,  1.0,  0.0]),
    BONE_SPINE:           np.array([0.0,  1.0,  0.0]),
    BONE_CHEST:           np.array([0.0,  1.0,  0.0]),
    BONE_NECK:            np.array([0.0,  1.0,  0.0]),
    BONE_HEAD:            np.array([0.0,  1.0,  0.0]),
    BONE_LEFT_SHOULDER:   np.array([-1.0, 0.0,  0.0]),
    BONE_LEFT_UPPER_ARM:  np.array([-1.0, 0.0,  0.0]),
    BONE_LEFT_LOWER_ARM:  np.array([-1.0, 0.0,  0.0]),
    BONE_LEFT_HAND:       np.array([-1.0, 0.0,  0.0]),
    BONE_RIGHT_SHOULDER:  np.array([1.0,  0.0,  0.0]),
    BONE_RIGHT_UPPER_ARM: np.array([1.0,  0.0,  0.0]),
    BONE_RIGHT_LOWER_ARM: np.array([1.0,  0.0,  0.0]),
    BONE_RIGHT_HAND:      np.array([1.0,  0.0,  0.0]),
    BONE_LEFT_UPPER_LEG:  np.array([0.0, -1.0,  0.0]),
    BONE_LEFT_LOWER_LEG:  np.array([0.0, -1.0,  0.0]),
    BONE_RIGHT_UPPER_LEG: np.array([0.0, -1.0,  0.0]),
    BONE_RIGHT_LOWER_LEG: np.array([0.0, -1.0,  0.0]),
}


# --- Math helpers ---

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    return v / n


def quat_from_two_vectors(v_from, v_to):
    """Quaternion that rotates v_from to v_to. Returns (x, y, z, w)."""
    v_from = normalize(v_from)
    v_to = normalize(v_to)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)

    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.999999:
        ortho = np.cross(np.array([1.0, 0.0, 0.0]), v_from)
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(np.array([0.0, 1.0, 0.0]), v_from)
        ortho = normalize(ortho)
        return np.array([ortho[0], ortho[1], ortho[2], 0.0])

    axis = np.cross(v_from, v_to)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w])
    return q / np.linalg.norm(q)


def quat_inverse(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def mp_to_unity(landmark):
    """
    Convert MediaPipe world coord to Unity left-hand coord system.
    MediaPipe: X = subject's left, Y = down, Z = toward camera
    Unity:     X = character's right, Y = up, Z = forward (away from camera)
    Negate all three axes to get correct mapping.
    """
    return np.array([-landmark.x, -landmark.y, -landmark.z])


def midpoint_lm(lm, idx_a, idx_b):
    a = mp_to_unity(lm[idx_a])
    b = mp_to_unity(lm[idx_b])
    return (a + b) * 0.5


# --- Bone rotation computation ---

def compute_bone_rotations(pose_world_landmarks):
    """
    Compute 17 bone local quaternions from MediaPipe pose world landmarks.

    Uses direct rotation computation per bone rather than parent-chain propagation,
    which avoids error accumulation and gives cleaner results.

    Returns (list of 17 (x,y,z,w) quaternions, hip_position).
    """
    lm = pose_world_landmarks

    hip_center = midpoint_lm(lm, MP_LEFT_HIP, MP_RIGHT_HIP)
    shoulder_center = midpoint_lm(lm, MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER)
    left_shoulder = mp_to_unity(lm[MP_LEFT_SHOULDER])
    right_shoulder = mp_to_unity(lm[MP_RIGHT_SHOULDER])
    left_elbow = mp_to_unity(lm[MP_LEFT_ELBOW])
    right_elbow = mp_to_unity(lm[MP_RIGHT_ELBOW])
    left_wrist = mp_to_unity(lm[MP_LEFT_WRIST])
    right_wrist = mp_to_unity(lm[MP_RIGHT_WRIST])
    left_hip = mp_to_unity(lm[MP_LEFT_HIP])
    right_hip = mp_to_unity(lm[MP_RIGHT_HIP])
    left_knee = mp_to_unity(lm[MP_LEFT_KNEE])
    right_knee = mp_to_unity(lm[MP_RIGHT_KNEE])
    left_ankle = mp_to_unity(lm[MP_LEFT_ANKLE])
    right_ankle = mp_to_unity(lm[MP_RIGHT_ANKLE])
    nose = mp_to_unity(lm[MP_NOSE])

    torso_up = normalize(shoulder_center - hip_center)
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    rots = [identity.copy() for _ in range(NUM_BONES)]

    # --- Spine chain ---
    # Hips: torso tilt relative to world up
    rots[BONE_HIPS] = quat_from_two_vectors(np.array([0, 1, 0]), torso_up)

    # Spine/Chest: identity (follow hips)
    rots[BONE_SPINE] = identity.copy()
    rots[BONE_CHEST] = identity.copy()

    # --- Neck and Head ---
    # Use nose position relative to shoulder center to derive pitch/yaw.
    # The nose naturally sits forward, so we subtract that baseline.
    # We compute angles as offsets from a neutral standing pose.
    nose_offset = nose - shoulder_center

    # Project onto torso's local axes
    # torso_right = direction from right shoulder to left shoulder (Unity: left is -X)
    torso_right = normalize(right_shoulder - left_shoulder)
    torso_forward = normalize(np.cross(torso_up, torso_right))

    # Nose position in torso-local space
    nose_local_x = np.dot(nose_offset, torso_right)    # left-right
    nose_local_y = np.dot(nose_offset, torso_up)        # up-down
    nose_local_z = np.dot(nose_offset, torso_forward)   # forward-back

    # Head yaw: nose left-right offset relative to forward distance
    head_yaw = np.arctan2(nose_local_x, max(abs(nose_local_z), 0.05))
    # Head pitch: nose up-down relative to expected height
    # Measured neutral: nose is ~0.15 above shoulders in torso-local space
    neutral_nose_height = 0.15
    pitch_offset = nose_local_y - neutral_nose_height
    head_pitch = np.arctan2(pitch_offset, max(abs(nose_local_z), 0.05))

    # Split full rotation between neck (40%) and head (60%)
    # No attenuation — pass through full detected angles
    neck_pitch = head_pitch * 0.4
    neck_yaw = head_yaw * 0.4
    head_pitch_local = head_pitch * 0.6
    head_yaw_local = head_yaw * 0.6

    # Also compute head roll from ear tilt
    left_ear = mp_to_unity(lm[7])
    right_ear = mp_to_unity(lm[8])
    ear_delta = left_ear - right_ear
    # Project ear delta onto torso's up and right to get roll angle
    ear_right = np.dot(ear_delta, torso_right)
    ear_up = np.dot(ear_delta, torso_up)
    head_roll = np.arctan2(ear_up, max(abs(ear_right), 0.01))
    # Baseline: ears are roughly level, so roll is ~0 when head is straight
    # Subtract baseline (ears are on a horizontal line → roll ≈ 0)
    head_roll = -(head_roll - 0.0)  # negate so tilting right = positive roll

    neck_roll = head_roll * 0.4
    head_roll_local = head_roll * 0.6

    # Build neck quaternion (pitch X, yaw Y, roll Z)
    cp, sp = np.cos(neck_pitch/2), np.sin(neck_pitch/2)
    cy, sy = np.cos(neck_yaw/2), np.sin(neck_yaw/2)
    cr, sr = np.cos(neck_roll/2), np.sin(neck_roll/2)
    # Combine: yaw * pitch * roll (Y * X * Z order)
    rots[BONE_NECK] = quat_multiply(
        np.array([0, sy, 0, cy]),
        quat_multiply(np.array([sp, 0, 0, cp]), np.array([0, 0, sr, cr]))
    )

    # Build head quaternion
    cp, sp = np.cos(head_pitch_local/2), np.sin(head_pitch_local/2)
    cy, sy = np.cos(head_yaw_local/2), np.sin(head_yaw_local/2)
    cr, sr = np.cos(head_roll_local/2), np.sin(head_roll_local/2)
    rots[BONE_HEAD] = quat_multiply(
        np.array([0, sy, 0, cy]),
        quat_multiply(np.array([sp, 0, 0, cp]), np.array([0, 0, sr, cr]))
    )

    # --- Arms: compute directly, no parent chain ---
    # Each arm bone rotation = how much it deviates from its rest direction

    # Left arm
    l_upper_dir = normalize(left_elbow - left_shoulder)
    l_lower_dir = normalize(left_wrist - left_elbow)
    rots[BONE_LEFT_SHOULDER] = identity.copy()
    rots[BONE_LEFT_UPPER_ARM] = quat_from_two_vectors(
        REST_DIRECTIONS[BONE_LEFT_UPPER_ARM], l_upper_dir)
    # Lower arm: rotation relative to upper arm direction
    # In T-pose rest, lower arm continues in the same direction as upper arm
    rots[BONE_LEFT_LOWER_ARM] = quat_from_two_vectors(l_upper_dir, l_lower_dir)
    rots[BONE_LEFT_HAND] = identity.copy()

    # Right arm
    r_upper_dir = normalize(right_elbow - right_shoulder)
    r_lower_dir = normalize(right_wrist - right_elbow)
    rots[BONE_RIGHT_SHOULDER] = identity.copy()
    rots[BONE_RIGHT_UPPER_ARM] = quat_from_two_vectors(
        REST_DIRECTIONS[BONE_RIGHT_UPPER_ARM], r_upper_dir)
    # Lower arm relative to upper
    rots[BONE_RIGHT_LOWER_ARM] = quat_from_two_vectors(r_upper_dir, r_lower_dir)
    rots[BONE_RIGHT_HAND] = identity.copy()

    # --- Legs: same direct approach ---
    l_upper_leg_dir = normalize(left_knee - left_hip)
    l_lower_leg_dir = normalize(left_ankle - left_knee)
    rots[BONE_LEFT_UPPER_LEG] = quat_from_two_vectors(
        REST_DIRECTIONS[BONE_LEFT_UPPER_LEG], l_upper_leg_dir)
    rots[BONE_LEFT_LOWER_LEG] = quat_from_two_vectors(l_upper_leg_dir, l_lower_leg_dir)

    r_upper_leg_dir = normalize(right_knee - right_hip)
    r_lower_leg_dir = normalize(right_ankle - right_knee)
    rots[BONE_RIGHT_UPPER_LEG] = quat_from_two_vectors(
        REST_DIRECTIONS[BONE_RIGHT_UPPER_LEG], r_upper_leg_dir)
    rots[BONE_RIGHT_LOWER_LEG] = quat_from_two_vectors(r_upper_leg_dir, r_lower_leg_dir)

    return rots, hip_center


# --- Packet building ---

def build_packet(sequence, blendshapes, bone_rotations, hip_position,
                 face_confidence, body_confidence):
    """Build a 512-byte binary UDP packet."""
    timestamp = time.time() % 100000.0

    data = struct.pack('<I', MAGIC)
    data += struct.pack('<I', sequence)
    data += struct.pack('<f', timestamp)

    for i in range(NUM_BLENDSHAPES):
        val = blendshapes[i] if i < len(blendshapes) else 0.0
        data += struct.pack('<f', float(val))

    for i in range(NUM_BONES):
        q = bone_rotations[i] if i < len(bone_rotations) else np.array([0, 0, 0, 1])
        data += struct.pack('<ffff', float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    data += struct.pack('<fff', float(hip_position[0]), float(hip_position[1]), float(hip_position[2]))
    data += struct.pack('<ff', float(face_confidence), float(body_confidence))

    assert len(data) == PACKET_SIZE, f"Packet size mismatch: {len(data)} != {PACKET_SIZE}"
    return data


# --- Blendshape extraction from FaceLandmarker result ---

def extract_blendshapes(face_result):
    """
    Extract 52 ARKit blendshapes from FaceLandmarker result.
    FaceLandmarker directly outputs ARKit-compatible blendshape scores.
    """
    bs = [0.0] * NUM_BLENDSHAPES

    if not face_result.face_blendshapes or len(face_result.face_blendshapes) == 0:
        return bs

    # face_blendshapes[0] is the first (and usually only) face
    for category in face_result.face_blendshapes[0]:
        name = category.category_name
        # MediaPipe uses underscore format like "browDownLeft" or "_neutral"
        # Skip the _neutral category
        if name.startswith("_"):
            continue
        idx = _ARKIT_NAME_TO_IDX.get(name)
        if idx is not None:
            bs[idx] = category.score

    return bs


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="MediaPipe tracker for VLive-Arena")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--port", type=int, default=11574, help="UDP port to send to")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="UDP target IP")
    parser.add_argument("--no-body", action="store_true", help="Disable body tracking")
    parser.add_argument("--no-face", action="store_true", help="Disable face tracking")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--show-camera", action="store_true", help="Show camera preview with landmarks")
    args = parser.parse_args()

    print(f"MediaPipe Tracker starting...")
    print(f"  Camera: {args.camera}")
    print(f"  Target: {args.ip}:{args.port}")
    print(f"  Body: {'disabled' if args.no_body else 'enabled'}")
    print(f"  Face: {'disabled' if args.no_face else 'enabled'}")

    # Model paths (relative to this script's directory)
    models_dir = os.path.join(_script_dir, "models")
    face_model_path = os.path.join(models_dir, "face_landmarker.task")
    pose_model_path = os.path.join(models_dir, "pose_landmarker.task")

    # Initialize FaceLandmarker
    face_landmarker = None
    if not args.no_face:
        if not os.path.exists(face_model_path):
            print(f"ERROR: Face model not found: {face_model_path}", file=sys.stderr)
            print("Download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", file=sys.stderr)
            sys.exit(1)

        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        face_landmarker = FaceLandmarker.create_from_options(face_options)
        print("  Face landmarker loaded.")

    # Initialize PoseLandmarker
    pose_landmarker = None
    if not args.no_body:
        if not os.path.exists(pose_model_path):
            print(f"ERROR: Pose model not found: {pose_model_path}", file=sys.stderr)
            print("Download it from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task", file=sys.stderr)
            sys.exit(1)

        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        print("  Pose landmarker loaded.")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.ip, args.port)

    sequence = 0
    fps_counter = 0
    fps_timer = time.time()
    frame_timestamp_ms = 0

    print("Tracking started. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame", file=sys.stderr)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms += 33  # ~30fps stepping

            blendshapes = [0.0] * NUM_BLENDSHAPES
            bone_rotations = [np.array([0, 0, 0, 1], dtype=np.float64)] * NUM_BONES
            hip_position = np.array([0.0, 0.0, 0.0])
            face_confidence = 0.0
            body_confidence = 0.0

            # --- Face tracking ---
            if face_landmarker is not None:
                face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                    face_confidence = 1.0
                    blendshapes = extract_blendshapes(face_result)

            # --- Body tracking ---
            if pose_landmarker is not None:
                pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                if pose_result.pose_world_landmarks and len(pose_result.pose_world_landmarks) > 0:
                    body_confidence = 1.0
                    world_lm = pose_result.pose_world_landmarks[0]
                    bone_rotations, hip_position = compute_bone_rotations(world_lm)

            # Camera preview with landmarks
            if args.show_camera:
                preview = frame.copy()
                h, w = preview.shape[:2]

                # Draw pose landmarks
                if pose_landmarker is not None and pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    plm = pose_result.pose_landmarks[0]
                    # Draw connections
                    connections = [
                        (11,13),(13,15),(12,14),(14,16),  # arms
                        (11,12),(11,23),(12,24),(23,24),  # torso
                        (23,25),(25,27),(24,26),(26,28),  # legs
                        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),  # face outline
                    ]
                    for a, b in connections:
                        if a < len(plm) and b < len(plm):
                            x1, y1 = int(plm[a].x * w), int(plm[a].y * h)
                            x2, y2 = int(plm[b].x * w), int(plm[b].y * h)
                            cv2.line(preview, (x1,y1), (x2,y2), (0,255,0), 2)
                    for i, lm in enumerate(plm):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(preview, (cx, cy), 3, (0,0,255), -1)

                # Draw face mesh dots
                if face_landmarker is not None and face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                    flm = face_result.face_landmarks[0]
                    for lm in flm:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(preview, (cx, cy), 1, (255,200,0), -1)

                # Status text
                cv2.putText(preview, f"Face: {face_confidence:.0f} Body: {body_confidence:.0f}",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow("MediaPipe Tracker", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Build and send packet
            packet = build_packet(
                sequence, blendshapes, bone_rotations, hip_position,
                face_confidence, body_confidence
            )
            sock.sendto(packet, target)
            sequence += 1

            # FPS reporting
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 5.0:
                fps = fps_counter / elapsed
                print(f"FPS: {fps:.1f} | Face: {face_confidence:.1f} | Body: {body_confidence:.1f}")
                fps_counter = 0
                fps_timer = time.time()

    except KeyboardInterrupt:
        print("\nStopping tracker...")
    finally:
        cap.release()
        sock.close()
        if face_landmarker is not None:
            face_landmarker.close()
        if pose_landmarker is not None:
            pose_landmarker.close()
        if args.show_camera:
            cv2.destroyAllWindows()
        print("Tracker stopped.")


if __name__ == "__main__":
    main()
